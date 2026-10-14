"""How I convinced myself the output is right. Run after `python run.py`:

    python eval/eval_harness.py [--out output]

Checks, in order of how much I trust them:
  1. Grounding audit   - every justified row re-checked against HAND-labelled note facts
                         (not the LLM's extraction). Any failure = a fabricated justification.
  2. Planted anomalies - spikes injected into the real data at known weeks, including traps
                         that sit right on top of misleading notes. Known answers, unbiased.
  3. Labelled weeks    - 20 hand-labelled route-weeks (episodes, traps, calm weeks).
  4. Note extraction   - the LLM's reading of each note vs my reading.
  5. Retrieval         - would pure embedding search have found the right note on its own?
  6. Threshold sweep   - how the flag set moves if the thresholds move.
Writes eval/eval_report.md; exits non-zero if a guardrail check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from costwatch.config import CACHE_DIR, THRESHOLDS, Thresholds  # noqa: E402
from costwatch.flags import add_flags  # noqa: E402
from costwatch.justify import judge_week  # noqa: E402
from costwatch.metrics import add_own_history_baseline, add_peer_baseline, load_shipments, weekly_costs  # noqa: E402
from costwatch.notes import extract_note_facts, load_notes  # noqa: E402
from costwatch.retrieval import NoteIndex, query_for  # noqa: E402

GOLD_NOTES = json.loads((ROOT / "eval" / "gold_note_facts.json").read_text())
GOLD_LABELS = pd.read_csv(ROOT / "eval" / "gold_labels.csv", keep_default_na=False)


def verdicts_for(shipments: pd.DataFrame, index: NoteIndex, t: Thresholds) -> pd.DataFrame:
    """Deterministic part of the pipeline (numbers, flags, note matching) on a shipments frame."""
    w = weekly_costs(shipments)
    w = add_peer_baseline(add_own_history_baseline(w, t.history_weeks))
    w = add_flags(w.sort_values(["route", "week_of"], ignore_index=True), t)
    flagged, ids = [], []
    for row in w.itertuples(index=False):
        row = pd.Series(row._asdict())
        v = judge_week(row, index, t)
        flagged.append(v.flagged)
        ids.append(v.matched.note_id if v.matched else "")
    return w.assign(flagged=flagged, matched_note_id=ids)


# ---------------------------------------------------------------- 1. grounding audit
def grounding_audit(out: pd.DataFrame, notes: pd.DataFrame) -> tuple[list[str], int]:
    failures = []
    just = out[out.flagged == "No (justified)"]
    for r in just.itertuples():
        nid = r.matched_note_id
        g = GOLD_NOTES.get(nid)
        note = notes.set_index("note_id").loc[nid] if nid in set(notes.note_id) else None
        wk = date.fromisoformat(r.week_of)
        wk_end = wk + pd.Timedelta(days=6)
        if g is None or note is None:
            failures.append(f"{r.route} {r.week_of}: cites unknown note {nid!r}")
            continue
        if note.applies_to not in (r.route, "All Routes"):
            failures.append(f"{r.route} {r.week_of}: {nid} applies to {note.applies_to}")
        if not g["raises_cost"]:
            failures.append(f"{r.route} {r.week_of}: {nid} does not describe a cost increase ({g['why']})")
        elif not (date.fromisoformat(g["window_start"]) <= wk_end and date.fromisoformat(g["window_end"]) >= wk):
            failures.append(f"{r.route} {r.week_of}: week outside {nid} window {g['window_start']}..{g['window_end']}")
        if nid not in r.reason:
            failures.append(f"{r.route} {r.week_of}: reason does not name {nid}")
    for r in out[out.flagged != "No (justified)"].itertuples():
        if r.matched_note_id:
            failures.append(f"{r.route} {r.week_of}: matched_note_id set on a {r.flagged!r} row")
    return failures, len(just)


# ---------------------------------------------------------------- 2. planted anomalies
TRAPS = [
    # (route, week_of, factor, expected_flagged, expected_note, what it tests)
    ("Mumbai-Delhi", "2024-07-29", 1.25, "Yes", "", "spike in N005's maintenance week; N005 says costs were NOT affected"),
    ("Kolkata-Bhubaneswar", "2025-06-09", 1.25, "Yes", "", "spike in the week of N008, which says no disruptions"),
    ("Chennai-Bangalore", "2025-03-24", 1.25, "Yes", "", "spike right after the flood; N009 says back to normal"),
    ("Ahmedabad-Mumbai", "2025-01-27", 1.25, "Yes", "", "spike the week AFTER the festival note's week"),
    ("Delhi-Jaipur", "2025-05-12", 1.25, "Yes", "", "spike inside the diesel window, but far bigger than the note's 5-7%"),
    ("Mumbai-Pune", "2024-03-11", 1.25, "Yes", "", "spike in the week of N004 (toll on routes not in the dataset)"),
    ("Delhi-Jaipur", "2025-10-27", 1.25, "Yes", "", "spike in the week of N010 (costs absorbed, no rate change)"),
    ("Delhi-Chennai", "2025-02-24", 1.25, "Yes", "", "spike in the flood weeks, but on a different route"),
    ("Chennai-Bangalore", "2025-03-03", 1.15, "No (justified)", "N001", "extra cost inside the real flood window stays justified"),
]


def planted_tests(base: pd.DataFrame, index: NoteIndex, t: Thresholds, n_random: int = 40, seed: int = 7):
    def plant(route, week, factor):
        s = base.copy()
        s["freight_cost_inr"] = s["freight_cost_inr"].astype(float)
        m = (s.route == route) & (s.week_of == pd.Timestamp(week))
        s.loc[m, "freight_cost_inr"] = s.loc[m, "freight_cost_inr"] * factor
        return s

    trap_rows = []
    for route, week, factor, exp_flag, exp_note, what in TRAPS:
        v = verdicts_for(plant(route, week, factor), index, t)
        got = v[(v.route == route) & (v.week_of == pd.Timestamp(week))].iloc[0]
        ok = got.flagged == exp_flag and got.matched_note_id == exp_note
        trap_rows.append((route, week, f"+{(factor - 1) * 100:.0f}%", exp_flag, got.flagged, got.matched_note_id or "-", "PASS" if ok else "FAIL", what))

    # Random spikes in calm weeks: does the detector catch a +20% one-off? (one at a time)
    clean = verdicts_for(base, index, t)
    calm = clean[(clean.flagged == "No") & (clean.own_history_weeks == t.history_weeks)
                 & (clean.week_of < "2025-05-01")]  # before any noted event or creep
    rng = np.random.default_rng(seed)
    picks = calm.iloc[rng.choice(len(calm), size=n_random, replace=False)]
    caught, missed = 0, []
    for p in picks.itertuples():
        v = verdicts_for(plant(p.route, p.week_of, 1.20), index, t)
        got = v[(v.route == p.route) & (v.week_of == p.week_of)].iloc[0]
        if got.flagged == "Yes":
            caught += 1
        else:
            missed.append(f"{p.route} {p.week_of:%Y-%m-%d}: was {p.vs_own_history_pct:+.1f}% vs own history before "
                          f"the spike, {got.vs_own_history_pct:+.1f}% after (threshold {t.own_history_pct:g}%)")
    return trap_rows, caught, len(picks), missed


# ---------------------------------------------------------------- 5. retrieval
def retrieval_check(index: NoteIndex) -> list[tuple]:
    rows = []
    for g in GOLD_LABELS[GOLD_LABELS.expected_note_id != ""].itertuples():
        wk = date.fromisoformat(g.week_of)
        pure = [n.note_id for n, _ in index.search(query_for(g.route, wk))]
        hybrid = [c.note.note_id for c in index.candidates_for_week(g.route, wk, query_for(g.route, wk))]
        rows.append((g.route, g.week_of, g.expected_note_id,
                     pure.index(g.expected_note_id) + 1, ", ".join(pure[:3]),
                     (hybrid.index(g.expected_note_id) + 1) if g.expected_note_id in hybrid else "-",
                     ", ".join(hybrid)))
    return rows


def md_table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "output")
    a = ap.parse_args()

    out = pd.read_csv(a.out / "cost_watch_output.csv", keep_default_na=False)
    notes_df = pd.read_csv(ROOT / "data" / "context_notes.csv")
    notes = extract_note_facts(load_notes(ROOT / "data" / "context_notes.csv"), None,
                               CACHE_DIR / "note_extractions.json", THRESHOLDS.history_weeks)
    index = NoteIndex(notes, CACHE_DIR / "fastembed")
    base = load_shipments(ROOT / "data" / "shipment_records.csv")
    report = ["# Evaluation report", "", f"Output checked: `{a.out.name}/cost_watch_output.csv`", ""]
    hard_fail = False

    # 1
    failures, n_just = grounding_audit(out, notes_df)
    hard_fail |= bool(failures)
    report += ["## 1. Grounding audit (guardrail)", "",
               f"Justified rows: **{n_just}**. Rows failing a re-check against hand-labelled note facts "
               f"(wrong route, wrong window, note that does not raise cost, note id missing from reason): "
               f"**{len(failures)}**.", ""] + [f"- {f}" for f in failures] + [""]

    # 2
    trap_rows, caught, n_random, missed = planted_tests(base, index, THRESHOLDS)
    trap_fail = sum(r[6] == "FAIL" for r in trap_rows)
    hard_fail |= trap_fail > 0
    report += ["## 2. Planted anomalies (known answers)", "",
               "Each test multiplies every shipment's cost in one route-week, re-runs the deterministic "
               "pipeline, and checks the verdict for that week.", "",
               md_table(["route", "week", "planted", "expected", "got", "note", "result", "trap"], trap_rows), "",
               f"Random one-off +20% spikes in calm weeks (seeded, {n_random} trials): **{caught}/{n_random} caught**.", ""]
    report += [f"- missed: {m}" for m in missed] + ([""] if missed else [])

    # 3
    merged = GOLD_LABELS.merge(out, on=["route", "week_of"], how="left", suffixes=("", "_got"))
    lab_rows, lab_ok = [], 0
    for r in merged.itertuples():
        ok = r.flagged == r.expected_flagged and r.matched_note_id == r.expected_note_id
        lab_ok += ok
        lab_rows.append((r.route, r.week_of, r.expected_flagged, r.expected_note_id or "-", r.flagged,
                         r.matched_note_id or "-", "PASS" if ok else "FAIL", r.why))
    report += ["## 3. Hand-labelled route-weeks", "",
               f"**{lab_ok}/{len(merged)}** match. Labels come from reading the notes and each route's weekly "
               "series; they include every noted episode, the trap notes, and calm controls. This set was "
               "written by the same person who built the rules, so treat it as a sanity check; section 2 is "
               "the less biased test.", "",
               md_table(["route", "week", "expected", "exp note", "got", "got note", "result", "why"], lab_rows), ""]

    # 4
    ext_rows, ext_ok = [], 0
    for n in notes:
        g = GOLD_NOTES[n.note_id]
        llm_raises = n.usable_for_justification
        ok = llm_raises == g["raises_cost"]
        if ok and g["raises_cost"]:
            ok = (str(n.window_start), str(n.window_end)) == (g["window_start"], g["window_end"])
        ext_ok += ok
        window = f"{n.window_start}..{n.window_end}" if n.window_start else "-"
        ext_rows.append((n.note_id, n.cost_effect, n.timing, window,
                         "yes" if llm_raises else "no", "yes" if g["raises_cost"] else "no",
                         "; ".join(n.problems) or "-", "PASS" if ok else "FAIL"))
    report += ["## 4. Note extraction (LLM reading vs my reading)", "",
               f"**{ext_ok}/{len(notes)}** notes read correctly (same can-justify decision and, for "
               "cost-raising notes, the same window).", "",
               md_table(["note", "LLM effect", "LLM timing", "resolved window", "usable", "gold", "validation problems", "result"], ext_rows), ""]

    # 5
    report += ["## 5. Retrieval", "",
               "Rank of the correct note for each justified episode: pure embedding search over all 10 notes "
               "vs the hybrid search the pipeline uses (route + time filters, then embeddings).", "",
               md_table(["route", "week", "gold note", "pure rank", "pure top-3", "hybrid rank", "hybrid candidates"],
                        retrieval_check(index)), "",
               "Pure semantic search does not know which week a note covers, which is why time and route are "
               "hard filters and the final say is the rule-based eligibility check.", ""]

    # 6
    sweep = []
    for own in (8, 10, 12, 15):
        for peer in (10, 12, 15):
            t = replace(THRESHOLDS, own_history_pct=own, peer_excess_pct=peer)
            v = verdicts_for(base, index, t)
            m = GOLD_LABELS.merge(v.assign(week_of=v.week_of.dt.strftime("%Y-%m-%d")), on=["route", "week_of"])
            acc = int(((m.flagged == m.expected_flagged) & (m.matched_note_id == m.expected_note_id)).sum())
            mark = " (chosen)" if (own, peer) == (THRESHOLDS.own_history_pct, THRESHOLDS.peer_excess_pct) else ""
            sweep.append((f"{own}%", f"{peer} pts{mark}", int((v.flagged == "Yes").sum()),
                          int((v.flagged == "No (justified)").sum()), f"{acc}/{len(m)}"))
    report += ["## 6. Threshold sensitivity", "",
               md_table(["own-history threshold", "peer-excess threshold", "flagged Yes", "justified", "labelled weeks correct"], sweep),
               ""]

    summary = (f"grounding failures: {len(failures)} | traps: {len(trap_rows) - trap_fail}/{len(trap_rows)} | "
               f"random spikes caught: {caught}/{n_random} | labelled: {lab_ok}/{len(merged)} | "
               f"note extraction: {ext_ok}/{len(notes)}")
    report.insert(3, f"**Summary:** {summary}\n")
    (ROOT / "eval" / "eval_report.md").write_text("\n".join(report))
    print(summary)
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
