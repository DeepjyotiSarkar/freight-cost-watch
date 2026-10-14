"""End-to-end run: numbers -> flags -> note matching -> reasons -> CSV + run log."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import CACHE_DIR, LLMSettings, Thresholds
from .explain import build_reason
from .flags import add_flags
from .justify import judge_week
from .llm import LLMClient
from .metrics import build_weekly_table
from .notes import extract_note_facts, load_notes
from .output import to_output_frame, write_csv
from .retrieval import NoteIndex


def run(
    shipments_path: Path,
    notes_path: Path,
    out_dir: Path,
    thresholds: Thresholds,
    llm_settings: LLMSettings | None,
    refresh_llm: bool = False,
    llm_reasons: bool = False,
    cache_path: Path = CACHE_DIR / "note_extractions.json",
) -> dict:
    """llm_reasons=False (default): reasons are written by code from the verdict's evidence.
    llm_reasons=True: the LLM rewrites each flagged week's reason under a validator (experimental,
    see README "Explanations" for why it is off by default)."""
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Numbers and flags (deterministic).
    weekly = add_flags(build_weekly_table(shipments_path, thresholds.history_weeks), thresholds)

    # 2. Notes -> validated facts (LLM once per note, cached).
    llm = None
    if llm_settings is not None:
        llm = LLMClient(llm_settings)
        if not llm.ping():
            print(f"[warn] LLM at {llm_settings.base_url} not reachable; using cached note facts.",
                  file=sys.stderr)
            llm = None
    notes = extract_note_facts(load_notes(notes_path), llm, cache_path, thresholds.history_weeks, refresh_llm)
    notes_by_id = {n.note_id: n for n in notes}
    index = NoteIndex(notes, CACHE_DIR / "fastembed")

    # 3. Verdict + reason per route-week.
    flagged, note_ids, reasons, sources, audit = [], [], [], [], []
    for row in weekly.itertuples(index=False):
        row = pd.Series(row._asdict())
        v = judge_week(row, index, thresholds)
        reason, source = build_reason(row, v, thresholds, llm if llm_reasons else None, notes_by_id)
        flagged.append(v.flagged)
        note_ids.append(v.matched.note_id if v.matched else "")
        reasons.append(reason)
        sources.append(source)
        if row.anomaly:
            audit.append({
                "route": row.route, "week_of": row.week_of.date().isoformat(), "verdict": v.flagged,
                "matched_note_id": v.matched.note_id if v.matched else "",
                "own_trigger": bool(row.own_trigger), "peer_trigger": bool(row.peer_trigger),
                "retrieved_note_ids": v.retrieved_ids,
                "rejected": [{"note_id": r.note_id, "why": r.why} for r in v.rejections],
                "reason_source": source,
            })

    results = weekly.assign(flagged=flagged, matched_note_id=note_ids, reason=reasons, reason_source=sources)
    out = to_output_frame(results)
    write_csv(out, out_dir / "cost_watch_output.csv")
    write_csv(out[out["flagged"] != "No"], out_dir / "flagged_weeks.csv")
    results.to_csv(out_dir / "weekly_metrics_full.csv", index=False)

    src = pd.Series(sources)
    log = {
        "rows": len(out),
        "anomalies": int(weekly["anomaly"].sum()),
        "flagged_yes": int((out["flagged"] == "Yes").sum()),
        "justified": int((out["flagged"] == "No (justified)").sum()),
        "thresholds": asdict(thresholds),
        "llm_reasons": llm_reasons,
        "llm": llm.ledger.to_json() if llm else None,
        "explanations": {
            "llm": int((src == "llm").sum()),
            "template_fallback": int(src.str.startswith("template_fallback").sum()),
            "template": int((src == "template").sum()),
        },
        "note_facts": [n.to_dict() for n in notes],
        "anomaly_audit": audit,
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "run_log.json").write_text(json.dumps(log, indent=2, default=str))
    return log
