"""Ask questions about the findings in plain English (stretch goal).

    python ask.py "why did Chennai-Bangalore get pricier in March 2025?"
    python ask.py "is anything unusual going on with Mumbai-Pune lately?"
    python ask.py "which routes had unexplained cost rises?"

The assistant answers ONLY from its own result rows. Route and time are picked out of the
question with plain string matching, and the matching rows are handed to the LLM. Its answer is
discarded if it mentions a note id, percentage or cause that is not in those rows, or if it leaves
out a category that is there (unexplained weeks, or the note behind a justified week). Under every
answer, the flagged weeks it was based on are listed by code, so the evidence is always complete.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from costwatch.config import OUTPUT_DIR, RUNS_DIR, LLMSettings  # noqa: E402
from costwatch.explain import CAUSE_WORDS  # noqa: E402
from costwatch.llm import LLMClient, LLMUnavailable  # noqa: E402

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
MAX_ROWS = 25

SYSTEM = (
    "You answer questions about freight cost alerts using ONLY the result rows provided. Each row is "
    "one route-week with its cost per tonne-km, comparison vs its own history and vs similar routes, "
    "a verdict (Yes = unexplained rise, No (justified) = explained by a note, No = normal) and a reason. "
    "Cite week dates and note IDs from the rows. Do not add causes, numbers or notes that are not in "
    "the rows. If some weeks are justified and others unexplained, say which are which. If the rows "
    "do not answer the question, say so. Answer in at most 5 sentences."
)


def pick_routes(q: str, routes: list[str]) -> list[str]:
    ql = q.lower()
    exact = [r for r in routes if r.lower() in ql]
    if exact:
        return exact
    scores = {r: sum(c.lower() in ql for c in r.split("-")) for r in routes}
    best = max(scores.values())
    return [r for r, s in scores.items() if s == best] if best else []


def pick_period(q: str, weeks: pd.Series) -> pd.Series:
    """Boolean mask over week_of for months/years named in the question (all weeks if none)."""
    ql = q.lower()
    years = [int(y) for y in re.findall(r"\b(20\d\d)\b", ql)]
    months = [i + 1 for i, m in enumerate(MONTHS) if re.search(rf"\b{m[:3]}(?:{m[3:]})?\b", ql)]
    if "late" in ql and not months:
        months = [9, 10, 11, 12]
    if ("lately" in ql or "recent" in ql) and not months and not years:
        return weeks >= weeks.max() - pd.Timedelta(weeks=12)
    mask = pd.Series(True, index=weeks.index)
    # A week belongs to a month if any of its 7 days falls in it.
    week_end = weeks + pd.Timedelta(days=6)
    if months:
        mask &= weeks.dt.month.isin(months) | week_end.dt.month.isin(months)
    if years:
        mask &= weeks.dt.year.isin(years) | week_end.dt.year.isin(years)
    return mask


def select_rows(q: str, df: pd.DataFrame) -> pd.DataFrame:
    routes = pick_routes(q, sorted(df.route.unique()))
    sel = df[pick_period(q, df.week_of)]
    if routes:
        sel = sel[sel.route.isin(routes)]
    interesting = sel[sel.flagged != "No"]
    if routes and len(sel) <= MAX_ROWS:
        return sel
    if len(interesting):
        return interesting.head(MAX_ROWS)
    step = -(-len(sel) // MAX_ROWS)  # nothing flagged: spread the sample over the whole period
    return sel.iloc[::step]


def rows_block(rows: pd.DataFrame) -> str:
    return "\n".join(
        f"- {r.route} | week of {r.week_of:%Y-%m-%d} | {r.cost_per_tonne_km:.2f} INR/tonne-km | "
        f"{r.vs_own_history_pct:+.1f}% vs own history | {r.vs_similar_pct:+.1f}% vs similar routes | "
        f"verdict: {r.flagged} {r.matched_note_id or ''} | reason: {r.reason}"
        for r in rows.itertuples()
    )


def grounded(answer: str, rows: pd.DataFrame) -> list[str]:
    problems = []
    ids = {i for i in rows.matched_note_id if i} | set(re.findall(r"\bN\d{3}\b", " ".join(rows.reason)))
    for nid in set(re.findall(r"\bN\d{3}\b", answer)):
        if nid not in ids:
            problems.append(f"note {nid} not in the rows")
    nums = [abs(x) for x in pd.concat([rows.vs_own_history_pct, rows.vs_similar_pct]).dropna()]
    nums += [float(x) for x in re.findall(r"\d+(?:\.\d+)?", " ".join(rows.reason))]
    for tok in re.findall(r"\d+(?:\.\d+)?(?=\s*%)", answer):
        x = float(tok)
        if not any(abs(x - n) <= 0.06 or (x == int(x) and x == round(n)) for n in nums):
            problems.append(f"number {tok}% not in the rows")
    text = " ".join(rows.reason).lower()
    for w in CAUSE_WORDS:
        if re.search(rf"\b{w}", answer, re.IGNORECASE) and w not in text:
            problems.append(f"cause '{w}' not in the rows")
    # Completeness: do not tell half the story.
    if (rows.flagged == "Yes").any() and not re.search(r"unexplained|no (matching )?note|not justified", answer, re.IGNORECASE):
        problems.append("leaves out the unexplained weeks")
    for nid in {i for i in rows.matched_note_id if i}:
        if nid not in answer:
            problems.append(f"leaves out justified note {nid}")
    return problems


def evidence(rows: pd.DataFrame) -> str:
    hits = rows[rows.flagged != "No"]
    if hits.empty:
        return "Nothing was flagged for that route and period; weekly costs stayed within the normal range."
    return "\n".join(f"- {r.route}, week of {r.week_of:%Y-%m-%d}: {r.flagged}. {r.reason}" for r in hits.itertuples())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--results", type=Path, default=OUTPUT_DIR / "weekly_metrics_full.csv")
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(a.results, parse_dates=["week_of"], keep_default_na=False,
                     na_values={"vs_own_history_pct": [""], "vs_similar_pct": [""]})
    rows = select_rows(a.question, df)
    if rows.empty:
        print("I have no results matching that route or period.")
        return

    llm = None if a.no_llm else LLMClient(LLMSettings())
    answer, source = None, "rows"
    if llm and llm.ping():
        try:
            reply = " ".join(llm.chat(SYSTEM, f"Question: {a.question}\n\nRows:\n{rows_block(rows)}",
                                      purpose="qa", max_tokens=220).split())
            problems = grounded(reply, rows)
            answer, source = (reply, "llm") if not problems else (None, "rows (LLM answer rejected: " + "; ".join(problems) + ")")
        except LLMUnavailable:
            pass
    if answer:
        print(answer + "\n\nEvidence (from the results, listed by code):")
    print(evidence(rows))
    print(f"\n[answered from {len(rows)} result rows; summary source: {source}]")
    if llm:
        RUNS_DIR.mkdir(exist_ok=True)
        with open(RUNS_DIR / "ask_log.jsonl", "a") as f:
            f.write(json.dumps({"question": a.question, "source": source, **llm.ledger.summary()}) + "\n")


if __name__ == "__main__":
    main()
