"""Reproducibility check: N untouched runs on the same input, then diff.

    python eval/repro_check.py --runs 3              # runs the pipeline 3x (fresh LLM calls each time)
    python eval/repro_check.py --compare-only DIR...  # just compare existing run folders

"Fresh" means --refresh-llm: every run re-reads every note with the LLM instead of using the
cache, so this tests the LLM's stability, not just the cache.
Pass criterion (from the brief): flags, numbers, verdicts and cited note ids identical.
Explanation wording may differ and is reported, not failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MUST_MATCH = ["route", "week_of", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes", "flagged", "matched_note_id"]


def digest(df: pd.DataFrame, cols: list[str]) -> str:
    return hashlib.sha256(df[cols].to_csv(index=False).encode()).hexdigest()[:16]


def compare(dirs: list[Path]) -> tuple[bool, str]:
    outs = [pd.read_csv(d / "cost_watch_output.csv", keep_default_na=False, dtype=str) for d in dirs]
    logs = [json.loads((d / "run_log.json").read_text()) for d in dirs]
    lines = ["# Reproducibility check", "", f"Runs compared: {', '.join(str(d.relative_to(ROOT)) for d in dirs)}", ""]
    lines += ["| run | rows | flagged Yes | justified | hash of numbers+verdicts | hash of full file | LLM calls | LLM explanations used | template fallbacks |",
              "|---|---|---|---|---|---|---|---|---|"]
    for d, o, lg in zip(dirs, outs, logs):
        llm = (lg.get("llm") or {}).get("summary", {})
        lines.append(f"| {d.name} | {len(o)} | {(o.flagged == 'Yes').sum()} | {(o.flagged == 'No (justified)').sum()} | "
                     f"`{digest(o, MUST_MATCH)}` | `{digest(o, list(o.columns))}` | {llm.get('llm_calls', 0)} | "
                     f"{lg['explanations']['llm']} | {lg['explanations']['template_fallback']} |")
    same_core = len({digest(o, MUST_MATCH) for o in outs}) == 1
    same_notes = len({json.dumps([(n["note_id"], n["usable_for_justification"], n["window_start"], n["window_end"])
                                  for n in lg["note_facts"]]) for lg in logs}) == 1
    reason_diff = int(sum((outs[0].reason != o.reason).sum() for o in outs[1:]))
    lines += ["",
              f"- Numbers, flags, verdicts and cited note ids identical across runs: **{'YES' if same_core else 'NO'}**",
              f"- LLM note readings (usable? + window) identical across runs: **{'YES' if same_notes else 'NO'}**",
              f"- `reason` cells that differ from run 1 (wording only; allowed by the brief): **{reason_diff}**", ""]
    if not same_core:
        base = outs[0][MUST_MATCH]
        for d, o in zip(dirs[1:], outs[1:]):
            diff = base.compare(o[MUST_MATCH])
            lines += [f"### Differences vs {d.name}", "```", diff.to_string(), "```"]
    return same_core and same_notes, "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--dir", type=Path, default=ROOT / "runs" / "repro")
    ap.add_argument("--compare-only", type=Path, nargs="+")
    ap.add_argument("--llm-reasons", action="store_true", help="also re-write reasons with the LLM each run")
    a = ap.parse_args()
    if a.compare_only:
        dirs = [d.resolve() for d in a.compare_only]
    else:
        dirs = []
        for i in range(1, a.runs + 1):
            d = a.dir / f"run{i}"
            cmd = [sys.executable, str(ROOT / "run.py"), "--refresh-llm", "--out", str(d)]
            subprocess.run(cmd + (["--llm-reasons"] if a.llm_reasons else []), check=True)
            dirs.append(d)
    ok, text = compare(dirs)
    (ROOT / "runs" / "repro_report.md").write_text(text)
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
