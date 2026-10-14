"""Write the token/cost log for one full run.

    python eval/token_log.py runs/repro/run1/run_log.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Published reference rate (USD per 1M tokens), checked 2026-10-14: Groq, Llama 3.1 8B Instant.
REFERENCE = ("Groq Llama 3.1 8B Instant", 0.05, 0.08)


def main(path: Path) -> str:
    log = json.loads(path.read_text())
    s = log["llm"]["summary"]
    name, rin, rout = REFERENCE
    ref_cost = s["input_tokens"] / 1e6 * rin + s["output_tokens"] / 1e6 * rout
    rows = [
        "| purpose | LLM calls | input tokens | output tokens |",
        "|---|---|---|---|",
    ]
    for purpose, p in s["by_purpose"].items():
        rows.append(f"| {purpose} | {p['calls']} | {p['input_tokens']:,} | {p['output_tokens']:,} |")
    rows.append(f"| **total** | **{s['llm_calls']}** | **{s['input_tokens']:,}** | **{s['output_tokens']:,}** |")
    text = "\n".join([
        "# Token / cost log - one full run",
        "",
        f"Source: `{path.relative_to(ROOT)}` (fresh run: note cache ignored, every note re-read by the LLM).",
        f"Input: {log['rows']} route-weeks from the full shipment file; {log['anomalies']} anomalous weeks "
        f"({log['flagged_yes']} unexplained, {log['justified']} justified).",
        "",
        *rows,
        "",
        f"- Model: `{s['model']}` (Qwen2.5-3B-Instruct Q4_K_M, open weights) on a local llama.cpp server, CPU only.",
        f"- Cost at that setup: **$0.00** (no API). Wall time inside the LLM: {s['wall_seconds_in_llm']:.0f} s on 2 CPU cores.",
        f"- Same tokens at {name}'s published rate (${rin}/M in, ${rout}/M out): **${ref_cost:.5f}**.",
        ("- Reasons: LLM rewrites - "
         f"{log['explanations']['llm']} accepted, {log['explanations']['template_fallback']} rejected by the validator."
         if log.get("llm_reasons") else
         f"- Reasons: written by code for all {log['rows']} rows (no LLM calls)."),
        "- A repeat run with the note cache in place makes zero LLM calls.",
        "- For comparison, the experimental `--llm-reasons` run (runs/experiments/) used 29 calls, "
        "11,588 input / 2,680 output tokens.",
    ])
    return text


if __name__ == "__main__":
    p = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "runs" / "repro" / "run1" / "run_log.json"
    out = main(p)
    (ROOT / "runs" / "token_cost_log.md").write_text(out + "\n")
    print(out)
