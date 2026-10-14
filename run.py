"""Run the cost watcher.

    python run.py                      # LLM at $LLM_BASE_URL (default: local Ollama)
    python run.py --no-llm             # no LLM: cached note facts + template explanations
    python run.py --refresh-llm        # ignore the note cache and re-read every note with the LLM
    python run.py --llm-reasons        # experimental: LLM rewrites flagged-week reasons (validated)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from costwatch.config import DATA_DIR, OUTPUT_DIR, THRESHOLDS, LLMSettings  # noqa: E402
from costwatch.pipeline import run  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shipments", type=Path, default=DATA_DIR / "shipment_records.csv")
    p.add_argument("--notes", type=Path, default=DATA_DIR / "context_notes.csv")
    p.add_argument("--out", type=Path, default=OUTPUT_DIR)
    p.add_argument("--no-llm", action="store_true", help="skip the LLM entirely")
    p.add_argument("--refresh-llm", action="store_true", help="re-extract note facts instead of using the cache")
    p.add_argument("--llm-reasons", action="store_true", help="experimental: LLM rewrites flagged-week reasons")
    a = p.parse_args()

    log = run(a.shipments, a.notes, a.out, THRESHOLDS, None if a.no_llm else LLMSettings(), a.refresh_llm,
              a.llm_reasons)
    llm = (log["llm"] or {}).get("summary", {})
    print(json.dumps({
        "rows": log["rows"], "flagged_yes": log["flagged_yes"], "justified": log["justified"],
        "llm_calls": llm.get("llm_calls", 0), "input_tokens": llm.get("input_tokens", 0),
        "output_tokens": llm.get("output_tokens", 0), "explanations": log["explanations"],
        "output": str(a.out / "cost_watch_output.csv"),
    }, indent=2))


if __name__ == "__main__":
    main()
