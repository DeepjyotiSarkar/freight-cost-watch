# Reproducibility check

Runs compared: runs/repro/run1, runs/repro/run2, runs/repro/run3

| run | rows | flagged Yes | justified | hash of numbers+verdicts | hash of full file | LLM calls | LLM explanations used | template fallbacks |
|---|---|---|---|---|---|---|---|---|
| run1 | 728 | 15 | 4 | `c89f29c8cb205d73` | `0afbad7629c85ec3` | 10 | 0 | 0 |
| run2 | 728 | 15 | 4 | `c89f29c8cb205d73` | `0afbad7629c85ec3` | 10 | 0 | 0 |
| run3 | 728 | 15 | 4 | `c89f29c8cb205d73` | `0afbad7629c85ec3` | 10 | 0 | 0 |

- Numbers, flags, verdicts and cited note ids identical across runs: **YES**
- LLM note readings (usable? + window) identical across runs: **YES**
- `reason` cells that differ from run 1 (wording only; allowed by the brief): **0**
