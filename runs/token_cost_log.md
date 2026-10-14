# Token / cost log - one full run

Source: `runs/repro/run1/run_log.json` (fresh run: note cache ignored, every note re-read by the LLM).
Input: 728 route-weeks from the full shipment file; 19 anomalous weeks (15 unexplained, 4 justified).

| purpose | LLM calls | input tokens | output tokens |
|---|---|---|---|
| note_extraction | 10 | 7,706 | 754 |
| **total** | **10** | **7,706** | **754** |

- Model: `qwen2.5-3b-instruct` (Qwen2.5-3B-Instruct Q4_K_M, open weights) on a local llama.cpp server, CPU only.
- Cost at that setup: **$0.00** (no API). Wall time inside the LLM: 244 s on 2 CPU cores.
- Same tokens at Groq Llama 3.1 8B Instant's published rate ($0.05/M in, $0.08/M out): **$0.00045**.
- Reasons: written by code for all 728 rows (no LLM calls).
- A repeat run with the note cache in place makes zero LLM calls.
- For comparison, the experimental `--llm-reasons` run (runs/experiments/) used 29 calls, 11,588 input / 2,680 output tokens.
