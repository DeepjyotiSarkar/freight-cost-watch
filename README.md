# Freight Cost Watch

An assistant that tracks cost per tonne-km for every route, week by week, flags rises that look
out of the ordinary, and only calls a rise "justified" when a context note genuinely covers that
route and that week. Everything that decides a verdict is deterministic code. The LLM does the
parts that need language - reading the free-text notes into checkable facts, and answering questions
about the findings - and every LLM output is validated before it is used.

```
python run.py                 # full run -> output/cost_watch_output.csv
python -m pytest eval -q      # 23 unit tests (core numbers, output contract, guardrails)
python eval/eval_harness.py   # grounding audit, planted anomalies, labelled weeks -> eval/eval_report.md
python ask.py "why did Chennai-Bangalore get pricier in March 2025?"
```

## Results at a glance

`output/cost_watch_output.csv` has all 728 route-weeks (7 routes x 104 weeks) in the exact
8-column contract. `output/flagged_weeks.csv` is the 19-row subset that is not plain "No".

| Route | Weeks | Verdict | Why |
|---|---|---|---|
| Ahmedabad-Mumbai | 20 Jan 2025 | No (justified), N002 | festival-week surcharge on this corridor, that week |
| Chennai-Bangalore | 24 Feb, 3 Mar 2025 | No (justified), N001 | flood, 24 Feb - 8 Mar |
| Chennai-Bangalore | 10 Mar, 17 Mar 2025 | **Yes** | still ~+34% vs peers, but N001 ends 8 Mar ("passable again by Mar 9") and N009 only says the route is back to normal |
| Delhi-Jaipur | 11 Nov, 18 Nov 2024 | **Yes** | +35% / +28% spike, no note for this route or time |
| Mumbai-Delhi | 12 May 2025 | No (justified), N003 | diesel +5-7% from 5 May; observed +11.6%, the 4.6-point remainder is within weekly noise |
| Mumbai-Pune | Sep - Dec 2025 (10 weeks) | **Yes** | slow creep: 12-28 points above its usual gap to Delhi-Jaipur; N006 (stable demand) and N010 (cost absorbed, no rate change) rejected |
| Mumbai-Delhi | 22 Dec 2025 | **Yes** | borderline: 12.2 points above its usual peer gap while only +3.2% vs own history |

All three rows in `sample_output_format_v2.csv` are reproduced exactly (same numbers, same verdicts).

## How it works

```
shipments.csv ─► weekly cost/tonne-km ─► 8-week own baseline ─► flag rule ─┐
                 (Mon-Sun weeks)          + same-week peer baseline         │ anomalous weeks only
                                                                            ▼
context_notes.csv ─► LLM reads each note ONCE ─► code validates ─► hybrid retrieval ─► eligibility rules ─► verdict
                     (effect, dates, size)       (fail closed)     (route+time filter,   (scope, effect,
                                                                    then embeddings)      window, size)
                                                                                           │
                                     plain-English reason (quotes the note + window, or the notes ─► CSV
                                     checked and why each does not count)
```

### 1. Core numbers (`src/costwatch/metrics.py`)
- **Week**: `week_of` = Monday of the shipment date; weeks run Mon-Sun.
- **Cost per tonne-km** for a route-week = total freight cost / total (tonnes x km). This
  shipment-weighted ratio is what reproduces the sample rows; the mean of per-shipment ratios does not.
- **vs own history**: mean of the route's previous 8 *calendar* weekly values, current week excluded.
  Every route has all 104 weeks, so this is always 8 weeks except at the start: week 1 has no
  history (`n/a (no prior weeks)`, never flagged) and weeks 2-8 use the 1-7 weeks available. Those
  rows say so in `reason`. Nothing is padded or extrapolated.
- **vs similar routes**: simple mean of the other same-`route_type` routes' values that week, the
  route itself excluded. Short and Long have only two routes each, so the peer is a single route.

### 2. How a week gets flagged (`src/costwatch/flags.py`)
A week is anomalous when cost is **rising** and **unusual**:
- rising: >= +3% vs its own 8-week average (about 1 sd of normal weekly noise), and
- unusual: >= +10% vs its own 8-week average (~3.5 sd), **or** a gap vs peers at least 12 points
  above that route's *usual* gap (~3.4 sd).

The "usual gap" matters. Raw peer gaps are not comparable across routes: Mumbai-Pune (150 km) sits
about 11% above Delhi-Jaipur (280 km) in every calm week, because fixed costs spread over fewer km.
A raw "+20% vs peers" rule would treat that as normal on one route and miss a real jump on the other.
Comparing each route to its own usual gap (median of its past gaps, past weeks only) is also what
catches the Mumbai-Pune creep: the 8-week own average climbs along with a slow creep, so the
own-history signal stays small (+4-9%), while the gap to peers keeps widening.

Thresholds were set from measured noise in the calm Mar-Oct 2024 period
(`python eval/calibrate_thresholds.py`) and then frozen. The sweep in the eval report shows the
labelled results hold for most nearby settings.

### 3. Reading the notes (`src/costwatch/notes.py`)
The LLM reads each of the 10 notes once and returns JSON: does it raise costs, when (single week /
explicit range / lasting change from a date), by how much, and a supporting quote. Code then:
- validates every field (enums, dates near the note date, plausible ranges, and the quote must be
  near-verbatim note text, so a paraphrased or invented "evidence" fails);
- cross-checks with a simple rule: if the note text says costs were *not* affected / absorbed /
  normal and the LLM still says "increase", the note is excluded;
- resolves the window in code. A lasting change (diesel) can explain a rise vs the 8-week average only
  for 8 weeks; after that the average has caught up, so N003 covers 5 May - 29 Jun 2025.

Anything that fails is **fail closed**: the note can be mentioned, never used as justification.
Results are cached by note content + prompt version, so re-runs cost zero tokens unless a note changes.

### 4. Matching a note to a flagged week (`retrieval.py`, `justify.py`)
Retrieval is hybrid: route scope and time proximity are hard filters on the extracted facts, then
`bge-small-en-v1.5` embeddings (local, ONNX) rank what is left. A note justifies a flagged week only if
1. it applies to this route or to all routes;
2. it says costs went up, for routes in this dataset;
3. its window overlaps this Mon-Sun week;
4. it can account for what was seen: a network-wide note hit the peers too, so it cannot explain a
   gap vs peers; and if it states a size (5-7%), the rise left after subtracting it must no longer be
   an anomaly by our own rule.

No note passes -> `Yes`, blank `matched_note_id`, and the reason names the closest notes checked and
why each does not count. We never guess.

### 5. Explanations (`explain.py`)
Each reason is written by code from the verdict's own evidence: for a justified week it quotes the
note, its date and window, and the rise seen; for an unexplained week it says which comparison made
it unusual and names the closest notes checked and why each does not count (e.g. "N001 ... does not
count because its window (24 Feb 2025 to 08 Mar 2025) does not cover the week of 10 Mar 2025").
The note text and its supporting quote come from the LLM's validated reading.

I also built and tested an LLM rewrite of these reasons (`--llm-reasons`, off by default): the
LLM rewords the draft and a validator rejects any rewrite that adds a note id, number, date, quote
or cause, drops a note id or window, or asserts a cause for an unexplained week. On a full run the
validator caught 1 of 19 (a moved date), but reading the 18 it accepted, 9 still misstated something
a pattern check cannot see - e.g. "similar routes, which are 13.2 points higher than usual", which
inverts who is above normal. Verdicts were unaffected, but a reader would be misled, so the shipped
reasons are code-written. Details: [`runs/experiments/README.md`](runs/experiments/README.md).

### 6. Ask questions (`ask.py`, stretch goal)
Picks route(s) and months/years out of the question by string matching and hands the matching
result rows to the LLM. Its summary is shown only if every note id, percentage and cause in it is in
those rows **and** it does not leave out part of the story (unexplained weeks, or the note behind a
justified week). Under every answer, code lists the flagged weeks it was based on.

```
$ python ask.py "what happened on Delhi-Jaipur in November 2024?"
In November 2024, there was an unexplained rise in the cost of freight from Delhi to Jaipur ...
Evidence (from the results, listed by code):
- Delhi-Jaipur, week of 2024-11-11: Yes. Unexplained, flagged for review. Cost is +35.5% ...
- Delhi-Jaipur, week of 2024-11-18: Yes. Unexplained, flagged for review. Cost is +27.6% ...
```
For "why did Chennai-Bangalore get pricier in March 2025?" the model's summary credited the flood
for all of March, skipping the two unexplained weeks after 8 Mar; the completeness check rejected
it and only the evidence list was shown. Q&A calls are logged in `runs/ask_log.jsonl`.

## How I convinced myself it is right

`python eval/eval_harness.py` writes [`eval/eval_report.md`](eval/eval_report.md). Current result:

> grounding failures: **0** | trap spikes: **9/9** | random spikes caught: **39/40** | labelled weeks: **20/20** | note extraction: **10/10**

1. **Independent re-implementation**: `eval/test_core.py` recomputes every number with plain loops and
   dicts straight from the brief's wording and matches the pipeline on all 728 rows to 1e-9; it also
   checks the sample rows, Monday bucketing, no look-ahead, and peer self-exclusion.
2. **Grounding audit** (the check the graders said they would run): every justified row is re-checked
   against *hand-labelled* note facts in `eval/gold_note_facts.json`, not against the LLM's reading.
3. **Planted anomalies with known answers**: +25% spikes injected on top of every misleading note
   (N005 "costs not affected", N008, N009, N010, N004, the week after the festival, a diesel-window
   spike far bigger than 5-7%, the flood weeks on a different route) all come back unexplained; extra
   cost inside the real flood window stays justified. 40 random +20% one-off spikes in calm weeks:
   39 caught; the miss was a week that sat well below its average before the spike.
4. **Hand-labelled weeks** (`eval/gold_labels.csv`): every noted episode, the traps, and calm
   controls. I wrote these myself, so I treat them as a sanity check, not an unbiased test.
5. **Guardrail unit tests** (`eval/test_guardrails.py`): wrong route, wrong window, "costs not
   affected", an LLM saying "increase" against the note text, network-wide note vs peer gap, oversized
   rise vs stated size, invented evidence, and (for `--llm-reasons`) a rewrite that adds a cause,
   moves a date, invents a note or number, or drops the checked note.

## Reproducibility

`python eval/repro_check.py --runs 3` runs the full pipeline three times **with fresh LLM calls**
(`--refresh-llm`: the note cache is ignored and every note is re-read by the LLM) and diffs the outputs.
Result in [`runs/repro_report.md`](runs/repro_report.md): all three output files are **byte-identical** (SHA-256 `0afbad7629c85ec3...`), and the LLM's readings of all 10 notes were identical field by field.

The LLM runs at temperature 0 with a fixed seed (42), i.e. greedy decoding; nothing else is random
(embeddings are a fixed ONNX model on CPU). The one place an LLM could move a verdict is its reading
of a note, so that reading is validated field by field and then cached: a normal re-run reuses it,
and a different model's reading is scored against the hand labels by the eval before you trust it.
With `--llm-reasons` the wording of flagged reasons may vary between runs; verdicts and note ids
cannot, because code sets them.

## Cost of running it

One full fresh run over the whole shipment file (from `runs/repro/run1/run_log.json`;
also in [`runs/token_cost_log.md`](runs/token_cost_log.md)):

| purpose | LLM calls | input tokens | output tokens |
|---|---|---|---|
| note reading | 10 | 7,706 | 754 |
| **total** | **10** | **7,706** | **754** |

Model: **Qwen2.5-3B-Instruct** (open weights), Q4_K_M GGUF, served locally on CPU with
llama.cpp's OpenAI-compatible server - the same weights as Ollama's `qwen2.5:3b-instruct`.
Cost at that setup: **$0**. For reference, the same tokens at Groq's published rate for Llama 3.1 8B
Instant ($0.05 / M input, $0.08 / M output) would be about **$0.00045** per run.
LLM calls scale with the number of notes, not shipments: one reading per note, cached, so a repeat
run with unchanged notes makes **zero** LLM calls. The experimental `--llm-reasons` mode adds one call
per anomalous week (19 here; 29 calls, 11.6k input / 2.7k output tokens in total for that run).

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# A) local, free, open-source (default)
ollama pull qwen2.5:3b-instruct
python run.py

# B) Groq free tier (or any OpenAI-compatible endpoint)
export LLM_BASE_URL=https://api.groq.com/openai/v1 LLM_MODEL=llama-3.1-8b-instant LLM_API_KEY=...
export LLM_USD_PER_M_INPUT=0.05 LLM_USD_PER_M_OUTPUT=0.08
python run.py

# C) no LLM at all: committed note cache, identical output
python run.py --no-llm
```
`--refresh-llm` re-reads the notes with whatever model is configured (the eval then scores that
model's reading against the hand labels). The first run downloads the ~65 MB embedding model.
If the LLM endpoint is unreachable, the run warns and continues from the note cache.

## Trade-offs and limitations
- **Thresholds are judgment calls.** They are tied to measured noise and the sweep shows the labelled
  results are stable, but a team should tune them to how many alerts they want to review.
- **Single-peer comparisons are noisy.** Short and Long have one peer each, so one odd peer moves the
  "similar routes" baseline. Mumbai-Delhi 22 Dec 2025 is flagged mainly because Delhi-Chennai never
  picked up the May diesel rise; its reason says the unusual part is the gap to similar routes,
  with own cost only +3.2% vs its 8-week average.
- **Strict windows over generous ones.** Chennai-Bangalore stayed expensive for two weeks after the
  note's stated end. A human might connect that to the flood; the brief says a wrong window does not
  count and a confident wrong reason is worse than none, so those weeks stay unexplained and visible.
- **Ongoing changes**: a lasting rise (diesel) is treated as explaining rises only while the 8-week
  baseline catches up. A permanent shift in a route's peer gap is not re-based automatically.
- **Retrieval at this size** is not where correctness comes from: with 10 notes, the filters and the
  eligibility rules do the work. The eval shows pure embedding search would rank the diesel note only
  3rd for Mumbai-Delhi, because embeddings do not know dates. The design is there for 10k notes.
- **Rule-based cross-check** for "no impact" phrases is a cheap second opinion, not complete. It only
  ever removes notes from use, so its failure mode is "unexplained", never "fabricated".

What I would do next: per-transporter breakdown of flagged weeks (is one carrier driving the creep?),
alerts on episodes rather than single weeks, and a human-confirmed note-facts table in production.

## Repo layout
```
run.py / ask.py                 entry points
src/costwatch/  config.py       every threshold and model setting
                metrics.py      weekly cost/tonne-km, own and peer baselines
                flags.py        anomaly rule
                notes.py        LLM note reading + validation + window rules
                retrieval.py    embeddings + hybrid search
                justify.py      eligibility rules -> verdict
                explain.py      reasons (code-written; optional LLM rewrite + validator)
                llm.py          OpenAI-compatible client + token ledger
                output.py       CSV contract
eval/           test_core.py, test_guardrails.py, eval_harness.py, repro_check.py,
                calibrate_thresholds.py, gold_labels.csv, gold_note_facts.json, eval_report.md
output/         cost_watch_output.csv (deliverable), flagged_weeks.csv, weekly_metrics_full.csv, run_log.json
runs/           repro/run{1,2,3}/, repro_report.md, token_cost_log.md
cache/          note_extractions.json (LLM readings of the notes)
```

Note on the sample file: row 3 of `sample_output_format_v2.csv` has an unquoted comma inside
`reason`, so it parses as 9 fields. Our CSV quotes text fields where needed to keep exactly 8 columns.
