# Experiment: letting the LLM write the reasons (`--llm-reasons`)

`llm_rewritten_reasons/` is one full fresh run with `python run.py --refresh-llm --llm-reasons`
(Qwen2.5-3B-Instruct, temperature 0): 29 LLM calls, 11,588 input / 2,680 output tokens.

In this mode code writes a fact-complete draft for each flagged week, the LLM rewrites it, and a
validator rejects any rewrite that adds a note id, number, date, quote or cause, drops a note id or
date window, or asserts a cause for an unexplained week.

**What the validator caught:** 1 of 19. Mumbai-Delhi, week of 12 May 2025: the rewrite moved the
diesel note's window start from 5 May to 12 May (both dates were in the draft, so only the
window-span check caught it) and paraphrased the note inside quotation marks. The code draft was used.

**What it did not catch:** reading the 18 accepted rewrites by hand, 9 still misstate something:
- 8 Mumbai-Pune weeks say cost is up vs "similar routes, which are 13.2 points higher than usual" -
  the subject is inverted; it is this route that is above its usual gap, not the peers.
- 2 weeks (Mumbai-Delhi 22 Dec, Mumbai-Pune 8 Dec 2025) describe N010's "costs absorbed without a
  rate change" as if it were part of what happened on the route, when the verdict says N010 was
  rejected because it reports no cost increase.

None of this touched a verdict or a note id (those are set by code; `matched_note_id`, `flagged` and
all numbers in this run are identical to the default output). But a reader would be misled, and the
brief scores trustworthiness of explanations, so the default is the code-written reason. The LLM is
still used where language understanding is actually needed: reading the notes and answering questions.
A larger model or an LLM-as-judge consistency pass could make this mode the default later; it would
have to pass the same manual review first.
