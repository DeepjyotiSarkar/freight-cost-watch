"""Plain-English reasons.

Default: the reason for every week is the code-written draft below. It quotes the note and its
window, or names the notes checked and why each does not count, so it is fully auditable.

Optional (--llm-reasons, off by default): the LLM rewrites flagged weeks' drafts.
1. Code writes a fact-complete draft: verdict, numbers, the note (id, date, window, its words)
   or, for unexplained weeks, which notes were checked and why each does not count.
2. The LLM rewrites the draft body into plainer English (temperature 0).
3. The rewrite is accepted only if it adds nothing: every note id, number, date, quote and cause
   word in it must already be in the draft, it must keep every note id and date window the draft
   names, and it may not assert a cause for an unexplained week. Otherwise the draft itself is
   used, and the swap is counted in the run log.
The verdict prefix ("Justified by N002." / "Unexplained, flagged for review.") is always set by
code, so the LLM can change the wording but never the verdict.

Why it is off by default: on a full run with Qwen2.5-3B, 18/19 rewrites passed the validator,
but reading them showed 9 of the 18 still misstated something a pattern check cannot see (e.g.
"similar routes, which are 13.2 points higher than usual" - the subject inverted - or describing a
rejected note's content as if it explained the rise). See runs/experiments/.
"""

from __future__ import annotations

import re

import pandas as pd

from .config import Thresholds
from .justify import Rejection, Verdict
from .llm import LLMClient, LLMUnavailable
from .notes import is_quote_of

CAUSE_WORDS = [
    "diesel", "fuel", "toll", "flood", "rain", "monsoon", "weather", "festival", "surcharge",
    "strike", "demand", "maintenance", "accident", "congestion", "holiday", "shortage",
    "detour", "repair", "inflation", "tracking", "compliance",
]
CAUSAL_CLAIM = re.compile(
    r"(?<!not )(?<!n't )\b(due to|caused by|because of|driven by|explained by|attributable to|"
    r"the result of|resulted from)\b",
    re.IGNORECASE,
)
NOTE_ID = re.compile(r"\bN\d{3}\b")
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
MONTH_DAY = re.compile(
    r"\b(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|"
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})\b",
    re.IGNORECASE,
)
SPAN = re.compile(r"\b\d{1,2} [A-Z][a-z]{2} \d{4} to \d{1,2} [A-Z][a-z]{2} \d{4}\b")
QUOTED = re.compile(r'"([^"]{8,})"')
PCT = re.compile(r"[+-]?\d+(?:\.\d+)?(?=\s*(?:%|percent|percentage points|points?\b))")

SYSTEM = (
    "You rewrite freight cost alerts into clear, plain English for a logistics manager. "
    "Keep every note ID, date and number exactly as written in the draft. Do not add any cause, "
    "date, number or note that is not in the draft, and do not guess at reasons. "
    "Write one to three short sentences. Reply with the rewritten text only."
)


def _pct(x: float) -> str:
    return f"{x:+.1f}%"


def _d(x) -> str:
    return f"{x:%d %b %Y}"


# ------------------------------------------------------------------ deterministic draft
def _trigger_text(row: pd.Series) -> str:
    own = f"{_pct(row.vs_own_history_pct)} vs its own 8-week average"
    if row.own_trigger and not row.peer_trigger:
        return f"cost is {own}, a jump well beyond normal weekly variation"
    peer = (
        f"{_pct(row.vs_similar_pct)} vs similar {row.route_type} routes, {row.peer_excess_pct:.1f} points "
        f"above this route's usual gap of {_pct(row.usual_peer_gap_pct)}"
    )
    if row.own_trigger:
        return f"cost is {own} and {peer}"
    return f"cost is {peer}, while only {own}: the unusual part is the widening gap to similar routes"


def _rejection_text(r: Rejection, notes_by_id: dict) -> str:
    n = notes_by_id.get(r.note_id)
    quote = f', which says "{n.evidence}"' if n is not None and n.evidence and not n.problems else ""
    return f"{r.note_id} ({r.note_date.isoformat()}{quote}) does not count because {r.why}"


def draft(row: pd.Series, v: Verdict, t: Thresholds, notes_by_id: dict) -> tuple[str, str]:
    """Return (prefix, body). prefix carries the verdict; body carries the evidence."""
    if v.flagged == "No":
        if pd.isna(row.own_history_avg):
            return "Not flagged:", "first week in the data, so there is no history to compare against."
        thin = (
            f" (8-week baseline uses only {row.own_history_weeks} prior week{'s' if row.own_history_weeks != 1 else ''})"
            if row.own_history_weeks < t.history_weeks else ""
        )
        gap = f", usual gap {_pct(row.usual_peer_gap_pct)}" if pd.notna(row.usual_peer_gap_pct) else ""
        return "Not flagged:", (
            f"{_pct(row.vs_own_history_pct)} vs its own past average{thin} and "
            f"{_pct(row.vs_similar_pct)} vs similar routes{gap}; within normal range."
        )

    if v.flagged == "No (justified)":
        m = v.matched
        body = (
            f"Note {m.note_id} (dated {m.note_date.isoformat()}, applies to {m.applies_to}) says: \"{m.text}\" "
            f"Its window, {_d(m.window_start)} to {_d(m.window_end)}, covers the week of {_d(row.week_of)}, "
            f"when cost was {_pct(row.vs_own_history_pct)} vs this route's past average."
        )
        if m.magnitude_high_pct is not None:
            body += (
                f" The note sizes the effect at {m.magnitude_low_pct:g}-{m.magnitude_high_pct:g}%; the remaining "
                f"{row.vs_own_history_pct - m.magnitude_high_pct:.1f} points are within normal weekly variation."
            )
        return f"Justified by {m.note_id}.", body

    body = _trigger_text(row)[0].upper() + _trigger_text(row)[1:] + "."
    if v.rejections:
        body += " Notes checked: " + "; ".join(_rejection_text(r, notes_by_id) for r in v.rejections[:2]) + "."
    else:
        body += " No note covers this route and week."
    return "Unexplained, flagged for review.", body


def template_reason(row: pd.Series, v: Verdict, t: Thresholds, notes_by_id: dict) -> str:
    prefix, body = draft(row, v, t, notes_by_id)
    return f"{prefix} {body}"


# ------------------------------------------------------------------ validation of the rewrite
def _numbers(text: str) -> list[float]:
    return [abs(float(x)) for x in PCT.findall(text)] + [float(x) for x in re.findall(r"(?<![\d.])\d+\.\d+(?![\d%])", text)]


def _month_days(text: str) -> set[tuple[str, int]]:
    out = set()
    for m in MONTH_DAY.finditer(text):
        day, mon = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
        out.add((mon[:3].lower(), int(day)))
    for iso in ISO_DATE.findall(text):
        y, mo, dd = iso.split("-")
        out.add((["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"][int(mo) - 1], int(dd)))
    return out


def _norm_dates(s: str) -> str:
    return re.sub(r"\b0(\d)\b", r"\1", " ".join(s.split())).lower()


def validate_rewrite(rewrite: str, draft_body: str, verdict: str) -> list[str]:
    """Problems with an LLM rewrite of draft_body. Empty list = adds nothing, drops nothing key."""
    problems = []
    # Date windows must survive intact: a model can otherwise move a start date onto another
    # date that happens to appear in the draft (seen in testing: "05 May" -> "12 May").
    for span in SPAN.findall(draft_body):
        if _norm_dates(span) not in _norm_dates(rewrite):
            problems.append(f"drops or alters the window '{span}'")
    # Anything the rewrite puts in quotes must be a quote from the draft, not a paraphrase.
    for q in QUOTED.findall(rewrite):
        if not is_quote_of(q, draft_body):
            problems.append("misquotes the note")
    if not rewrite or len(rewrite) > 700:
        problems.append("empty or too long")
    ids_d, ids_r = set(NOTE_ID.findall(draft_body)), set(NOTE_ID.findall(rewrite))
    if ids_r - ids_d:
        problems.append(f"adds note ids {sorted(ids_r - ids_d)}")
    if ids_d - ids_r:
        problems.append(f"drops note ids {sorted(ids_d - ids_r)}")
    allowed = _numbers(draft_body)
    for x in _numbers(rewrite):
        if not any(abs(x - n) <= 0.051 or (x == int(x) and x == round(n)) for n in allowed):
            problems.append(f"number {x:g} is not in the draft")
    extra_dates = _month_days(rewrite) - _month_days(draft_body)
    if extra_dates:
        problems.append(f"dates not in the draft: {sorted(extra_dates)}")
    for w in CAUSE_WORDS:
        if re.search(rf"\b{w}", rewrite, re.IGNORECASE) and not re.search(rf"\b{w}", draft_body, re.IGNORECASE):
            problems.append(f"adds cause '{w}'")
    if verdict == "Yes" and CAUSAL_CLAIM.search(rewrite):
        problems.append("asserts a cause for an unexplained week")
    if verdict == "Yes" and "no note covers" in draft_body.lower() and "no note" not in rewrite.lower():
        problems.append("drops 'no note covers this week'")
    if re.search(r"\bjustified\b", rewrite, re.IGNORECASE) and verdict == "Yes" and not re.search(r"\bnot justified\b", rewrite, re.IGNORECASE):
        problems.append("calls an unexplained week justified")
    if verdict == "No (justified)" and re.search(r"\bunexplained\b", rewrite, re.IGNORECASE):
        problems.append("calls a justified week unexplained")
    return problems


def build_reason(
    row: pd.Series, v: Verdict, t: Thresholds, llm: LLMClient | None, notes_by_id: dict
) -> tuple[str, str]:
    """Return (reason, source) where source is 'template', 'llm' or 'template_fallback:<why>'."""
    prefix, body = draft(row, v, t, notes_by_id)
    if v.flagged == "No" or llm is None:
        return f"{prefix} {body}", "template"
    try:
        rewrite = llm.chat(SYSTEM, f"Draft:\n{body}", purpose="explanation", max_tokens=160)
    except LLMUnavailable as e:
        return f"{prefix} {body}", f"template_fallback:llm_unavailable:{e}"
    rewrite = " ".join(rewrite.split()).strip('"')
    problems = validate_rewrite(rewrite, body, v.flagged)
    if problems:
        return f"{prefix} {body}", "template_fallback:" + " | ".join(problems)
    return f"{prefix} {rewrite}", "llm"
