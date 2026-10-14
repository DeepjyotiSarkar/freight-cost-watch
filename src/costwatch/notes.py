"""Turn free-text context notes into checkable facts.

The LLM reads each note ONCE and returns structured fields (does it raise costs? from when
to when? by how much?). Every field is then validated in code, and a cheap rule-based
cross-check acts as a second opinion. Anything that fails is "fail closed": the note can
still be mentioned, but it can never be used to justify a flag.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .llm import LLMClient, LLMUnavailable, parse_json_object

PROMPT_VERSION = "note-facts-v4"
ALL_ROUTES = "All Routes"

COST_EFFECTS = {"increase", "no_increase", "decrease", "unclear"}
TIMINGS = {"single_week", "date_range", "ongoing_from_date", "unspecified"}

SYSTEM_PROMPT = """You read short logistics notes and extract facts as JSON. Be literal: use only what the note says.

Return exactly these keys:
- "cost_effect": "increase" only if the note says the event raised (or pushed up) freight or transport costs/rates. Use "no_increase" if it says costs were not affected, were absorbed, rates did not change, conditions were normal/stable/improved, or the affected routes are not in the dataset. Use "decrease" if costs fell. Otherwise "unclear".
- "timing": "single_week" if it describes one week (e.g. "this week", "a festival week", "for about a week"); "date_range" ONLY if the note names both a start date and an end date; "ongoing_from_date" if a lasting change starts on a date (e.g. a price rise "starting this week"); otherwise "unspecified".
- "start_date": YYYY-MM-DD. Use the note date unless the note names another start date. Take the year from the note date.
- "end_date": YYYY-MM-DD for "date_range" (the last affected day, taking the year from the note date), else null.
- "magnitude_low_pct", "magnitude_high_pct": numbers if the note states a percentage effect on cost (e.g. "5-7%" -> 5 and 7), else null.
- "affects_routes_in_dataset": false only if the note says the affected routes are not part of this dataset, else true.
- "evidence": the shortest exact phrase copied from the note that supports cost_effect."""

FEW_SHOT = """Example 1
Note date: 2023-04-03 | Applies to: Pune-Nashik
Note: A truckers' strike on the Pune-Nashik road from Apr 3 to Apr 10 pushed up freight rates.
{"cost_effect": "increase", "timing": "date_range", "start_date": "2023-04-03", "end_date": "2023-04-10", "magnitude_low_pct": null, "magnitude_high_pct": null, "affects_routes_in_dataset": true, "evidence": "pushed up freight rates"}

Example 2
Note date: 2023-08-14 | Applies to: All Routes
Note: Port congestion was widely reported, but road freight rates stayed unchanged.
{"cost_effect": "no_increase", "timing": "single_week", "start_date": "2023-08-14", "end_date": null, "magnitude_low_pct": null, "magnitude_high_pct": null, "affects_routes_in_dataset": true, "evidence": "road freight rates stayed unchanged"}

Example 3
Note date: 2023-10-02 | Applies to: All Routes
Note: Highway toll rates were raised by about 3% from this week.
{"cost_effect": "increase", "timing": "ongoing_from_date", "start_date": "2023-10-02", "end_date": null, "magnitude_low_pct": 3, "magnitude_high_pct": 3, "affects_routes_in_dataset": true, "evidence": "toll rates were raised by about 3%"}
"""

# Rule-based second opinion. If the note text contains one of these and the LLM still says
# "increase", the two disagree and the note is excluded from justification (fail closed).
NO_IMPACT_CUES = re.compile(
    r"not (?:significantly )?affected|without (?:a|any) rate change|no rate change|"
    r"rates? (?:stayed|remained) (?:unchanged|stable)|absorbed|no (?:major|significant) disruptions?|"
    r"remained (?:normal|stable)|returned to normal|not part of this dataset|improved road",
    re.IGNORECASE,
)


@dataclass
class NoteFacts:
    note_id: str
    note_date: date
    applies_to: str
    text: str
    # LLM-extracted fields
    cost_effect: str = "unclear"
    timing: str = "unspecified"
    start_date: date | None = None
    end_date: date | None = None
    magnitude_low_pct: float | None = None
    magnitude_high_pct: float | None = None
    affects_routes_in_dataset: bool = True
    evidence: str = ""
    # Resolved in code
    window_start: date | None = None
    window_end: date | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def usable_for_justification(self) -> bool:
        return (
            not self.problems
            and self.cost_effect == "increase"
            and self.affects_routes_in_dataset
            and self.window_start is not None
        )

    def applies_to_route(self, route: str) -> bool:
        return self.applies_to in (route, ALL_ROUTES)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, date):
                d[k] = v.isoformat()
        d["usable_for_justification"] = self.usable_for_justification
        return d


def load_notes(path: Path) -> list[NoteFacts]:
    df = pd.read_csv(path, parse_dates=["date"])
    return [
        NoteFacts(note_id=r.note_id, note_date=r.date.date(), applies_to=r.applies_to.strip(), text=r.note.strip())
        for r in df.itertuples()
    ]


def _monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def is_quote_of(evidence: str, text: str, min_overlap: float = 0.9) -> bool:
    """Near-verbatim check: >= 90% of the evidence's characters appear, in order, in the note.

    Tolerates a small model writing "pushed up" for "pushing up", but not a paraphrase.
    """
    ev, tx = _norm(evidence), _norm(text)
    if len(ev) < 8:
        return False
    if ev in tx:
        return True
    blocks = difflib.SequenceMatcher(None, ev, tx, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks if b.size >= 3) / len(ev) >= min_overlap


def _parse_date(v) -> date | None:
    if v in (None, "", "null"):
        return None
    return date.fromisoformat(str(v)[:10])


def resolve_window(n: NoteFacts, history_weeks: int) -> tuple[date | None, date | None]:
    """When does this note plausibly explain a rise? Decided by code, not the LLM.

    single_week / unspecified : the Monday-Sunday week containing the start date.
    date_range                : start to end, as stated.
    ongoing_from_date         : a lasting level change (e.g. diesel) shows up as a rise vs the
                                trailing 8-week average only until that average catches up,
                                so it can explain rises for `history_weeks` weeks from its start.
    """
    if n.start_date is None:
        return None, None
    if n.timing == "date_range" and n.end_date:
        return n.start_date, n.end_date
    if n.timing == "ongoing_from_date":
        start = _monday(n.start_date)
        return n.start_date, start + timedelta(weeks=history_weeks) - timedelta(days=1)
    start = _monday(n.start_date)
    return start, start + timedelta(days=6)


def validate(n: NoteFacts) -> list[str]:
    """Code-level checks on what the LLM returned. Any problem -> note cannot justify."""
    problems = []
    if n.cost_effect not in COST_EFFECTS:
        problems.append(f"invalid cost_effect {n.cost_effect!r}")
    if n.timing not in TIMINGS:
        problems.append(f"invalid timing {n.timing!r}")
    if n.start_date is None:
        problems.append("missing start_date")
    elif abs((n.start_date - n.note_date).days) > 14:
        problems.append(f"start_date {n.start_date} is far from note date {n.note_date}")
    if n.timing == "date_range":
        if n.end_date is None:
            problems.append("date_range without end_date")
        elif n.start_date and not (0 <= (n.end_date - n.start_date).days <= 120):
            problems.append(f"implausible range {n.start_date}..{n.end_date}")
    lo, hi = n.magnitude_low_pct, n.magnitude_high_pct
    if (lo is None) != (hi is None) or (lo is not None and not (0 < lo <= hi <= 100)):
        problems.append(f"implausible magnitude {lo}..{hi}")
    if not is_quote_of(n.evidence, n.text):
        problems.append("evidence is not a quote from the note")
    if n.cost_effect == "increase" and NO_IMPACT_CUES.search(n.text):
        problems.append("LLM says 'increase' but the note text says costs were not affected")
    return problems


def _cache_key(n: NoteFacts) -> str:
    raw = f"{PROMPT_VERSION}|{n.note_id}|{n.note_date}|{n.applies_to}|{n.text}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _user_prompt(n: NoteFacts) -> str:
    return (
        f"{FEW_SHOT}\nNow this note. Reply with the JSON object only.\n"
        f"Note date: {n.note_date.isoformat()} | Applies to: {n.applies_to}\nNote: {n.text}"
    )


def _apply_extraction(n: NoteFacts, data: dict) -> None:
    n.cost_effect = str(data.get("cost_effect", "unclear")).strip().lower()
    n.timing = str(data.get("timing", "unspecified")).strip().lower()
    # The prompt tells the model to default start_date to the note date; enforce that default here.
    n.start_date = _parse_date(data.get("start_date")) or n.note_date
    n.end_date = _parse_date(data.get("end_date"))
    lo, hi = data.get("magnitude_low_pct"), data.get("magnitude_high_pct")
    n.magnitude_low_pct = float(lo) if lo is not None else None
    n.magnitude_high_pct = float(hi) if hi is not None else None
    n.affects_routes_in_dataset = bool(data.get("affects_routes_in_dataset", True))
    n.evidence = str(data.get("evidence", "")).strip()
    # A "range" with no end date is under-specified; read it as the narrowest window
    # (the week of the start date). Narrower windows can only justify fewer weeks.
    if n.timing == "date_range" and n.end_date is None:
        n.timing = "single_week"


def extract_note_facts(
    notes: list[NoteFacts],
    llm: LLMClient | None,
    cache_path: Path,
    history_weeks: int,
    refresh: bool = False,
) -> list[NoteFacts]:
    """Fill each note's facts from cache or the LLM, then validate and resolve its window.

    The cache is keyed on prompt version + note content, so a changed note or prompt is
    re-read automatically and an unchanged one costs zero tokens on re-runs.
    """
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for n in notes:
        key = _cache_key(n)
        entry = None if refresh else cache.get(key)
        if entry is None:
            if llm is None:
                n.problems = ["no cached extraction and no LLM available"]
                continue
            try:
                reply = llm.chat(SYSTEM_PROMPT, _user_prompt(n), purpose="note_extraction",
                                 max_tokens=220, json_mode=True)
            except LLMUnavailable as e:
                n.problems = [f"LLM unavailable: {e}"]
                continue
            entry = {"note_id": n.note_id, "model": llm.s.model, "prompt_version": PROMPT_VERSION, "reply": reply}
            cache[key] = entry
        try:
            _apply_extraction(n, parse_json_object(entry["reply"]))
        except (ValueError, TypeError) as e:
            n.problems = [f"unparseable LLM reply: {e}"]
            continue
        n.problems = validate(n)
        n.window_start, n.window_end = resolve_window(n, history_weeks)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
    return notes
