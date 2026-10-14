"""Decide whether a flagged week is justified by a note. Deterministic rules over extracted facts.

A note justifies a flagged route-week only if ALL of these hold:
  1. scope     : it applies to this route, or to all routes;
  2. effect    : its (validated) facts say it raised costs, for routes in this dataset;
  3. window    : its time window overlaps this Monday-Sunday week;
  4. size/kind : it can account for what was actually seen -
                 - a network-wide note (All Routes) hit the peer routes too, so it cannot
                   explain a gap vs similar routes;
                 - if the note states a size (e.g. 5-7%), what is left after subtracting it
                   must no longer be an anomaly by our own flag rule.
If no note passes, the week stays flagged as unexplained. We never guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from .config import Thresholds
from .notes import ALL_ROUTES, NoteFacts
from .retrieval import Candidate, NoteIndex, query_for


@dataclass
class Rejection:
    note_id: str
    note_date: date
    why: str


@dataclass
class Verdict:
    flagged: str  # "Yes" | "No (justified)" | "No"
    matched: NoteFacts | None = None
    rejections: list[Rejection] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)


def _fmt_window(n: NoteFacts) -> str:
    if n.window_start == n.window_end:
        return f"{n.window_start:%d %b %Y}"
    return f"{n.window_start:%d %b %Y} to {n.window_end:%d %b %Y}"


def check_note(n: NoteFacts, row: pd.Series, t: Thresholds) -> str | None:
    """Return None if the note justifies this flagged week, else a plain-English reason why not."""
    week_of = row.week_of.date()
    week_end = week_of + pd.Timedelta(days=6)
    if not n.applies_to_route(row.route):
        return f"applies to {n.applies_to}, not {row.route}"
    if not n.affects_routes_in_dataset:
        return "it concerns routes that are not in this dataset"
    if n.cost_effect != "increase":
        return "it reports no cost increase"
    if n.problems:
        return "its extracted facts failed validation, so it is not used as evidence"
    if not (n.window_start <= week_end and n.window_end >= week_of):
        return f"its window ({_fmt_window(n)}) does not cover the week of {week_of:%d %b %Y}"
    if n.applies_to == ALL_ROUTES and row.peer_trigger:
        return (
            "it affected all routes, so it cannot explain why this route moved away from similar routes"
        )
    if n.magnitude_high_pct is not None:
        residual = row.vs_own_history_pct - n.magnitude_high_pct
        if residual >= t.own_history_pct:
            return (
                f"it accounts for at most {n.magnitude_high_pct:g}%, leaving {residual:.1f} points "
                f"of the {row.vs_own_history_pct:.1f}% rise unexplained"
            )
    return None


def judge_week(row: pd.Series, index: NoteIndex, t: Thresholds) -> Verdict:
    if not row.anomaly:
        return Verdict(flagged="No")

    candidates: list[Candidate] = index.candidates_for_week(row.route, row.week_of.date(), query_for(row.route, row.week_of.date()))
    accepted, rejections = [], []
    for c in candidates:
        why = check_note(c.note, row, t)
        if why is None:
            accepted.append(c)
        else:
            rejections.append(Rejection(c.note.note_id, c.note.note_date, why))

    retrieved = [c.note.note_id for c in candidates]
    if accepted:
        # Prefer a route-specific note over a network-wide one, then the closer semantic match.
        best = sorted(accepted, key=lambda c: (c.note.applies_to == ALL_ROUTES, -c.semantic_score))[0]
        return Verdict("No (justified)", best.note, rejections, retrieved)
    # Order rejections by time proximity so the explanation mentions the most relevant one first.
    gap = {c.note.note_id: c.days_from_window for c in candidates}
    rejections.sort(key=lambda r: (gap[r.note_id], r.note_id))
    return Verdict("Yes", None, rejections, retrieved)
