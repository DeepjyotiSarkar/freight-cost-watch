"""Guardrail unit tests: the rules that stop a wrong or invented justification.

Run: python -m pytest eval -q
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from costwatch.config import THRESHOLDS  # noqa: E402
from costwatch.explain import validate_rewrite  # noqa: E402
from costwatch.justify import check_note  # noqa: E402
from costwatch.notes import NoteFacts, is_quote_of, resolve_window, validate  # noqa: E402


def note(**kw) -> NoteFacts:
    base = dict(note_id="N900", note_date=date(2025, 2, 24), applies_to="Chennai-Bangalore",
                text="Flooding raised trip costs from Feb 24 to Mar 8.", cost_effect="increase",
                timing="date_range", start_date=date(2025, 2, 24), end_date=date(2025, 3, 8),
                evidence="raised trip costs")
    base.update(kw)
    n = NoteFacts(**base)
    n.problems = validate(n)
    n.window_start, n.window_end = resolve_window(n, THRESHOLDS.history_weeks)
    return n


def row(route="Chennai-Bangalore", week="2025-03-03", own=20.0, peer_trigger=False) -> pd.Series:
    return pd.Series({"route": route, "week_of": pd.Timestamp(week), "vs_own_history_pct": own,
                      "peer_trigger": peer_trigger})


# ---- check_note: when a note may justify a flagged week
def test_note_in_scope_and_window_justifies():
    assert check_note(note(), row(), THRESHOLDS) is None


def test_wrong_route_rejected():
    assert "applies to" in check_note(note(), row(route="Delhi-Chennai"), THRESHOLDS)


def test_week_after_window_rejected():
    assert "window" in check_note(note(), row(week="2025-03-10"), THRESHOLDS)


def test_note_saying_costs_unaffected_rejected():
    n = note(cost_effect="no_increase", text="Maintenance caused delays; costs were not significantly affected.",
             evidence="costs were not significantly affected")
    assert "no cost increase" in check_note(n, row(), THRESHOLDS)


def test_llm_saying_increase_against_text_is_overruled():
    n = note(text="Delays were reported, but costs were not significantly affected.", evidence="Delays were reported")
    assert n.problems and not n.usable_for_justification
    assert check_note(n, row(), THRESHOLDS) is not None


def test_network_wide_note_cannot_explain_gap_vs_peers():
    n = note(applies_to="All Routes", timing="ongoing_from_date", end_date=None)
    assert "all routes" in check_note(n, row(peer_trigger=True), THRESHOLDS)


def test_magnitude_residual_rejects_oversized_rise():
    n = note(applies_to="All Routes", timing="ongoing_from_date", end_date=None,
             magnitude_low_pct=5.0, magnitude_high_pct=7.0)
    assert check_note(n, row(own=11.6), THRESHOLDS) is None          # 4.6 pts left: noise
    assert "accounts for at most 7%" in check_note(n, row(own=30.0), THRESHOLDS)  # 23 pts left


def test_invented_evidence_fails_validation():
    n = note(evidence="diesel prices doubled")
    assert "evidence is not a quote from the note" in n.problems


def test_quote_check_tolerates_inflection_not_paraphrase():
    t = "Diesel prices rose nationwide starting this week, pushing up transportation costs across all routes by roughly 5-7%."
    assert is_quote_of("pushed up transportation costs across all routes by roughly 5-7%", t)
    assert not is_quote_of("fuel became more expensive everywhere", t)


# ---- validate_rewrite: the LLM may reword, never add or drop facts
DRAFT = ("Cost is +22.1% vs its own 8-week average. Notes checked: N001 (2025-02-24) does not count because "
         "its window (24 Feb 2025 to 08 Mar 2025) does not cover the week of 10 Mar 2025.")


def test_faithful_rewrite_accepted():
    ok = ("This week's cost is 22.1% above its 8-week average. N001 (2025-02-24) doesn't apply: its window, "
          "24 Feb 2025 to 8 Mar 2025, does not cover the week of 10 Mar 2025.")
    assert validate_rewrite(ok, DRAFT, "Yes") == []


def test_rewrite_adding_a_cause_rejected():
    bad = ("Cost is 22.1% above average, likely due to diesel prices. N001 (2025-02-24) window 24 Feb 2025 to "
           "08 Mar 2025 does not cover 10 Mar 2025.")
    probs = validate_rewrite(bad, DRAFT, "Yes")
    assert any("diesel" in p for p in probs) and any("asserts a cause" in p for p in probs)


def test_rewrite_moving_a_date_rejected():
    bad = "Cost is +22.1%. N001's window, 24 Feb 2025 to 10 Mar 2025, does not cover this week."
    assert any("window" in p for p in validate_rewrite(bad, DRAFT, "Yes"))


def test_rewrite_inventing_a_note_or_number_rejected():
    bad = ("Cost is +25.0% vs average. N001 and N004 do not apply; N001's window 24 Feb 2025 to 08 Mar 2025 "
           "misses 10 Mar 2025.")
    probs = validate_rewrite(bad, DRAFT, "Yes")
    assert any("N004" in p for p in probs) and any("25" in p for p in probs)


def test_rewrite_dropping_the_checked_note_rejected():
    assert any("drops note ids" in p for p in validate_rewrite("Cost is +22.1%; no explanation found.", DRAFT, "Yes"))
