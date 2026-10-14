"""Decide which route-weeks look out of the ordinary. Deterministic; no LLM."""

from __future__ import annotations

import pandas as pd

from .config import Thresholds


def add_flags(weekly: pd.DataFrame, t: Thresholds) -> pd.DataFrame:
    """A week is an anomaly when cost is rising AND it is unusual vs own history or vs peers.

    rising        : this week's cost/tonne-km is above the route's own trailing average by more
                    than normal weekly noise (>= t.min_rise_pct).
    own trigger   : rise vs own trailing average >= t.own_history_pct.
    peer trigger  : gap vs peers exceeds the route's *usual* gap by >= t.peer_excess_pct points.
                    The usual gap is the median of the route's past weekly gaps (past weeks only).
                    Raw peer gaps are not comparable across routes: a 150 km haul is structurally
                    dearer per tonne-km than a 280 km one, so Mumbai-Pune sits ~10% above
                    Delhi-Jaipur in every calm week. Comparing against the route's own usual gap
                    is what catches slow creep that the 8-week own-history average absorbs.
    """
    weekly = weekly.sort_values(["route", "week_of"]).copy()

    by_route = weekly.groupby("route")["vs_similar_pct"]
    weekly["usual_peer_gap_pct"] = by_route.transform(
        lambda s: s.shift(1).expanding(min_periods=t.peer_baseline_min_weeks).median()
    )
    weekly["peer_excess_pct"] = weekly["vs_similar_pct"] - weekly["usual_peer_gap_pct"]

    weekly["rising"] = weekly["vs_own_history_pct"] >= t.min_rise_pct
    weekly["own_trigger"] = weekly["vs_own_history_pct"] >= t.own_history_pct
    weekly["peer_trigger"] = weekly["peer_excess_pct"] >= t.peer_excess_pct
    weekly["anomaly"] = weekly["rising"] & (weekly["own_trigger"] | weekly["peer_trigger"])
    return weekly.reset_index(drop=True)
