"""Core numbers: weekly cost per tonne-km and the two baselines defined in the brief.

Everything here is plain pandas and fully deterministic. No LLM touches these numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "shipment_id", "origin", "destination", "route_type", "quantity_tonnes",
    "distance_km", "freight_cost_inr", "shipment_date",
}


def load_shipments(path: Path) -> pd.DataFrame:
    """Read and validate shipments; add route, week_of (Monday) and tonne_km."""
    df = pd.read_csv(path, parse_dates=["shipment_date"])
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"shipment file is missing columns: {sorted(missing)}")
    if df["shipment_id"].duplicated().any():
        raise ValueError("duplicate shipment_id values found")
    bad = (df["quantity_tonnes"] <= 0) | (df["distance_km"] <= 0) | (df["freight_cost_inr"] <= 0)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} rows have non-positive quantity, distance or cost")
    if df.groupby(["origin", "destination"])["route_type"].nunique().max() > 1:
        raise ValueError("a route appears with more than one route_type")

    df["route"] = df["origin"] + "-" + df["destination"]
    # Weeks run Monday-Sunday; week_of is that Monday.
    df["week_of"] = df["shipment_date"] - pd.to_timedelta(df["shipment_date"].dt.weekday, unit="D")
    df["tonne_km"] = df["quantity_tonnes"] * df["distance_km"]
    return df


def weekly_costs(shipments: pd.DataFrame) -> pd.DataFrame:
    """One row per (route, week). Cost per tonne-km = total cost / total tonne-km.

    This is the shipment-weighted ratio (sum of costs over sum of tonne-km), which is
    what "total freight cost / (quantity x distance)" means for a group of shipments.
    It reproduces the three rows in sample_output_format_v2.csv exactly; the mean of
    per-shipment ratios does not.
    """
    weekly = (
        shipments.groupby(["route", "route_type", "week_of"], as_index=False)
        .agg(
            n_shipments=("shipment_id", "size"),
            freight_cost_inr=("freight_cost_inr", "sum"),
            tonne_km=("tonne_km", "sum"),
        )
        .sort_values(["route", "week_of"], ignore_index=True)
    )
    weekly["cost_per_tonne_km"] = weekly["freight_cost_inr"] / weekly["tonne_km"]
    return weekly


def add_own_history_baseline(weekly: pd.DataFrame, history_weeks: int = 8) -> pd.DataFrame:
    """Trailing average of the route's weekly cost/tonne-km over the previous N calendar weeks.

    Uses only weeks strictly before the current week (no look-ahead). The window is
    calendar-based, so a week with no shipments is simply absent, never padded.
    With fewer than N prior weeks, all available prior weeks are used.
    """
    window = pd.Timedelta(weeks=history_weeks)
    out = []
    for _, g in weekly.groupby("route", sort=True):
        s = g.set_index("week_of")["cost_per_tonne_km"]
        # closed="left": interval [t - 8w, t) -> the 8 Mondays before t, excluding t itself.
        roll = s.rolling(window, closed="left")
        g = g.copy()
        g["own_history_avg"] = roll.mean().to_numpy()
        g["own_history_weeks"] = roll.count().fillna(0).astype(int).to_numpy()
        out.append(g)
    weekly = pd.concat(out, ignore_index=True)
    weekly["vs_own_history_pct"] = (weekly["cost_per_tonne_km"] / weekly["own_history_avg"] - 1) * 100
    return weekly


def add_peer_baseline(weekly: pd.DataFrame) -> pd.DataFrame:
    """Average cost/tonne-km in the same week across the *other* routes of the same route_type.

    A simple mean of the other routes' weekly values (each route counts once), with the
    route itself excluded. This also matches the sample rows exactly.
    """
    grp = weekly.groupby(["route_type", "week_of"])["cost_per_tonne_km"]
    total, count = grp.transform("sum"), grp.transform("count")
    peers = count - 1
    weekly["peer_routes_n"] = peers.astype(int)
    weekly["peer_avg"] = np.where(peers > 0, (total - weekly["cost_per_tonne_km"]) / peers.where(peers > 0), np.nan)
    weekly["vs_similar_pct"] = (weekly["cost_per_tonne_km"] / weekly["peer_avg"] - 1) * 100
    return weekly


def build_weekly_table(shipments_path: Path, history_weeks: int = 8) -> pd.DataFrame:
    weekly = weekly_costs(load_shipments(shipments_path))
    weekly = add_own_history_baseline(weekly, history_weeks)
    weekly = add_peer_baseline(weekly)
    return weekly.sort_values(["route", "week_of"], ignore_index=True)
