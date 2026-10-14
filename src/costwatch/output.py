"""Write results in the exact column contract of sample_output_format_v2.csv."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

OUTPUT_COLUMNS = [
    "route", "week_of", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes",
    "flagged", "matched_note_id", "reason",
]


def own_history_text(pct: float) -> str:
    if pd.isna(pct):
        return "n/a (no prior weeks)"
    return f"{pct:+.1f}% vs this route's past average"


def similar_routes_text(pct: float) -> str:
    if pd.isna(pct):
        return "n/a (no other routes of this type)"
    return f"{pct:+.1f}% vs similar-length routes this week"


def to_output_frame(results: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "route": results["route"],
        "week_of": results["week_of"].dt.strftime("%Y-%m-%d"),
        "cost_per_tonne_km": results["cost_per_tonne_km"].round(2).map(lambda x: f"{x:.2f}"),
        "vs_own_history": results["vs_own_history_pct"].map(own_history_text),
        "vs_similar_routes": results["vs_similar_pct"].map(similar_routes_text),
        "flagged": results["flagged"],
        "matched_note_id": results["matched_note_id"].fillna(""),
        "reason": results["reason"],
    })
    return out[OUTPUT_COLUMNS]


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Text fields are quoted when needed, so commas inside `reason` never shift columns.

    (Row 3 of the provided sample has an unquoted comma in `reason`, which makes it parse as
    9 fields; we keep the 8-column contract by quoting.)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
