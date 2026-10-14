"""Core-logic tests. Run: python -m pytest eval -q

The main check is an independent, loop-based re-implementation of the brief's definitions
(plain Python, no pandas groupby/rolling) compared against the pipeline on all 728 rows.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from costwatch.config import THRESHOLDS  # noqa: E402
from costwatch.flags import add_flags  # noqa: E402
from costwatch.metrics import build_weekly_table, load_shipments, weekly_costs  # noqa: E402
from costwatch.notes import NoteFacts, resolve_window  # noqa: E402
from costwatch.output import OUTPUT_COLUMNS  # noqa: E402

SHIPMENTS = ROOT / "data" / "shipment_records.csv"


@pytest.fixture(scope="module")
def weekly() -> pd.DataFrame:
    return build_weekly_table(SHIPMENTS)


def naive_reference() -> dict[tuple[str, date], dict]:
    """Brief's definitions, written as literally as possible with dicts and loops."""
    cost, tkm, rtype = defaultdict(float), defaultdict(float), {}
    with open(SHIPMENTS) as f:
        for r in csv.DictReader(f):
            d = datetime.strptime(r["shipment_date"], "%Y-%m-%d").date()
            monday = d - timedelta(days=d.weekday())
            key = (f'{r["origin"]}-{r["destination"]}', monday)
            cost[key] += float(r["freight_cost_inr"])
            tkm[key] += float(r["quantity_tonnes"]) * float(r["distance_km"])
            rtype[key[0]] = r["route_type"]
    cpt = {k: cost[k] / tkm[k] for k in cost}
    out = {}
    for (route, wk), v in cpt.items():
        prior = [cpt[(route, wk - timedelta(weeks=i))] for i in range(1, 9) if (route, wk - timedelta(weeks=i)) in cpt]
        peers = [c for (r2, w2), c in cpt.items() if w2 == wk and r2 != route and rtype[r2] == rtype[route]]
        out[(route, wk)] = {
            "cpt": v,
            "own": (v / (sum(prior) / len(prior)) - 1) * 100 if prior else None,
            "peer": (v / (sum(peers) / len(peers)) - 1) * 100 if peers else None,
            "prior_n": len(prior),
        }
    return out


def test_matches_independent_reimplementation(weekly):
    ref = naive_reference()
    assert len(ref) == len(weekly) == 728
    for r in weekly.itertuples():
        e = ref[(r.route, r.week_of.date())]
        assert r.cost_per_tonne_km == pytest.approx(e["cpt"], rel=1e-12)
        assert r.own_history_weeks == e["prior_n"]
        if e["own"] is None:
            assert pd.isna(r.vs_own_history_pct)
        else:
            assert r.vs_own_history_pct == pytest.approx(e["own"], abs=1e-9)
        assert r.vs_similar_pct == pytest.approx(e["peer"], abs=1e-9)


def test_sample_rows_reproduced_exactly(weekly):
    with open(ROOT / "data" / "sample_output_format_v2.csv") as f:
        rows = list(csv.reader(f))[1:]
    for r in rows:
        w = weekly[(weekly.route == r[0]) & (weekly.week_of == r[1])].iloc[0]
        assert f"{w.cost_per_tonne_km:.2f}" == r[2]
        assert r[3].startswith(f"{w.vs_own_history_pct:+.1f}%")
        assert r[4].startswith(f"{w.vs_similar_pct:+.1f}%")


def test_weeks_start_on_monday(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text(
        "shipment_id,origin,destination,route_type,material,quantity_tonnes,distance_km,freight_cost_inr,shipment_date,transporter\n"
        "A,X,Y,Short,M,1,1,10,2025-01-05,T\n"   # Sunday -> week of Mon 2024-12-30
        "B,X,Y,Short,M,1,1,10,2025-01-06,T\n"   # Monday -> week of 2025-01-06
    )
    s = load_shipments(p)
    assert list(s.week_of.dt.date) == [date(2024, 12, 30), date(2025, 1, 6)]


def test_weekly_cost_is_total_cost_over_total_tonne_km(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text(
        "shipment_id,origin,destination,route_type,material,quantity_tonnes,distance_km,freight_cost_inr,shipment_date,transporter\n"
        "A,X,Y,Short,M,10,100,1000,2025-01-06,T\n"   # 1.0 per tkm
        "B,X,Y,Short,M,1,100,500,2025-01-07,T\n"     # 5.0 per tkm
    )
    w = weekly_costs(load_shipments(p))
    assert w.cost_per_tonne_km.iloc[0] == pytest.approx(1500 / 1100)  # not the mean of 1.0 and 5.0


def test_no_look_ahead(weekly):
    """Changing a later week must not change any earlier week's baselines."""
    df = pd.read_csv(SHIPMENTS)
    df.loc[df.shipment_date >= "2025-06-01", "freight_cost_inr"] *= 3
    tmp = ROOT / "eval" / "_tmp_shipments.csv"
    df.to_csv(tmp, index=False)
    try:
        changed = build_weekly_table(tmp)
    finally:
        tmp.unlink()
    before = weekly.week_of < "2025-05-26"
    pd.testing.assert_series_equal(
        weekly.loc[before, "vs_own_history_pct"], changed.loc[before, "vs_own_history_pct"]
    )


def test_peer_average_excludes_route_itself(weekly):
    w = weekly[weekly.week_of == "2025-01-20"].set_index("route")
    # Medium routes: Ahmedabad-Mumbai, Chennai-Bangalore, Kolkata-Bhubaneswar
    expected = (w.loc["Chennai-Bangalore", "cost_per_tonne_km"] + w.loc["Kolkata-Bhubaneswar", "cost_per_tonne_km"]) / 2
    assert w.loc["Ahmedabad-Mumbai", "peer_avg"] == pytest.approx(expected)


def test_flags_never_fire_without_a_rise(weekly):
    f = add_flags(weekly, THRESHOLDS)
    assert not (f.anomaly & (f.vs_own_history_pct < THRESHOLDS.min_rise_pct)).any()


def test_ongoing_note_window_spans_history_length():
    n = NoteFacts("X", date(2025, 5, 5), "All Routes", "t", timing="ongoing_from_date", start_date=date(2025, 5, 5))
    assert resolve_window(n, 8) == (date(2025, 5, 5), date(2025, 6, 29))


def test_output_contract():
    out = ROOT / "output" / "cost_watch_output.csv"
    if not out.exists():
        pytest.skip("run `python run.py` first")
    with open(out) as f:
        rows = list(csv.reader(f))
    assert rows[0] == OUTPUT_COLUMNS
    assert all(len(r) == 8 for r in rows)
    for r in rows[1:]:
        assert r[5] in {"Yes", "No (justified)", "No"}
        assert (r[6] != "") == (r[5] == "No (justified)")
        if r[6]:
            assert r[6] in r[7], "justified reason must name its note"
