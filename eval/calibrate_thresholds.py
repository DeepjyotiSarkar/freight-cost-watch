"""Where the flag thresholds come from. Run: python eval/calibrate_thresholds.py

Measures normal week-to-week noise in the calm period (Jan-Oct 2024: no cost-raising notes,
no visible episodes) and prints what 1 sd looks like, so the thresholds in config.py can be
read as "about 3 sd". Thresholds were set from these numbers, then frozen.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from costwatch.config import THRESHOLDS  # noqa: E402
from costwatch.metrics import build_weekly_table  # noqa: E402

w = build_weekly_table(ROOT / "data" / "shipment_records.csv")
calm = w[(w.week_of >= "2024-03-01") & (w.week_of < "2024-11-01")]  # skip the thin-baseline first 8 weeks
own_sd = calm.groupby("route").vs_own_history_pct.std()
gap = calm.groupby("route").vs_similar_pct
print("Calm period Mar-Oct 2024, per route:")
print(f"{'route':22} {'sd vs own 8wk avg':>18} {'usual gap vs peers':>19} {'sd of gap':>10}")
for r in own_sd.index:
    print(f"{r:22} {own_sd[r]:17.2f}% {gap.median()[r]:+18.1f}% {gap.std()[r]:9.2f}")
pooled_own = float(np.sqrt((own_sd ** 2).mean()))
pooled_gap = float(np.sqrt((gap.std() ** 2).mean()))
print(f"\npooled sd vs own history : {pooled_own:.2f}%  -> min_rise {THRESHOLDS.min_rise_pct}% ~ "
      f"{THRESHOLDS.min_rise_pct / pooled_own:.1f} sd, own threshold {THRESHOLDS.own_history_pct}% ~ "
      f"{THRESHOLDS.own_history_pct / pooled_own:.1f} sd")
print(f"pooled sd of peer gap    : {pooled_gap:.2f} pts -> peer-excess threshold {THRESHOLDS.peer_excess_pct} pts ~ "
      f"{THRESHOLDS.peer_excess_pct / pooled_gap:.1f} sd")
