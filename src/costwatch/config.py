"""All tunable settings in one place. Nothing else in the code hard-codes a number."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
OUTPUT_DIR = ROOT / "output"
RUNS_DIR = ROOT / "runs"


@dataclass(frozen=True)
class Thresholds:
    # Baseline definitions fixed by the brief.
    history_weeks: int = 8

    # Flag rule (see README "How a week gets flagged"). Noise levels come from
    # eval/calibrate_thresholds.py on the calm Mar-Oct 2024 period.
    # "Rising" means above the route's own 8-week average by more than one typical weekly
    # wobble (pooled 1 sd = 2.9%), not merely a hair above it.
    min_rise_pct: float = 3.0
    # A jump of >= 10% over the route's own 8-week average is a ~3.5 sd move.
    own_history_pct: float = 10.0
    # Gap vs peers is compared with the route's *usual* gap (Mumbai-Pune normally sits ~11%
    # above Delhi-Jaipur). Gap noise is 3.6 points (pooled 1 sd); 12 points is ~3.4 sd.
    peer_excess_pct: float = 12.0
    # Weeks of peer-gap history needed before we trust a route's "usual gap".
    peer_baseline_min_weeks: int = 8


@dataclass(frozen=True)
class LLMSettings:
    """Any OpenAI-compatible endpoint: Ollama, llama.cpp server, Groq, vLLM..."""

    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "qwen2.5:3b-instruct"))
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "not-needed-for-local"))
    temperature: float = 0.0  # greedy decoding: same prompt -> same text on the same build
    seed: int = 42  # belt and braces for servers that sample even at temperature 0
    timeout_s: int = 300
    # Published price per 1M tokens for the model in use (USD). Local models cost 0.
    usd_per_m_input: float = float(os.getenv("LLM_USD_PER_M_INPUT", "0"))
    usd_per_m_output: float = float(os.getenv("LLM_USD_PER_M_OUTPUT", "0"))


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # open-source, 33M params, runs on CPU via ONNX

THRESHOLDS = Thresholds()
