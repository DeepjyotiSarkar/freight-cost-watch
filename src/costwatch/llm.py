"""Tiny client for any OpenAI-compatible chat endpoint, with a token/cost ledger.

Works with Ollama (http://localhost:11434/v1), llama.cpp's server, Groq, vLLM, etc.
No SDK dependency: one POST per call.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field

import requests

from .config import LLMSettings


class LLMUnavailable(RuntimeError):
    pass


@dataclass
class CallRecord:
    purpose: str
    input_tokens: int
    output_tokens: int
    seconds: float


@dataclass
class TokenLedger:
    model: str
    usd_per_m_input: float
    usd_per_m_output: float
    calls: list[CallRecord] = field(default_factory=list)

    def add(self, rec: CallRecord) -> None:
        self.calls.append(rec)

    def summary(self) -> dict:
        tin = sum(c.input_tokens for c in self.calls)
        tout = sum(c.output_tokens for c in self.calls)
        by_purpose: dict[str, dict] = {}
        for c in self.calls:
            p = by_purpose.setdefault(c.purpose, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
            p["calls"] += 1
            p["input_tokens"] += c.input_tokens
            p["output_tokens"] += c.output_tokens
        return {
            "model": self.model,
            "llm_calls": len(self.calls),
            "input_tokens": tin,
            "output_tokens": tout,
            "usd_cost_at_configured_rate": round(
                tin / 1e6 * self.usd_per_m_input + tout / 1e6 * self.usd_per_m_output, 6
            ),
            "wall_seconds_in_llm": round(sum(c.seconds for c in self.calls), 1),
            "by_purpose": by_purpose,
        }

    def to_json(self) -> dict:
        return {"summary": self.summary(), "calls": [asdict(c) for c in self.calls]}


class LLMClient:
    def __init__(self, settings: LLMSettings):
        self.s = settings
        self.ledger = TokenLedger(settings.model, settings.usd_per_m_input, settings.usd_per_m_output)

    def ping(self) -> bool:
        try:
            r = requests.get(f"{self.s.base_url}/models", timeout=5,
                             headers={"Authorization": f"Bearer {self.s.api_key}"})
            return r.ok
        except requests.RequestException:
            return False

    def chat(self, system: str, user: str, purpose: str, max_tokens: int = 200, json_mode: bool = False) -> str:
        body = {
            "model": self.s.model,
            "temperature": self.s.temperature,
            "seed": self.s.seed,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        t0 = time.time()
        try:
            r = requests.post(
                f"{self.s.base_url}/chat/completions",
                json=body,
                timeout=self.s.timeout_s,
                headers={"Authorization": f"Bearer {self.s.api_key}"},
            )
            r.raise_for_status()
        except requests.RequestException as e:
            raise LLMUnavailable(str(e)) from e
        data = r.json()
        usage = data.get("usage") or {}
        self.ledger.add(CallRecord(
            purpose=purpose,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            seconds=round(time.time() - t0, 2),
        ))
        return data["choices"][0]["message"]["content"] or ""


def parse_json_object(text: str) -> dict:
    """Parse the first JSON object in a reply (tolerates ```json fences)."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start : end + 1])
