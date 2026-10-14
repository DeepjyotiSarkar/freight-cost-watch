"""Hybrid retrieval over context notes: metadata filters first, then semantic ranking.

Embeddings are good at "what is this note about" (flood vs diesel vs festival) and bad at
"does it cover the week of 2025-03-10". So route scope and time proximity are hard filters
on structured fields, and the embedding score only orders what survives.

With 10 notes a numpy matrix *is* the vector store. The interface (add / query with a
metadata filter) is the same one Qdrant or Chroma expose, so swapping one in is contained.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from .config import EMBEDDING_MODEL
from .notes import NoteFacts


@dataclass
class Candidate:
    note: NoteFacts
    semantic_score: float
    days_from_window: int  # 0 = the note's window overlaps the week


class NoteIndex:
    def __init__(self, notes: list[NoteFacts], cache_dir: Path):
        from fastembed import TextEmbedding  # local ONNX model, deterministic on CPU

        self.notes = notes
        self._model = TextEmbedding(EMBEDDING_MODEL, cache_dir=str(cache_dir))
        docs = [f"Route: {n.applies_to}. Date: {n.note_date:%d %B %Y}. {n.text}" for n in notes]
        self._matrix = self._embed(docs)

    def _embed(self, texts: list[str]) -> np.ndarray:
        m = np.vstack(list(self._model.embed(texts))).astype(np.float64)
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    def search(self, query: str, keep=lambda n: True) -> list[tuple[NoteFacts, float]]:
        """Pure semantic search with an optional metadata filter, best first."""
        scores = self._matrix @ self._embed([query])[0]
        ranked = sorted(
            ((n, float(s)) for n, s in zip(self.notes, scores) if keep(n)),
            key=lambda x: (-round(x[1], 6), x[0].note_id),  # stable tie-break
        )
        return ranked

    def candidates_for_week(
        self, route: str, week_of: date, query: str, max_gap_days: int = 56
    ) -> list[Candidate]:
        """Notes for this route (or all routes) whose time window is within ~8 weeks of the week."""
        week_end = week_of + timedelta(days=6)

        def gap(n: NoteFacts) -> int:
            start = n.window_start or n.note_date
            end = n.window_end or n.note_date
            if end < week_of:
                return (week_of - end).days
            if start > week_end:
                return (start - week_end).days
            return 0

        keep = lambda n: n.applies_to_route(route) and gap(n) <= max_gap_days
        return [Candidate(n, s, gap(n)) for n, s in self.search(query, keep)]


def query_for(route: str, week_of: date) -> str:
    origin, _, destination = route.partition("-")
    return (
        f"Why did freight costs rise on the {route} route ({origin} to {destination}) "
        f"in the week of {week_of:%d %B %Y}? Possible reasons: fuel or diesel prices, tolls, "
        f"floods or weather disruption, festival surcharges, road works."
    )
