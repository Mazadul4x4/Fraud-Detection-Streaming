"""Online (real-time) version of the per-card history features.

Training computes these features with pandas over the whole dataset
(`src/training/preprocess.py::add_card_history_features`). At serving time we
see ONE transaction at a time, so we keep each card's recent history in memory
and compute exactly the same values.

`tests/test_card_history.py` checks that both implementations agree
(protection against training/serving skew).

This in-memory store is a stepping stone: in Phase 7-8 the same features will
come from Spark Structured Streaming and the Feast online store (Redis).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from src.training.preprocess import MAX_GAP_SECONDS

WINDOWS = {"1h": timedelta(hours=1), "24h": timedelta(hours=24)}
LONGEST_WINDOW = max(WINDOWS.values())


@dataclass
class _CardState:
    recent: deque = field(default_factory=deque)  # (timestamp, amount) within the longest window
    last_time: datetime | None = None
    total_count: int = 0
    total_amount: float = 0.0


class CardHistoryStore:
    """Keeps per-card state and produces history features for a new transaction."""

    def __init__(self) -> None:
        self._cards: dict[str, _CardState] = defaultdict(_CardState)

    def features(self, card: str, time: datetime, amount: float) -> dict[str, float]:
        """Features for a transaction, using ONLY earlier transactions of this card."""
        state = self._cards[card]
        out: dict[str, float] = {}

        for name, length in WINDOWS.items():
            start = time - length
            # Same rule as pandas rolling(closed="left"): start <= t < current time
            window = [a for (t, a) in state.recent if start <= t < time]
            out[f"card_txn_count_{name}"] = float(len(window))
            out[f"card_amt_sum_{name}"] = float(sum(window))

        if state.last_time is None:
            gap = MAX_GAP_SECONDS
        else:
            gap = min((time - state.last_time).total_seconds(), MAX_GAP_SECONDS)
        out["seconds_since_last_txn"] = float(gap)

        if state.total_count == 0:
            out["amt_vs_card_mean"] = 1.0
        else:
            out["amt_vs_card_mean"] = amount / (state.total_amount / state.total_count)
        return out

    def add(self, card: str, time: datetime, amount: float) -> None:
        """Record a transaction so that later transactions can see it."""
        state = self._cards[card]
        state.recent.append((time, amount))
        while state.recent and state.recent[0][0] < time - LONGEST_WINDOW:
            state.recent.popleft()
        state.last_time = time
        state.total_count += 1
        state.total_amount += amount

    def features_and_add(self, card: str, time: datetime, amount: float) -> dict[str, float]:
        """Features for a new transaction, then record it (one call per scored transaction)."""
        features = self.features(card, time, amount)
        self.add(card, time, amount)
        return features

    def __len__(self) -> int:
        return len(self._cards)

    # --- State export / import (used by the Spark streaming job) -------------
    def export_state(self, card: str) -> dict:
        """Plain-Python snapshot of one card's state (to persist between batches)."""
        state = self._cards[card]
        return {
            "recent_times": [t for (t, _) in state.recent],
            "recent_amounts": [a for (_, a) in state.recent],
            "last_time": state.last_time,
            "total_count": state.total_count,
            "total_amount": state.total_amount,
        }

    def import_state(self, card: str, snapshot: dict) -> None:
        """Restore one card's state from `export_state` output."""
        self._cards[card] = _CardState(
            recent=deque(zip(snapshot["recent_times"], snapshot["recent_amounts"])),
            last_time=snapshot["last_time"],
            total_count=int(snapshot["total_count"]),
            total_amount=float(snapshot["total_amount"]),
        )
