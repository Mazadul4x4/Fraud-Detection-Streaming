"""Online card-history state stored in Redis.

The API keeps no history in its own memory any more: each card's state (recent
transactions + running totals) lives in Redis under `card_state:<card number>`.
For every scored transaction the API reads the card's state, computes the
features with the SAME `CardHistoryStore` code used everywhere else (training
parity is unit-tested), then writes the updated state back - atomically.

Why Redis directly instead of Feast (see docs/model_card.md and the README):
  * Feast (0.66) requires pandas < 3, while the project is pinned to pandas 3.
  * Feast serves feature values computed at a card's LAST event; time-window
    features (txns in the last 1h/24h) then go stale between events and would
    differ from training. Storing the state and computing at request time keeps
    training and serving features identical.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import redis

from src.api.card_history import CardHistoryStore

KEY_PREFIX = "card_state:"


def _to_epoch(dt: datetime) -> float:
    return dt.replace(tzinfo=timezone.utc).timestamp()


def _to_dt(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).replace(tzinfo=None)


def encode_snapshot(snapshot: dict) -> str:
    """CardHistoryStore snapshot -> compact JSON (times as epoch seconds)."""
    return json.dumps({
        "recent": [[_to_epoch(t), a] for t, a in zip(snapshot["recent_times"], snapshot["recent_amounts"])],
        "last_time": _to_epoch(snapshot["last_time"]) if snapshot["last_time"] else None,
        "total_count": snapshot["total_count"],
        "total_amount": snapshot["total_amount"],
    })


def decode_snapshot(raw: str | bytes) -> dict:
    data = json.loads(raw)
    return {
        "recent_times": [_to_dt(t) for t, _ in data["recent"]],
        "recent_amounts": [a for _, a in data["recent"]],
        "last_time": _to_dt(data["last_time"]) if data["last_time"] is not None else None,
        "total_count": data["total_count"],
        "total_amount": data["total_amount"],
    }


class RedisCardHistory:
    """Same interface as CardHistoryStore.features_and_add, backed by Redis."""

    def __init__(self, client: redis.Redis):
        self.client = client

    @classmethod
    def from_url(cls, url: str) -> "RedisCardHistory":
        return cls(redis.Redis.from_url(url))

    def features_and_add(self, card: str, time: datetime, amount: float) -> dict[str, float]:
        key = KEY_PREFIX + card
        with self.client.pipeline() as pipe:
            while True:
                try:
                    # WATCH + MULTI/EXEC: if another request updates this card in the
                    # meantime, the write is rejected and we retry with fresh state.
                    pipe.watch(key)
                    raw = pipe.get(key)
                    store = CardHistoryStore()
                    if raw is not None:
                        store.import_state(card, decode_snapshot(raw))
                    features = store.features_and_add(card, time, amount)
                    pipe.multi()
                    pipe.set(key, encode_snapshot(store.export_state(card)))
                    pipe.execute()
                    return features
                except redis.WatchError:
                    continue

    def put_snapshots(self, snapshots: dict[str, dict]) -> None:
        """Bulk-write card states (used by the backfill)."""
        with self.client.pipeline(transaction=False) as pipe:
            for card, snapshot in snapshots.items():
                pipe.set(KEY_PREFIX + card, encode_snapshot(snapshot))
            pipe.execute()

    def card_count(self) -> int:
        return sum(1 for _ in self.client.scan_iter(match=KEY_PREFIX + "*", count=1000))

    def clear(self) -> int:
        keys = list(self.client.scan_iter(match=KEY_PREFIX + "*", count=1000))
        return self.client.delete(*keys) if keys else 0

    def __len__(self) -> int:
        return self.card_count()
