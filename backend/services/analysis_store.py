from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any, Dict, Optional
import uuid


@dataclass(frozen=True)
class StoredAnalysis:
    created_at: datetime
    expires_at: datetime
    payload: Dict[str, Any]  

class AnalysisStore:
    """
    Simple in-memory TTL store for analysis artifacts.
    - Thread-safe for a single-process Flask server.
    - NOT shared across multiple workers/processes (use Redis later for prod).
    """

    def __init__(self, ttl_seconds: int = 1800, max_items: int = 5000):
        self._ttl = int(ttl_seconds)
        self._max_items = int(max_items)
        self._lock = RLock()
        self._data: Dict[str, StoredAnalysis] = {}

    def put(self, payload: Dict[str, Any]) -> str:
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=self._ttl)
        analysis_id = uuid.uuid4().hex  # stable, URL-safe

        with self._lock:
            self._evict_expired_locked(now)
            self._evict_if_full_locked()

            self._data[analysis_id] = StoredAnalysis(
                created_at=now,
                expires_at=expires_at,
                payload=payload,
            )

        return analysis_id

    def get(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        now = datetime.now(timezone.utc)
        with self._lock:
            item = self._data.get(analysis_id)
            if item is None:
                return None
            if item.expires_at <= now:
                # expired
                del self._data[analysis_id]
                return None
            return item.payload

    def delete(self, analysis_id: str) -> bool:
        with self._lock:
            return self._data.pop(analysis_id, None) is not None

    def stats(self) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        with self._lock:
            self._evict_expired_locked(now)
            return {
                "items": len(self._data),
                "ttl_seconds": self._ttl,
                "max_items": self._max_items,
            }

    def cleanup(self) -> int:
        now = datetime.now(timezone.utc)
        with self._lock:
            before = len(self._data)
            self._evict_expired_locked(now)
            return before - len(self._data)

    # -----------------
    # internal helpers
    # -----------------
    def _evict_expired_locked(self, now: datetime) -> None:
        expired_keys = [k for k, v in self._data.items() if v.expires_at <= now]
        for k in expired_keys:
            del self._data[k]

    def _evict_if_full_locked(self) -> None:
        # If we exceed max_items, evict oldest first.
        # (Simple + good enough for MVP.)
        if len(self._data) < self._max_items:
            return

        # sort by created_at ascending and evict 10% (or at least 1)
        items = sorted(self._data.items(), key=lambda kv: kv[1].created_at)
        n_evict = max(1, int(self._max_items * 0.1))
        for k, _ in items[:n_evict]:
            self._data.pop(k, None)
