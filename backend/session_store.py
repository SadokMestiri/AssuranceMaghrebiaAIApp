"""
In-memory conversation session store.
Each session holds a capped list of turns (user question + assistant answer + context snapshot).
Sessions expire after TTL_SECONDS of inactivity.
"""
from __future__ import annotations

import threading
import time
from typing import Any

_TTL_SECONDS: int = 1800       # 30 minutes of inactivity → session evicted
_MAX_TURNS: int = 10           # turns kept per session (older ones dropped)
_MAX_SESSIONS: int = 500       # hard cap; oldest evicted when reached

_lock = threading.Lock()
_store: dict[str, dict[str, Any]] = {}
# _store[session_id] = {"turns": [...], "last_access": float}


def get_history(session_id: str) -> list[dict[str, Any]]:
    """Return the turn list for a session (empty list if session doesn't exist)."""
    with _lock:
        _evict_expired()
        session = _store.get(session_id)
        if not session:
            return []
        session["last_access"] = time.monotonic()
        return list(session["turns"])


def append_turn(
    session_id: str,
    user_question: str,
    assistant_answer: str,
    context: dict[str, Any],
    intent: str = "",
) -> None:
    """Append one completed turn to the session history."""
    with _lock:
        _evict_expired()
        if session_id not in _store:
            if len(_store) >= _MAX_SESSIONS:
                _evict_oldest()
            _store[session_id] = {"turns": [], "last_access": time.monotonic()}

        session = _store[session_id]
        session["turns"].append({
            "user": user_question,
            "assistant": str(assistant_answer)[:600],
            "intent": intent,
            "context": {k: v for k, v in context.items() if v is not None and k != "history"},
        })
        session["turns"] = session["turns"][-_MAX_TURNS:]
        session["last_access"] = time.monotonic()


def _evict_expired() -> None:
    now = time.monotonic()
    expired = [k for k, v in _store.items() if now - v["last_access"] > _TTL_SECONDS]
    for k in expired:
        del _store[k]


def _evict_oldest() -> None:
    if not _store:
        return
    oldest_key = min(_store, key=lambda k: _store[k]["last_access"])
    del _store[oldest_key]
