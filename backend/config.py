"""
Central configuration — all env-var reads and defaults live here.
Import from this module instead of calling os.getenv() in business logic.
"""
from __future__ import annotations

import os

# ─── Qdrant ──────────────────────────────────────────────────────────────────
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://qdrant:6333").rstrip("/")
QDRANT_COLLECTIONS: list[str] = [
    item.strip()
    for item in os.getenv(
        "QDRANT_COLLECTIONS", "business_rules,kpi_history,alert_history"
    ).split(",")
    if item.strip()
]

# ─── Ollama ──────────────────────────────────────────────────────────────────
OLLAMA_HOST: str = os.getenv(
    "OLLAMA_HOST", "http://host.docker.internal:11434"
).rstrip("/")
OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_CHAT_MODEL: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3")
OLLAMA_TIMEOUT_SECONDS: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "35"))

# ─── Next.js internal API ─────────────────────────────────────────────────────
NEXTJS_API_URL: str = os.getenv(
    "NEXTJS_API_URL", "http://host.docker.internal:3000"
).rstrip("/")

# ─── Data year range ─────────────────────────────────────────────────────────
DATA_YEAR_FROM: int = int(os.getenv("DATA_YEAR_FROM", "2019"))
DATA_YEAR_TO: int = int(os.getenv("DATA_YEAR_TO", "2025"))

# ─── Alert thresholds ────────────────────────────────────────────────────────
ALERTE_IMPAYE_RATE_PCT: float = float(os.getenv("ALERTE_IMPAYE_RATE_PCT", "2.0"))
ALERTE_PRODUCTION_DROP_PCT: float = float(
    os.getenv("ALERTE_PRODUCTION_DROP_PCT", "15.0")
)

# ─── Agent behaviour ─────────────────────────────────────────────────────────
AGENT_INTENT_MIN_CONFIDENCE: float = float(
    os.getenv("AGENT_INTENT_MIN_CONFIDENCE", "0.45")
)
AGENT_LLM_MIN_CONFIDENCE: float = float(
    os.getenv("AGENT_LLM_MIN_CONFIDENCE", "0.42")
)
AGENT_LOW_CONFIDENCE_POLICY: str = (
    os.getenv("AGENT_LOW_CONFIDENCE_POLICY", "proceed").strip().lower()
)
AGENT_FORCE_DETERMINISTIC: bool = os.getenv(
    "AGENT_FORCE_DETERMINISTIC", "false"
).strip().lower() in {"1", "true", "yes", "on"}
