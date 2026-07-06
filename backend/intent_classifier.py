"""
Trained intent classifier — inference only.

Loads models/intent_classifier.pkl (built by train_intent_classifier.py).
Returns calibrated class probabilities so confidence thresholds are meaningful.
Returns None when the pkl is absent so agent_graph falls back to Ollama + keywords.
"""
from __future__ import annotations

import pickle
import re
import unicodedata
from pathlib import Path
from typing import Any

_MODEL_PATH = Path(__file__).resolve().parent / "models" / "intent_classifier.pkl"

# Module-level cache — pkl loaded once on first call, never reloaded mid-run.
_cache: dict[str, Any] | None | bool = False   # False = not yet attempted


def _load() -> dict[str, Any] | None:
    global _cache
    if _cache is False:
        if not _MODEL_PATH.exists():
            _cache = None
        else:
            try:
                with _MODEL_PATH.open("rb") as fh:
                    _cache = pickle.load(fh)
            except Exception:
                _cache = None
    return _cache  # type: ignore[return-value]


def _normalize(text: str) -> str:
    text = text.lower()
    nfkd = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in nfkd if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def classify(
    question: str,
    min_confidence: float = 0.55,
) -> tuple[str, float, dict[str, float]] | None:
    """
    Classify a question using the trained model.

    Returns:
        (intent, confidence, all_probabilities) when confidence >= min_confidence
        None when the pkl is missing or confidence is below threshold
            → caller should fall back to Ollama / keyword classifier

    confidence is the max class probability from CalibratedClassifierCV.
    all_probabilities is a {class: probability} dict for introspection.
    """
    model = _load()
    if model is None:
        return None

    pipeline = model["pipeline"]
    classes  = model["classes"]

    try:
        x_norm = _normalize(question)
        proba  = pipeline.predict_proba([x_norm])[0]
    except Exception:
        return None

    best_idx    = int(proba.argmax())
    confidence  = float(proba[best_idx])
    intent      = classes[best_idx]
    all_proba   = {c: round(float(p), 4) for c, p in zip(classes, proba)}

    if confidence < min_confidence:
        return None

    return intent, confidence, all_proba


def is_available() -> bool:
    """True if the pkl exists and loaded successfully."""
    return _load() is not None


def model_report() -> dict[str, Any] | None:
    """Return the training report embedded in the pkl, or None."""
    model = _load()
    if model is None:
        return None
    return model.get("report", {})


def reload() -> bool:
    """Force re-read from disk (call after retraining)."""
    global _cache
    _cache = False
    return is_available()
