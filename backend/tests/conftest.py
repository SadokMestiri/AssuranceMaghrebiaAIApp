"""Shared test fixtures.

The classifier unit tests validate the *deterministic keyword* routing (Path 3
of classify_question). They must therefore run independently of the optional
trained-model pickle (Path 1) and the Ollama LLM (Path 2), whose availability
differs between a developer machine, the Docker stack and the CI runner. This
autouse fixture forces the deterministic path so the tests assert the keyword
classifier's behaviour reproducibly everywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest  # type: ignore[reportMissingImports]

sys.path.append(str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def _force_deterministic_classifier(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        import agent_graph
    except Exception:
        return
    # Skip Path 1 (model pkl) and Path 2 (Ollama) → deterministic keyword path.
    monkeypatch.setattr(agent_graph, "FORCE_DETERMINISTIC", True, raising=False)
