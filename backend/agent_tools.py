"""
Thin shim — all logic lives in the tools/ package.
This module re-exports the public API so existing callers (agent_graph, tests) need no changes.

NOTE: tests that monkeypatch agent_tools._query_dataframe or agent_tools.importlib
must be updated to patch the symbol in the module that actually owns it:
  - tools.sql._query_dataframe  (for sql_tool / data_query_tool)
  - tools.forecast._query_dataframe  (for forecast_tool)
  - tools.forecast.importlib  (for forecast_tool's model-load path)
"""
from __future__ import annotations

import importlib  # re-exported so existing monkeypatches that target agent_tools.importlib resolve

from tools import TOOL_REGISTRY, run_tool  # noqa: F401  (used by agent_graph)
from tools.kpi import kpi_tool  # noqa: F401
from tools.rag import rag_tool  # noqa: F401
from tools.alerte import alerte_tool  # noqa: F401
from tools.forecast import forecast_tool  # noqa: F401
from tools.anomaly import anomaly_tool  # noqa: F401
from tools.drift import drift_tool  # noqa: F401
from tools.explain import explain_tool  # noqa: F401
from tools.segmentation import segmentation_tool  # noqa: F401
from tools.client import client_tool  # noqa: F401
from tools.sql import sql_tool, data_query_tool  # noqa: F401
from tools.ml_predict import ml_predict_tool  # noqa: F401
from tools.dim import dim_tool  # noqa: F401
from db import query_dataframe as _query_dataframe  # noqa: F401  (re-exported for legacy monkeypatches)
