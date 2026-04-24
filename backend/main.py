from datetime import datetime, timezone
import logging
import os
from typing import Any

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from db import get_db
from agent_router import perform_agent_warmup, router as agent_router
from geo_router import router as geo_router
from kpi_router import router as kpi_router
from ml_router import router as ml_router


logger = logging.getLogger("maghrebia.backend")

app = FastAPI(
    title="Maghrebia CEO Platform API",
    description="Phase 3a KPI backend aligned with TDSP roadmap.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(kpi_router, prefix="/api/v1")
app.include_router(geo_router, prefix="/api/v1")
app.include_router(ml_router, prefix="/api/v1")
app.include_router(agent_router, prefix="/api/v1")


@app.on_event("startup")
def startup_warmup() -> None:
    auto_warmup = os.getenv("AGENT_AUTO_WARMUP_ON_STARTUP", "false").strip().lower() == "true"
    if not auto_warmup:
        return

    auto_preindex = os.getenv("AGENT_AUTO_PREINDEX_ON_STARTUP", "false").strip().lower() == "true"
    max_docs = int(os.getenv("AGENT_STARTUP_PREINDEX_DOCS", "250"))
    strict = os.getenv("AGENT_STARTUP_WARMUP_STRICT", "false").strip().lower() == "true"

    report = perform_agent_warmup(
        preindex=auto_preindex,
        max_docs_per_collection=max_docs,
        strict=strict,
    )
    logger.info("Agent startup warmup status=%s", report.get("status"))


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "maghrebia-backend",
        "status": "running",
        "docs": "/docs",
        "version": "0.1.0",
    }


@app.get("/health")
def health(db: Session = Depends(get_db)) -> dict[str, Any]:
    db.execute(text("SELECT 1"))
    return {
        "status": "ok",
        "service": "maghrebia-backend",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
