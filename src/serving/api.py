import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

import torch
from fastapi import FastAPI, HTTPException
from gliner import GLiNER

from src.serving.cache import InMemoryCache
from src.serving.schemas import (
    BenchmarkSummary,
    EntitySpan,
    ExtractionRequest,
    ExtractionResponse,
    HealthResponse,
    SystemBenchmarkRow,
)

logger = logging.getLogger(__name__)


import os
from huggingface_hub import login

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

_model: GLiNER | None = None
_model_version: str = ""
_start_time: float = 0.0
_cache: InMemoryCache = InMemoryCache(max_size=1024)

DEFAULT_ENTITY_TYPES = [
    "PLAYER", "TEAM", "POSITION", "STAT", "INJURY",
    "TRADE_ASSET", "GAME_EVENT", "VENUE", "COACH", "AWARD",
]

# Load from HuggingFace Hub — no local path needed
MODEL_ID = "myarapat/gliner-sports-ner"
BENCHMARK_PATH = Path("data/benchmark_results.json")


# ---------------------------------------------------------------------------
# Lifespan — model loads once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load GLiNER model once at startup, release on shutdown."""
    global _model, _model_version, _start_time

    logger.info(f"Loading model from {MODEL_ID}...")
    _model = GLiNER.from_pretrained(MODEL_ID, load_tokenizer=True)
    _model_version = MODEL_ID
    _start_time = time.time()
    logger.info("Model loaded and ready")

    yield

    # Teardown — free GPU memory on shutdown
    logger.info("Shutting down — releasing model")
    del _model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GLiNER Sports NER API",
    description="Named entity recognition for sports articles.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/extract", response_model=ExtractionResponse)
async def extract(request: ExtractionRequest) -> ExtractionResponse:
    """
    Extract named entities from a sports text passage.
    Checks LRU cache before running inference.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    entity_types = request.entity_types or DEFAULT_ENTITY_TYPES

    # Build cache key from request params
    cache_key = f"{request.text}|{sorted(entity_types)}|{request.threshold}"

    # Check cache first
    cached = _cache.get(cache_key)
    if cached is not None:
        cached["cache_hit"] = True
        return ExtractionResponse(**cached)

    # Run inference
    start = time.perf_counter()
    raw_entities = _model.predict_entities(
        request.text,
        entity_types,
        threshold=request.threshold,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    # Map to EntitySpan schema
    entities = [
        EntitySpan(
            text=ent["text"],
            label=ent["label"],
            start=ent["start"],
            end=ent["end"],
            score=round(ent["score"], 4),
        )
        for ent in raw_entities
    ]

    response_data = {
        "entities": [e.model_dump() for e in entities],
        "latency_ms": round(latency_ms, 2),
        "model_version": _model_version,
        "cache_hit": False,
    }

    # Store in cache
    _cache.set(cache_key, response_data)

    return ExtractionResponse(
        entities=entities,
        latency_ms=round(latency_ms, 2),
        model_version=_model_version,
        cache_hit=False,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service health and uptime."""
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        model_version=_model_version,
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@app.get("/benchmark", response_model=BenchmarkSummary)
async def benchmark() -> BenchmarkSummary:
    """Return pre-computed benchmark results."""
    if not BENCHMARK_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Benchmark results not found. Run src/evaluation/benchmark.py first."
        )

    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        data = json.load(f)

    results = [SystemBenchmarkRow(**row) for row in data["results"]]

    return BenchmarkSummary(
        results=results,
        test_set_size=data["test_set_size"],
        evaluated_at=data["evaluated_at"],
    )