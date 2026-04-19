import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from gliner import GLiNER

from src.serving.cache import InMemoryCache
from src.serving.schemas import (
    BenchmarkSummary,
    ExtractionRequest,
    ExtractionResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)

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

MODEL_PATH = "checkpoints/gliner-sports"


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the GLiNER model once at startup and release on shutdown."""
    global _model, _model_version, _start_time

    # TODO: Call GLiNER.from_pretrained(MODEL_PATH) and assign to _model.
    # TODO: Set _model_version from model config or checkpoint directory name.
    # TODO: Set _start_time = time.time().
    # TODO: Log that the model is ready.

    _start_time = time.time()
    yield

    # TODO: Any teardown logic (e.g., clearing GPU memory).


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GLiNER Sports NER API",
    description="Named entity recognition for sports articles.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/extract", response_model=ExtractionResponse)
async def extract(request: ExtractionRequest) -> ExtractionResponse:
    """
    Extract named entities from a sports text passage.

    Checks the in-memory LRU cache before running model inference.

    Args:
        request: ExtractionRequest with text, optional entity_types, and threshold.

    Returns:
        ExtractionResponse with entities, latency_ms, model_version, and cache_hit.

    Raises:
        HTTPException 503 if the model is not yet loaded.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    entity_types = request.entity_types or DEFAULT_ENTITY_TYPES

    # TODO: Check _cache.get(request.text, entity_types, request.threshold).
    # TODO: If cache hit, return cached ExtractionResponse with cache_hit=True.

    # TODO: Record start time, call _model.predict_entities(request.text, entity_types,
    #       threshold=request.threshold), record end time.
    # TODO: Map raw predictions to List[EntitySpan].
    # TODO: Store result in _cache.
    # TODO: Build and return ExtractionResponse.
    raise NotImplementedError


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Return service health status and uptime.

    Returns:
        HealthResponse indicating model load status and uptime.
    """
    # TODO: Return HealthResponse with status="ok", model_loaded=(_model is not None),
    #       model_version=_model_version, uptime_seconds=time.time() - _start_time.
    raise NotImplementedError


@app.get("/benchmark", response_model=BenchmarkSummary)
async def benchmark() -> BenchmarkSummary:
    """
    Return the most recent benchmark comparison across all three systems.

    Reads pre-computed benchmark results from disk if available.

    Returns:
        BenchmarkSummary with per-system metrics.

    Raises:
        HTTPException 404 if no benchmark results have been generated yet.
    """
    # TODO: Load benchmark results JSON from a known path (e.g. checkpoints/benchmark.json).
    # TODO: Deserialize into BenchmarkSummary and return.
    # TODO: Raise 404 if the file does not exist.
    raise NotImplementedError
