from typing import List, Optional

from pydantic import BaseModel, Field


class ExtractionRequest(BaseModel):
    """Request body for the /extract endpoint."""

    text: str = Field(..., description="Sports article or passage to extract entities from.")
    entity_types: Optional[List[str]] = Field(
        default=None,
        description="Entity types to extract. Defaults to all 10 sports entity types.",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for a predicted entity to be returned.",
    )


class EntitySpan(BaseModel):
    """A single predicted named entity span."""

    text: str = Field(..., description="Surface form of the entity as it appears in the text.")
    label: str = Field(..., description="Entity type label (e.g. PLAYER, TEAM).")
    start: int = Field(..., description="Character offset of the entity start (inclusive).")
    end: int = Field(..., description="Character offset of the entity end (exclusive).")
    score: float = Field(..., description="Model confidence score for this prediction.")


class ExtractionResponse(BaseModel):
    """Response body for the /extract endpoint."""

    entities: List[EntitySpan] = Field(..., description="List of extracted entity spans.")
    latency_ms: float = Field(..., description="Total inference latency in milliseconds.")
    model_version: str = Field(..., description="Identifier of the model used for extraction.")
    cache_hit: bool = Field(default=False, description="Whether the result was served from cache.")


class SystemBenchmarkRow(BaseModel):
    """Benchmark metrics for a single inference system."""

    system: str
    f1: float
    precision: float
    recall: float
    p50_latency_ms: float
    p99_latency_ms: float
    total_cost_usd: Optional[float] = None


class BenchmarkSummary(BaseModel):
    """Response body for the /benchmark endpoint."""

    results: List[SystemBenchmarkRow] = Field(
        ..., description="Per-system benchmark results."
    )
    test_set_size: int = Field(..., description="Number of examples in the test set.")
    evaluated_at: str = Field(..., description="ISO-8601 timestamp of the benchmark run.")


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str = Field(..., description="'ok' if the service is healthy.")
    model_loaded: bool = Field(..., description="Whether the NER model is loaded and ready.")
    model_version: str = Field(..., description="Identifier of the currently loaded model.")
    uptime_seconds: float = Field(..., description="Seconds since the service started.")
