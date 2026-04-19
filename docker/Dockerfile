# ── Stage 1: dependency builder ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ ./src/
COPY demo/ ./demo/

# Copy fine-tuned model checkpoint (must exist at build time or be mounted at runtime)
COPY checkpoints/ ./checkpoints/

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
