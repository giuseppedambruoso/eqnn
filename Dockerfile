# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Builder stage: resolve and install dependencies with Poetry
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    POETRY_REQUESTS_TIMEOUT=120 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ENV PATH="$POETRY_HOME/bin:$PATH"

RUN curl -sSL https://install.python-poetry.org | python3 -

# Fewer concurrent downloads is more resilient on flaky/constrained networks
# (e.g. Docker Desktop + WSL2 NAT) than poetry's default parallelism.
RUN poetry config installer.max-workers 4

WORKDIR /app

# Copy only dependency manifests first so Docker can cache this layer
COPY pyproject.toml poetry.lock README.md ./

# Install dependencies only (no root package yet) — keeps the layer cacheable.
# Retries in a loop: transient SSL/connection drops mid-download shouldn't
# fail the whole build when the fix is just "try again".
RUN bash -c '\
    for i in $(seq 1 5); do \
        poetry install --no-root --only main --no-ansi && exit 0; \
        echo "poetry install failed (attempt $i/5), retrying in 5s..."; \
        sleep 5; \
    done; \
    exit 1'

# Now copy the rest of the source code and install the project itself
COPY src ./src
RUN poetry install --only-root --no-ansi

# ---------------------------------------------------------------------------
# Runtime stage: slim image with just the venv + source, no build toolchain
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Runtime shared libs needed by opencv-python / matplotlib / pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    WANDB_MODE=online \
    WANDB_DIR=/app/wandb

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser README.md pyproject.toml ./

# Directories the training job writes/reads at runtime; mount these as
# volumes (see docker-compose.yml) to persist data/results across runs.
RUN mkdir -p data outputs multirun wandb \
    && chown -R appuser:appuser data outputs multirun wandb

VOLUME ["/app/data", "/app/outputs", "/app/multirun", "/app/wandb"]

USER appuser

# WANDB_API_KEY must be provided at runtime, e.g.:
#   docker run -e WANDB_API_KEY=... eqnn TRAINING.epochs=10
ENTRYPOINT ["python3", "-m", "src.main"]
CMD []
