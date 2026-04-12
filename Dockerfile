FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Force CPU-only torch — avoids pulling CUDA (~4GB saved)
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install torch CPU-only FIRST to lock the lightweight variant,
# then install the rest. This prevents pip from upgrading to the
# CUDA build when sentence-transformers pulls torch as a dependency.
RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Pre-download the embedding model into the image layer so it doesn't
# fetch at runtime (no internet needed in prod, faster cold start).
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY *.py .
COPY website.html .

# DO NOT bake faiss_index into the image — let it be created fresh at
# runtime or mount it as a volume. Baking it risks stale state and bloat.

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "-m", "uvicorn", "client:app", "--host", "0.0.0.0", "--port", "8080"]