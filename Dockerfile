# ─────────────────────────────────────────────────────────────────────────────
# NyayBot — Multi-stage Dockerfile for Hugging Face Spaces (Docker SDK)
# ─────────────────────────────────────────────────────────────────────────────
#
# Stage layout:
#   frontend-builder  — Node 22 slim: installs npm deps, runs `vite build`
#   backend-base      — Python 3.11 slim: installs all Python deps (heavy layer,
#                       cached aggressively so code changes don't retrigger it)
#   runner            — Python 3.11 slim: runtime-only image; copies:
#                         - Python site-packages from backend-base
#                         - Built frontend dist/ from frontend-builder
#                         - data/indexes/ from the host (via COPY, tracked in LFS)
#                         - Application source (src/, data/)
#
# Index strategy — Option A (LFS):
#   The .pkl/.faiss files in data/indexes/ are committed to the repo under Git
#   LFS (see .gitattributes). HF Spaces clones the repo including LFS objects
#   before building, so COPY data/indexes/ below finds them already on disk.
#   This avoids re-running the 30–60 min indexing step on every image build.
#   See .gitattributes for LFS rules and README for rebuild instructions.
#
# HF Spaces notes:
#   - app_port must match the port uvicorn binds (7860, set in README.md YAML).
#   - Secrets (GEMINI_API_KEY, SUPABASE_*, etc.) are set in the HF Space's
#     Settings → Variables and Secrets panel — never baked into the image.
#   - HF free tier: 2 vCPU, 16 GB RAM, no GPU.
#     Set EMBEDDING_PROVIDER=api and HF_TOKEN so the BGE-M3 embedding is served
#     via HF Inference API instead of loading locally.  This avoids the ~2 GB
#     torch + sentence-transformers layer and the OOM that comes with it.
#     torch / sentence-transformers are intentionally excluded from requirements.txt.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Frontend build ───────────────────────────────────────────────────
FROM node:22-slim AS frontend-builder

WORKDIR /app/frontend

# Copy package files first — Docker caches this layer until package.json changes
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --prefer-offline

# Copy frontend source and build
# VITE_API_BASE_URL is intentionally left empty: the backend and frontend are
# served from the same uvicorn process on the same origin, so all API calls
# use relative URLs (/query, /health, /history/...) with no host prefix.
COPY frontend/ ./
RUN VITE_API_BASE_URL="" npm run build


# ── Stage 2: Python dependency installation ───────────────────────────────────
# Separate stage so Python deps (3-4 GB) are cached independently of source.
# Changing a .py file only invalidates the final COPY layers, not pip install.
FROM python:3.11-slim AS backend-base

WORKDIR /app

# System deps needed by some wheels (faiss-cpu, psycopg2-binary)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first — cached until requirements.txt changes
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


# ── Stage 3: Runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runner

# Non-root user required by HF Spaces
RUN useradd -m -u 1000 appuser

WORKDIR /app

# System runtime libs (libpq for psycopg2, libgomp for faiss CPU BLAS)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from backend-base
COPY --from=backend-base /usr/local/lib/python3.11/site-packages \
                         /usr/local/lib/python3.11/site-packages
COPY --from=backend-base /usr/local/bin /usr/local/bin

# Copy Vite build output from frontend-builder
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Copy application source (small, changes frequently — last for cache efficiency)
COPY src/ ./src/
COPY data/indexes/ ./data/indexes/

# Give the non-root user ownership
RUN chown -R appuser:appuser /app
USER appuser

# HF Spaces exposes port 7860 (set app_port: 7860 in README.md YAML)
EXPOSE 7860

# Uvicorn startup:
#   --host 0.0.0.0  required for HF Spaces networking
#   --port 7860     must match app_port in README YAML
#   --workers 1     single worker — BGE-M3 + reranker share one GPU/RAM budget;
#                   multiple workers would each load their own model copy and OOM
#   --timeout-keep-alive 120  long timeout for first-query model cold-load
CMD ["uvicorn", "src.backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--timeout-keep-alive", "120"]
