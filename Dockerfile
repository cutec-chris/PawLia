# ── Stage 1: build ────────────────────────────────────────────────────────
FROM python:3.13-alpine AS builder

RUN apk add --no-cache \
        gcc \
        g++ \
        musl-dev \
        python3-dev \
        openssl-dev \
        libffi-dev \
        olm-dev

COPY requirements.txt ./
# python-olm must link against system libolm; needs --no-build-isolation
# so the env var reaches olm_build.py inside pip's subprocess.
RUN pip install --no-cache-dir --prefix=/install cffi \
    && PYTHON_OLM_USE_SYSTEM_LIB=1 \
       pip install --no-cache-dir --no-build-isolation --prefix=/install python-olm \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────
FROM python:3.13-alpine

# Runtime system deps:
#   nodejs/npm  — AgentSkills
#   olm         — E2EE for Matrix
RUN apk add --no-cache \
        nodejs \
        npm \
        openssl \
        libffi \
        olm

WORKDIR /app

# Copy compiled Python packages from builder
COPY --from=builder /install /usr/local

# Application code
COPY pawlia/ pawlia/
COPY skills/ skills/
RUN mkdir -p skills/user

# Install deps + compile workflows for all pre-bundled skills
RUN python -m pawlia.install_skill_deps

# Session data lives in a volume
VOLUME ["/app/session"]

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "pawlia", "--mode", "server"]
