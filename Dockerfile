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

# Build python-olm against system libolm first (needs --no-build-isolation
# so PYTHON_OLM_USE_SYSTEM_LIB reaches the cffi build script)
RUN pip install --no-cache-dir cffi setuptools \
    && PYTHON_OLM_USE_SYSTEM_LIB=1 pip install --no-cache-dir --no-build-isolation python-olm

COPY requirements.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
# Copy the already-built python-olm into /install so it's available at runtime
RUN cp -r /usr/local/lib/python3.13/site-packages/_libolm* \
          /usr/local/lib/python3.13/site-packages/olm \
          /install/lib/python3.13/site-packages/ 2>/dev/null || true


# ── Stage 2: runtime ──────────────────────────────────────────────────────
FROM python:3.13-alpine

# Runtime system deps:
#   nodejs/npm  — AgentSkills
#   olm         — E2EE for Matrix (C library for python-olm)
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
RUN python -m pawlia.install_skill_deps --no-compile

# Session data lives in a volume
VOLUME ["/app/session"]

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "pawlia", "--mode", "server"]
