# ── Stage 1: build ────────────────────────────────────────────────────────
FROM alpine:3.20 AS builder

# System deps:
#   python3 / py3-pip — runtime
#   py3-olm          — E2EE for Matrix (visible to venv via --system-site-packages)
#   gcc/g++/musl-dev/python3-dev/libffi-dev — compile non-wheel packages
RUN apk add --no-cache \
        python3 \
        py3-pip \
        py3-olm \
        py3-numpy \
        gcc \
        g++ \
        musl-dev \
        python3-dev \
        openssl-dev \
        libffi-dev

# Create a venv that can see system py3-olm so pip accepts matrix-nio[e2e]
RUN python3 -m venv --system-site-packages /venv

COPY requirements.txt requirements-e2e.txt ./
RUN /venv/bin/pip install --no-cache-dir -r requirements.txt -r requirements-e2e.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────
FROM alpine:3.20

# Runtime system deps:
#   python3   — interpreter
#   py3-olm   — E2EE (visible to venv via --system-site-packages)
#   nodejs/npm — AgentSkills
#   bash      — skill harnesses & the bash tool run under real bash; busybox
#               ash lacks arrays/[[ ]]/brace-expansion and caused LLM thrashing
RUN apk add --no-cache \
        python3 \
        py3-olm \
        nodejs \
        npm \
        py3-numpy \
        openssl \
        openssh-client \
        libffi \
        bubblewrap \
        bash \
        git

WORKDIR /app

# Copy the venv from the builder
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# Application code
COPY pawlia/ pawlia/
COPY skills/ skills/
RUN mkdir -p skills/user

# Install deps + compile workflows for all pre-bundled skills
RUN python3 -m pawlia.install_skill_deps --no-compile

# Session data lives in a volume
VOLUME ["/app/session"]

ENV PYTHONUNBUFFERED=1

CMD ["python3", "-m", "pawlia", "--mode", "server"]
