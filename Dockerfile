FROM python:3.13-slim

WORKDIR /app

# System dependencies:
#   nodejs/npm  — AgentSkills
RUN apt-get update && apt-get install -y --no-install-recommends \
        nodejs \
        npm \
    && rm -rf /var/lib/apt/lists/*

# Base Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and built-in assets
COPY pawlia/ pawlia/
COPY skills/ skills/

# Ensure user skills directory exists (may be empty if gitignored)
RUN mkdir -p skills/user

# Install deps + compile workflows for all pre-bundled skills
RUN python -m pawlia.install_skill_deps

# Session data lives in a volume
VOLUME ["/app/session"]

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "pawlia", "--mode", "server"]
