# CI/CD Pipeline for Picoclaw

This repository uses **Forgejo Actions** for continuous integration and testing.

## CI Workflow

The CI workflow is defined in `.github/workflows/ci.yml` and performs the following steps:

### Trigger
- Runs on every push to the `main` branch
- Runs on every pull request targeting the `main` branch

### Jobs

#### Test Job
1. **Container Setup**: Uses `node:20-alpine` container for lightweight execution
2. **Install Python**: Installs Python 3 and pip using `apk add --no-cache python3 py3-pip git`
3. **Checkout code**: Checks out the repository code using Forgejo's checkout action
4. **Create CI Config**: Generates minimal `config.yaml` with required providers/agents sections
5. **Install dependencies**: Installs all required Python packages from `source/requirements.txt`
6. **Run tests**: Executes test suite using pytest
7. **Upload test results**: Archives test results as artifacts (even if tests fail)

### Test Coverage
The CI pipeline runs the following test categories:
- Basic functionality tests
- LLM integration tests (mocked)
- Memory fact extraction and consolidation tests
- Session management tests
- Date change handling tests

## Local Testing

To run tests locally:

```bash
# Using the virtual environment
.venv/bin/python -m pytest tests/ -v

# Or with system Python (after installing dependencies)
python -m pytest tests/ -v
```

## Requirements

All dependencies are listed in `source/requirements.txt`:
- bottle==0.13.4
- requests==2.32.5
- pytest==9.0.2
- pytest-asyncio==1.3.0
- unittest2 (for enhanced unit testing)

## Artifacts

Test results are uploaded as artifacts and include:
- `.pytest_cache` - Pytest cache files
- Test result XML files (if generated)
- Pytest output files

## Status Badge

You can add a CI status badge to your README:

```markdown
[![CI Status](https://your-forgejo-instance.com/your-username/picoclaw/actions/workflows/ci.yml/badge.svg)](https://your-forgejo-instance.com/your-username/picoclaw/actions/workflows/ci.yml)
```

## Forgejo vs GitHub Actions

This workflow uses **Forgejo Actions** instead of GitHub Actions:
- **Self-hosted**: Runs on your own Forgejo instance
- **Open Source**: No dependency on GitHub's infrastructure
- **Compatible**: Uses the same workflow syntax as GitHub Actions
- **Actions URLs**: Uses `https://code.forgejo.org/actions/` instead of `actions/`

## Troubleshooting

If tests fail in CI but pass locally:
1. Check Python version (CI uses 3.14)
2. Verify all dependencies are installed
3. Check for environment-specific issues
4. Review the CI logs for detailed error information
5. Ensure your Forgejo instance has the required actions runners configured

## Forgejo Setup

To use this CI pipeline:
1. Ensure your Forgejo instance has actions enabled
2. Configure at least one runner with Docker support
3. Ensure the runner can pull the `node:20-alpine` image
4. Push this workflow to your repository
5. The pipeline will automatically run on pushes and PRs

## Container Details

This workflow uses a **Node.js 20 Alpine** container:
- **Base Image**: `node:20-alpine` (lightweight Alpine Linux)
- **Python**: Python 3 installed via `apk add python3 py3-pip`
- **Git**: Required for checkout, installed via `apk add git`
- **Config**: Auto-generates `config.yaml` for CI environment
- **Benefits**: Smaller image size, faster startup, minimal dependencies

## Why Node.js Container for Python Tests?

While it may seem unusual to use a Node.js container for Python tests, this approach has advantages:
- Alpine Linux is extremely lightweight (~5MB base image)
- Node.js container includes necessary build tools
- Python 3 is easily installable via apk
- Faster container startup and execution
- Reduced resource usage on CI runners

## Virtual Environment Benefits

The workflow uses Python virtual environments for isolation:
- **Isolation**: Dependencies don't conflict with system packages
- **Reproducibility**: Exact same environment every run
- **Cleanliness**: No system package modifications
- **Best Practice**: Follows Python packaging standards
- **Enhanced Testing**: Includes unittest2 for advanced unit testing features

## Container + Virtual Environment = Best of Both Worlds

- **Container**: Lightweight, portable execution environment
- **Virtual Environment**: Clean Python dependency management
- **Result**: Reliable, reproducible test runs