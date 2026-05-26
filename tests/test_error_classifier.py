import pytest

from pawlia.agents.error_classifier import (
    ErrorCategory,
    classify_error,
    is_retryable,
    should_compact,
)


# ── classify_error ─────────────────────────────────────────────────────

class _StatusExc(Exception):
    def __init__(self, msg, status_code):
        super().__init__(msg)
        self.status_code = status_code


def test_context_overflow_by_text():
    cat, _ = classify_error(ValueError("context_length_exceeded"))
    assert cat == ErrorCategory.context_overflow


def test_context_overflow_glm_1214():
    cat, _ = classify_error(ValueError("messages parameter is illegal '1214'"))
    assert cat == ErrorCategory.context_overflow


def test_context_overflow_413():
    cat, _ = classify_error(_StatusExc("payload too large", 413))
    assert cat == ErrorCategory.context_overflow


def test_auth_error_401():
    cat, _ = classify_error(_StatusExc("unauthorized", 401))
    assert cat == ErrorCategory.auth_error


def test_auth_error_403():
    cat, _ = classify_error(_StatusExc("forbidden", 403))
    assert cat == ErrorCategory.auth_error


def test_auth_error_by_text():
    cat, _ = classify_error(ValueError("invalid api key provided"))
    assert cat == ErrorCategory.auth_error


def test_rate_limit_429():
    cat, _ = classify_error(_StatusExc("too many requests", 429))
    assert cat == ErrorCategory.rate_limit


def test_rate_limit_by_text():
    cat, _ = classify_error(ValueError("rate limit exceeded, try again in 10s"))
    assert cat == ErrorCategory.rate_limit


def test_server_error_500():
    cat, _ = classify_error(_StatusExc("internal server error", 500))
    assert cat == ErrorCategory.server_error


def test_server_error_503():
    cat, _ = classify_error(_StatusExc("service unavailable", 503))
    assert cat == ErrorCategory.server_error


def test_format_error_400():
    cat, _ = classify_error(_StatusExc("bad request", 400))
    assert cat == ErrorCategory.format_error


def test_timeout_by_text():
    cat, _ = classify_error(RuntimeError("request timed out"))
    assert cat == ErrorCategory.timeout


def test_timeout_by_type():
    cat, _ = classify_error(TimeoutError("deadline exceeded"))
    assert cat == ErrorCategory.timeout


def test_unknown():
    cat, _ = classify_error(ValueError("something completely unexpected"))
    assert cat == ErrorCategory.unknown


def test_auth_takes_priority_over_rate_limit():
    # 403 looks like forbidden, not rate_limit
    cat, _ = classify_error(_StatusExc("forbidden access", 403))
    assert cat == ErrorCategory.auth_error


# ── is_retryable ───────────────────────────────────────────────────────

@pytest.mark.parametrize("cat,expected", [
    (ErrorCategory.rate_limit, True),
    (ErrorCategory.timeout, True),
    (ErrorCategory.server_error, True),
    (ErrorCategory.unknown, True),
    (ErrorCategory.auth_error, False),
    (ErrorCategory.format_error, False),
    (ErrorCategory.context_overflow, False),
])
def test_is_retryable(cat, expected):
    assert is_retryable(cat) == expected


# ── should_compact ─────────────────────────────────────────────────────

def test_should_compact_only_for_context_overflow():
    assert should_compact(ErrorCategory.context_overflow) is True
    assert should_compact(ErrorCategory.rate_limit) is False
    assert should_compact(ErrorCategory.server_error) is False
