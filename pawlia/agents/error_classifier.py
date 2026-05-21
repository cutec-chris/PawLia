"""Lightweight API error classification for pawlia.

Distinguishes the 6 most important failure categories so the retry
loop can pick the correct recovery strategy instead of blindly
re-trying or giving up.

Categories
----------
- ``context_overflow`` — prompt too large → compact / summarize
- ``rate_limit``       — throttled → wait + retry
- ``auth_error``       — bad key → do NOT retry, surface to user
- ``timeout``          — network / server slow → retry with backoff
- ``server_error``     — 5xx → retry with backoff
- ``format_error``     — 400 bad request → surface to user
"""

from __future__ import annotations

import enum
from typing import Optional


class ErrorCategory(enum.Enum):
    context_overflow = "context_overflow"
    rate_limit = "rate_limit"
    auth_error = "auth_error"
    timeout = "timeout"
    server_error = "server_error"
    format_error = "format_error"
    unknown = "unknown"


# ── Pattern lists ──────────────────────────────────────────────────────

_CONTEXT_OVERFLOW_HINTS = (
    "context_length_exceeded",
    "prompt exceeds max length",
    "maximum context length",
    "maximum context",
    "context window",
    "please reduce the length of the messages or completion",
    "too many tokens",
    "context length exceeded",
    "truncating input",
    "exceeds the max_model_len",
    "max_model_len",
    "prompt length",
    "input is too long",
    "maximum model length",
    "超过最大长度",
    "上下文长度",
    "input token",
    "exceeds the maximum number of input tokens",
    # ZhiPu GLM error 1214: triggered by accumulated multi-turn tool messages
    "messages parameter is illegal",
    "'1214'",
    "\"1214\"",
)

_RATE_LIMIT_HINTS = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "throttled",
    "requests per minute",
    "tokens per minute",
    "requests per day",
    "try again in",
    "please retry after",
    "resource_exhausted",
    "throttlingexception",
    "too many concurrent requests",
    "servicequotaexceededexception",
    "usage limit",
    "key limit exceeded",
)

_AUTH_HINTS = (
    "invalid api key",
    "invalid_api_key",
    "authentication",
    "unauthorized",
    "forbidden",
    "invalid token",
    "token expired",
    "token revoked",
    "access denied",
)

_TIMEOUT_HINTS = (
    "timed out",
    "turn timed out",
    "request timed out",
    "deadline exceeded",
    "operation timed out",
    "upstream timed out",
)

_SERVER_DISCONNECT_HINTS = (
    "server disconnected",
    "peer closed connection",
    "connection reset by peer",
    "connection was closed",
    "network connection lost",
    "unexpected eof",
    "incomplete chunked read",
)

_TRANSPORT_ERROR_TYPES = frozenset({
    "ReadTimeout", "ConnectTimeout", "PoolTimeout",
    "ConnectError", "RemoteProtocolError",
    "ConnectionError", "ConnectionResetError",
    "ConnectionAbortedError", "BrokenPipeError",
    "TimeoutError", "ReadError",
    "ServerDisconnectedError",
    "SSLError", "SSLZeroReturnError",
})


def classify_error(exc: BaseException) -> tuple[ErrorCategory, str]:
    """Classify an exception into a recovery category.

    Returns ``(category, detail_message)``.
    """
    text = str(exc).lower()
    exc_type = type(exc).__name__

    # Status code extraction (walks cause chain)
    status_code = _extract_status_code(exc)

    # Context overflow
    if any(h in text for h in _CONTEXT_OVERFLOW_HINTS):
        return ErrorCategory.context_overflow, _short_detail(text, 200)

    # Auth errors — check before rate_limit because some 403s look like both
    if status_code == 401:
        return ErrorCategory.auth_error, _short_detail(text, 200)
    if status_code == 403:
        return ErrorCategory.auth_error, _short_detail(text, 200)
    if any(h in text for h in _AUTH_HINTS):
        return ErrorCategory.auth_error, _short_detail(text, 200)

    # Rate limiting
    if status_code == 429:
        return ErrorCategory.rate_limit, _short_detail(text, 200)
    if any(h in text for h in _RATE_LIMIT_HINTS):
        return ErrorCategory.rate_limit, _short_detail(text, 200)

    # Payload too large → treat as context overflow
    if status_code == 413 or "request entity too large" in text or "payload too large" in text:
        return ErrorCategory.context_overflow, _short_detail(text, 200)

    # Server errors
    if status_code is not None and 500 <= status_code < 600:
        return ErrorCategory.server_error, _short_detail(text, 200)

    # Format errors — 400 that is NOT context overflow
    if status_code == 400:
        return ErrorCategory.format_error, _short_detail(text, 200)

    # Timeout patterns
    if any(h in text for h in _TIMEOUT_HINTS):
        return ErrorCategory.timeout, _short_detail(text, 200)

    # Server disconnect → timeout (could be context overflow on large
    # sessions, but we treat as timeout to avoid expensive summarization)
    if any(h in text for h in _SERVER_DISCONNECT_HINTS):
        return ErrorCategory.timeout, _short_detail(text, 200)

    # Transport error types
    if exc_type in _TRANSPORT_ERROR_TYPES or isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return ErrorCategory.timeout, _short_detail(text, 200)

    # Tool-use errors from OpenAI-compatible APIs (model writes tool call
    # as JSON instead of using structured tool_calls)
    if "tool_use_failed" in text or (
        "tool choice is none" in text and "called a tool" in text
    ):
        # Special category: the model tried to call a tool despite
        # tool_choice="none". Caller handles this by injecting a
        # synthetic ToolMessage.
        return ErrorCategory.format_error, _short_detail(text, 200)

    return ErrorCategory.unknown, _short_detail(text, 200)


def is_retryable(category: ErrorCategory) -> bool:
    """Return True if the error category should be retried."""
    return category in {
        ErrorCategory.rate_limit,
        ErrorCategory.timeout,
        ErrorCategory.server_error,
        ErrorCategory.unknown,
    }


def should_compact(category: ErrorCategory) -> bool:
    """Return True if the prompt should be compacted before retry."""
    return category == ErrorCategory.context_overflow


def _extract_status_code(exc: BaseException) -> Optional[int]:
    """Walk the exception and its cause chain for an HTTP status code."""
    current: BaseException | None = exc
    for _ in range(5):
        if current is None:
            break
        code = getattr(current, "status_code", None)
        if isinstance(code, int):
            return code
        code = getattr(current, "status", None)
        if isinstance(code, int) and 100 <= code < 600:
            return code
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        if current is exc:
            break
    return None


def _short_detail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."
