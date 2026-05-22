"""Shared configuration helpers for audio processing components."""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("pawlia.audio.config")


def get_float_config(
    cfg: Dict[str, Any],
    key: str,
    default: float,
    context: str = "",
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    value = cfg.get(key, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        logger.warning(
            "%s: invalid voip.%s=%r, using default %s",
            context, key, value, default,
        )
        return default

    if minimum is not None and value < minimum:
        logger.warning(
            "%s: voip.%s=%s below minimum %s, using default %s",
            context, key, value, minimum, default,
        )
        return default
    if maximum is not None and value > maximum:
        logger.warning(
            "%s: voip.%s=%s above maximum %s, using default %s",
            context, key, value, maximum, default,
        )
        return default
    return value


def get_int_config(
    cfg: Dict[str, Any],
    key: str,
    default: int,
    context: str = "",
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    value = cfg.get(key, default)
    try:
        value = int(value)
    except (TypeError, ValueError):
        logger.warning(
            "%s: invalid voip.%s=%r, using default %s",
            context, key, value, default,
        )
        return default

    if minimum is not None and value < minimum:
        logger.warning(
            "%s: voip.%s=%s below minimum %s, using default %s",
            context, key, value, minimum, default,
        )
        return default
    if maximum is not None and value > maximum:
        logger.warning(
            "%s: voip.%s=%s above maximum %s, using default %s",
            context, key, value, maximum, default,
        )
        return default
    return value


def get_bool_config(
    cfg: Dict[str, Any],
    key: str,
    default: bool,
    context: str = "",
) -> bool:
    value = cfg.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    logger.warning(
        "%s: invalid boolean voip.%s=%r — using default %r",
        context, key, value, default,
    )
    return default
