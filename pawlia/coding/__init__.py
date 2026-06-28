"""Coding backends for skill script generation and debugging.

This package re-exports the public API of :mod:`pawlia.coding.coding`
(``run_implement``, ``run_fix``, backend detection) and exposes the new
long-lived opencode daemon under :mod:`pawlia.coding.opencode_daemon`.

History
-------
The original layout was a single ``pawlia/coding.py`` file. The package
form was introduced when the persistent opencode daemon was added —
a single file could not host the daemon module (Python would not let
``from pawlia.coding.opencode_daemon import …`` work when
``pawlia/coding.py`` shadowed the package). The flat file is now at
``pawlia/coding/coding.py`` and is the source of truth for the public
functions; this ``__init__`` just re-exports them so call sites can
keep using ``from pawlia.coding import run_implement``.

A few underscored helpers (``_build_task_prompt``) are also re-exported
because the unit tests reference them directly. Treat them as internal.
"""

from pawlia.coding.coding import (  # noqa: F401  (re-export public API)
    run_implement,
    run_fix,
    detect_backend,
    install_backend,
    backend_available,
    _build_task_prompt,
)

__all__ = [
    "run_implement",
    "run_fix",
    "detect_backend",
    "install_backend",
    "backend_available",
    "_build_task_prompt",
]
