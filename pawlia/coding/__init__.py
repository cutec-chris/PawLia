"""Coding backend for skill script generation and debugging.

Single in-process path: a direct LLM call through ``LLMFactory`` resolves
the agent type ``coder`` and writes the model's fenced file blocks to
disk. The previous opencode/aider CLI backends and the persistent
opencode daemon were removed because they ran against their own opaque
model configuration, never PawLia's configured ``coder`` model, and
returned ``ok=True`` unconditionally on daemon errors — making failures
look like successes.

Public API: :func:`run_implement` and :func:`run_fix`. The SkillRunner's
direct-passthrough shells out to ``creator.py implement|fix`` (CLI), so
the in-process path is transparent to that flow.
"""

from pawlia.coding.coding import (  # noqa: F401  (re-export public API)
    run_implement,
    run_fix,
    _build_task_prompt,
)

__all__ = [
    "run_implement",
    "run_fix",
    "_build_task_prompt",
]
