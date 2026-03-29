import os
from functools import lru_cache


_SYSTEM_PROMPTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "prompts",
    "system",
)


@lru_cache(maxsize=None)
def _read_system_prompt(name: str) -> str:
    path = os.path.join(_SYSTEM_PROMPTS_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_system_prompt(name: str, **replacements: object) -> str:
    """Load a prompt fragment from pawlia/prompts/system.

    Replacement tokens use the form <<token_name>> inside the prompt files.
    """
    text = _read_system_prompt(name)
    for key, value in replacements.items():
        text = text.replace(f"<<{key}>>", str(value))
    return text