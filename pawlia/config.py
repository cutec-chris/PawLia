"""Configuration loading for PawLia."""

import json
import os
from typing import Any, Dict, Optional

import yaml


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load config from the given path, CWD, or project root.

    Tries config.yaml / config.yml before config.json.
    Returns an empty dict if no config file is found.
    """
    candidates = []
    if path:
        candidates.append(path)

    for base in (os.getcwd(), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        candidates.append(os.path.join(base, "config.yaml"))
        candidates.append(os.path.join(base, "config.yml"))
        candidates.append(os.path.join(base, "config.json"))

    for candidate in candidates:
        if os.path.isfile(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                if candidate.endswith((".yaml", ".yml")):
                    return yaml.safe_load(f) or {}
                return json.load(f)

    return {}
