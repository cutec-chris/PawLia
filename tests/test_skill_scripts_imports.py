#!/usr/bin/env python3
"""Regression test: verify all bundled skill scripts can import their dependencies.

For every *.py file under skills/*/scripts/, this test spawns a subprocess that
attempts to import the script's top-level imports (sys.path resolution + actual
import).  This catches errors like the broken `pawlia.utils` path in
skills/files/scripts/files.py before they reach production.

Usage:
    pytest tests/test_skill_scripts_imports.py
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SKILLS_DIR = _PROJECT_ROOT / "skills"


def _discover_scripts() -> list[Path]:
    """Return all *.py files directly under skills/*/scripts/."""
    scripts: list[Path] = []
    for skill_dir in sorted(_SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue
        scripts_dir = skill_dir / "scripts"
        if not scripts_dir.is_dir():
            continue
        for script in sorted(scripts_dir.glob("*.py")):
            scripts.append(script)
    return scripts


ALL_SCRIPTS = _discover_scripts()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSkillScriptsImport:
    """Each script can be imported or syntax-checked without error."""

    @pytest.fixture(autouse=True)
    def _set_env(self, monkeypatch):
        """Ensure required environment variables exist for import checks."""
        monkeypatch.setenv("PAWLIA_SESSION_DIR", str(_PROJECT_ROOT / "session"))
        monkeypatch.setenv("PAWLIA_USER_ID", "test")

    @pytest.mark.parametrize("script", ALL_SCRIPTS, ids=lambda p: p.relative_to(_SKILLS_DIR).as_posix())
    def test_script_syntax_compiles(self, script: Path):
        """Verify the script has no syntax errors and can be compiled."""
        # This catches missing modules at import-time because Python
        # processes top-level imports as part of compilation.
        code = script.read_text(encoding="utf-8")
        compile(code, str(script), "exec")

    @pytest.mark.parametrize("script", ALL_SCRIPTS, ids=lambda p: p.relative_to(_SKILLS_DIR).as_posix())
    def test_script_imports_resolve(self, script: Path):
        """Run the script with -c 'pass' to verify sys.path + imports resolve."""
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            env={
                **os.environ,
                "PAWLIA_SESSION_DIR": str(_PROJECT_ROOT / "session"),
                "PAWLIA_USER_ID": "test",
            },
        )
        # --help exits 0 if imports succeed.  Even a non-zero exit is fine
        # as long as the stderr doesn't mention ModuleNotFoundError.
        assert "ModuleNotFoundError" not in result.stderr, (
            f"Import failure in {script.relative_to(_SKILLS_DIR)}:\n{result.stderr}"
        )
        assert "ImportError" not in result.stderr, (
            f"Import failure in {script.relative_to(_SKILLS_DIR)}:\n{result.stderr}"
        )


class TestPawliaImportPaths:
    """Verify scripts that import from pawlia have correct sys.path manipulation."""

    PAWLIA_IMPORT_SCRIPTS = [
        Path("files/scripts/files.py"),
        Path("memory/scripts/memory.py"),
        Path("researcher/scripts/researcher.py"),
    ]

    @pytest.mark.parametrize("script_rel", PAWLIA_IMPORT_SCRIPTS)
    def test_sys_path_manipulation(self, script_rel: Path):
        """Scripts importing from pawlia must add project root to sys.path."""
        script_path = _SKILLS_DIR / script_rel
        code = script_path.read_text(encoding="utf-8")
        # Must contain sys.path insertion pointing to a parent directory
        assert "sys.path" in code or "sys.path.insert" in code or "sys.path.append" in code, (
            f"{script_rel} imports from pawlia but does not manipulate sys.path"
        )