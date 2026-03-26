"""Install runtime dependencies and compile workflows for AgentSkills.

Handles both steps in one call: pip/npm deps first, then workflow compilation.
Called during Docker build and when skills are uploaded at runtime.

Usage::

    python -m pawlia.install_skill_deps [skills_dir]
    python -m pawlia.install_skill_deps [skills_dir] --no-compile
    python -m pawlia.install_skill_deps [skills_dir] --force
"""

import asyncio
import logging
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional

from pawlia.utils import collect_skill_dirs, parse_frontmatter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("install_skill_deps")


def _install_deps(skills_dir: str) -> None:
    """Install pip and npm dependencies for all skills."""
    if not os.path.isdir(skills_dir):
        logger.info("No skills directory at %s", skills_dir)
        return

    for skill_path in collect_skill_dirs(skills_dir):
        skill_name = os.path.basename(skill_path)

        # ── pip: requirements.txt ──
        req_txt = os.path.join(skill_path, "requirements.txt")
        if os.path.isfile(req_txt):
            logger.info("Installing pip deps for skill '%s'...", skill_name)
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "-r", req_txt],
                    check=True, capture_output=True, text=True,
                )
                logger.info("pip install -r requirements.txt for '%s' → OK", skill_name)
            except subprocess.CalledProcessError as e:
                logger.warning("pip install for '%s' failed: %s", skill_name, e.stderr.strip())

        # ── npm: openclaw.install steps ──
        skill_md = os.path.join(skill_path, "SKILL.md")
        fm = parse_frontmatter(skill_md) or {}
        steps = fm.get("metadata", {}).get("openclaw", {}).get("install", [])
        if not steps:
            continue

        logger.info("Installing npm deps for skill '%s'...", skill_name)
        for step in steps:
            kind = step.get("kind")
            package = step.get("package")
            if not kind or not package:
                continue
            if kind == "node":
                # Strip version spec: "pkg@1.0" → "pkg", "@org/pkg@1.0" → "@org/pkg"
                if package.startswith("@"):
                    parts = package[1:].split("@")
                    pkg_name = "@" + parts[0]
                else:
                    pkg_name = package.split("@")[0]
                node_modules = os.path.join(skill_path, "node_modules", pkg_name)
                if os.path.isdir(node_modules):
                    logger.debug("npm package '%s' already installed — skipping", package)
                    continue
                npm = shutil.which("npm")
                if not npm:
                    logger.warning("npm not found — skipping '%s'", package)
                    continue
                try:
                    subprocess.run(
                        [npm, "install", package],
                        cwd=skill_path, check=True,
                        capture_output=True, text=True,
                    )
                    logger.info("npm install %s → OK", package)
                except subprocess.CalledProcessError as e:
                    logger.warning("npm install %s failed: %s", package, e.stderr.strip())
            else:
                logger.debug("Unknown install kind '%s' for '%s' — skipping", kind, package)


async def _compile_workflows(
    skills_dir: str,
    config: Optional[Dict[str, Any]] = None,
    *,
    force: bool = False,
) -> None:
    """Compile workflows for all skills that need it."""
    if config is None:
        from pawlia.config import load_config
        config = load_config()

    from pawlia.skills.compiler import compile_all
    results = await compile_all(skills_dir, config, force=force)
    if results:
        logger.info("Compiled workflows: %s", ", ".join(results.keys()))


async def install_skills(
    skills_dir: str,
    config: Optional[Dict[str, Any]] = None,
    *,
    compile: bool = True,
    force: bool = False,
) -> None:
    """Install deps + compile workflows for all skills in a directory.

    This is the single entry point — called from Docker build, app startup
    (for workspace skills), and the web skill-upload handler.
    """
    _install_deps(skills_dir)
    if compile:
        await _compile_workflows(skills_dir, config, force=force)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Install skill deps + compile workflows")
    parser.add_argument("skills_dir", nargs="?", default=None)
    parser.add_argument("--no-compile", action="store_true", help="Skip workflow compilation")
    parser.add_argument("--force", action="store_true", help="Force recompilation of all workflows")
    args = parser.parse_args()

    skills_dir = args.skills_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "skills"
    )
    asyncio.run(install_skills(
        skills_dir,
        compile=not args.no_compile,
        force=args.force,
    ))
