"""Install runtime dependencies for pre-bundled AgentSkills.

Called during Docker build to set up npm packages declared in SKILL.md
under ``metadata.openclaw.install``.

Usage::

    python -m pawlia.install_skill_deps [skills_dir]
"""

import logging
import os
import shutil
import subprocess
import sys

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("install_skill_deps")


def _parse_frontmatter(skill_md: str) -> dict:
    try:
        with open(skill_md, encoding="utf-8") as f:
            content = f.read()
        parts = content.split("---", 2)
        if len(parts) < 3:
            return {}
        return yaml.safe_load(parts[1]) or {}
    except Exception:
        return {}


def _collect_skill_dirs(skills_dir: str) -> list:
    """Collect skill directories: direct children + skills/user/*."""
    candidates = []
    if not os.path.isdir(skills_dir):
        return candidates

    for entry in os.listdir(skills_dir):
        entry_path = os.path.join(skills_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        if os.path.isfile(os.path.join(entry_path, "SKILL.md")):
            candidates.append(entry_path)

    user_dir = os.path.join(skills_dir, "user")
    if os.path.isdir(user_dir):
        for entry in os.listdir(user_dir):
            entry_path = os.path.join(user_dir, entry)
            if os.path.isdir(entry_path) and os.path.isfile(
                os.path.join(entry_path, "SKILL.md")
            ):
                candidates.append(entry_path)

    return candidates


def install_all_skill_deps(skills_dir: str) -> None:
    if not os.path.isdir(skills_dir):
        logger.info("No skills directory at %s", skills_dir)
        return

    for skill_path in _collect_skill_dirs(skills_dir):
        skill_md = os.path.join(skill_path, "SKILL.md")
        fm = _parse_frontmatter(skill_md)
        steps = fm.get("metadata", {}).get("openclaw", {}).get("install", [])
        if not steps:
            continue

        logger.info("Installing deps for skill '%s'...", os.path.basename(skill_path))
        for step in steps:
            kind = step.get("kind")
            package = step.get("package")
            if not kind or not package:
                continue
            if kind == "node":
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


if __name__ == "__main__":
    skills_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "skills"
    )
    install_all_skill_deps(skills_dir)
