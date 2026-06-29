"""Coding backend for skill script generation and debugging.

Single in-process path: a direct LLM call through ``LLMFactory`` resolves
the agent type ``coder`` and writes the model's fenced file blocks to
disk. Success is measured by files actually written (``ok = len(files_written) > 0``).

The skill-creator and the SkillRunner direct-passthrough use the public
``run_implement`` / ``run_fix`` entry points. Both go through
``_run_llm``; there is no CLI fallback, no long-lived daemon, and no
auto-detection to misfire on a stale binary in the image.

The ``coder`` agent type is configured under ``agents.coder`` in
``config.yaml``; it falls back to the first defined model if unset.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _references_dir() -> Path:
    return _PROJECT_ROOT / "skills" / "skill-creator" / "references"


def _build_task_prompt(
    skill_md_body: str,
    task: str,
    existing_files: Dict[str, str],
    references: Dict[str, str],
    mode: str,
    error_output: str = "",
    failing_command: str = "",
) -> str:
    """Build the task prompt for the LLM."""
    parts = [f"# Task\n{task}\n"]

    if mode == "fix":
        parts.append(f"## Error\n{error_output}\n")
        if failing_command:
            parts.append(f"## Failing command\n```\n{failing_command}\n```\n")

    parts.append("## SKILL.md instructions\n```")
    parts.append(skill_md_body)
    parts.append("```\n")

    if existing_files:
        parts.append("## Existing files\n")
        for fname, content in existing_files.items():
            parts.append(f"### {fname}\n```")
            parts.append(content)
            parts.append("```\n")

    if references:
        parts.append("## References\n")
        for rname, content in references.items():
            parts.append(f"### {rname}\n```")
            parts.append(content[:2000])
            parts.append("```\n")

    if mode == "implement":
        parts.append(
            "## Rules\n"
            "- Scripts must output JSON with `success` (bool) + result fields\n"
            "- Exit 0 on success, non-zero on failure\n"
            "- Read config from PAWLIA_SKILL_CONFIG env var, credentials from CRED_* env vars\n"
            "- Scripts receive PAWLIA_SESSION_DIR and PAWLIA_USER_ID env vars\n"
            "- Propagate real upstream errors (status code + body), never generic messages\n"
            "- Do NOT pre-format user-facing text — return structured data only\n"
        )
    elif mode == "fix":
        parts.append(
            "## Rules\n"
            "- Fix the root cause, not the symptom\n"
            "- Preserve the existing CLI interface (argparse args)\n"
            "- Keep the same JSON output format\n"
            "- Add error handling for the specific failure mode\n"
        )

    return "\n".join(parts)


def _collect_skill_files(skill_path: Path) -> Dict[str, str]:
    """Collect existing script files from a skill directory."""
    files = {}
    scripts_dir = skill_path / "scripts"
    if scripts_dir.is_dir():
        for f in scripts_dir.iterdir():
            if f.is_file() and not f.name.startswith("."):
                try:
                    files[f"scripts/{f.name}"] = f.read_text(encoding="utf-8")
                except OSError:
                    pass
    for name in ("harness.sh", "harness.py", "harness.mjs"):
        p = skill_path / name
        if p.is_file():
            try:
                files[name] = p.read_text(encoding="utf-8")
            except OSError:
                pass
    return files


def _collect_references(skill_path: Path) -> Dict[str, str]:
    """Collect reference files from the skill and from skill-creator."""
    refs = {}
    refs_dir = skill_path / "references"
    if refs_dir.is_dir():
        for f in refs_dir.iterdir():
            if f.is_file() and f.suffix == ".md":
                try:
                    refs[f.name] = f.read_text(encoding="utf-8")
                except OSError:
                    pass
    creator_refs = _references_dir()
    if creator_refs.is_dir():
        for name in ("patterns.md", "design-principles.md"):
            p = creator_refs / name
            if p.is_file():
                try:
                    refs[f"creator/{name}"] = p.read_text(encoding="utf-8")
                except OSError:
                    pass
    return refs


def _parse_skill_md(skill_path: Path) -> Optional[str]:
    """Extract SKILL.md body (after frontmatter)."""
    skill_md = skill_path / "SKILL.md"
    if not skill_md.is_file():
        return None
    content = skill_md.read_text(encoding="utf-8")
    parts = content.split("---", 2)
    return parts[2].strip() if len(parts) >= 3 else content.strip()


# ── LLM backend (single in-process path) ────────────────────────────────

def _run_llm(
    skill_path: Path,
    task_prompt: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Direct LLM call for code generation.

    Resolves the ``coder`` agent via :class:`LLMFactory` and writes the
    model's fenced file blocks to ``skill_path``. ``ok`` is derived from
    the number of files actually written, so a model that only produces
    text without a parseable fence block is reported as a failure.
    """
    from pawlia.llm import LLMFactory

    factory = LLMFactory(config)
    llm = factory.get("coder")

    system = (
        "You are a code generator for PawLia skills. "
        "Output ONLY file contents. For each file, output:\n"
        "```<filename>\n<content>\n```\n"
        "Generate complete, working code. No explanations."
    )

    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=task_prompt),
    ]

    logger.info("Running LLM backend for code generation...")
    try:
        response = llm.invoke(messages)
        content = response.content or ""
    except Exception as e:
        return {"ok": False, "backend": "llm", "error": str(e)}

    files_written = _extract_and_write_files(content, skill_path)

    return {
        "ok": len(files_written) > 0,
        "backend": "llm",
        "output": content[-2000:],
        "files_written": files_written,
    }


def _extract_and_write_files(content: str, skill_path: Path) -> List[str]:
    """Extract ````<filename>` blocks from LLM output and write them.

    The regex matches the language tag of any fenced code block; we treat
    it as the filename unless it is a bare language identifier (``python``,
    ``bash``, …) with no path separator. Filenames without a ``/`` land in
    ``scripts/`` so a ``fix`` reply that just dumps a single ``.py`` block
    still ends up where the harness expects it. Paths starting with
    ``scripts/`` get their parent directories created on demand; paths
    elsewhere only write if the parent directory already exists.
    """
    import re

    written = []
    pattern = re.compile(r"```(\S+)\n(.*?)```", re.DOTALL)
    for match in pattern.finditer(content):
        filename = match.group(1).strip()
        file_content = match.group(2)

        if not filename or filename in ("python", "bash", "sh", "javascript",
                                         "json", "yaml", "text", "md"):
            continue

        if "/" in filename:
            target = skill_path / filename
            if not target.parent.is_dir() and not filename.startswith("scripts/"):
                # Refuse to silently create arbitrary parent dirs.
                continue
        else:
            target = skill_path / "scripts" / filename

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(file_content, encoding="utf-8")
            written.append(str(target.relative_to(skill_path)))
            logger.info("Wrote %s", target)
        except OSError as e:
            logger.warning("Failed to write %s: %s", target, e)

    return written


# ── Public API ─────────────────────────────────────────────────────────

def run_implement(
    skill_path: Path,
    task: str,
    config: Dict[str, Any],
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Implement scripts for a skill via the configured LLM.

    ``user_id`` is accepted for backward compatibility (the previous
    daemon-based backend used it to pin a session per user) and is
    ignored by the in-process path.
    """
    skill_md_body = _parse_skill_md(skill_path)
    existing_files = _collect_skill_files(skill_path)
    references = _collect_references(skill_path)

    task_prompt = _build_task_prompt(
        skill_md_body=skill_md_body or "",
        task=task,
        existing_files=existing_files,
        references=references,
        mode="implement",
    )

    logger.info("Implementing via llm backend: %s", task[:100])
    return _run_llm(skill_path, task_prompt, config)


def run_fix(
    skill_path: Path,
    error: str,
    command: str,
    config: Dict[str, Any],
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Fix a broken skill via the configured LLM.

    ``user_id`` is accepted for backward compatibility and ignored by the
    in-process path.
    """
    skill_md_body = _parse_skill_md(skill_path)
    existing_files = _collect_skill_files(skill_path)
    references = _collect_references(skill_path)

    task = (
        f"Fix the following error in this skill.\n"
        f"Failing command: {command}\n"
        f"Error output: {error}"
    )

    task_prompt = _build_task_prompt(
        skill_md_body=skill_md_body or "",
        task=task,
        existing_files=existing_files,
        references=references,
        mode="fix",
        error_output=error,
        failing_command=command,
    )

    logger.info("Fixing via llm backend: %s", error[:100])
    return _run_llm(skill_path, task_prompt, config)
