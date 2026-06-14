"""Compile SKILL.md into structured workflow.yaml using a large LLM.

Usage::

    python -m pawlia.skills.compiler --skill browser
    python -m pawlia.skills.compiler --all
    python -m pawlia.skills.compiler --all --force
"""

import argparse
import asyncio
import logging
import os
import re
import sys
from datetime import date
from typing import Any, Dict, Optional

import yaml

from pawlia.prompt_utils import load_system_prompt
from pawlia.skills.workflow_schema import CompiledWorkflow
from pawlia.utils import collect_skill_dirs, parse_frontmatter

logger = logging.getLogger(__name__)

# Skills that should never be compiled into workflow.yaml because their
# tasks involve free-form, multi-line text that breaks rigid building-block
# command substitution (e.g. implement/fix with arbitrary task descriptions).
_SKIP_COMPILE = {"skill-creator"}

_ANGLE_PLACEHOLDER_RE = re.compile(r"<([a-zA-Z_][a-zA-Z0-9_]*)>")

# Directory placeholders the executor actually resolves (pawlia/skills/executor.py
# _substitute). Any *other* placeholder used as a script-path prefix (e.g. a
# {skill_dir} invented from a `SKILL_DIR="<scripts_dir>"` shell variable) is left
# unresolved and breaks the command at runtime — so the lint rejects it.
_ALLOWED_PATH_VARS = {"scripts_dir", "skills_root"}


def _extract_yaml(text: str) -> str:
    """Extract YAML from LLM output, stripping think tags and markdown fences."""
    from pawlia.agents.base import BaseAgent

    text = BaseAgent.strip_thinking(text)

    # Handle unclosed <think> — find first line starting with "skill:"
    if "<think>" in text or "<thinking>" in text:
        for i, line in enumerate(text.split("\n")):
            if line.strip().startswith("skill:"):
                text = "\n".join(text.split("\n")[i:])
                break

    text = text.strip()

    # Strip markdown fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]

    return text.strip()

def _build_user_prompt(
    skill_name: str,
    version: str,
    instructions: str,
    scripts: list[str],
    today: str,
) -> str:
    parts = [
        f"# Skill: {skill_name}  (version {version})",
        f"# Date: {today}",
        "",
        "## SKILL.md content",
        instructions,
        "",
        f"## Available scripts: {', '.join(scripts) if scripts else '(none)'}",
    ]
    return "\n".join(parts)


def _normalize_placeholders(value: Any) -> Any:
    """Recursively normalize <param> placeholders to {param}."""
    if isinstance(value, str):
        return _ANGLE_PLACEHOLDER_RE.sub(r"{\1}", value)
    if isinstance(value, list):
        return [_normalize_placeholders(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_placeholders(v) for k, v in value.items()}
    return value


def _lint_compiled_workflow(
    compiled: CompiledWorkflow,
    *,
    today: str,
    scripts: list[str],
) -> list[str]:
    """Return semantic issues that should trigger a compiler retry."""
    issues: list[str] = []

    if compiled.compiled_at != today:
        issues.append(f"compiled_at must be {today}, got {compiled.compiled_at}")

    for workflow in compiled.workflows:
        for block in workflow.building_blocks:
            command = block.command or ""
            if "[" in command or "]" in command:
                issues.append(
                    f"block {block.id} uses markdown-style optional syntax in command: {command}"
                )
            if scripts and not any(script in command for script in scripts):
                issues.append(
                    f"block {block.id} command does not reference a known script: {command}"
                )
            # A script invoked through a path prefix must use {scripts_dir} (the
            # only script-dir placeholder the executor resolves). Catch invented
            # prefixes like {skill_dir}/foo.mjs that would never be substituted.
            for script in scripts:
                for match in re.finditer(r"\{(\w+)\}/" + re.escape(script), command):
                    var = match.group(1)
                    if var not in _ALLOWED_PATH_VARS:
                        issues.append(
                            f"block {block.id} invokes '{script}' via unresolved path "
                            f"placeholder {{{var}}} — use {{scripts_dir}}/{script} "
                            f"(only {{scripts_dir}}/{{skills_root}} are resolved at runtime)"
                        )

    return issues


async def compile_skill(
    skill_path: str,
    llm: Any,
    *,
    force: bool = False,
    max_retries: int = 2,
) -> Optional[CompiledWorkflow]:
    """Compile a single skill's SKILL.md into workflow.yaml.

    Returns the compiled workflow on success, None on skip/failure.
    """
    skill_md = os.path.join(skill_path, "SKILL.md")
    workflow_path = os.path.join(skill_path, "workflow.yaml")

    metadata = parse_frontmatter(skill_md)
    if not metadata or not metadata.get("name"):
        logger.warning("No valid frontmatter in %s — skipping", skill_md)
        return None

    skill_name = metadata["name"]
    if skill_name in _SKIP_COMPILE:
        logger.info("Skill '%s' is blacklisted from compilation — skipping", skill_name)
        return None

    version = str(metadata.get("metadata", {}).get("version", "1.0"))

    # Check if already compiled and up-to-date
    if not force and os.path.isfile(workflow_path):
        try:
            with open(workflow_path, encoding="utf-8") as f:
                existing = yaml.safe_load(f)
            if existing and existing.get("version") == version:
                logger.info("Skill '%s' already compiled (v%s) — skipping", skill_name, version)
                return CompiledWorkflow(**existing)
        except Exception:
            pass  # re-compile on any parse error

    # Load instructions
    with open(skill_md, encoding="utf-8") as f:
        content = f.read()
    parts = content.split("---", 2)
    instructions = parts[2].strip() if len(parts) >= 3 else content.strip()

    # List scripts
    scripts_dir = os.path.join(skill_path, "scripts")
    scripts: list[str] = []
    if os.path.isdir(scripts_dir):
        scripts = sorted(os.listdir(scripts_dir))

    today = date.today().isoformat()
    user_prompt = _build_user_prompt(skill_name, version, instructions, scripts, today)

    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=load_system_prompt("skills/compiler_system.md")),
        HumanMessage(content=user_prompt),
    ]

    for attempt in range(1, max_retries + 1):
        logger.info("Compiling skill '%s' v%s (attempt %d/%d) ...", skill_name, version, attempt, max_retries)

        try:
            response = await llm.ainvoke(messages)
        except Exception as exc:
            logger.error("LLM error compiling '%s': %s", skill_name, exc)
            if attempt < max_retries:
                continue
            return None

        raw_content = (response.content or "").strip()
        if not raw_content or (raw_content.startswith("<think") and "</think" not in raw_content):
            logger.error(
                "Empty or truncated LLM response for '%s' — max_tokens is likely too low",
                skill_name,
            )
            if attempt < max_retries:
                continue
            return None

        # Check if the compiler judged the skill unsuitable for workflow compilation
        if raw_content == "UNSUITABLE" or raw_content.splitlines()[0].strip() == "UNSUITABLE":
            logger.info("Compiler judged skill '%s' as unsuitable for workflow compilation — skipping", skill_name)
            return None

        raw = _extract_yaml(raw_content)

        # Parse and validate
        try:
            data = _normalize_placeholders(yaml.safe_load(raw))
            compiled = CompiledWorkflow(**data)
        except Exception as exc:
            logger.error(
                "Failed to parse compiler output for '%s' (attempt %d): %s\n--- raw output (last 500) ---\n%s",
                skill_name, attempt, exc, raw[-500:],
            )
            if attempt < max_retries:
                continue
            return None

        issues = _lint_compiled_workflow(compiled, today=today, scripts=scripts)
        if issues:
            logger.error(
                "Semantic compiler issues for '%s' (attempt %d): %s",
                skill_name,
                attempt,
                "; ".join(issues),
            )
            if attempt < max_retries:
                messages = messages + [
                    HumanMessage(
                        content=(
                            "Your previous YAML was schema-valid but semantically invalid. "
                            "Regenerate it and fix these issues exactly:\n- "
                            + "\n- ".join(issues)
                            + "\nDo not use bracketed optional shell syntax like [--foo ...]. "
                            "Invoke every script via {scripts_dir}/<script>. "
                            "Every command must be directly executable as written."
                        )
                    )
                ]
                continue
            return None

        # Write workflow.yaml
        with open(workflow_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        logger.info("Compiled skill '%s' → %s", skill_name, workflow_path)
        return compiled

    return None


async def compile_all(
    skills_dir: str,
    config: Dict[str, Any],
    *,
    force: bool = False,
    skill_filter: Optional[str] = None,
) -> Dict[str, CompiledWorkflow]:
    """Compile all (or one) skills in a directory."""
    from pawlia.llm import LLMFactory

    llm_factory = LLMFactory(config)
    llm = llm_factory.get("compiler")

    results: Dict[str, CompiledWorkflow] = {}

    for skill_path in collect_skill_dirs(skills_dir):
        skill_name = os.path.basename(skill_path)
        if skill_filter and skill_name != skill_filter:
            continue

        compiled = await compile_skill(skill_path, llm, force=force)
        if compiled:
            results[compiled.skill] = compiled

    return results


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Compile SKILL.md → workflow.yaml")
    parser.add_argument("--skill", default=None, help="Compile a single skill by name")
    parser.add_argument("--all", action="store_true", help="Compile all skills")
    parser.add_argument("--force", action="store_true", help="Re-compile even if up-to-date")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--skills-dir", default=None, help="Directory to look for skills in (default: bundled skills/)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.skill and not args.all:
        parser.error("Specify --skill <name> or --all")

    from pawlia.config import load_config

    config = load_config(args.config)

    if args.skills_dir:
        skills_dir = os.path.abspath(args.skills_dir)
    else:
        # __file__ is pawlia/skills/compiler.py → 3 levels up to project root
        pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        skills_dir = os.path.join(pkg_dir, "skills")

    results = await compile_all(
        skills_dir, config,
        force=args.force,
        skill_filter=args.skill,
    )

    if results:
        print(f"Compiled {len(results)} skill(s): {', '.join(results.keys())}")
    else:
        print("No skills compiled.")


if __name__ == "__main__":
    asyncio.run(_main())
