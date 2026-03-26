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

from pawlia.skills.workflow_schema import CompiledWorkflow
from pawlia.utils import collect_skill_dirs, parse_frontmatter

logger = logging.getLogger(__name__)


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


COMPILER_PROMPT = """\
You are a workflow compiler. Output ONLY valid YAML — no explanation, \
no markdown fences. Analyse a skill's SKILL.md and produce a workflow.yaml that a \
small language model can follow step-by-step.

## Input
- The full SKILL.md content (instructions, examples, error recovery)
- The list of available scripts in the skill's scripts/ directory

## Output
Produce **only** valid YAML (no markdown fences, no explanation) that matches \
this exact schema:

```
skill: <skill_name>
version: "<version from SKILL.md>"
compiled_at: "<today>"
compiled_by: "<your model name>"

workflows:
  - id: <unique_id>
    trigger: "<when should this workflow be chosen — 1 sentence>"
    max_steps: <int, safety limit>
    goal_check:                    # optional
      prompt: "<question to check if user goal was reached>"
      max_retries: 2

    building_blocks:
      - id: <block_id>
        command: "<bash command template with {param} placeholders>"
        description: "<1 sentence: what this block does>"
        status_desc: "<short user-facing status with {param} placeholders, e.g. 'Öffne {url}'>"
        verify:                     # optional
          exit_code: 0
          output_contains: []       # strings that MUST appear in stdout
          output_not_contains: []   # strings that must NOT appear in stdout
          output_regex: null        # optional regex
        on_error: <block_id>        # optional: block to run on failure
```

## Rules
1. One workflow per distinct procedure. One workflow for all use-cases is fine.
2. Each workflow MUST list every possible action as a building_block.
3. Use ONLY commands from the SKILL.md. Do NOT invent new ones.
4. ALL placeholders use CURLY braces: {url}, {element_id}, {scripts_dir}, etc. \
Convert <angle_brackets> from SKILL.md to {curly_braces}.
5. Map error-recovery from SKILL.md to on_error references.
6. verify: For output_not_contains use EXACT error strings from the SKILL.md \
error table. Leave output_contains EMPTY if unsure — never guess output strings.
7. status_desc: Short, German, with {param} placeholders. Shown to the user.
8. Include goal_check for multi-step interactive skills. Skip for simple lookups.
9. Be COMPACT. No quotes around strings unless YAML requires them. \
Omit optional fields that are null/empty.
10. PREFER fewer workflows. If all commands belong to the same script, \
use ONE workflow with all building_blocks. Only split into multiple workflows \
when the procedures are truly independent (different scripts, different triggers).
"""


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
        SystemMessage(content=COMPILER_PROMPT),
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

        raw = _extract_yaml(raw_content)

        # Parse and validate
        try:
            data = yaml.safe_load(raw)
            compiled = CompiledWorkflow(**data)
        except Exception as exc:
            logger.error(
                "Failed to parse compiler output for '%s' (attempt %d): %s\n--- raw output (last 500) ---\n%s",
                skill_name, attempt, exc, raw[-500:],
            )
            if attempt < max_retries:
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
