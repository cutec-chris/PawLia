"""AgentSkill loading and discovery from SKILL.md files."""

import logging
import os
from typing import Any, Dict, List, Optional

import yaml


logger = logging.getLogger(__name__)


class AgentSkill:
    """A single agent skill loaded from a SKILL.md file."""

    def __init__(self, skill_path: str, metadata: Dict[str, Any]):
        self.skill_path = skill_path
        self.metadata = metadata
        self.name: str = metadata.get("name", "")
        self.description: str = metadata.get("description", "")
        self.scripts_dir = os.path.join(skill_path, "scripts")
        self.instructions = self._load_instructions()

    def _load_instructions(self) -> str:
        """Load the Markdown body from SKILL.md (after YAML frontmatter)."""
        skill_md = os.path.join(self.skill_path, "SKILL.md")
        with open(skill_md, encoding="utf-8") as f:
            content = f.read()
        parts = content.split("---", 2)
        return parts[2].strip() if len(parts) >= 3 else content.strip()

    def as_openai_spec(self) -> Dict[str, Any]:
        """OpenAI tool spec for the ChatAgent (only name + description + query param)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The task or query for this skill",
                        }
                    },
                    "required": ["query"],
                },
            },
        }


class SkillLoader:
    """Discovers and loads AgentSkills from a directory."""

    @staticmethod
    def discover(
        skills_dir: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, AgentSkill]:
        """Discover all valid skills in the given directory.

        Scans both top-level entries and one level of subdirectories
        (e.g. ``skills/user/bahn/``), so user-provided skills inside
        a ``user/`` folder are loaded the same way as built-in ones.

        Returns a dict mapping skill name -> AgentSkill.
        """
        config = config or {}
        skills: Dict[str, AgentSkill] = {}

        if not os.path.isdir(skills_dir):
            logger.debug("Skills directory not found: %s", skills_dir)
            return skills

        # Collect candidate directories: direct children + skills/user/*
        candidates: List[str] = []
        for entry in os.listdir(skills_dir):
            entry_path = os.path.join(skills_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            if os.path.isfile(os.path.join(entry_path, "SKILL.md")):
                candidates.append(entry_path)

        # User-provided skills live in skills/user/
        user_dir = os.path.join(skills_dir, "user")
        if os.path.isdir(user_dir):
            for sub in os.listdir(user_dir):
                sub_path = os.path.join(user_dir, sub)
                if os.path.isdir(sub_path) and os.path.isfile(
                    os.path.join(sub_path, "SKILL.md")
                ):
                    candidates.append(sub_path)

        for skill_path in candidates:
            skill_md = os.path.join(skill_path, "SKILL.md")
            try:
                metadata = _parse_frontmatter(skill_md)
                if not metadata or not metadata.get("name"):
                    continue

                skill_name = metadata["name"]

                # Check required config
                required = metadata.get("metadata", {}).get("requires_config", [])
                if required:
                    skill_cfg = config.get("skill-config", {}).get(skill_name, {})
                    missing = [k for k in required if k not in skill_cfg]
                    if missing:
                        logger.info(
                            "Skipping skill '%s': missing config keys: %s",
                            skill_name, ", ".join(missing),
                        )
                        continue

                skill = AgentSkill(skill_path, metadata)
                skills[skill.name] = skill
                logger.debug("Loaded skill: %s", skill.name)

            except Exception as e:
                logger.error("Error loading skill %s: %s", skill_path, e)

        return skills


def _parse_frontmatter(skill_md_path: str) -> Optional[Dict[str, Any]]:
    """Parse YAML frontmatter from a SKILL.md file."""
    with open(skill_md_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    frontmatter_lines: List[str] = []
    in_frontmatter = False

    for line in lines:
        if line.strip() == "---":
            if in_frontmatter:
                break
            in_frontmatter = True
            continue
        if in_frontmatter:
            frontmatter_lines.append(line)

    if not frontmatter_lines:
        return None

    try:
        return yaml.safe_load("\n".join(frontmatter_lines))
    except Exception as e:
        logger.error("Error parsing YAML in %s: %s", skill_md_path, e)
        return None
