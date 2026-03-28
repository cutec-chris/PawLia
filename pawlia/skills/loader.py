"""AgentSkill loading and discovery from SKILL.md files."""

import logging
import os
from functools import cached_property
from typing import Any, Dict, Optional

import yaml

from pawlia.utils import collect_skill_dirs, parse_frontmatter


logger = logging.getLogger(__name__)


class AgentSkill:
    """A single agent skill loaded from a SKILL.md file."""

    def __init__(self, skill_path: str, metadata: Dict[str, Any], workspace_dir: Optional[str] = None):
        self.skill_path = skill_path
        self.metadata = metadata
        cwd_mode = metadata.get("metadata", {}).get("openclaw", {}).get("cwd", "skill")
        if cwd_mode == "workspace" and workspace_dir:
            self.base_dir = workspace_dir
        else:
            self.base_dir = skill_path
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

    @cached_property
    def workflow(self) -> Optional["CompiledWorkflow"]:
        """Load workflow.yaml if present and version matches."""
        from pawlia.skills.workflow_schema import CompiledWorkflow

        path = os.path.join(self.skill_path, "workflow.yaml")
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            compiled = CompiledWorkflow(**data)
            skill_version = str(
                self.metadata.get("metadata", {}).get("version", "")
            )
            if skill_version and compiled.version != skill_version:
                logger.info(
                    "Workflow for '%s' is outdated (skill=%s, workflow=%s)",
                    self.name, skill_version, compiled.version,
                )
                return None
            return compiled
        except Exception as exc:
            logger.warning("Failed to load workflow for '%s': %s", self.name, exc)
            return None

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
                            "minLength": 1,
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }


class SkillLoader:
    """Discovers and loads AgentSkills from a directory."""

    @staticmethod
    def discover(
        skills_dir: str,
        config: Optional[Dict[str, Any]] = None,
        workspace_dir: Optional[str] = None,
        require_workflow: bool = False,
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

        for skill_path in collect_skill_dirs(skills_dir):
            skill_md = os.path.join(skill_path, "SKILL.md")
            try:
                metadata = parse_frontmatter(skill_md)
                if not metadata or not metadata.get("name"):
                    continue

                skill_name = metadata["name"]

                # Check required config
                required = metadata.get("metadata", {}).get("requires_config", [])
                if required:
                    skill_config_root = config.get("skill-config") or {}
                    skill_cfg = skill_config_root.get(skill_name, {})
                    missing = [k for k in required if k not in skill_cfg]
                    if missing:
                        logger.info(
                            "Skipping skill '%s': missing config keys: %s",
                            skill_name, ", ".join(missing),
                        )
                        continue

                skill = AgentSkill(skill_path, metadata, workspace_dir=workspace_dir)

                if require_workflow and skill.workflow is None:
                    logger.info(
                        "Skipping skill '%s': no compiled workflow",
                        skill_name,
                    )
                    continue

                skills[skill.name] = skill
                logger.debug("Loaded skill: %s", skill.name)

            except Exception as e:
                logger.error("Error loading skill %s: %s", skill_path, e)

        return skills
