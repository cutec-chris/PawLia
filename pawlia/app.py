"""App - central state holder for PawLia.

Holds shared LLMs, tool registry, and skills.
Provides a factory for creating ChatAgents per user session.
"""

import logging
import os
import threading
from typing import Any, Callable, Dict, Optional

from pawlia.config import load_config
from pawlia.llm import LLMFactory
from pawlia.memory import MemoryManager
from pawlia.tools.base import ToolRegistry
from pawlia.tools.bash import BashTool
from pawlia.skills.loader import AgentSkill, SkillLoader
from pawlia.agents.chat import ChatAgent
from pawlia.agents.skill_runner import SkillRunnerAgent
from pawlia.scheduler import Scheduler


class App:
    """Central application state.

    Holds shared resources (LLMs, tools, skills) and provides
    a factory for creating ChatAgent instances per user/interface.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None,
                 config_path: Optional[str] = None):
        self.config = config
        self.config_path = config_path
        self.logger = logger or logging.getLogger("pawlia.app")

        # Session directory (same location as legacy system)
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.session_dir = config.get("session_dir", os.path.join(pkg_dir, "session"))
        self.memory = MemoryManager(self.session_dir, logger=self.logger.getChild("memory"))

        # LLM factory — instances are created lazily and cached
        self.llm = LLMFactory(config)

        # Tools
        self.tools = ToolRegistry()
        self.tools.register(BashTool())

        # Skills — bundled (shared across all users, read-only)
        skills_dir = os.path.join(pkg_dir, "skills")
        require_workflow = config.get("workflow", {}).get("require_compiled", False)
        self._bundled_skills: Dict[str, AgentSkill] = SkillLoader.discover(
            skills_dir, config, require_workflow=require_workflow,
        )
        self._skills_lock = threading.Lock()

        if self._bundled_skills:
            self.logger.info("Loaded bundled skills: %s", ", ".join(self._bundled_skills.keys()))
        else:
            self.logger.info("No bundled skills loaded")

        # Backwards-compatible accessor (used by test scripts)
        self.skills = self._bundled_skills

        # Scheduler for proactive reminders / event notifications
        self.scheduler = Scheduler(self.session_dir, config=self.config)
        self.scheduler.set_app(self)

    def _discover_user_workspace_skills(self, user_id: str) -> Dict[str, AgentSkill]:
        """Discover workspace skills for a single user.

        Returns only the skills from ``session/<user_id>/workspace/skills/``.
        Thread-safe via ``_skills_lock``.
        """
        allow_workspace = self.config.get("skill-install", {}).get("allow_workspace", True)
        if not allow_workspace or not os.path.isdir(self.session_dir):
            return {}

        workspace_dir = os.path.join(self.session_dir, user_id, "workspace")
        workspace_skills_dir = os.path.join(workspace_dir, "skills")
        if not os.path.isdir(workspace_skills_dir):
            return {}

        require_workflow = self.config.get("workflow", {}).get("require_compiled", False)

        with self._skills_lock:
            return SkillLoader.discover(
                workspace_skills_dir, self.config,
                workspace_dir=workspace_dir,
                require_workflow=require_workflow,
            )

    def _build_user_skills(self, user_id: str) -> Dict[str, AgentSkill]:
        """Return bundled skills + this user's workspace skills.

        Each user gets their own copy so workspace skills are isolated.
        """
        skills = dict(self._bundled_skills)
        user_skills = self._discover_user_workspace_skills(user_id)
        skills.update(user_skills)
        return skills

    def make_agent(self, user_id: str = "default", **kwargs) -> ChatAgent:
        """Create a new ChatAgent for a user session.

        Each agent gets its own SkillRunner factory bound to the user context.
        Skills are scoped per user: bundled skills + this user's workspace skills.
        Extra kwargs are forwarded to ChatAgent (e.g. on_interim).
        """
        session = self.memory.load_session(user_id)

        # Build per-user skill set (bundled + user's own workspace skills)
        user_skills = self._build_user_skills(user_id)

        # Resolve LLMs – honour per-session model override
        if session.model_override:
            chat_llm = self.llm.get_with_model(session.model_override)
            vision_llm = chat_llm
        else:
            chat_llm = self.llm.get("chat")
            vision_llm = self.llm.get("vision")

        def make_runner(skill: AgentSkill) -> SkillRunnerAgent:
            skill_config_root = self.config.get("skill-config") or {}
            skill_cfg = skill_config_root.get(skill.name, {})
            return SkillRunnerAgent(
                llm=self.llm.get(f"skill.{skill.name}"),
                skill=skill,
                tool_registry=self.tools,
                context={
                    "skill_config": skill_cfg,
                    "user_id": user_id,
                    "session_dir": self.session_dir,
                    "session": session,
                    "config_path": self.config_path,
                },
            )

        def refresh_user_skills() -> None:
            """Re-discover this user's workspace skills and update the agent's dict."""
            fresh = self._build_user_skills(user_id)
            user_skills.clear()
            user_skills.update(fresh)

        agent = ChatAgent(
            llm=chat_llm,
            skills=user_skills,
            skill_runner_factory=make_runner,
            logger=self.logger.getChild(f"chat.{user_id}"),
            memory=self.memory,
            session=session,
            vision_llm=vision_llm,
            **kwargs,
        )
        # Let the agent resolve per-thread model overrides at run() time
        agent._llm_resolver = self.llm.get_with_model
        # Resolve config keys (e.g. "fast") to actual model names
        agent._model_name_resolver = self.llm.resolve_model_name
        # Let the ChatAgent re-discover this user's workspace skills after each skill call
        agent._skills_refresher = refresh_user_skills
        return agent


def create_app(config_path: Optional[str] = None,
               logger: Optional[logging.Logger] = None) -> App:
    """Load config and create an App instance."""
    from pawlia.config import resolve_config_path
    config_path = config_path or resolve_config_path()
    config = load_config(config_path)
    return App(config, logger=logger, config_path=config_path)
