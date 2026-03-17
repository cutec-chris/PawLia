"""App - central state holder for PawLia.

Holds shared LLMs, tool registry, and skills.
Provides a factory for creating ChatAgents per user session.
"""

import logging
import os
from typing import Any, Callable, Dict, Optional

from pawlia.config import load_config
from pawlia.llm import LLMFactory
from pawlia.memory import MemoryManager
from pawlia.tools.base import ToolRegistry
from pawlia.tools.bash import BashTool
from pawlia.tools.reminder import ReminderTool
from pawlia.skills.loader import AgentSkill, SkillLoader
from pawlia.agents.chat import ChatAgent
from pawlia.agents.skill_runner import SkillRunnerAgent
from pawlia.scheduler import Scheduler


class App:
    """Central application state.

    Holds shared resources (LLMs, tools, skills) and provides
    a factory for creating ChatAgent instances per user/interface.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
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
        self.tools.register(ReminderTool())

        # Skills
        skills_dir = os.path.join(pkg_dir, "skills")
        self.skills: Dict[str, AgentSkill] = SkillLoader.discover(skills_dir, config)
        if self.skills:
            self.logger.info("Loaded skills: %s", ", ".join(self.skills.keys()))
        else:
            self.logger.info("No skills loaded")

        # Scheduler for proactive reminders / event notifications
        self.scheduler = Scheduler(self.session_dir)

    def make_agent(self, user_id: str = "default", **kwargs) -> ChatAgent:
        """Create a new ChatAgent for a user session.

        Each agent gets its own SkillRunner factory bound to the user context.
        Extra kwargs are forwarded to ChatAgent (e.g. on_interim).
        """
        session = self.memory.load_session(user_id)

        # Resolve LLMs – honour per-session model override
        if session.model_override:
            chat_llm = self.llm.get_with_model(session.model_override)
            vision_llm = chat_llm
        else:
            chat_llm = self.llm.get("chat")
            vision_llm = self.llm.get("vision")

        def make_runner(skill: AgentSkill) -> SkillRunnerAgent:
            skill_cfg = self.config.get("skill-config", {}).get(skill.name, {})
            return SkillRunnerAgent(
                llm=self.llm.get(f"skill.{skill.name}"),
                skill=skill,
                tool_registry=self.tools,
                context={
                    "skill_config": skill_cfg,
                    "user_id": user_id,
                    "session_dir": self.session_dir,
                    "session": session,
                },
            )

        agent = ChatAgent(
            llm=chat_llm,
            skills=self.skills,
            skill_runner_factory=make_runner,
            logger=self.logger.getChild(f"chat.{user_id}"),
            memory=self.memory,
            session=session,
            vision_llm=vision_llm,
            **kwargs,
        )
        # Let the agent resolve per-thread model overrides at run() time
        agent._llm_resolver = self.llm.get_with_model
        return agent


def create_app(config_path: Optional[str] = None,
               logger: Optional[logging.Logger] = None) -> App:
    """Load config and create an App instance."""
    config = load_config(config_path)
    return App(config, logger=logger)
