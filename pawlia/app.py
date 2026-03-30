"""App - central state holder for PawLia.

Holds shared LLMs, tool registry, and skills.
Provides a factory for creating ChatAgents per user session.
"""

import logging
import os
from typing import Any, Callable, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pawlia.config import load_config
from pawlia.llm import LLMFactory
from pawlia.memory import MemoryManager
from pawlia.prompt_utils import load_system_prompt
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

        # Skills — built-in (already installed+compiled during Docker build)
        skills_dir = os.path.join(pkg_dir, "skills")
        require_workflow = config.get("workflow", {}).get("require_compiled", False)
        self.skills: Dict[str, AgentSkill] = SkillLoader.discover(
            skills_dir, config, require_workflow=require_workflow,
        )

        # Also discover skills placed in any session workspace (session/<user>/workspace/skills/)
        # Requires skill-install.allow_workspace: true in config (default: false)
        # Deps + workflows are installed/compiled at upload time, not here.
        allow_workspace = config.get("skill-install", {}).get("allow_workspace", False)
        if allow_workspace and os.path.isdir(self.session_dir):
            for user_entry in os.listdir(self.session_dir):
                workspace_dir = os.path.join(self.session_dir, user_entry, "workspace")
                workspace_skills_dir = os.path.join(workspace_dir, "skills")
                if os.path.isdir(workspace_skills_dir):
                    workspace_skills = SkillLoader.discover(
                        workspace_skills_dir, config,
                        workspace_dir=workspace_dir,
                        require_workflow=require_workflow,
                    )
                    self.skills.update(workspace_skills)

        if self.skills:
            self.logger.info("Loaded skills: %s", ", ".join(self.skills.keys()))
        else:
            self.logger.info("No skills loaded")

        # Scheduler for proactive reminders / event notifications
        self.scheduler = Scheduler(self.session_dir, config=self.config)
        self.scheduler.set_app(self)
        self.scheduler.set_llm_formatter(self._format_notification)

    async def _format_notification(self, user_id: str, raw_message: str) -> str:
        """Pass a raw notification through the LLM for personalized delivery.

        The LLM receives the raw data (reminder text, script output, etc.)
        and produces a natural, personalized message for the user.
        If the LLM is busy (e.g. handling a chat request on a local model),
        the scheduler's timeout + fallback ensures the raw message still
        gets delivered.
        """
        session = self.memory.load_session(user_id)
        llm = self.llm.get("chat")

        # Minimal prompt — keep it short so local models respond fast
        system = load_system_prompt("notifications/formatter.md")

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=raw_message),
        ]

        response = await llm.ainvoke(messages)
        return response.content or raw_message

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
        # Let the agent fall back to default LLMs when an override is unreachable
        agent._fallback_resolver = self.llm.get
        return agent


def create_app(config_path: Optional[str] = None,
               logger: Optional[logging.Logger] = None) -> App:
    """Load config and create an App instance."""
    config = load_config(config_path)
    return App(config, logger=logger, config_path=config_path)
