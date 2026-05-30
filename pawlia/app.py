"""App - central state holder for PawLia.

Holds shared LLMs, tool registry, and skills.
Provides a factory for creating ChatAgents per user session.
"""

import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from pawlia.config import load_config
from pawlia.llm import LLMFactory
from pawlia.memory import MemoryManager
from pawlia.tools.base import ToolRegistry
from pawlia.tools.bash import BashTool
from pawlia.skills.loader import AgentSkill, SkillLoader
from pawlia.agents.chat import ChatAgent
from pawlia.agents.router import RouterAgent
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
        self.session_dir = os.path.abspath(
            config.get("session_dir", os.path.join(pkg_dir, "session"))
        )
        self.memory = MemoryManager(self.session_dir, logger=self.logger.getChild("memory"))
        self._pkg_dir = pkg_dir
        self._skills_lock = threading.Lock()
        self._workspace_skills_cache: Dict[str, tuple] = {}  # user_id → (skills_dict, dir_mtime)
        self._initialize_runtime()

        # Scheduler for proactive reminders / event notifications
        self.scheduler = Scheduler(self.session_dir, config=self.config)
        self.scheduler.set_app(self)

    def _initialize_runtime(self) -> None:
        """Initialize config-driven runtime components."""
        # LLM factory — instances are created lazily and cached
        self.llm = LLMFactory(self.config)

        # Tools
        self.tools = ToolRegistry()
        self.tools.register(BashTool())

        # Skills — bundled (shared across all users, read-only)
        skills_dir = os.path.join(self._pkg_dir, "skills")
        require_workflow = self.config.get("workflow", {}).get("require_compiled", False)
        self._bundled_skills = SkillLoader.discover(
            skills_dir, self.config, require_workflow=require_workflow,
        )

        if self._bundled_skills:
            self.logger.info("Loaded bundled skills: %s", ", ".join(self._bundled_skills.keys()))
        else:
            self.logger.info("No bundled skills loaded")

        # Backwards-compatible accessor (used by test scripts)
        self.skills = self._bundled_skills

    def reload(self) -> Dict[str, Any]:
        """Reload config-driven app state without restarting the process.

        Refreshes config, LLM factory, tools, bundled skills, and scheduler
        settings. Long-lived interface listener settings such as ports, tokens,
        and session_dir still require a full process restart.
        """
        new_config = load_config(self.config_path)
        old_session_dir = self.session_dir
        self.config = new_config
        warnings: List[str] = []

        new_session_dir = os.path.abspath(
            new_config.get("session_dir", old_session_dir)
        )
        if new_session_dir != old_session_dir:
            warnings.append(
                "session_dir changed in config but stays unchanged until process restart"
            )

        self._initialize_runtime()
        self.scheduler.reload_config(self.config)
        self.scheduler.set_app(self)
        self._workspace_skills_cache.clear()

        self.logger.info(
            "Reloaded config from %s (%d bundled skills, %d model definitions)",
            self.config_path or "(auto-discovered)",
            len(self._bundled_skills),
            len(self.config.get("models") or {}),
        )

        return {
            "config_path": self.config_path,
            "warnings": warnings,
            "bundled_skills": sorted(self._bundled_skills.keys()),
            "model_count": len(self.config.get("models") or {}),
        }

    def _workspace_skills_mtime(self, workspace_skills_dir: str) -> float:
        """Get the newest mtime of any SKILL.md in the workspace skills tree."""
        max_mtime = 0.0
        for root, dirs, files in os.walk(workspace_skills_dir):
            for f in files:
                if f == "SKILL.md":
                    try:
                        mt = os.path.getmtime(os.path.join(root, f))
                        if mt > max_mtime:
                            max_mtime = mt
                    except OSError:
                        pass
        return max_mtime

    def _discover_user_workspace_skills(self, user_id: str) -> Dict[str, AgentSkill]:
        """Discover workspace skills for a single user.

        Returns only the skills from ``session/<user_id>/workspace/skills/``.
        Cached by directory mtime — only re-discovers when SKILL.md files change.
        Thread-safe via ``_skills_lock``.
        """
        allow_workspace = self.config.get("skill-install", {}).get("allow_workspace", True)
        if not allow_workspace or not os.path.isdir(self.session_dir):
            return {}

        workspace_dir = os.path.join(self.session_dir, user_id, "workspace")
        workspace_skills_dir = os.path.join(workspace_dir, "skills")
        if not os.path.isdir(workspace_skills_dir):
            return {}

        current_mtime = self._workspace_skills_mtime(workspace_skills_dir)

        with self._skills_lock:
            cached = self._workspace_skills_cache.get(user_id)
            if cached is not None:
                cached_skills, cached_mtime = cached
                if cached_mtime == current_mtime:
                    return cached_skills

            require_workflow = self.config.get("workflow", {}).get("require_compiled", False)
            skills = SkillLoader.discover(
                workspace_skills_dir, self.config,
                workspace_dir=workspace_dir,
                require_workflow=require_workflow,
            )
            self._workspace_skills_cache[user_id] = (skills, current_mtime)
            return skills

    def _build_user_skills(self, user_id: str, disabled: Optional[List[str]] = None) -> Dict[str, AgentSkill]:
        """Return bundled + workspace skills, minus any session-disabled ones."""
        skills = dict(self._bundled_skills)
        user_skills = self._discover_user_workspace_skills(user_id)
        skills.update(user_skills)
        if disabled:
            for name in disabled:
                skills.pop(name, None)
        return skills

    def run_instruction(self, instruction: str, user_id: str = "default") -> RouterAgent:
        """Create a normal agent for a scheduled natural-language instruction.

        Automations should run through the same chat/skill dispatcher as an
        interactive turn. That lets instructions such as "Nutze den perplexica
        Skill ..." call the real skill instead of forcing the model to invent
        shell commands inside a virtual Bash-only skill.
        """
        return self.make_agent(user_id)

    def make_agent(self, user_id: str = "default", **kwargs) -> RouterAgent:
        """Create a backend-dispatching agent for a user session.

        The returned agent keeps PawLia's logging/memory layer stable while
        routing each request to either the local ChatAgent stack or Hermes,
        depending on the selected model's provider backend.
        """
        session = self.memory.load_session(user_id)

        # Build per-user skill set (bundled + user's own workspace skills, minus disabled)
        user_skills = self._build_user_skills(user_id, disabled=session.disabled_skills)

        def make_runner(skill: AgentSkill, thread_id: Optional[str] = None) -> SkillRunnerAgent:
            skill_config_root = self.config.get("skill-config") or {}
            skill_cfg = skill_config_root.get(skill.name, {})
            agent_overrides = self.memory.effective_agent_overrides(session, thread_id)
            agent_type = f"skill.{skill.name}"
            model_name = self.llm.default_model_name(agent_type, agent_overrides=agent_overrides)
            max_tool_turns = self.llm.max_tool_turns_for_model(model_name)
            return SkillRunnerAgent(
                llm=self.llm.get(agent_type, agent_overrides=agent_overrides),
                skill=skill,
                tool_registry=self.tools,
                context={
                    "skill_config": skill_cfg,
                    "user_id": user_id,
                    "session_dir": self.session_dir,
                    "session": session,
                    "config_path": self.config_path,
                },
                max_tool_turns=max_tool_turns,
            )

        def refresh_user_skills() -> None:
            """Re-discover this user's workspace skills and update the agent's dict."""
            fresh = self._build_user_skills(user_id)
            user_skills.clear()
            user_skills.update(fresh)

        def make_local_agent() -> ChatAgent:
            overrides = self.memory.effective_agent_overrides(session, None)
            chat_llm = self.llm.get("chat", agent_overrides=overrides)
            vision_llm = self.llm.get("vision", agent_overrides=overrides)
            ws_search_cfg = self.config.get("workspace-search", {})
            # Model-size-aware tool-turn budget
            chat_model_name = self.llm.default_model_name("chat", agent_overrides=overrides)
            chat_max_turns = self.llm.max_tool_turns_for_model(chat_model_name)
            # No direct file tools: reads/list/grep route through the `files`
            # skill (isolated runner context) so raw file content never lands
            # in the slim main loop. Plumbing is kept (the kwarg below) for a
            # possible bounded direct-read exception later.
            direct_tools: dict = {}

            agent = ChatAgent(
                llm=chat_llm,
                skills=user_skills,
                skill_runner_factory=make_runner,
                logger=self.logger.getChild(f"chat.{user_id}"),
                memory=self.memory,
                session=session,
                vision_llm=vision_llm,
                workspace_search_cfg=ws_search_cfg,
                max_tool_turns=chat_max_turns,
                direct_tools=direct_tools,
                **kwargs,
            )
            # Resolve session/thread-specific agent selectors at run() time.
            agent._agent_llm_resolver = (
                lambda agent_type, thread_id=None:
                self.llm.get(
                    agent_type,
                    agent_overrides=self.memory.effective_agent_overrides(session, thread_id),
                )
            )
            agent._context_window_resolver = (
                lambda agent_type, thread_id=None:
                self.llm.context_size_for_model(
                    self.llm.default_model_name(
                        agent_type,
                        agent_overrides=self.memory.effective_agent_overrides(session, thread_id),
                    )
                )
            )
            # Resolve config keys (e.g. "fast") to actual model names
            agent._model_name_resolver = self.llm.resolve_model_name
            # Let the ChatAgent re-discover this user's workspace skills after each skill call
            agent._skills_refresher = refresh_user_skills
            return agent

        return RouterAgent(
            user_id=user_id,
            llm_factory=self.llm,
            memory=self.memory,
            session=session,
            skills=user_skills,
            local_agent_factory=make_local_agent,
            logger=self.logger.getChild(f"router.{user_id}"),
            on_interim=kwargs.get("on_interim"),
        )


def create_app(config_path: Optional[str] = None,
               logger: Optional[logging.Logger] = None) -> App:
    """Load config and create an App instance."""
    from pawlia.config import resolve_config_path
    config_path = config_path or resolve_config_path()
    config = load_config(config_path)
    return App(config, logger=logger, config_path=config_path)
