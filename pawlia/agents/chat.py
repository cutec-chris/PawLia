"""ChatAgent - conversational dispatcher that delegates work to SkillRunners.

The ChatAgent has NO tools of its own. It only knows about available skills
(via their OpenAI function specs). When the LLM decides a skill is needed,
the ChatAgent spawns a SkillRunnerAgent to do the actual work, then
incorporates the result into its final response.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional

# Callback types
InterimCallback = Callable[[str], Awaitable[None]]
SkillStartCallback = Callable[[str, str], Awaitable[None]]  # (skill_name, query)

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

from pawlia.agents.base import BaseAgent
from pawlia.skills.loader import AgentSkill

if TYPE_CHECKING:
    from pawlia.memory import MemoryManager, Session


DEFAULT_SYSTEM_PROMPT = (
    "You are PawLia, a helpful AI assistant.\n\n"
    "IMPORTANT: You have skills (tools) available. "
    "When a user asks for information that a skill can provide "
    "(routes, train connections, searches, file operations, etc.), "
    "you MUST call the matching skill. NEVER guess or make up answers — "
    "always use the skill to get real data.\n"
    "Only answer directly for simple conversation (greetings, opinions, "
    "general knowledge).\n\n"
    "When you learn a persistent fact or preference about the user "
    "(name, language, habits, preferences, etc.), "
    "use the files skill to append it to memory/memory.md."
)


class ChatAgent(BaseAgent):
    """Dispatcher agent - no tools, only skill descriptions.

    For every task that requires tools, it delegates to a SkillRunnerAgent
    via the ``skill_runner_factory`` callback.
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        skills: Dict[str, AgentSkill],
        skill_runner_factory: Callable[[AgentSkill], Any],
        logger: Optional[logging.Logger] = None,
        memory: Optional["MemoryManager"] = None,
        session: Optional["Session"] = None,
        on_interim: Optional[InterimCallback] = None,
        vision_llm: Optional[ChatOpenAI] = None,
    ):
        super().__init__(llm, logger)
        self.skills = skills
        self.skill_runner_factory = skill_runner_factory
        self.memory = memory
        self.session = session
        self.on_interim = on_interim
        self.on_skill_start: Optional[SkillStartCallback] = None  # (skill_name, query)
        self.on_skill_step: Optional[InterimCallback] = None      # (step_description)
        self._idle_task: Optional[asyncio.Task] = None

        # Bind skill specs as "tools" so the LLM can call them
        skill_specs = [s.as_openai_spec() for s in skills.values()]
        if skill_specs:
            self.bound_llm = llm.bind_tools(skill_specs, tool_choice="auto")
            self.vision_bound_llm = (vision_llm or llm).bind_tools(skill_specs, tool_choice="auto")
        else:
            self.bound_llm = llm
            self.vision_bound_llm = vision_llm or llm

    async def run(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        images: Optional[List[str]] = None,
    ) -> str:
        """Process user input and return a response.

        1. Send to LLM with skill specs as available functions
        2. If LLM calls a skill -> spawn SkillRunnerAgent
        3. Feed skill result back to LLM for final answer

        ``images`` is an optional list of base64 data-URIs
        (e.g. ``data:image/png;base64,…``).
        """
        if system_prompt:
            prompt = system_prompt
        elif self.memory and self.session:
            prompt = self.memory.build_system_prompt(self.session)
        else:
            prompt = DEFAULT_SYSTEM_PROMPT

        messages: List[BaseMessage] = [SystemMessage(content=prompt)]

        # Replay recent exchanges as structured message pairs so the
        # model treats them as its own prior turns (better than flat text).
        if self.session:
            for user_text, bot_text in self.session.exchanges:
                messages.append(HumanMessage(content=user_text))
                messages.append(AIMessage(content=bot_text))

        # Build multimodal content when images are present
        if images:
            self.logger.debug("Sending %d image(s) to LLM", len(images))
            content: List[Dict[str, Any]] = [{"type": "text", "text": user_input or "What's in this image?"}]
            for data_uri in images:
                content.append({"type": "image_url", "image_url": {"url": data_uri}})
            messages.append(HumanMessage(content=content))
        else:
            messages.append(HumanMessage(content=user_input))

        # Turn 1: LLM decides whether to call a skill or answer directly
        active_llm = self.vision_bound_llm if images else self.bound_llm
        response = await self._invoke(messages, llm=active_llm)
        self.logger.debug("LLM response: tool_calls=%s, content=%s",
                          bool(response.tool_calls),
                          repr(response.content[:200]) if response.content else "(empty)")

        if not response.tool_calls:
            result = self.extract_text(response)
            await self._persist(user_input, result, track_similarity=True)
            return result

        # Send interim message if the LLM included text alongside tool calls
        interim = self.extract_text(response)
        if interim and self.on_interim:
            try:
                await self.on_interim(interim)
            except Exception as exc:
                self.logger.debug("on_interim callback error: %s", exc)

        # Skill calls detected -> execute via SkillRunners
        messages.append(response)  # AIMessage with tool_calls

        for tool_call in response.tool_calls:
            skill_name = tool_call["name"]
            query = tool_call.get("args", {}).get("query", "")
            skill = self.skills.get(skill_name)

            if skill:
                self.logger.info("Delegating to skill '%s': %s", skill_name, query[:80])
                if self.on_skill_start:
                    try:
                        await self.on_skill_start(skill_name, query)
                    except Exception as exc:
                        self.logger.debug("on_skill_start error: %s", exc)
                runner = self.skill_runner_factory(skill)
                runner.on_step = self.on_skill_step
                result = await runner.run(query=query)
            else:
                self.logger.warning("Unknown skill called: %s", skill_name)
                result = f"Error: Unknown skill '{skill_name}'."

            messages.append(ToolMessage(
                content=result,
                tool_call_id=tool_call.get("id", ""),
            ))

        # Turn 2: LLM formulates final answer incorporating skill results
        # Use unbound LLM (no tools) for the final response
        final = await self._invoke(messages, llm=self.llm)
        result = self.extract_text(final)
        await self._persist(user_input, result, track_similarity=False)
        return result

    async def _persist(
        self, user_input: str, response: str, *, track_similarity: bool = True,
    ) -> None:
        """Save exchange to daily log and schedule summarization if needed."""
        if not (self.memory and self.session):
            return

        self.memory.append_exchange(
            self.session, user_input, response,
            track_similarity=track_similarity,
        )

        reason = self.memory.should_summarize(self.session)
        if reason:
            self.logger.info("Summarizing conversation (trigger: %s)", reason)
            self._schedule_background_summary()
        else:
            self._schedule_idle_summary()

    def _schedule_background_summary(self) -> None:
        """Run summarization as a background task (non-blocking)."""
        try:
            asyncio.create_task(self._summarize_conversation())
        except RuntimeError:
            pass  # no running event loop

    async def _summarize_conversation(self) -> None:
        """Ask the LLM to summarize the conversation history."""
        if not (self.memory and self.session):
            return

        history = self.session.daily_history.strip()
        if not history:
            return

        prior = self.session.summary.strip()
        context = ""
        if prior:
            context = f"Previous summary:\n{prior}\n\n"

        messages = [
            SystemMessage(content=(
                "Summarize this conversation in 2-4 short bullet points.\n"
                "Keep ONLY:\n"
                "- User preferences and personal facts\n"
                "- Decisions made or tasks completed\n"
                "- Open/unanswered requests\n"
                "DISCARD:\n"
                "- Specific numbers, routes, or data (the user can ask again)\n"
                "- Failed attempts, errors, or debugging details\n"
                "- Greetings and small talk\n"
                "Write in the user's language. Maximum 4 lines."
            )),
            HumanMessage(content=(
                f"{context}Conversation to summarize:\n{history}"
            )),
        ]

        response = await self._invoke(messages, llm=self.llm)
        summary = self.extract_text(response)

        if summary:
            self.memory.summarize(self.session, summary)

    def _schedule_idle_summary(self) -> None:
        """(Re)start a background timer that summarizes after idle timeout."""
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()

        from pawlia.memory import IDLE_TIMEOUT_SECONDS

        async def _idle_watcher():
            await asyncio.sleep(IDLE_TIMEOUT_SECONDS)
            if not (self.memory and self.session):
                return
            reason = self.memory.should_summarize(self.session)
            if reason:
                self.logger.info("Summarizing conversation (trigger: idle)")
                await self._summarize_conversation()

        try:
            self._idle_task = asyncio.create_task(_idle_watcher())
        except RuntimeError:
            pass  # no running event loop (e.g. in tests)
