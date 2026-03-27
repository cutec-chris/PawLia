"""ChatAgent - conversational dispatcher that delegates work to SkillRunners.

The ChatAgent has NO tools of its own. It only knows about available skills
(via their OpenAI function specs). When the LLM decides a skill is needed,
the ChatAgent spawns a SkillRunnerAgent to do the actual work, then
incorporates the result into its final response.
"""

import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional, Tuple

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
    "You have skills (tools) available. "
    "When a user asks for something a skill can handle, "
    "you MUST call the matching skill. NEVER guess or make up answers.\n"
    "Only answer directly for simple conversation (greetings, opinions)."
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
        self.on_skill_done: Optional[InterimCallback] = None      # (skill_name)

        # Bind skill specs as "tools" so the LLM can call them
        self._skill_specs = [s.as_openai_spec() for s in skills.values()]
        if self._skill_specs:
            self.bound_llm = llm.bind_tools(self._skill_specs, tool_choice="auto")
            self.vision_bound_llm = (vision_llm or llm).bind_tools(self._skill_specs, tool_choice="auto")
        else:
            self.bound_llm = llm
            self.vision_bound_llm = vision_llm or llm

        # Resolver for per-thread model overrides: model_name -> ChatOpenAI
        # Set by App.make_agent after construction.
        self._llm_resolver: Optional[Callable[[str], Any]] = None

    async def run(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        images: Optional[List[str]] = None,
        thread_id: Optional[str] = None,
        on_skill_start: Optional[SkillStartCallback] = None,
        on_skill_step: Optional[InterimCallback] = None,
        on_skill_done: Optional[InterimCallback] = None,
    ) -> str:
        """Process user input and return a response.

        1. Send to LLM with skill specs as available functions
        2. If LLM calls a skill -> spawn SkillRunnerAgent
        3. Feed skill result back to LLM for final answer

        ``images`` is an optional list of base64 data-URIs
        (e.g. ``data:image/png;base64,…``).

        ``thread_id`` isolates the context window: the model only sees exchanges
        from that thread (seeded from the last 5 main-session exchanges on first use).

        Optional per-call callbacks override instance-level attributes to avoid
        race conditions when the same agent is shared across concurrent requests.
        """
        # Resolve callbacks: per-call overrides > instance attributes
        _on_skill_start = on_skill_start or self.on_skill_start
        _on_skill_step = on_skill_step or self.on_skill_step
        _on_skill_done = on_skill_done or self.on_skill_done
        if system_prompt:
            prompt = system_prompt
        elif self.memory and self.session:
            prompt = self.memory.build_system_prompt(
                self.session, skills=self.skills,
            )
        else:
            prompt = DEFAULT_SYSTEM_PROMPT

        messages: List[BaseMessage] = [SystemMessage(content=prompt)]

        # Replay recent exchanges.  For threads, use the thread-specific window
        # instead of the main session history.
        if self.session and self.memory:
            if thread_id:
                exchanges = self.memory.get_thread_context(self.session, thread_id)
            else:
                exchanges = self.session.exchanges
            for exchange in exchanges:
                # Unpack 2-tuple or 3-tuple (old format compatibility)
                if len(exchange) == 2:
                    user_text, bot_text = exchange  # type: ignore
                    tool_calls_info = None
                else:
                    user_text, bot_text, tool_calls_info = exchange  # type: ignore

                messages.append(HumanMessage(content=user_text))

                # Restore tool calls if present
                if tool_calls_info:
                    # Create AIMessage with reconstructed tool_calls
                    reconstructed_tool_calls = []
                    for tc in tool_calls_info:
                        reconstructed_tool_calls.append({
                            "name": tc["name"],
                            "args": tc["args"],
                            "id": f"restored_{len(reconstructed_tool_calls)}",
                        })
                    messages.append(AIMessage(
                        content=bot_text,
                        tool_calls=reconstructed_tool_calls,
                    ))
                    # Add ToolMessage for each restored tool call with its result
                    for i, tc in enumerate(tool_calls_info):
                        messages.append(ToolMessage(
                            content=tc["result"],
                            tool_call_id=f"restored_{i}",
                        ))
                else:
                    messages.append(AIMessage(content=bot_text))

        # Resolve the LLMs to use for this call.
        # A thread-specific model override takes priority over the session default.
        bound_llm, unbound_llm = self._resolve_llms(thread_id, images=bool(images))

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
        active_llm = bound_llm
        response = await self._invoke(messages, llm=active_llm)
        self.logger.debug("LLM response: tool_calls=%s, content=%s",
                          bool(response.tool_calls),
                          repr(response.content[:200]) if response.content else "(empty)")

        if not response.tool_calls:
            result = self.extract_text(response)
            await self._persist(user_input, result, track_similarity=True, thread_id=thread_id)
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

        # Collect tool call information for persistent storage
        tool_calls_info: List[Dict[str, Any]] = []

        for tool_call in response.tool_calls:
            skill_name = tool_call["name"]
            query = tool_call.get("args", {}).get("query", "")
            skill = self.skills.get(skill_name)

            if skill:
                self.logger.info("Delegating to skill '%s': %s", skill_name, query[:80])
                if _on_skill_start:
                    try:
                        await _on_skill_start(skill_name, query)
                    except Exception as exc:
                        self.logger.debug("on_skill_start error: %s", exc)
                runner = self.skill_runner_factory(skill)
                runner.on_step = _on_skill_step
                result = await runner.run(query=query)
                if _on_skill_done:
                    try:
                        await _on_skill_done(skill_name)
                    except Exception as exc:
                        self.logger.debug("on_skill_done error: %s", exc)
            else:
                self.logger.warning("Unknown skill called: %s", skill_name)
                result = f"Error: Unknown skill '{skill_name}'."

            # Store tool call info for persistent storage
            tool_calls_info.append({
                "name": skill_name,
                "args": tool_call.get("args", {}),
                "result": result,
            })

            messages.append(ToolMessage(
                content=result,
                tool_call_id=tool_call.get("id", ""),
            ))

        # Turn 2: LLM formulates final answer incorporating skill results
        # Use unbound LLM (no tools) for the final response
        self.logger.debug("Turn 2: sending %d messages to LLM for final answer", len(messages))
        final = await self._invoke(messages, llm=unbound_llm)
        self.logger.debug("Turn 2 response: content=%s",
                          repr(final.content[:200]) if final.content else "(empty)")
        result = self.extract_text(final)
        await self._persist(
            user_input, result,
            track_similarity=False,
            thread_id=thread_id,
            tool_calls_info=tool_calls_info,
        )
        return result

    def _resolve_llms(
        self, thread_id: Optional[str], *, images: bool = False
    ) -> Tuple[Any, Any]:
        """Return (bound_llm, unbound_llm) for this call.

        Checks for a thread-specific model override first; falls back to the
        agent's default LLMs.
        """
        if thread_id and self._llm_resolver and self.memory and self.session:
            model = self.memory.get_thread_model_override(self.session, thread_id)
            if model:
                llm = self._llm_resolver(model)
                bound = llm.bind_tools(self._skill_specs, tool_choice="auto") if self._skill_specs else llm
                return bound, llm

        return (self.vision_bound_llm if images else self.bound_llm), self.llm

    async def _persist(
        self,
        user_input: str,
        response: str,
        *,
        track_similarity: bool = True,
        thread_id: Optional[str] = None,
        tool_calls_info: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Save exchange to daily log and schedule summarization if needed."""
        if not (self.memory and self.session):
            return

        if thread_id:
            # Thread exchanges go to a separate log; main session is unchanged.
            self.memory.append_thread_exchange(
                self.session, thread_id, user_input, response, tool_calls_info
            )
            return

        self.memory.append_exchange(
            self.session, user_input, response,
            track_similarity=track_similarity,
            tool_calls_info=tool_calls_info,
        )

        # Summarization is handled by the Scheduler based on idle time.
