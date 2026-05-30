"""ScriptedLLM — the single deterministic LLM test double for pawlia.

Every agent (ChatAgent, SkillRunnerAgent, WorkflowExecutor) takes its LLM by
constructor injection and only ever touches a small *duck-typed* surface:
``invoke`` / ``ainvoke`` / ``astream`` / ``bind_tools`` plus a few read-only
attributes (``model_name``, ``model``, ``temperature``) and a no-op
``set_on_fallback``. This double implements exactly that surface, so it can
drive whole turns through the real agent loop, real tool dispatch and real
memory — no internal mocking.

Design: **route-matching, not a flat response queue.**

The routing key is the text of the *last user (Human) message* in the prompt.
Tool results are deliberately ignored for routing, so the key stays stable
across a multi-step turn (user asks -> model calls tool -> tool result comes
back -> model answers): both invocations route to the same trigger, which then
yields its scripted replies in sequence.  The same trigger therefore always
produces the same scripted reply, which makes the double robust against the
agent's nudge/retry wrappers
(``_invoke_with_tool_retry``, the chat-nudge loop, the skill-runner retry) that
legitimately re-invoke the LLM mid-turn. A flat queue would have those extra
invocations silently eat scripted replies; route-matching does not.

A route may script several replies *in sequence* (e.g. a tool-call turn, then a
final-answer turn); the last reply is "sticky" and repeats if the route fires
more often than scripted. An unscripted prompt raises ``AssertionError`` loudly
rather than returning an empty string — an unrouted turn is a test bug.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Pattern, Union

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
)

Trigger = Union[str, Pattern, Callable[[str], bool]]
ToolCall = dict  # {"id", "name", "args"}


# ---------------------------------------------------------------------------
# Scripted reply
# ---------------------------------------------------------------------------
@dataclass
class Reply:
    """One scripted LLM response: free text and/or one or more tool calls."""

    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)

    def to_ai_message(self, ids: List[int]) -> AIMessage:
        msg = AIMessage(content=self.text)
        if self.tool_calls:
            calls = []
            for tc in self.tool_calls:
                call = dict(tc)
                if not call.get("id"):
                    call["id"] = f"call_{ids[0]}"
                    ids[0] += 1
                call.setdefault("type", "tool_call")
                calls.append(call)
            msg.tool_calls = calls
        return msg


@dataclass
class _Route:
    matcher: Callable[[str], bool]
    replies: List[Reply]
    cursor: int = 0

    def next_reply(self) -> Reply:
        i = min(self.cursor, len(self.replies) - 1)
        self.cursor += 1
        return self.replies[i]


# ---------------------------------------------------------------------------
# The double
# ---------------------------------------------------------------------------
class ScriptedLLM:
    """Deterministic, route-matching fake satisfying the full LLM surface."""

    def __init__(
        self,
        *,
        model_name: str = "scripted",
        temperature: float = 0.0,
        bound_tools: Optional[list] = None,
        _shared: Optional["_Shared"] = None,
    ):
        self.model_name = model_name
        self.model = model_name
        self.temperature = temperature
        self.bound_tools = list(bound_tools or [])
        # Routes, default route, call log and id counter are shared between a
        # root ScriptedLLM and every view returned by bind_tools(), so the
        # script applies regardless of which handle the agent invokes.
        self._s = _shared or _Shared()

    # -- scripting API ------------------------------------------------------
    def on(self, trigger: Trigger, *replies: Reply) -> "ScriptedLLM":
        """Route: when the last user/tool message matches *trigger*, return the
        given replies in sequence (last one sticky)."""
        self._s.routes.append(_Route(_as_matcher(trigger), list(replies) or [Reply()]))
        return self

    def on_text(self, trigger: Trigger, text: str) -> "ScriptedLLM":
        return self.on(trigger, Reply(text=text))

    def on_tool_then_text(
        self, trigger: Trigger, *, skill: str, args: Optional[dict] = None,
        answer: str = "",
    ) -> "ScriptedLLM":
        """First invocation for this route -> a tool call; next -> the answer."""
        return self.on(
            trigger,
            Reply(tool_calls=[_mk_call(skill, args or {})]),
            Reply(text=answer),
        )

    def default(self, *replies: Reply) -> "ScriptedLLM":
        """Fallback when no route matches (otherwise an unscripted prompt raises)."""
        self._s.default = _Route(lambda _k: True, list(replies) or [Reply()])
        return self

    @staticmethod
    def tool(skill: str, **args: Any) -> Reply:
        """Ergonomic helper: a reply that calls *skill* with keyword args."""
        return Reply(tool_calls=[_mk_call(skill, args)])

    # -- introspection for assertions --------------------------------------
    @property
    def calls(self) -> List[List[BaseMessage]]:
        """Full prompt (message list) of every invocation, in order."""
        return self._s.calls

    @property
    def call_count(self) -> int:
        return len(self._s.calls)

    # -- duck-typed LLM surface --------------------------------------------
    def invoke(self, messages, **kwargs) -> AIMessage:
        self._s.calls.append(list(messages))
        return self._reply_for(messages).to_ai_message(self._s.ids)

    async def ainvoke(self, messages, **kwargs) -> AIMessage:
        return self.invoke(messages, **kwargs)

    async def astream(self, messages, **kwargs):
        self._s.calls.append(list(messages))
        reply = self._reply_for(messages)
        if reply.tool_calls:
            chunk = AIMessageChunk(content="")
            chunk.tool_calls = reply.to_ai_message(self._s.ids).tool_calls
            yield chunk
            return
        for piece in _chunkify(reply.text):
            yield AIMessageChunk(content=piece)

    def bind_tools(self, specs, tool_choice=None, **kwargs) -> "ScriptedLLM":
        # Return a VIEW that shares routes / default / call log / id counter.
        return ScriptedLLM(
            model_name=self.model_name,
            temperature=self.temperature,
            bound_tools=specs,
            _shared=self._s,
        )

    def set_on_fallback(self, cb) -> None:  # satisfies chat.py fallback wiring
        pass

    # -- internals ----------------------------------------------------------
    def _reply_for(self, messages) -> Reply:
        key = _routing_key(messages)
        for route in self._s.routes:
            if route.matcher(key):
                return route.next_reply()
        if self._s.default is not None:
            return self._s.default.next_reply()
        raise AssertionError(
            "ScriptedLLM: no route matched the last user/tool message:\n"
            f"  {key!r}\n"
            "Add an .on(...) route for it, or set .default(...)."
        )


@dataclass
class _Shared:
    routes: List[_Route] = field(default_factory=list)
    default: Optional[_Route] = None
    calls: List[List[BaseMessage]] = field(default_factory=list)
    ids: List[int] = field(default_factory=lambda: [0])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mk_call(name: str, args: dict) -> ToolCall:
    return {"name": name, "args": dict(args), "id": ""}


def _as_matcher(trigger: Trigger) -> Callable[[str], bool]:
    if callable(trigger) and not isinstance(trigger, (str, re.Pattern)):
        return trigger
    if isinstance(trigger, re.Pattern):
        return lambda key: bool(trigger.search(key))
    needle = str(trigger).lower()
    return lambda key: needle in key.lower()


def _message_text(msg: BaseMessage) -> str:
    """Stringify message content, flattening multimodal (vision) lists."""
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(content)


def _routing_key(messages) -> str:
    """The last *user* message text — stable across a turn's tool loop."""
    for msg in reversed(list(messages)):
        if isinstance(msg, HumanMessage):
            return _message_text(msg)
    # Fall back to the very last message if no human turn is present.
    return _message_text(messages[-1]) if messages else ""


def _chunkify(text: str, size: int = 8) -> List[str]:
    if not text:
        return [""]
    return [text[i:i + size] for i in range(0, len(text), size)]
