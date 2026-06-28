"""Base agent class with shared LLM invocation and thinking-tag cleanup."""

import asyncio
import logging
import os
import random
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, List, Optional, Tuple

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

from pawlia.agents.error_classifier import (
    ErrorCategory,
    classify_error,
    is_retryable,
    should_compact,
)
from pawlia.utils import run_sync_in_thread


_SURROGATE_RE = re.compile(r'[\ud800-\udfff]')
_RE_ROLE_ALTERNATION_VIOLATION = re.compile(
    r"(tool|assistant)\s*(tool|assistant)", re.IGNORECASE
)


_RE_THINK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL)
_RE_TOOL_CALL_LEAKED = re.compile(r"<tool_call>.*?(?:</tool_call>|$)", re.DOTALL)
# Chat-template tokens that some models leak into their output
_RE_CHAT_TOKENS = re.compile(r"<\|.*?\|>.*", re.DOTALL)
# Pattern for tool call in failed_generation from API errors
_RE_TOOL_CALL = re.compile(r'\{.*?"name"\s*:.*?"args"\s*:.*?\}', re.DOTALL)

# Internal-context markers that must never reach a user. These are framing the
# agent injects into its OWN context — replayed skill summaries
# (``[Earlier skill use — internal context:]`` from
# ``ChatAgent._format_replayed_assistant_turn``) and trust headers
# (``[Report from …]`` / ``Trust: INTERNAL`` from ``_wrap_with_trust_header``).
# Small models routinely parrot these back into their answers. This is the
# single source of truth — the call/TTS path (call_core._for_tts) imports it.
_INTERNAL_MARKER_PREFIXES = (
    "[Earlier skill use",
    "[Report from `",
    "[internal context",
    "Trust: INTERNAL",
    "Trust: EXTERNAL",
    "Raw outside data",
    "Treat with skepticism",
    "This information comes from the user",
    "Cross-check with what you know",
    "when in conflict, follow this source",
)
# Line-anchored matcher for the per-sentence TTS filter and the line scrubber.
# Mirrors the prefixes above; ``---`` on its own line is a separator the trust
# wrapper emits and is only stripped when adjacent to a dropped block.
_RE_INTERNAL_LINE = re.compile(
    r"^\s*(?:"
    r"\[Earlier skill use"
    r"|\[Report from `"
    r"|\[internal context"
    r"|Trust: (?:INTERNAL|EXTERNAL)"
    r"|Raw outside data"
    r"|Treat with skepticism"
    r"|This information comes from the user.s own"
    r"|Cross-check with what you know"
    r"|when in conflict, follow this source"
    r")",
    re.IGNORECASE,
)
# Markers that introduce a trailing block whose continuation lines (bullets /
# annotations / blanks) must be dropped along with the marker line itself.
_RE_INTERNAL_BLOCK_START = re.compile(
    r"^\s*(?:\[Earlier skill use|\[internal context)", re.IGNORECASE
)

_LOG_DIR: Optional[str] = None  # set by enable_prompt_logging()


def enable_prompt_logging() -> None:
    """Enable prompt logging into ``log/`` inside the project directory."""
    global _LOG_DIR
    # Two levels up from pawlia/agents/base.py → project root
    _LOG_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "log"
    )
    os.makedirs(_LOG_DIR, exist_ok=True)


def log_prompt(messages: List[BaseMessage], name: str = "prompt") -> None:
    """Write the full message list to ``log/<name>.log``.

    Overwrites the file each time so it always contains the last context.
    *name* defaults to ``"prompt"``; skill executors pass the skill name.
    """
    if not _LOG_DIR:
        return
    try:
        path = os.path.join(_LOG_DIR, f"{name}.log")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"--- {datetime.now().isoformat()} ---\n\n")
            for msg in messages:
                role = msg.__class__.__name__.replace("Message", "").upper()
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                f.write(f"[{role}]\n{content}\n\n")
    except OSError:
        pass


class BaseAgent(ABC):
    """Abstract base for all agents."""

    def __init__(self, llm: ChatOpenAI, logger: Optional[logging.Logger] = None):
        self.llm = llm
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.log_name: str = "prompt"  # overridden by SkillRunnerAgent

    @abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> str:
        """Execute the agent's main task and return a text result."""

    async def _invoke(self, messages: List[BaseMessage],
                      llm: Optional[ChatOpenAI] = None,
                      max_retries: int = 3) -> Tuple[AIMessage, List[BaseMessage]]:
        """Invoke an LLM (default: self.llm) with the given messages.

        Classifies API errors and applies the correct recovery strategy:
        - context_overflow → compact and retry
        - rate_limit / timeout / server_error → jittered backoff + retry
        - auth_error / format_error → surface immediately, no retry
        - tool-use-failed → inject synthetic ToolMessage, retry

        Returns ``(response, messages)`` where *messages* is the (possibly
        compacted/summarized) message list used for the successful LLM call.
        Callers in a tool loop should use the returned *messages* to keep
        the stored context bounded instead of appending to the original list.
        """
        messages = self._sanitize_messages(list(messages))
        log_prompt(messages, name=self.log_name)
        target = llm or self.llm

        def _call() -> AIMessage:
            try:
                return target.invoke(messages)
            except StopIteration as exc:
                raise RuntimeError("LLM invoke exhausted iterator") from exc

        retries = 0
        while True:
            try:
                response = await run_sync_in_thread(_call)
                reasoning = (response.additional_kwargs or {}).get("reasoning_content", "")
                # Only embed reasoning when the model returned plain text (no tool
                # calls). When tool_calls are present the content field is typically
                # empty and some providers (e.g. GLM) reject non-null content
                # alongside tool_calls. In that case just strip reasoning_content
                # from additional_kwargs so it doesn't pollute the next API call.
                if reasoning and not response.tool_calls:
                    new_kwargs = {k: v for k, v in (response.additional_kwargs or {}).items()
                                  if k != "reasoning_content"}
                    response = response.model_copy(update={
                        "content": f"<think>{reasoning}</think>" + (response.content or ""),
                        "additional_kwargs": new_kwargs,
                    })
                elif reasoning:
                    new_kwargs = {k: v for k, v in (response.additional_kwargs or {}).items()
                                  if k != "reasoning_content"}
                    response = response.model_copy(update={"additional_kwargs": new_kwargs})
                return response, messages
            except Exception as exc:
                category, detail = classify_error(exc)

                # Tool-use-failed: model embedded a tool call as JSON text
                # despite tool_choice="none" or no tools bound.
                if category == ErrorCategory.format_error and (
                    "tool_use_failed" in detail or "tool choice is none" in detail
                ):
                    retries += 1
                    if retries >= max_retries:
                        self.logger.warning(
                            "Max retries (%d) reached for tool-use-failed error",
                            max_retries,
                        )
                        raise

                    tool_name = self._extract_failed_tool_call(detail) or "unknown_tool"
                    self.logger.info(
                        "Model output a tool call as JSON: '%s' (attempt %d/%d), "
                        "injecting tool result",
                        tool_name, retries, max_retries,
                    )
                    tool_call_id = f"synthetic_{retries}"
                    messages = list(messages) + [
                        ToolMessage(
                            content=(
                                f"Tool '{tool_name}' was called but cannot be executed "
                                f"at this stage. The previous tool output is already "
                                f"complete — just answer the user with plain text now."
                            ),
                            tool_call_id=tool_call_id,
                        ),
                    ]
                    continue

                # Context overflow → compact messages and retry once
                if should_compact(category):
                    compacted = self._compact_messages(messages)
                    if compacted is not messages and len(compacted) < len(messages):
                        self.logger.info(
                            "Context overflow: compacted %d → %d messages, retrying",
                            len(messages), len(compacted),
                        )
                        messages = compacted
                        retries += 1
                        continue
                    self.logger.warning(
                        "Context overflow: cannot compact further, surfacing"
                    )
                    raise

                # Non-retryable errors (auth, format, permanent) → surface
                if not is_retryable(category):
                    self.logger.warning(
                        "Non-retryable API error (%s): %s", category.value, detail,
                    )
                    raise

                # Retryable: rate_limit, timeout, server_error, unknown
                retries += 1
                if retries >= max_retries:
                    self.logger.warning(
                        "Max retries (%d) reached for %s: %s",
                        max_retries, category.value, detail,
                    )
                    raise

                delay = self._jittered_backoff(retries, category)
                self.logger.info(
                    "API error (%s, attempt %d/%d): retrying in %.1fs — %s",
                    category.value, retries, max_retries, delay, detail,
                )
                await asyncio.sleep(delay)

    _TOOL_RESULT_COMPRESS_THRESHOLD = 200
    _TOOL_RESULT_KEEP_BATCHES = 2
    _TOOL_RESULT_SUMMARY_LIMIT = 120

    @staticmethod
    def _compress_tool_results(messages: List[BaseMessage]) -> List[BaseMessage]:
        """Summarize old tool result content to reduce token usage.

        Keeps the most recent N tool-result batches intact; older tool
        messages have their content replaced with a short summary.
        Prevents GLM error 1214 (accumulated multi-turn tool messages)
        and generally reduces context pressure.

        A "batch" is a run of consecutive ToolMessages following an
        AIMessage with ``tool_calls``.
        """
        batches: List[List[int]] = []
        current_batch: List[int] = []

        for i, msg in enumerate(messages):
            if isinstance(msg, ToolMessage):
                current_batch.append(i)
            else:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
        if current_batch:
            batches.append(current_batch)

        if len(batches) <= BaseAgent._TOOL_RESULT_KEEP_BATCHES:
            return messages

        old_batch_indices = set()
        for batch in batches[:-BaseAgent._TOOL_RESULT_KEEP_BATCHES]:
            old_batch_indices.update(batch)

        if not old_batch_indices:
            return messages

        result = list(messages)
        for idx in old_batch_indices:
            msg = result[idx]
            if not isinstance(msg, ToolMessage):
                continue
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if len(content) <= BaseAgent._TOOL_RESULT_COMPRESS_THRESHOLD:
                continue
            first_line = content.split("\n", 1)[0]
            if len(first_line) > BaseAgent._TOOL_RESULT_SUMMARY_LIMIT:
                first_line = first_line[:BaseAgent._TOOL_RESULT_SUMMARY_LIMIT - 3].rstrip() + "..."
            lines = content.count("\n") + 1
            summary = (
                f"[Tool result compressed — {lines} lines, "
                f"{len(content)} chars]\n{first_line}"
            )
            result[idx] = ToolMessage(content=summary, tool_call_id=msg.tool_call_id)

        return result

    @staticmethod
    def _sanitize_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
        """Strip surrogate characters and repair role alternation in-place.

        Four passes:
          1. Strip surrogate characters from message content.
          2. Drop orphaned ToolMessages (no preceding AI(tool_calls)).
          3. Compress old tool result content to reduce token pressure.
          4. Merge consecutive same-role messages (AI/Human only — ToolMessages
             must each keep their own tool_call_id).
        """
        result: List[BaseMessage] = []
        for msg in messages:
            content = msg.content
            if isinstance(content, str) and _SURROGATE_RE.search(content):
                cleaned = _SURROGATE_RE.sub("\ufffd", content)
                if isinstance(msg, HumanMessage):
                    msg = HumanMessage(content=cleaned)
                elif isinstance(msg, AIMessage):
                    msg = AIMessage(content=cleaned, tool_calls=getattr(msg, "tool_calls", []))
                elif isinstance(msg, ToolMessage):
                    msg = ToolMessage(content=cleaned, tool_call_id=msg.tool_call_id)
                elif isinstance(msg, SystemMessage):
                    msg = SystemMessage(content=cleaned)
            result.append(msg)

        # Pass 2: drop orphaned tool messages (no matching AIMessage with tool_calls)
        # A ToolMessage is valid if there is a preceding AIMessage with a
        # tool_call whose id matches — it does NOT have to be the immediately
        # previous message (parallel tool results are separate ToolMessages).
        known_tool_ids: set = set()
        repaired: List[BaseMessage] = []
        for msg in result:
            if isinstance(msg, AIMessage):
                known_tool_ids = set()
                for tc in (getattr(msg, "tool_calls", None) or []):
                    tc_id = tc.get("id") if isinstance(tc, dict) else None
                    if tc_id:
                        known_tool_ids.add(tc_id)
                repaired.append(msg)
            elif isinstance(msg, ToolMessage):
                if msg.tool_call_id in known_tool_ids:
                    repaired.append(msg)
            else:
                known_tool_ids = set()
                repaired.append(msg)

        # Pass 3: compress old tool result content
        repaired = BaseAgent._compress_tool_results(repaired)

        # Pass 4: merge consecutive same-role messages (except ToolMessages —
        # each must keep its own tool_call_id to match the parent AIMessage).
        merged: List[BaseMessage] = []
        for msg in repaired:
            if isinstance(msg, ToolMessage):
                merged.append(msg)
                continue
            if merged and type(merged[-1]) is type(msg):
                prev = merged[-1]
                prev_content = prev.content if isinstance(prev.content, str) else ""
                cur_content = msg.content if isinstance(msg.content, str) else ""
                if isinstance(prev, AIMessage) and isinstance(msg, AIMessage):
                    prev_tc = getattr(prev, "tool_calls", []) or []
                    cur_tc = getattr(msg, "tool_calls", []) or []
                    combined = (prev_content + "\n\n" + cur_content).strip()
                    merged[-1] = AIMessage(content=combined, tool_calls=prev_tc + cur_tc)
                elif isinstance(prev, HumanMessage) and isinstance(msg, HumanMessage):
                    combined = (prev_content + "\n\n" + cur_content).strip()
                    merged[-1] = HumanMessage(content=combined)
                else:
                    merged.append(msg)
            else:
                merged.append(msg)

        return merged

    @staticmethod
    def _compact_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
        """Drop the middle of the conversation to reduce context size.

        Keeps system message + first exchange + last 4 exchanges intact.
        """
        if len(messages) <= 4:
            return messages
        system = messages[:1]
        tail = messages[-4:]
        return system + [
            HumanMessage(content="[Earlier conversation compacted due to context limit]"),
        ] + tail

    @staticmethod
    def _jittered_backoff(attempt: int, category: ErrorCategory = ErrorCategory.unknown) -> float:
        base = 5.0 if category == ErrorCategory.rate_limit else 1.5
        delay = min(base * (1.5 ** (attempt - 1)), 30.0)
        return delay * (0.5 + random.random())

    @staticmethod
    def _extract_failed_tool_call(error_str: str) -> Optional[str]:
        """Extract the failed tool call name from an error message."""
        # Look for "name": "something" pattern in the failed_generation
        name_match = re.search(r'"name"\s*:\s*"([^"]+)"', error_str)
        if name_match:
            return name_match.group(1)
        return None

    @staticmethod
    def strip_thinking(text: str) -> str:
        """Remove <think>/<thinking> blocks and leaked chat-template tokens."""
        text = _RE_THINK.sub("", text)
        # Handle unclosed tags (model started thinking but response got cut)
        for tag in ("</think>", "</thinking>"):
            if tag in text:
                text = text[text.find(tag) + len(tag):]
        # Strip leaked <tool_call>…</tool_call> blocks (model forced to respond without tools)
        text = _RE_TOOL_CALL_LEAKED.sub("", text)
        # Strip chat-template tokens like <|endoftext|><|im_start|>user ...
        text = _RE_CHAT_TOKENS.sub("", text)
        return text.lstrip("\n").rstrip()

    @staticmethod
    def extract_text(response: AIMessage) -> str:
        """Extract plain text from an AIMessage, stripping thinking tags."""
        content = response.content if isinstance(response.content, str) else ""
        return BaseAgent.strip_thinking(content)

    @staticmethod
    def sanitize_output(text: str) -> str:
        """Strip leaked internal-context markers from user-facing output.

        Small models parrot the internal framing they see in their own replayed
        history — the ``[Earlier skill use — internal context:]`` skill summary
        and the ``[Report from …]`` / ``Trust: …`` trust headers. None of that is
        meant for the user. This is the single choke-point applied to every
        agent answer (final + interim) so all interfaces — text and voice — are
        covered, regardless of which model produced the leak.

        Conservative by design: only exact known marker prefixes are removed, so
        legitimate prose (a stray "Trust", a markdown ``---`` rule) survives. The
        contract is pinned by tests/test_output_sanitizer.py against real
        captured leaks.
        """
        if not text:
            return text

        lines = text.split("\n")
        cleaned: List[str] = []
        in_block = False       # inside a trailing [Earlier skill use]/[internal context] block
        just_dropped = False   # previous source line was removed as internal
        for line in lines:
            stripped = line.strip()
            if _RE_INTERNAL_BLOCK_START.match(line):
                # Start of a trailing summary block: drop it and the bullets /
                # blank lines that belong to it.
                in_block = True
                just_dropped = True
                continue
            if in_block:
                if stripped == "" or stripped.startswith(("- ", "* ", "•")) or stripped == "---":
                    just_dropped = True
                    continue
                in_block = False  # block ended; fall through to evaluate this line
            if _RE_INTERNAL_LINE.match(line):
                # A standalone trust-header / annotation line anywhere in the text.
                just_dropped = True
                continue
            if just_dropped and stripped == "---":
                # Separator left orphaned by a removed trust-header block; a bare
                # "---" elsewhere (a real markdown rule) is preserved.
                continue
            just_dropped = False
            cleaned.append(line)

        # Collapse the blank-line runs the removals may have left behind.
        result = "\n".join(cleaned)
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()
