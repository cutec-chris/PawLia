"""Base agent class with shared LLM invocation and thinking-tag cleanup."""

import asyncio
import logging
import os
import random
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, List, Optional

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


_RE_THINK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL)
_RE_TOOL_CALL_LEAKED = re.compile(r"<tool_call>.*?(?:</tool_call>|$)", re.DOTALL)
# Chat-template tokens that some models leak into their output
_RE_CHAT_TOKENS = re.compile(r"<\|.*?\|>.*", re.DOTALL)
# Pattern for tool call in failed_generation from API errors
_RE_TOOL_CALL = re.compile(r'\{.*?"name"\s*:.*?"args"\s*:.*?\}', re.DOTALL)

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
                      max_retries: int = 3) -> AIMessage:
        """Invoke an LLM (default: self.llm) with the given messages.

        Classifies API errors and applies the correct recovery strategy:
        - context_overflow → compact and retry
        - rate_limit / timeout / server_error → jittered backoff + retry
        - auth_error / format_error → surface immediately, no retry
        - tool-use-failed → inject synthetic ToolMessage, retry
        """
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
                return await run_sync_in_thread(_call)
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
