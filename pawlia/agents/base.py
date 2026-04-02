"""Base agent class with shared LLM invocation and thinking-tag cleanup."""

import asyncio
import logging
import os
import re
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


_RE_THINK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL)
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

        Runs synchronous ``llm.invoke`` in a thread to keep the event loop free.
        Retries up to *max_retries* times when the model calls a tool despite
        ``tool_choice="none"`` — a common failure mode for smaller models.
        """
        log_prompt(messages, name=self.log_name)
        target = llm or self.llm

        def _call() -> AIMessage:
            try:
                return target.invoke(messages)
            except StopIteration as exc:
                # Python 3.14+: StopIteration cannot propagate out of a thread
                # into a Future; wrap it so asyncio.to_thread works.
                raise RuntimeError("LLM invoke exhausted iterator") from exc

        retries = 0
        while True:
            try:
                return await asyncio.to_thread(_call)
            except Exception as exc:
                error_str = str(exc)
                # Detect "tool use failed" errors from OpenAI-compatible APIs.
                # This happens when the model embeds a tool call as JSON text
                # instead of using the structured tool_calls mechanism — or when
                # the model tries to call a tool despite tool_choice="none".
                if ("tool_use_failed" in error_str or
                    ("Tool choice is none" in error_str and "called a tool" in error_str)):
                    retries += 1
                    if retries >= max_retries:
                        self.logger.warning(
                            "Max retries (%d) reached for tool-use-failed error", max_retries)
                        raise

                    tool_name = self._extract_failed_tool_call(error_str) or "unknown_tool"
                    self.logger.info(
                        "Model output a tool call as JSON: '%s' (attempt %d/%d), "
                        "injecting tool result", tool_name, retries, max_retries)

                    # Add a ToolMessage to messages (not HumanMessage — the model
                    # responds with text when it "sees" a tool result in context)
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

                # Re-raise any other exceptions
                raise

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
        # Strip chat-template tokens like <|endoftext|><|im_start|>user ...
        text = _RE_CHAT_TOKENS.sub("", text)
        return text.lstrip("\n").rstrip()

    @staticmethod
    def extract_text(response: AIMessage) -> str:
        """Extract plain text from an AIMessage, stripping thinking tags."""
        content = response.content if isinstance(response.content, str) else ""
        return BaseAgent.strip_thinking(content)
