"""Base agent class with shared LLM invocation and thinking-tag cleanup."""

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI


_RE_THINK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL)
# Chat-template tokens that some models leak into their output
_RE_CHAT_TOKENS = re.compile(r"<\|.*?\|>.*", re.DOTALL)


class BaseAgent(ABC):
    """Abstract base for all agents."""

    def __init__(self, llm: ChatOpenAI, logger: Optional[logging.Logger] = None):
        self.llm = llm
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> str:
        """Execute the agent's main task and return a text result."""

    async def _invoke(self, messages: List[BaseMessage],
                      llm: Optional[ChatOpenAI] = None) -> AIMessage:
        """Invoke an LLM (default: self.llm) with the given messages.

        Runs synchronous ``llm.invoke`` in a thread to keep the event loop free.
        """
        target = llm or self.llm

        def _call() -> AIMessage:
            try:
                return target.invoke(messages)
            except StopIteration as exc:
                # Python 3.14+: StopIteration cannot propagate out of a thread
                # into a Future; wrap it so asyncio.to_thread works.
                raise RuntimeError("LLM invoke exhausted iterator") from exc

        return await asyncio.to_thread(_call)

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
