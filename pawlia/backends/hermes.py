"""Hermes backend adapter for PawLia.

Uses Hermes' stateful Responses API so Hermes keeps its own tool/runtime
context, while PawLia remains the interface and logging layer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import aiohttp


logger = logging.getLogger(__name__)


class HermesBackend:
    """Thin async client for a Hermes API server."""

    def __init__(
        self,
        *,
        model_name: str,
        provider_name: str,
        provider_cfg: Dict[str, Any],
        logger_: Optional[logging.Logger] = None,
    ):
        self.model_name = model_name
        self.provider_name = provider_name
        self.provider_cfg = dict(provider_cfg)
        self.logger = logger_ or logger

        self.base_url = str(self.provider_cfg.get("apiBase", "http://127.0.0.1:8642/v1")).rstrip("/")
        self.api_key = str(self.provider_cfg.get("apiKey", "") or "")
        self.timeout = int(self.provider_cfg.get("timeout", 600) or 600)
        self.conversation_namespace = str(self.provider_cfg.get("conversation_namespace", "pawlia"))
        self.store = bool(self.provider_cfg.get("store", True))

    def conversation_name(self, user_id: str, thread_id: Optional[str] = None) -> str:
        suffix = f"thread:{thread_id}" if thread_id else "main"
        return f"{self.conversation_namespace}:{user_id}:{suffix}"

    async def run(
        self,
        *,
        user_input: str,
        system_prompt: str,
        user_id: str,
        thread_id: Optional[str] = None,
        images: Optional[List[str]] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "input": [self._build_input_item(user_input, images=images)],
            "instructions": system_prompt,
            "conversation": self.conversation_name(user_id, thread_id),
            "store": self.store,
        }

        from pawlia.utils import PAWLIA_USER_AGENT
        headers = {"Content-Type": "application/json",
                   "User-Agent": PAWLIA_USER_AGENT}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        endpoint = f"{self.base_url}/responses"
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, json=payload, headers=headers) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise RuntimeError(
                        f"Hermes request failed ({resp.status}) via provider '{self.provider_name}': {body}"
                    )
                data = await resp.json()

        text = self._extract_text(data)
        if not text.strip():
            raise RuntimeError("Hermes returned no assistant text")
        return text

    def _build_input_item(
        self,
        user_input: str,
        *,
        images: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = []
        text = user_input or "What's in this image?"
        content.append({"type": "input_text", "text": text})
        for data_uri in images or []:
            content.append({"type": "input_image", "image_url": data_uri})
        return {"role": "user", "content": content}

    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> str:
        top_level = str(data.get("output_text", "") or "").strip()
        if top_level:
            return top_level
        parts: List[str] = []
        for item in data.get("output", []) or []:
            if item.get("type") != "message" or item.get("role") != "assistant":
                continue
            for content in item.get("content", []) or []:
                if content.get("type") == "output_text":
                    text = str(content.get("text", "") or "").strip()
                    if text:
                        parts.append(text)
        return "\n\n".join(parts).strip()
