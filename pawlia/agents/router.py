"""Backend-dispatching agent for PawLia.

Routes each request to either the existing local ChatAgent stack or a Hermes
backend based on the active model's provider backend.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, Optional

from pawlia.agents.chat import DEFAULT_SYSTEM_PROMPT, ChatAgent, _SENTENCE_RE
from pawlia.backends.hermes import HermesBackend


class _BackendLLMInfo:
    """Small metadata object so status views can inspect non-local backends."""

    def __init__(
        self,
        *,
        model: str,
        provider: str,
        backend: str,
        temperature: Optional[float] = None,
    ):
        self.model = model
        self.model_name = model
        self.provider_name = provider
        self.backend = backend
        self.temperature = temperature


class RouterAgent:
    """Dispatch requests to the configured backend while preserving PawLia UX."""

    def __init__(
        self,
        *,
        user_id: str,
        llm_factory: Any,
        memory: Any,
        session: Any,
        skills: Dict[str, Any],
        local_agent_factory: Callable[[], ChatAgent],
        logger: Any,
        on_interim: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        self.user_id = user_id
        self._llm_factory = llm_factory
        self.memory = memory
        self.session = session
        self._skills = skills
        self._local_agent_factory = local_agent_factory
        self._local_agent: Optional[ChatAgent] = None
        self._hermes_clients: Dict[str, HermesBackend] = {}
        self.logger = logger

        self.on_interim = on_interim
        self.on_skill_start: Optional[Callable[[str, str], Awaitable[None]]] = None
        self.on_skill_step: Optional[Callable[[str], Awaitable[None]]] = None
        self.on_skill_done: Optional[Callable[[str, str], Awaitable[None]]] = None
        self.on_model_change: Optional[Callable[[str], None]] = None
        self._on_fallback: Optional[Callable[[str, str], None]] = None  # propagated to chat agent

    @property
    def skills(self) -> Dict[str, Any]:
        return self._skills

    @property
    def pending_attachments(self) -> List[Dict[str, Any]]:
        """Expose the inner ChatAgent's attachment queue.

        Direct tools (e.g. ``attach_file``) queue onto the local ChatAgent, but
        interfaces drain ``agent.pending_attachments`` off the RouterAgent they
        hold. Without this proxy the queue is invisible and attachments are
        silently dropped (the "I attached it but you see nothing" bug). The
        hermes backend has no local agent and therefore no attachments.
        """
        if self._local_agent is not None:
            return self._local_agent.pending_attachments
        return []

    @property
    def llm(self) -> Any:
        meta = self.describe_backend(None)
        if meta["backend"] == "pawlia" and self._local_agent is not None:
            return self._local_agent.llm
        return _BackendLLMInfo(
            model=meta["resolved_model_name"],
            provider=meta["provider_name"],
            backend=meta["backend"],
            temperature=meta.get("temperature"),
        )

    @llm.setter
    def llm(self, value: Any) -> None:
        agent = self._ensure_local_agent()
        agent.llm = value

    def _ensure_local_agent(self) -> ChatAgent:
        if self._local_agent is None:
            self._local_agent = self._local_agent_factory()
        self._local_agent.on_interim = self.on_interim
        self._local_agent.on_skill_start = self.on_skill_start
        self._local_agent.on_skill_step = self.on_skill_step
        self._local_agent.on_skill_done = self.on_skill_done
        self._local_agent.on_model_change = self.on_model_change
        self._local_agent._on_fallback = self._on_fallback
        return self._local_agent

    def _agent_overrides(self, thread_id: Optional[str]) -> Dict[str, Any]:
        if self.memory and self.session:
            return self.memory.effective_agent_overrides(self.session, thread_id)
        return {}

    def _active_override_model(
        self,
        thread_id: Optional[str],
        *,
        agent_type: str = "chat",
    ) -> Optional[str]:
        overrides = self._agent_overrides(thread_id)
        if agent_type.startswith("skill."):
            skills = overrides.get("skills", {})
            if isinstance(skills, dict):
                value = skills.get(agent_type[len("skill."):])
                if isinstance(value, str) and value.strip():
                    return value.strip().split(",")[0].strip()
        value = overrides.get(agent_type)
        if isinstance(value, str) and value.strip():
            return value.strip().split(",")[0].strip()
        if agent_type != "default":
            default_value = overrides.get("default")
            if isinstance(default_value, str) and default_value.strip():
                return default_value.strip().split(",")[0].strip()
        return None

    def active_model_name(self, thread_id: Optional[str] = None, *, agent_type: str = "chat") -> str:
        overrides = self._agent_overrides(thread_id)
        return self._llm_factory.default_model_name(
            agent_type,
            agent_overrides=overrides,
        )

    def describe_backend(
        self,
        thread_id: Optional[str] = None,
        *,
        agent_type: str = "chat",
    ) -> Dict[str, Any]:
        selection = self.active_model_name(thread_id, agent_type=agent_type)
        model_cfg = self._llm_factory.get_model_config(selection)
        provider_name = self._llm_factory.get_provider_name_for_model(selection)
        provider_cfg = self._llm_factory.get_provider_config(provider_name)
        backend = self._llm_factory.get_backend_for_model(selection)
        return {
            "selection": selection,
            "backend": backend,
            "provider_name": provider_name,
            "provider_cfg": provider_cfg,
            "model_cfg": model_cfg,
            "resolved_model_name": str(model_cfg.get("model") or selection),
            "temperature": model_cfg.get("temperature"),
        }

    def list_skills(self, thread_id: Optional[str] = None) -> List[str]:
        if self.describe_backend(thread_id, agent_type="chat")["backend"] == "pawlia":
            return sorted(self._skills.keys())
        return []

    def build_system_prompt(
        self,
        *,
        mode: str = "chat",
        system_prompt: Optional[str] = None,
        thread_id: Optional[str] = None,
        extra_context: Optional[str] = None,
    ) -> str:
        if system_prompt:
            return system_prompt
        if self.memory and self.session:
            skills = self._skills if self.describe_backend(thread_id, agent_type="chat")["backend"] == "pawlia" else {}
            return self.memory.build_system_prompt(
                self.session,
                skills=skills,
                mode=mode,
                extra_context=extra_context,
            )
        return DEFAULT_SYSTEM_PROMPT

    def _hermes_client(self, thread_id: Optional[str] = None, *, agent_type: str = "chat") -> HermesBackend:
        meta = self.describe_backend(thread_id, agent_type=agent_type)
        provider_name = meta["provider_name"]
        if provider_name not in self._hermes_clients:
            self._hermes_clients[provider_name] = HermesBackend(
                model_name=meta["resolved_model_name"],
                provider_name=provider_name,
                provider_cfg=meta["provider_cfg"],
                logger_=self.logger.getChild(f"hermes.{provider_name}"),
            )
        client = self._hermes_clients[provider_name]
        client.model_name = meta["resolved_model_name"]
        return client

    async def run(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        images: Optional[List[str]] = None,
        thread_id: Optional[str] = None,
        on_skill_start: Optional[Callable[[str, str], Awaitable[None]]] = None,
        on_skill_step: Optional[Callable[[str], Awaitable[None]]] = None,
        on_skill_done: Optional[Callable[[str, str], Awaitable[None]]] = None,
    ) -> str:
        agent_type = "vision" if images else "chat"
        meta = self.describe_backend(thread_id, agent_type=agent_type)
        if meta["backend"] == "pawlia":
            agent = self._ensure_local_agent()
            return await agent.run(
                user_input,
                system_prompt=system_prompt,
                images=images,
                thread_id=thread_id,
                on_skill_start=on_skill_start,
                on_skill_step=on_skill_step,
                on_skill_done=on_skill_done,
            )

        prompt = self.build_system_prompt(
            mode="chat",
            system_prompt=system_prompt,
            thread_id=thread_id,
        )
        client = self._hermes_client(thread_id, agent_type=agent_type)
        result = await client.run(
            user_input=user_input,
            system_prompt=prompt,
            user_id=self.user_id,
            thread_id=thread_id,
            images=images,
        )
        await self._persist_hermes_turn(user_input, result, thread_id=thread_id)
        return result

    async def run_streamed(
        self,
        user_input: str,
        *,
        system_prompt: Optional[str] = None,
        images: Optional[List[str]] = None,
        thread_id: Optional[str] = None,
        on_sentence: Optional[Callable[[str], Awaitable[None]]] = None,
        on_skill_start: Optional[Callable[[str, str], Awaitable[None]]] = None,
        on_skill_step: Optional[Callable[[str], Awaitable[None]]] = None,
        on_skill_done: Optional[Callable[[str, str], Awaitable[None]]] = None,
        allow_skills: bool = True,
    ) -> str:
        agent_type = "vision" if images else "chat"
        meta = self.describe_backend(thread_id, agent_type=agent_type)
        if meta["backend"] == "pawlia":
            agent = self._ensure_local_agent()
            return await agent.run_streamed(
                user_input,
                system_prompt=system_prompt,
                images=images,
                thread_id=thread_id,
                on_sentence=on_sentence,
                on_skill_start=on_skill_start,
                on_skill_step=on_skill_step,
                on_skill_done=on_skill_done,
                allow_skills=allow_skills,
            )

        result = await self.run(
            user_input,
            system_prompt=system_prompt,
            images=images,
            thread_id=thread_id,
            on_skill_start=on_skill_start,
            on_skill_step=on_skill_step,
            on_skill_done=on_skill_done,
        )
        if on_sentence:
            for sentence in self._split_sentences(result):
                if sentence.strip():
                    await on_sentence(sentence.strip())
        return result

    async def _persist_hermes_turn(
        self,
        user_input: str,
        response: str,
        *,
        thread_id: Optional[str] = None,
    ) -> None:
        if not (self.memory and self.session):
            return
        if thread_id:
            self.memory.append_thread_exchange(self.session, thread_id, user_input, response)
            return
        self.memory.append_exchange(self.session, user_input, response, track_similarity=True)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        sentences: List[str] = []
        remaining = text.strip()
        while True:
            match = _SENTENCE_RE.search(remaining)
            if not match:
                break
            end = match.start() + 1
            sentences.append(remaining[:end].strip())
            remaining = remaining[end:].lstrip()
        if remaining:
            sentences.append(remaining)
        return sentences
