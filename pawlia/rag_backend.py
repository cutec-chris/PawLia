"""Abstract RAG backend interface and implementations.

Supported backends:
  - lightrag  (default) — LightRAG knowledge-graph RAG, powerful but slow
  - simple              — chunking + embedding + cosine similarity, no extra deps
  - mem0                — mem0 fact-extraction (requires: pip install mem0ai chromadb)

Select via skill-config:
  memory:
    rag_backend: simple   # lightrag | simple | mem0

All backends expose the same async interface:
  insert(text, doc_id)
  wait_for_indexed(doc_id, timeout, poll_interval)
  query(question)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Callable, Optional

logger = logging.getLogger("pawlia.rag_backend")


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class RagBackend(ABC):
    """Minimal async interface for indexing and querying documents."""

    @abstractmethod
    async def insert(self, text: str, doc_id: str) -> None:
        """Index *text* under *doc_id* (used for deduplication/updates)."""

    @abstractmethod
    async def wait_for_indexed(
        self, doc_id: str, timeout: int = 120, poll_interval: float = 5.0,
    ) -> bool:
        """Block until *doc_id* is indexed or timeout/failure. Returns True on success."""

    @abstractmethod
    async def query(self, question: str) -> str:
        """Return a natural-language answer synthesised from the index."""


# ---------------------------------------------------------------------------
# LightRAG backend
# ---------------------------------------------------------------------------

def _patch_tiktoken():
    """Allow special tokens that models like qwen emit."""
    try:
        import tiktoken.core
        _orig = tiktoken.core.Encoding.encode
        def _permissive(self, text, **kw):
            kw.setdefault("disallowed_special", ())
            return _orig(self, text, **kw)
        tiktoken.core.Encoding.encode = _permissive
    except Exception:
        pass


class LightRAGBackend(RagBackend):
    """Wraps a LightRAG instance.

    Parameters
    ----------
    index_path:
        Directory where LightRAG stores its graph/vector data.
    cfg:
        skill-config dict (embedding_provider, rag_model, …).
        An optional ``_query_model`` key (dict with provider/model/host)
        overrides the LLM used for query synthesis — used by memory.py to
        reuse the already-loaded chat model.
    llm_busy_check:
        Optional callable ``() -> bool``; if it returns True, embedding is
        aborted to avoid competing with a running chat request.
    think:
        Whether to pass ``think=True`` to the Ollama LLM. False during
        indexing (faster), None/True during query synthesis (let the model
        think if it wants to).
    max_async_llm / max_async_embedding:
        Override the config defaults.
    """

    def __init__(
        self,
        index_path: str,
        cfg: dict,
        llm_busy_check: Optional[Callable[[], bool]] = None,
        think: Optional[bool] = False,
        max_async_llm: Optional[int] = None,
        max_async_embedding: Optional[int] = None,
    ):
        _patch_tiktoken()
        self._index_path = index_path
        self._cfg = cfg
        self._llm_busy = llm_busy_check
        self._think = think
        self._max_async_llm = max_async_llm
        self._max_async_embedding = max_async_embedding
        self._rag = None

    # ── LightRAG wiring ─────────────────────────────────────────────────────

    def _build_embedding_func(self):
        import lightrag.utils
        import lightrag.llm.ollama
        import lightrag.llm.openai

        cfg = self._cfg
        provider = cfg.get("embedding_provider", "ollama")
        model = cfg["embedding_model"]
        dim = int(cfg["embedding_dim"])
        host = cfg.get("embedding_host", "http://localhost:11434")
        llm_busy = self._llm_busy

        if provider == "ollama":
            async def _ollama_embed(texts):
                import numpy as np
                if llm_busy and llm_busy():
                    raise RuntimeError("LLM busy — deferring memory embedding")
                try:
                    result = await lightrag.llm.ollama.ollama_embed(
                        texts, host=host, embed_model=model, max_token_size=8192,
                    )
                except Exception as e:
                    if "NaN" in str(e):
                        logger.warning("Ollama NaN embedding error, returning zero vectors: %s", e)
                        return np.zeros((len(texts), dim))
                    raise
                arr = np.array(result)
                if np.isnan(arr).any():
                    logger.warning("NaN in embeddings detected, replacing with 0.0")
                    arr = np.nan_to_num(arr, nan=0.0)
                return arr
            return lightrag.utils.EmbeddingFunc(embedding_dim=dim, func=_ollama_embed)

        api_key = cfg.get("embedding_api_key")
        base_url = cfg.get("embedding_base_url")

        async def _embed(texts):
            return await lightrag.llm.openai.openai_embed(
                texts, embed_model=model, api_key=api_key, base_url=base_url,
            )
        return lightrag.utils.EmbeddingFunc(embedding_dim=dim, func=_embed)

    def _build_llm_func(self):
        import lightrag.llm.ollama
        import lightrag.llm.openai

        cfg = self._cfg
        think = self._think

        # _query_model_override allows memory.py to use the already-loaded chat
        # model for query synthesis instead of forcing a separate rag_model load.
        override = cfg.get("_query_model")
        if override:
            provider = override.get("provider", "ollama")
            model = override.get("model", cfg.get("rag_model", "qwen3.5:latest"))
            host = override.get("host", cfg.get("embedding_host", "http://localhost:11434"))
        else:
            provider = cfg.get("rag_provider", cfg.get("embedding_provider", "ollama"))
            model = cfg.get("rag_model", "qwen3.5:latest")
            host = cfg.get("embedding_host", "http://localhost:11434")

        if provider == "ollama":
            async def _complete(prompt: str, system_prompt: str = "", **kw):
                options: dict = {"num_ctx": int(cfg.get("rag_numctx", 4096))}
                if think is not None:
                    options["think"] = think
                return await lightrag.llm.ollama.ollama_model_complete(
                    prompt, system_prompt=system_prompt,
                    host=host, options=options, **kw,
                )
            return _complete

        api_key = cfg.get("rag_api_key", cfg.get("embedding_api_key"))
        base_url = cfg.get("rag_base_url", cfg.get("embedding_base_url"))

        async def _complete(prompt, system_prompt=None, history_messages=[], **kw):
            return await lightrag.llm.openai.openai_complete_if_cache(
                model, prompt, system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key, base_url=base_url, **kw,
            )
        return _complete

    async def _get_rag(self):
        import lightrag
        import lightrag.kg.shared_storage

        if self._rag is not None:
            return self._rag

        os.makedirs(self._index_path, exist_ok=True)
        cfg = self._cfg

        # Resolve the model name for LightRAG's internal bookkeeping
        override = cfg.get("_query_model")
        model_name = (
            override.get("model") if override else cfg.get("rag_model", "qwen3.5:latest")
        )

        max_async_llm = self._max_async_llm or int(cfg.get("rag_max_async_llm", 1))
        max_async_emb = self._max_async_embedding or int(cfg.get("rag_max_async_embedding", 1))

        self._rag = lightrag.LightRAG(
            self._index_path,
            llm_model_func=self._build_llm_func(),
            llm_model_name=model_name,
            summary_max_tokens=8192,
            enable_llm_cache=False,
            llm_model_kwargs={},
            embedding_func=self._build_embedding_func(),
            default_llm_timeout=int(cfg.get("rag_timeout", 600)),
            default_embedding_timeout=int(cfg.get("rag_embedding_timeout", 120)),
            llm_model_max_async=max_async_llm,
            embedding_func_max_async=max_async_emb,
        )
        await self._rag.initialize_storages()
        await lightrag.kg.shared_storage.initialize_pipeline_status()
        return self._rag

    # ── RagBackend interface ────────────────────────────────────────────────

    async def insert(self, text: str, doc_id: str) -> None:
        rag = await self._get_rag()
        await rag.ainsert(text, ids=doc_id)

    async def wait_for_indexed(
        self, doc_id: str, timeout: int = 120, poll_interval: float = 5.0,
    ) -> bool:
        rag = await self._get_rag()
        elapsed = 0.0
        while elapsed < timeout:
            if self._llm_busy and self._llm_busy():
                logger.debug("LLM busy, stopping poll for %s", doc_id)
                return False
            status = await rag.doc_status.get_by_id(doc_id)
            if status:
                s = status.get("status")
                if s == "processed":
                    return True
                if s == "failed":
                    return False
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        return False

    async def query(self, question: str) -> str:
        import lightrag
        rag = await self._get_rag()
        return await rag.aquery(
            question,
            param=lightrag.QueryParam(mode="global", enable_rerank=False),
        )


# ---------------------------------------------------------------------------
# mem0 backend
# ---------------------------------------------------------------------------

class Mem0Backend(RagBackend):
    """mem0-based memory backend.

    Uses mem0's fact-extraction approach: documents are split into atomic
    facts by a small LLM call, then stored in a local ChromaDB vector store.
    Queries retrieve the most relevant facts, returned as a formatted string.

    Much faster than LightRAG for indexing because there is no full
    knowledge-graph construction — typically 2-5 LLM calls per document
    instead of 10-20+.

    Configuration keys (same section as lightrag):
      rag_model, embedding_model, embedding_host, embedding_dim
    """

    def __init__(
        self,
        index_path: str,
        cfg: dict,
        llm_busy_check: Optional[Callable[[], bool]] = None,
    ):
        self._index_path = index_path
        self._cfg = cfg
        self._llm_busy = llm_busy_check
        self._memory = None
        self._indexed: set[str] = set()

    def _build_config(self) -> dict:
        cfg = self._cfg
        llm_provider = cfg.get("rag_provider", cfg.get("embedding_provider", "ollama"))
        emb_provider = cfg.get("embedding_provider", "ollama")
        model = cfg.get("rag_model", "qwen3.5:latest")
        embedding_model = cfg.get("embedding_model", "bge-m3:latest")
        host = cfg.get("embedding_host", "http://localhost:11434")
        dim = int(cfg.get("embedding_dim", 1024))

        # LLM config — Ollama vs. OpenAI-compatible
        if llm_provider == "ollama":
            llm_cfg: dict = {
                "model": model,
                "ollama_base_url": host,
                "temperature": 0,
                "max_tokens": 2000,
            }
        else:
            llm_cfg = {
                "model": model,
                "api_key": cfg.get("rag_api_key", cfg.get("embedding_api_key", "")),
                "openai_base_url": cfg.get("rag_base_url", cfg.get("embedding_base_url", "")),
                "temperature": 0,
                "max_tokens": 2000,
            }
            llm_provider = "openai"  # mem0 uses "openai" for any OpenAI-compat provider

        # Embedder config — Ollama vs. OpenAI-compatible
        if emb_provider == "ollama":
            emb_cfg: dict = {
                "model": embedding_model,
                "ollama_base_url": host,
                "embedding_dims": dim,
            }
        else:
            emb_cfg = {
                "model": embedding_model,
                "api_key": cfg.get("embedding_api_key", ""),
                "openai_base_url": cfg.get("embedding_base_url", ""),
                "embedding_dims": dim,
            }
            emb_provider = "openai"

        return {
            "llm": {"provider": llm_provider, "config": llm_cfg},
            "embedder": {"provider": emb_provider, "config": emb_cfg},
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "pawlia_memory",
                    "path": os.path.join(self._index_path, "chroma"),
                },
            },
            "history_db_path": os.path.join(self._index_path, "history.db"),
        }

    def _get_memory(self):
        if self._memory is None:
            from mem0 import Memory
            self._memory = Memory.from_config(self._build_config())
        return self._memory

    async def insert(self, text: str, doc_id: str) -> None:
        if self._llm_busy and self._llm_busy():
            raise RuntimeError("LLM busy — deferring memory embedding")
        memory = self._get_memory()
        # mem0.add() is synchronous and blocking — run in thread pool
        await asyncio.to_thread(memory.add, text, user_id=doc_id)
        self._indexed.add(doc_id)

    async def wait_for_indexed(
        self, doc_id: str, timeout: int = 120, poll_interval: float = 5.0,
    ) -> bool:
        # insert() is synchronous/blocking so by the time it returns the
        # document is already indexed — nothing to poll.
        return doc_id in self._indexed

    async def query(self, question: str) -> str:
        memory = self._get_memory()
        results = await asyncio.to_thread(memory.search, query=question, limit=10)

        # mem0 returns {"results": [...]} where each item has a "memory" key
        items = results if isinstance(results, list) else results.get("results", [])
        if not items:
            return "Keine relevanten Erinnerungen gefunden."

        lines = []
        for r in items:
            if isinstance(r, dict) and "memory" in r:
                score = r.get("score", 0)
                lines.append(f"- {r['memory']}  (relevance: {score:.2f})")
            else:
                lines.append(f"- {r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Simple vector backend (numpy only, no external DB)
# ---------------------------------------------------------------------------

class SimpleVectorBackend(RagBackend):
    """Chunk → embed → cosine-similarity vector search.

    No external dependencies beyond numpy (already required by the project).
    Indexing is ~10–50x faster than LightRAG because there are no LLM calls —
    only embedding calls.

    Storage layout inside *index_path*:
      vectors.npy   — float32 array (N, dim)
      chunks.json   — [{text, doc_id, chunk_idx}]
    """

    CHUNK_SIZE = 800   # max chars per chunk
    TOP_K = 6          # chunks returned per query
    MIN_SCORE = 0.1    # minimum cosine similarity to include a chunk

    def __init__(
        self,
        index_path: str,
        cfg: dict,
        llm_busy_check: Optional[Callable[[], bool]] = None,
    ):
        self._index_path = index_path
        self._cfg = cfg
        self._llm_busy = llm_busy_check
        self._chunks: list[str] = []
        self._meta: list[dict] = []     # [{doc_id, chunk_idx}]
        self._vectors = None            # np.ndarray (N, dim) or None
        self._indexed: set[str] = set()
        self._loaded = False

    # ── persistence ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        import numpy as np
        vp = os.path.join(self._index_path, "vectors.npy")
        mp = os.path.join(self._index_path, "chunks.json")
        if os.path.exists(vp) and os.path.exists(mp):
            self._vectors = np.load(vp)
            with open(mp, encoding="utf-8") as f:
                data = json.load(f)
            self._chunks = [d["text"] for d in data]
            self._meta = [{"doc_id": d["doc_id"], "chunk_idx": d["chunk_idx"]} for d in data]
            self._indexed = {d["doc_id"] for d in data}

    def _save(self) -> None:
        import numpy as np
        os.makedirs(self._index_path, exist_ok=True)
        if self._vectors is not None:
            np.save(os.path.join(self._index_path, "vectors.npy"), self._vectors)
        data = [
            {"text": t, "doc_id": m["doc_id"], "chunk_idx": m["chunk_idx"]}
            for t, m in zip(self._chunks, self._meta)
        ]
        with open(os.path.join(self._index_path, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    # ── chunking ────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str) -> list[str]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[str] = []
        buf: list[str] = []
        buf_len = 0
        for para in paragraphs:
            if buf_len + len(para) > self.CHUNK_SIZE and buf:
                chunks.append("\n\n".join(buf))
                buf = [buf[-1]]   # one-paragraph overlap
                buf_len = len(buf[0])
            buf.append(para)
            buf_len += len(para)
        if buf:
            chunks.append("\n\n".join(buf))
        return chunks or [text[:self.CHUNK_SIZE]]

    # ── embedding ───────────────────────────────────────────────────────────

    async def _embed(self, texts: list[str]):
        import numpy as np
        cfg = self._cfg
        provider = cfg.get("embedding_provider", "ollama")
        model = cfg.get("embedding_model", "bge-m3:latest")
        host = cfg.get("embedding_host", "http://localhost:11434")
        dim = int(cfg.get("embedding_dim", 1024))

        if self._llm_busy and self._llm_busy():
            raise RuntimeError("LLM busy — deferring memory embedding")

        if provider == "ollama":
            import lightrag.llm.ollama
            try:
                result = await lightrag.llm.ollama.ollama_embed(
                    texts, host=host, embed_model=model, max_token_size=8192,
                )
            except Exception as e:
                if "NaN" in str(e):
                    return np.zeros((len(texts), dim))
                raise
            arr = np.array(result, dtype="float32")
        else:
            import lightrag.llm.openai
            result = await lightrag.llm.openai.openai_embed(
                texts,
                embed_model=model,
                api_key=cfg.get("embedding_api_key"),
                base_url=cfg.get("embedding_base_url"),
            )
            arr = np.array(result, dtype="float32")

        if np.isnan(arr).any():
            arr = np.nan_to_num(arr, nan=0.0)
        return arr

    # ── RagBackend interface ─────────────────────────────────────────────────

    async def insert(self, text: str, doc_id: str) -> None:
        import numpy as np
        self._load()

        # Remove existing chunks for this doc_id (re-index support)
        if doc_id in self._indexed:
            keep = [i for i, m in enumerate(self._meta) if m["doc_id"] != doc_id]
            self._chunks = [self._chunks[i] for i in keep]
            self._meta = [self._meta[i] for i in keep]
            self._vectors = self._vectors[keep] if self._vectors is not None and keep else None
            self._indexed.discard(doc_id)

        chunks = self._chunk_text(text)
        vectors = await self._embed(chunks)

        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            self._chunks.append(chunk)
            self._meta.append({"doc_id": doc_id, "chunk_idx": i})
            if self._vectors is None:
                self._vectors = vec.reshape(1, -1)
            else:
                self._vectors = np.vstack([self._vectors, vec.reshape(1, -1)])

        self._indexed.add(doc_id)
        self._save()
        logger.debug("SimpleVectorBackend: indexed %d chunks for %s", len(chunks), doc_id)

    async def wait_for_indexed(
        self, doc_id: str, timeout: int = 120, poll_interval: float = 5.0,
    ) -> bool:
        return doc_id in self._indexed

    async def query(self, question: str) -> str:
        import numpy as np
        self._load()

        if not self._chunks or self._vectors is None:
            return "Noch keine Dokumente indiziert."

        q_vec = (await self._embed([question]))[0]
        norms = np.linalg.norm(self._vectors, axis=1)
        q_norm = np.linalg.norm(q_vec)
        valid = norms > 0
        scores = np.zeros(len(self._chunks))
        if valid.any() and q_norm > 0:
            scores[valid] = (self._vectors[valid] @ q_vec) / (norms[valid] * q_norm)

        top_k = min(self.TOP_K, len(self._chunks))
        top_idx = np.argsort(scores)[::-1][:top_k]

        parts = [self._chunks[i] for i in top_idx if scores[i] >= self.MIN_SCORE]
        if not parts:
            return "Keine relevanten Informationen gefunden."
        return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_backend(
    index_path: str,
    cfg: dict,
    llm_busy_check: Optional[Callable[[], bool]] = None,
    *,
    think: Optional[bool] = False,
    max_async_llm: Optional[int] = None,
    max_async_embedding: Optional[int] = None,
) -> RagBackend:
    """Instantiate the configured RAG backend.

    Select via ``cfg["rag_backend"]``:
      - ``"lightrag"`` (default) — LightRAG knowledge-graph
      - ``"simple"``             — chunking + cosine similarity (numpy only)
      - ``"mem0"``               — mem0 fact extraction (requires mem0ai + chromadb)

    Parameters
    ----------
    think:
        Passed to LightRAGBackend. Use ``False`` for indexing (faster),
        ``None`` for querying (lets thinking models think freely).
    max_async_llm / max_async_embedding:
        Override config defaults (useful for query-only instances that can
        afford higher concurrency).
    """
    backend_name = cfg.get("rag_backend", "lightrag")

    if backend_name == "simple":
        logger.debug("Using SimpleVector backend at %s", index_path)
        return SimpleVectorBackend(index_path, cfg, llm_busy_check)

    if backend_name == "mem0":
        logger.debug("Using mem0 backend at %s", index_path)
        return Mem0Backend(index_path, cfg, llm_busy_check)

    logger.debug("Using LightRAG backend at %s", index_path)
    return LightRAGBackend(
        index_path, cfg, llm_busy_check,
        think=think,
        max_async_llm=max_async_llm,
        max_async_embedding=max_async_embedding,
    )
