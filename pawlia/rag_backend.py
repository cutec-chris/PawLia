"""Abstract RAG backend interface and implementations.

Supported backends:
  - markdown  (default) — Dream Wiki: LLM builds structured, interlinked wiki
  - lightrag            — LightRAG knowledge-graph RAG, powerful but slow
  - simple              — chunking + embedding + cosine similarity, no extra deps
  - mem0                — mem0 fact-extraction (requires: pip install mem0ai chromadb)

Select via skill-config:
  memory:
    rag_backend: lightrag   # markdown | lightrag | simple | mem0

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
import re
import unicodedata
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
                    # Use .func to bypass lightrag's hardcoded embedding_dim=1024 wrapper
                    result = await lightrag.llm.ollama.ollama_embed.func(
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
            param=lightrag.QueryParam(
                mode="hybrid",
                only_need_context=True,
            ),
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
                result = await lightrag.llm.ollama.ollama_embed.func(
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
# Markdown topic backend (no embeddings, no knowledge graph)
# ---------------------------------------------------------------------------

class MarkdownTopicBackend(RagBackend):
    """Topic-based markdown file backend — no embeddings or knowledge graphs.

    Uses an LLM to categorise conversation content into topics and writes
    one markdown file per topic.  Retrieval is simple keyword matching
    across the topic files — no vector search, no extra dependencies.

    Storage layout inside *index_path*:
      topics/          — one .md file per topic (slug-based filenames)
      doc_topics.json  — {doc_id: [slug, …]}  for tracking
    """

    # Size limit for query results (chars, ~8 K tokens).
    _MAX_RESULT_CHARS = 32_000

    # Common stop words (DE + EN) excluded from keyword matching.
    _STOP_WORDS = frozenset(
        "der die das und oder ein eine ist war hat haben was wie wer wo wann "
        "warum ich du wir sie er es mit von zu für auf in an bei nach über "
        "unter vor hinter zwischen nicht auch noch schon nur aber denn wenn "
        "dass weil als ob the a an and or is was are were has have what how "
        "who where when why i you we they he she it with from to for on at "
        "by about do did not also".split()
    )

    def __init__(
        self,
        index_path: str,
        cfg: dict,
        llm_busy_check: Optional[Callable[[], bool]] = None,
    ):
        self._index_path = index_path
        self._cfg = cfg
        self._llm_busy = llm_busy_check
        self._topics_dir = os.path.join(index_path, "topics")
        self._tracker_path = os.path.join(index_path, "doc_topics.json")
        self._indexed: set[str] = set()
        self._tracker: Optional[dict] = None

    # ── helpers ──────────────────────────────────────────────────────────────

    def _load_tracker(self) -> dict:
        if self._tracker is not None:
            return self._tracker
        if os.path.exists(self._tracker_path):
            try:
                with open(self._tracker_path, encoding="utf-8") as f:
                    self._tracker = json.load(f)
                self._indexed = set(self._tracker.keys())
                return self._tracker
            except Exception:
                pass
        self._tracker = {}
        return self._tracker

    def _save_tracker(self) -> None:
        os.makedirs(self._index_path, exist_ok=True)
        with open(self._tracker_path, "w", encoding="utf-8") as f:
            json.dump(self._tracker or {}, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _slugify(name: str) -> str:
        """Convert a topic name to a filesystem-safe slug."""
        slug = name.lower().strip()
        for old, new in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
            slug = slug.replace(old, new)
        slug = unicodedata.normalize("NFKD", slug)
        slug = slug.encode("ascii", "ignore").decode("ascii")
        slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
        return slug[:80] or "misc"

    # ── helpers ──────────────────────────────────────────────────────────────

    def _get_existing_topic_catalog(self) -> dict[str, str]:
        """Return {slug: title} for all existing topic files."""
        catalog: dict[str, str] = {}
        if not os.path.isdir(self._topics_dir):
            return catalog
        for fname in os.listdir(self._topics_dir):
            if not fname.endswith(".md"):
                continue
            slug = fname[:-3]
            filepath = os.path.join(self._topics_dir, fname)
            try:
                with open(filepath, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                title = first_line.lstrip("#").strip() or slug
            except Exception:
                title = slug
            catalog[slug] = title
        return catalog

    @staticmethod
    def _find_similar_slug(
        new_slug: str, existing_slugs: list[str], threshold: float = 0.7
    ) -> Optional[str]:
        """Return the most similar existing slug if similarity >= threshold, else None."""
        from difflib import SequenceMatcher
        best_slug: Optional[str] = None
        best_ratio = 0.0
        for existing in existing_slugs:
            ratio = SequenceMatcher(None, new_slug, existing).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_slug = existing
        return best_slug if best_ratio >= threshold else None

    # ── LLM calls ────────────────────────────────────────────────────────────

    async def _llm_call(self, system_prompt: str, user_prompt: str) -> str:
        """Make a single LLM call and return the stripped content string."""
        import urllib.request

        cfg = self._cfg
        provider = cfg.get("rag_provider", cfg.get("embedding_provider", "ollama"))
        model = cfg.get("rag_model", "qwen3.5:latest")
        host = cfg.get("embedding_host", "http://localhost:11434")

        if provider == "ollama":
            url = f"{host.rstrip('/')}/api/chat"
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {
                    "num_ctx": int(cfg.get("rag_numctx", 4096)),
                    "temperature": 0.1,
                },
            }
        else:
            base = cfg.get("rag_base_url", cfg.get("embedding_base_url", host))
            url = f"{base.rstrip('/')}/chat/completions"
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
            }

        body = json.dumps(payload).encode()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if provider != "ollama":
            api_key = cfg.get("rag_api_key", cfg.get("embedding_api_key", ""))
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        def _do():
            timeout = int(cfg.get("rag_timeout", 600))
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())

        result = await asyncio.to_thread(_do)

        if provider == "ollama":
            content = result.get("message", {}).get("content", "")
        else:
            content = (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    async def _llm_extract_topics(
        self, text: str, existing_topics: dict[str, str]
    ) -> list[dict]:
        """Extract topics from *text*, guided by the existing topic catalog."""
        existing_hint = ""
        if existing_topics:
            lines = "\n".join(
                f"- {slug}: {title}" for slug, title in sorted(existing_topics.items())
            )
            existing_hint = (
                f"\nBestehende Themen (verwende diese Slugs wenn der Inhalt dazu passt):\n"
                f"{lines}\n"
            )

        system_prompt = (
            "Du bist ein Assistent der Gesprächsprotokolle analysiert und nach "
            "Themen sortiert. Antworte NUR mit validem JSON."
        )
        user_prompt = (
            "Analysiere das folgende Gesprächsprotokoll und extrahiere die "
            "besprochenen Themen.\nFür jedes Thema erstelle eine Zusammenfassung "
            "mit den wichtigsten Informationen, Entscheidungen und Details.\n"
            + existing_hint
            + "\nAntworte NUR mit einem JSON-Array:\n"
            '[{"topic": "bestehender-slug-oder-neuer-name", "title": "Titel", '
            '"summary": "Markdown-Zusammenfassung"}]\n\n'
            f"Gesprächsprotokoll:\n{text}"
        )

        content = await self._llm_call(system_prompt, user_prompt)

        try:
            topics = json.loads(content)
            if isinstance(topics, list):
                return topics
        except json.JSONDecodeError:
            pass
        m = re.search(r"\[.*\]", content, re.DOTALL)
        if m:
            try:
                topics = json.loads(m.group())
                if isinstance(topics, list):
                    return topics
            except json.JSONDecodeError:
                pass

        logger.warning("MarkdownTopicBackend: could not parse LLM JSON, using raw")
        return [{"topic": "misc", "title": "Verschiedenes", "summary": content}]

    async def _llm_match_existing(
        self, new_slug: str, new_title: str, existing_topics: dict[str, str]
    ) -> Optional[str]:
        """Ask LLM if *new_slug* semantically matches an existing topic.

        Returns the matched existing slug or None.
        """
        if not existing_topics:
            return None
        lines = "\n".join(
            f"- {slug}: {title}" for slug, title in sorted(existing_topics.items())
        )
        system_prompt = "Antworte NUR mit validem JSON."
        user_prompt = (
            f"Bestehende Themen:\n{lines}\n\n"
            f"Neues Thema: \"{new_slug}\" (Titel: \"{new_title}\")\n\n"
            "Gehört das neue Thema inhaltlich zu einem der bestehenden? "
            'Antworte mit JSON: {"match": "bestehender-slug"} oder {"match": null}'
        )
        content = await self._llm_call(system_prompt, user_prompt)
        for pattern in (r'\{[^}]*"match"[^}]*\}', r"\{.*\}"):
            m = re.search(pattern, content, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                    match = data.get("match")
                    if match and match in existing_topics:
                        return match
                    break
                except (json.JSONDecodeError, AttributeError):
                    pass
        return None

    async def _consolidate_topics(self) -> None:
        """Merge duplicate or strongly overlapping topic files.

        Asks the LLM which topics should be combined, then merges the content
        of the dissolved files into the keeper file and updates the tracker.
        """
        catalog = self._get_existing_topic_catalog()
        if len(catalog) < 2:
            return

        lines = "\n".join(
            f"- {slug}: {title}" for slug, title in sorted(catalog.items())
        )
        system_prompt = "Antworte NUR mit validem JSON."
        user_prompt = (
            f"Hier sind alle gespeicherten Themen (slug: titel):\n{lines}\n\n"
            "Welche Themen sind inhaltlich gleich oder stark überlappend und "
            "sollten zusammengeführt werden?\n"
            "Antworte mit einem JSON-Array von Merge-Operationen, oder [] wenn keine nötig:\n"
            '[{"keep": "slug-behalten", "merge": ["slug-aufloesen1", "slug-aufloesen2"]}]'
        )
        content = await self._llm_call(system_prompt, user_prompt)

        merges: list[dict] = []
        try:
            data = json.loads(content)
            if isinstance(data, list):
                merges = data
        except json.JSONDecodeError:
            m = re.search(r"\[.*\]", content, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                    if isinstance(data, list):
                        merges = data
                except json.JSONDecodeError:
                    pass

        if not merges:
            logger.debug("MarkdownTopicBackend: consolidation — no merges needed")
            return

        tracker = self._load_tracker()
        merged_count = 0

        for op in merges:
            keep_slug = op.get("keep", "")
            merge_slugs = [s for s in op.get("merge", []) if s != keep_slug]
            if not keep_slug or not merge_slugs:
                continue

            keep_path = os.path.join(self._topics_dir, f"{keep_slug}.md")
            if not os.path.exists(keep_path):
                logger.warning(
                    "MarkdownTopicBackend: consolidate keep target missing: %s", keep_slug
                )
                continue

            for m_slug in merge_slugs:
                merge_path = os.path.join(self._topics_dir, f"{m_slug}.md")
                if not os.path.exists(merge_path):
                    continue
                try:
                    with open(merge_path, encoding="utf-8") as f:
                        merge_content = f.read()
                    # Drop the H1 header of the dissolved file before appending
                    body_lines = merge_content.split("\n")
                    if body_lines and body_lines[0].startswith("# "):
                        merge_content = "\n".join(body_lines[1:]).lstrip("\n")

                    with open(keep_path, "a", encoding="utf-8") as f:
                        f.write(f"\n\n{merge_content}")

                    os.remove(merge_path)
                    merged_count += 1
                    logger.info(
                        "MarkdownTopicBackend: merged '%s' → '%s'", m_slug, keep_slug
                    )

                    # Update tracker: replace dissolved slug with keeper everywhere
                    for doc_id, slugs in tracker.items():
                        if m_slug in slugs:
                            tracker[doc_id] = [
                                keep_slug if s == m_slug else s for s in slugs
                            ]
                except Exception as exc:
                    logger.warning(
                        "MarkdownTopicBackend: consolidation error for %s: %s", m_slug, exc
                    )

        if merged_count:
            self._save_tracker()
            logger.info(
                "MarkdownTopicBackend: consolidated %d topic file(s)", merged_count
            )

    # ── RagBackend interface ────────────────────────────────────────────────

    async def insert(self, text: str, doc_id: str) -> None:
        if self._llm_busy and self._llm_busy():
            raise RuntimeError("LLM busy — deferring memory indexing")

        os.makedirs(self._topics_dir, exist_ok=True)
        tracker = self._load_tracker()
        existing_topics = self._get_existing_topic_catalog()

        # Pass 1: extract topics, guided by the existing catalog (Stufen 1+5)
        topics = await self._llm_extract_topics(text, existing_topics)

        slugs: list[str] = []
        new_slugs_created: list[str] = []
        for t in topics:
            topic_name = t.get("topic", "misc")
            title = t.get("title", topic_name)
            summary = t.get("summary", "")
            if not summary:
                continue

            slug = self._slugify(topic_name)

            # Idee 4: string-similarity pre-filter (no LLM, catches typos/variants)
            if slug not in existing_topics:
                similar = self._find_similar_slug(slug, list(existing_topics.keys()))
                if similar:
                    logger.debug(
                        "MarkdownTopicBackend: '%s' → '%s' via string similarity",
                        slug, similar,
                    )
                    slug = similar

            # Stufe 2: semantic LLM matching for still-unknown slugs
            if slug not in existing_topics:
                matched = await self._llm_match_existing(slug, title, existing_topics)
                if matched:
                    logger.debug(
                        "MarkdownTopicBackend: '%s' → '%s' via LLM match", slug, matched
                    )
                    slug = matched

            is_new = slug not in existing_topics
            slugs.append(slug)
            filepath = os.path.join(self._topics_dir, f"{slug}.md")

            # Extract date from doc_id (chat_<user>_YYYY-MM-DD)
            date_str = doc_id.rsplit("_", 1)[-1] if "_" in doc_id else doc_id
            section = f"\n\n### {title} ({date_str})\n\n{summary}\n"

            if os.path.exists(filepath):
                with open(filepath, "a", encoding="utf-8") as f:
                    f.write(section)
            else:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"# {title}\n{section}")
                if is_new:
                    new_slugs_created.append(slug)
                    existing_topics[slug] = title  # keep catalog current for later iterations

        tracker[doc_id] = slugs
        self._indexed.add(doc_id)
        self._save_tracker()
        logger.info(
            "MarkdownTopicBackend: indexed %d topics for %s: %s",
            len(slugs), doc_id, slugs,
        )

        # Stufe 3: consolidate after every run that produced new topic files
        if new_slugs_created:
            logger.info(
                "MarkdownTopicBackend: %d new topic(s), running consolidation: %s",
                len(new_slugs_created), new_slugs_created,
            )
            try:
                await self._consolidate_topics()
            except Exception as exc:
                logger.warning("MarkdownTopicBackend: consolidation failed: %s", exc)

    async def wait_for_indexed(
        self, doc_id: str, timeout: int = 120, poll_interval: float = 5.0,
    ) -> bool:
        # insert() is synchronous — once it returns the doc is indexed.
        return doc_id in self._indexed

    async def query(self, question: str) -> str:
        if not os.path.isdir(self._topics_dir):
            return "Noch keine Dokumente indiziert."

        topic_files = sorted(
            f for f in os.listdir(self._topics_dir) if f.endswith(".md")
        )
        if not topic_files:
            return "Noch keine Dokumente indiziert."

        # Tokenise query, remove stop words
        query_words = set(re.split(r"\W+", question.lower())) - self._STOP_WORDS - {""}

        scored: list[tuple[int, str, str]] = []
        for fname in topic_files:
            filepath = os.path.join(self._topics_dir, fname)
            try:
                with open(filepath, encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                continue

            name_part = fname[:-3].replace("-", " ")
            search_text = name_part + " " + content.lower()
            score = sum(1 for w in query_words if w in search_text)
            # Boost filename / topic-name matches
            score += sum(2 for w in query_words if w in name_part)
            scored.append((score, fname, content))

        scored.sort(key=lambda x: x[0], reverse=True)

        has_matches = any(s > 0 for s, _, _ in scored)
        results: list[str] = []
        total = 0
        for score, _fname, content in scored:
            if has_matches and score == 0:
                break
            if total + len(content) > self._MAX_RESULT_CHARS:
                break
            results.append(content)
            total += len(content)

        if not results:
            return "Keine relevanten Informationen gefunden."
        return "\n\n---\n\n".join(results)


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
      - ``"markdown"`` (default) — Dream Wiki: structured, interlinked wiki
      - ``"lightrag"``           — LightRAG knowledge-graph
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
    from pawlia.dream_wiki import DreamWikiBackend

    backend_name = cfg.get("rag_backend", "markdown")

    if backend_name == "markdown":
        logger.debug("Using Dream Wiki backend at %s", index_path)
        return DreamWikiBackend(index_path, cfg, llm_busy_check)

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


# Legacy alias — existing code that references MarkdownTopicBackend keeps working.
MarkdownTopicBackend = None  # replaced by DreamWikiBackend; import from dream_wiki if needed
