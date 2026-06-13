"""Dream Wiki backend — Karpathy's LLM Wiki pattern for PawLia.

Instead of just indexing raw documents, the LLM incrementally builds
a persistent, structured wiki with cross-references, YAML frontmatter,
and ``[[wikilinks]]``.  The wiki is a compounding artifact: knowledge is
compiled once and kept current, not re-derived on every query.

Storage layout inside the user's workspace (``workspace/wiki/``):
  wiki/
    index.md              — Catalog of all pages (slug + one-line summary)
    log.md                — Chronological audit log of ingest/lint operations
    dreamed_files.json    — Tracking: which source files have been processed
    topics/
      {slug}.md           — One wiki page per topic/entity

Implements the RagBackend ABC from pawlia.rag_backend.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from typing import Callable, Optional

import yaml

logger = logging.getLogger("pawlia.dream_wiki")

from pawlia.utils import _STOP_WORDS, find_similar_slug as _find_similar_slug, slugify as _slugify


# (model_key) → context_length in tokens
_context_length_cache: dict[str, int] = {}

_DEFAULT_CTX = 4096
_RESERVED_TOKENS = 2000   # system prompt + JSON output headroom
_CHARS_PER_TOKEN = 3      # conservative estimate for mixed content


async def _get_model_context_length(cfg: dict) -> int:
    """Query Ollama /api/show for the model's native context length.

    Falls back to cfg['rag_numctx'] or _DEFAULT_CTX on any error.
    Result is cached per (host, model) key.
    """
    import urllib.request
    provider = cfg.get("rag_provider", cfg.get("embedding_provider", "ollama"))
    model = cfg.get("rag_model", "qwen3.5:latest")
    host = cfg.get("embedding_host", "http://localhost:11434")
    configured_ctx = int(cfg.get("rag_numctx", 0))

    if provider != "ollama":
        return configured_ctx or _DEFAULT_CTX

    cache_key = f"{host}|{model}"
    if cache_key in _context_length_cache:
        native = _context_length_cache[cache_key]
    else:
        try:
            body = json.dumps({"model": model}).encode()
            from pawlia.utils import PAWLIA_USER_AGENT
            req = urllib.request.Request(
                f"{host.rstrip('/')}/api/show",
                data=body,
                headers={"Content-Type": "application/json",
                         "User-Agent": PAWLIA_USER_AGENT},
                method="POST",
            )
            from pawlia.utils import run_sync_in_thread
            def _do():
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return json.loads(resp.read().decode())
            data = await run_sync_in_thread(_do)
            info = data.get("model_info", {})
            arch = info.get("general.architecture", "")
            native = int(info.get(f"{arch}.context_length", 0)) or _DEFAULT_CTX
        except Exception:
            native = _DEFAULT_CTX
        _context_length_cache[cache_key] = native

    if configured_ctx:
        # Honour the explicit config, but don't exceed what the model supports.
        effective = min(native, configured_ctx)
    else:
        # No explicit config: Ollama defaults to 2048 regardless of model max.
        # Use _DEFAULT_CTX (4096) as a safe practical baseline.
        effective = _DEFAULT_CTX
    return effective


async def _max_input_chars(cfg: dict) -> int:
    """Compute max input chars for _llm_analyze based on model context window."""
    ctx = await _get_model_context_length(cfg)
    return max(1000, (ctx - _RESERVED_TOKENS) * _CHARS_PER_TOKEN)


from pawlia.utils import rag_llm_call as _llm_call


_SENTINEL = object()

# Phrase-prefixes the LLM often adds in front of its JSON. Trim them so the
# remaining content is more likely to be a bare JSON array.
_JSON_PREAMBLE_RE = re.compile(
    r"^(?:"
    r"Hier (?:ist|sind) (?:der|die|das) JSON[^:]*:"
    r"|Here(?:'s| is) (?:the )?JSON[^:]*:"
    r"|Sure[,!]?\s*(?:here(?:'s| is)[^:]*)?:"
    r"|(?:JSON|Answer|Response|Output|Antwort|Ergebnis)\s*:\s*"
    r")\s*",
    re.IGNORECASE,
)


def _parse_json_array(content: str):
    """Try to parse a JSON array from LLM output, tolerating wrapping.

    Returns the parsed list, or _SENTINEL if no JSON array could be found.
    An empty list [] is a valid (successful) parse result.
    """
    if not content or not content.strip():
        return _SENTINEL

    # Strip code fences: ```json ... ``` or ``` ... ```
    stripped = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1",
                      content, flags=re.DOTALL)
    # Drop common preambles the model tacks on ("Here is the JSON:", etc.)
    stripped = _JSON_PREAMBLE_RE.sub("", stripped).strip()

    for candidate in (stripped, content.strip()):
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        m = re.search(r"\[.*\]", candidate, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass
        # Last resort: the model returned a single object where we wanted
        # an array — wrap it so callers still get a list.
        m = re.search(r"\{.*\}", candidate, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
                if isinstance(data, dict):
                    return [data]
            except json.JSONDecodeError:
                pass
    return _SENTINEL


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class DreamWikiBackend:
    """Structured wiki backend — LLM builds and maintains an interlinked wiki.

    Implements Karpathy's LLM Wiki pattern.  The RagBackend ABC methods
    (insert, wait_for_indexed, query) are present but this class is also
    used directly by the dream scheduler for the full ingest/lint cycle.
    """

    _MAX_RESULT_CHARS = 32_000

    def __init__(
        self,
        index_path: str,
        cfg: dict,
        llm_busy_check: Optional[Callable[[], bool]] = None,
        wiki_dir: Optional[str] = None,
    ):
        self._index_path = index_path
        self._cfg = cfg
        self._llm_busy = llm_busy_check
        if wiki_dir:
            self._wiki_dir = wiki_dir
        else:
            # Default: place wiki in workspace/wiki/ (sibling of memory_index)
            user_session = os.path.dirname(index_path)
            self._wiki_dir = os.path.join(user_session, "workspace", "wiki")
        self._topics_dir = os.path.join(self._wiki_dir, "topics")
        # Keep tracker in index_path (not in wiki) to keep the vault clean
        self._tracker_path = os.path.join(index_path, "dreamed_files.json")
        self._indexed: set[str] = set()
        self._tracker: Optional[dict] = None
        self._catalog_cache: Optional[dict[str, str]] = None
        self._catalog_cache_mtime: float = 0.0

    # ── Tracking ──────────────────────────────────────────────────────────────

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
        os.makedirs(self._wiki_dir, exist_ok=True)
        with open(self._tracker_path, "w", encoding="utf-8") as f:
            json.dump(self._tracker or {}, f, ensure_ascii=False, indent=2)

    # ── Wiki catalog ──────────────────────────────────────────────────────────

    def _iter_topic_files(self) -> list[str]:
        if not os.path.isdir(self._topics_dir):
            return []
        files: list[str] = []
        for dirpath, dirnames, filenames in os.walk(self._topics_dir):
            dirnames.sort()
            for fname in sorted(filenames):
                if fname.endswith(".md"):
                    files.append(os.path.join(dirpath, fname))
        return files

    def _topic_path(self, slug: str, entity_type: str = "topic") -> str:
        return os.path.join(self._topics_dir, entity_type, f"{slug}.md")

    def _find_topic_path(self, slug: str, entity_type: Optional[str] = None) -> Optional[str]:
        if entity_type:
            typed = self._topic_path(slug, entity_type)
            if os.path.exists(typed):
                return typed
        flat = os.path.join(self._topics_dir, f"{slug}.md")
        if os.path.exists(flat):
            return flat
        for path in self._iter_topic_files():
            if os.path.basename(path) == f"{slug}.md":
                return path
        return None

    def _slug_target(self, slug: str, entity_type: Optional[str] = None) -> str:
        path = self._find_topic_path(slug, entity_type)
        if not path:
            return slug
        rel = os.path.relpath(path, self._topics_dir).replace(os.sep, "/")
        return rel[:-3] if rel.endswith(".md") else rel

    def _ensure_topic_layout(self) -> None:
        """Move flat topic pages into type-based subdirectories when possible."""
        if not os.path.isdir(self._topics_dir):
            return
        for fname in sorted(os.listdir(self._topics_dir)):
            src = os.path.join(self._topics_dir, fname)
            if not os.path.isfile(src) or not fname.endswith(".md"):
                continue
            try:
                with open(src, encoding="utf-8") as f:
                    content = f.read()
                fm = self._parse_frontmatter(content) or {}
            except Exception:
                continue
            entity_type = fm.get("type", "topic")
            dst = self._topic_path(fname[:-3], entity_type)
            if src == dst or os.path.exists(dst):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.replace(src, dst)

    def _get_wiki_catalog(self) -> dict[str, str]:
        """Return {slug: title} for all existing wiki pages.

        Cached by topics-directory mtime: changes to page contents that update
        the title still invalidate via os.utime on rewrite (rewriting a file
        also bumps the parent dir mtime on POSIX when entries are
        added/removed). For in-place title edits, callers can drop the cache
        via _invalidate_catalog_cache().
        """
        self._ensure_topic_layout()
        if not os.path.isdir(self._topics_dir):
            return {}
        catalog: dict[str, str] = {}
        for filepath in self._iter_topic_files():
            slug = os.path.splitext(os.path.basename(filepath))[0]
            try:
                with open(filepath, encoding="utf-8") as f:
                    content = f.read()
                title = slug
                fm = self._parse_frontmatter(content)
                if fm and fm.get("title"):
                    title = fm["title"]
                else:
                    for line in content.split("\n"):
                        if line.startswith("# "):
                            title = line.lstrip("#").strip()
                            break
            except Exception:
                title = slug
            catalog[slug] = title
        return catalog

    def _invalidate_catalog_cache(self) -> None:
        self._catalog_cache = None
        self._catalog_cache_mtime = 0.0

    @staticmethod
    def _parse_frontmatter(text: str) -> Optional[dict]:
        stripped = text.lstrip()
        if not stripped.startswith("---"):
            return None
        parts = stripped.split("---", 2)
        if len(parts) < 3:
            return None
        try:
            return yaml.safe_load(parts[1]) or {}
        except Exception:
            return None

    def _build_frontmatter(self, slug: str, title: str, date_str: str,
                           tags: list[str] | None = None,
                           entity_type: str = "topic") -> str:
        fm = {
            "slug": slug,
            "title": title,
            "type": entity_type,
            "created": date_str,
            "updated": date_str,
            "tags": tags or [],
        }
        return f"---\n{yaml.dump(fm, allow_unicode=True, default_flow_style=False).strip()}\n---"

    # ── index.md and log.md ───────────────────────────────────────────────────

    def _read_index(self) -> str:
        path = os.path.join(self._wiki_dir, "index.md")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return f.read()
        return ""

    def _rebuild_index(self) -> None:
        """Rebuild index.md from the current wiki catalog."""
        catalog = self._get_wiki_catalog()
        # Group pages by type for a structured index
        typed: dict[str, list[tuple[str, str]]] = {}
        for slug, title in sorted(catalog.items()):
            filepath = self._find_topic_path(slug)
            entity_type = "topic"
            try:
                if filepath:
                    with open(filepath, encoding="utf-8") as f:
                        fm = self._parse_frontmatter(f.read())
                    if fm and fm.get("type"):
                        entity_type = fm["type"]
            except Exception:
                pass
            typed.setdefault(entity_type, []).append((slug, title))

        _TYPE_ORDER = ("person", "place", "object", "project", "topic")
        lines = ["# Wiki Index\n"]
        for etype in _TYPE_ORDER:
            entries = typed.pop(etype, [])
            if not entries:
                continue
            lines.append(f"\n## {etype.title()}\n")
            for slug, title in entries:
                lines.append(f"- {self._index_link(slug, title)}")
        # Any unknown types
        for etype, entries in typed.items():
            lines.append(f"\n## {etype.title()}\n")
            for slug, title in entries:
                lines.append(f"- {self._index_link(slug, title)}")
        lines.append(f"\n> {len(catalog)} pages")
        path = os.path.join(self._wiki_dir, "index.md")
        os.makedirs(self._wiki_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _append_log(self, entry_type: str, detail: str, slugs: list[str] | None = None) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        slug_list = f" | pages: {', '.join(slugs)}" if slugs else ""
        line = f"\n## [{timestamp}] {entry_type} | {detail}{slug_list}\n"
        path = os.path.join(self._wiki_dir, "log.md")
        os.makedirs(self._wiki_dir, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

    # ── LLM analysis ──────────────────────────────────────────────────────────

    async def _llm_analyze(self, text: str, wiki_index: str) -> list[dict]:
        """Analyze a chat log and return wiki actions (create/update pages)."""
        from pawlia.prompt_utils import load_system_prompt

        max_chars = await _max_input_chars(self._cfg)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[... truncated ...]"
            logger.debug("DreamWikiBackend: input truncated to %d chars (ctx=%d tokens)",
                         max_chars, max_chars // _CHARS_PER_TOKEN + _RESERVED_TOKENS)

        system_prompt = self._adapt_prompt_for_link_format(
            load_system_prompt("dream/analyze.md")
        )
        user_prompt = (
            f"## Aktueller Wiki-Index\n{wiki_index}\n\n"
            f"## Gesprächsprotokoll\n{text}"
        )

        content = await _llm_call(self._cfg, system_prompt, user_prompt,
                                  json_mode="array")
        actions = _parse_json_array(content)

        if actions is _SENTINEL:
            logger.warning("DreamWikiBackend: could not parse LLM JSON, skipping. Response: %s",
                           content[:300])
            return []
        return actions

    # ── Link format helpers ───────────────────────────────────────────────────

    @property
    def _link_format(self) -> str:
        """Return configured link format: 'wikilink' (default) or 'markdown'."""
        return self._cfg.get("wiki_link_format", "wikilink")

    def _md_link(self, slug: str, catalog: dict[str, str] | None = None) -> str:
        """Build a link for a slug using the configured format."""
        title = slug
        if catalog and slug in catalog:
            title = catalog[slug]
        else:
            filepath = self._find_topic_path(slug)
            if filepath and os.path.exists(filepath):
                try:
                    with open(filepath, encoding="utf-8") as f:
                        fm = self._parse_frontmatter(f.read())
                    if fm and fm.get("title"):
                        title = fm["title"]
                except Exception:
                    pass
        target = self._slug_target(slug)
        if self._link_format == "wikilink":
            return f"[[{target}|{title}]]" if title != slug else f"[[{target}]]"
        return f"[{title}]({target}.md)"

    def _index_link(self, slug: str, title: str) -> str:
        """Build a link for use in index.md (one level above topics/)."""
        target = self._slug_target(slug)
        if self._link_format == "wikilink":
            return f"[[{target}|{title}]]" if title != slug else f"[[{target}]]"
        return f"[{title}](topics/{target}.md)"

    def _has_link(self, text: str, slug: str) -> bool:
        """Check whether text already contains a link to slug in any format."""
        return bool(re.search(
            rf"\]\((?:[^)]*/)?{re.escape(slug)}\.md\)"
            rf"|\[\[(?:[^\]|#]*/)?{re.escape(slug)}(?:[#|\]])",
            text,
        ))

    def _adapt_prompt_for_link_format(self, prompt: str) -> str:
        """Replace markdown-link instructions with wikilink instructions when configured."""
        if self._link_format != "wikilink":
            return prompt
        replacements = [
            (
                "using **standard Markdown links**: `[Page Title](slug.md)`",
                "using **Obsidian wikilinks**: `[[slug]]` or `[[slug|Display Text]]`",
            ),
            (
                "Use `[Display Text](slug.md)` links (NOT `[[wikilinks]]`) to connect related pages",
                "Use `[[slug]]` or `[[slug|Display Text]]` wikilinks to connect related pages",
            ),
            (
                "Links use standard Markdown format: `[Title](slug.md)`",
                "Links use Obsidian wikilink format: `[[slug]]` or `[[slug|Display Text]]`",
            ),
        ]
        for old, new in replacements:
            prompt = prompt.replace(old, new)
        return prompt

    # ── Page management ───────────────────────────────────────────────────────

    async def _update_page(self, slug: str, title: str, content: str,
                           date_str: str, action: str,
                           tags: list[str] | None = None,
                           links: list[str] | None = None,
                           entity_type: str = "topic") -> None:
        filepath = self._find_topic_path(slug, entity_type)
        if not filepath:
            filepath = self._topic_path(slug, entity_type)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        catalog = self._get_wiki_catalog()
        link_section = ""
        if links:
            link_lines = [f"- {self._md_link(l, catalog)}" for l in links if l != slug]
            if link_lines:
                link_section = "\n\n## Related\n" + "\n".join(link_lines)

        if os.path.exists(filepath) and action == "update":
            with open(filepath, encoding="utf-8") as f:
                existing = f.read()

            fm = self._parse_frontmatter(existing)
            if fm:
                fm["updated"] = date_str
                if tags:
                    existing_tags = fm.get("tags", [])
                    fm["tags"] = list(set(existing_tags + tags))
                new_fm = f"---\n{yaml.dump(fm, allow_unicode=True, default_flow_style=False).strip()}\n---"
                body = existing.split("---", 2)
                if len(body) >= 3:
                    existing = new_fm + body[2]

            section = f"\n\n### Update ({date_str})\n\n{content}"
            if link_section:
                if "## Related" in existing:
                    for l in links:
                        if l != slug and not self._has_link(existing, l):
                            existing += f"\n- {self._md_link(l, catalog)}"
                else:
                    existing += link_section
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(existing + section)
        else:
            frontmatter = self._build_frontmatter(slug, title, date_str, tags,
                                                   entity_type=entity_type)
            page_content = f"{frontmatter}\n\n# {title}\n\n{content}"
            if link_section:
                page_content += link_section
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(page_content)

    # ── Consolidation / Lint ──────────────────────────────────────────────────

    async def consolidate(self) -> None:
        """Lint the wiki: merge overlapping pages, detect orphans, fix links."""
        catalog = self._get_wiki_catalog()
        if len(catalog) < 2:
            return

        from pawlia.prompt_utils import load_system_prompt

        page_summaries = []
        for slug, title in sorted(catalog.items()):
            filepath = self._find_topic_path(slug)
            try:
                if not filepath:
                    continue
                with open(filepath, encoding="utf-8") as f:
                    content = f.read()
                summary = content[:500]
                page_summaries.append(f"### {slug}: {title}\n{summary}\n")
            except Exception:
                continue

        system_prompt = self._adapt_prompt_for_link_format(
            load_system_prompt("dream/consolidate.md")
        )
        user_prompt = (
            f"## Wiki pages ({len(catalog)} total)\n\n"
            + "\n".join(page_summaries)
        )

        content = await _llm_call(self._cfg, system_prompt, user_prompt,
                                  json_mode="object")

        # Try parsing as a single JSON object first
        data = {}
        arrays = _parse_json_array(content)
        if arrays is not _SENTINEL and arrays:
            data = arrays[0] if isinstance(arrays[0], dict) else {}
        if not data:
            try:
                m = re.search(r"\{.*\}", content, re.DOTALL)
                if m:
                    data = json.loads(m.group())
            except (json.JSONDecodeError, AttributeError):
                pass

        if not data:
            logger.debug("DreamWikiBackend: consolidation — no action needed")
            return

        tracker = self._load_tracker()
        merged_count = 0

        # Process merges
        for op in data.get("merges", []):
            keep_slug = op.get("keep", "")
            merge_slugs = [s for s in op.get("merge", []) if s != keep_slug]
            if not keep_slug or not merge_slugs:
                continue

            keep_path = self._find_topic_path(keep_slug)
            if not keep_path or not os.path.exists(keep_path):
                logger.warning("DreamWikiBackend: consolidate keep target missing: %s", keep_slug)
                continue

            for m_slug in merge_slugs:
                merge_path = self._find_topic_path(m_slug)
                if not merge_path or not os.path.exists(merge_path):
                    continue
                try:
                    with open(merge_path, encoding="utf-8") as f:
                        merge_content = f.read()
                    body = merge_content
                    fm = self._parse_frontmatter(body)
                    if fm:
                        body = body.split("---", 2)[2]
                    body_lines = body.lstrip().split("\n")
                    if body_lines and body_lines[0].startswith("# "):
                        body = "\n".join(body_lines[1:]).lstrip("\n")

                    keep_title = catalog.get(keep_slug, keep_slug)
                    with open(keep_path, "a", encoding="utf-8") as f:
                        f.write(f"\n\n---\n*Merged from {self._md_link(m_slug, catalog)}*\n\n{body}")

                    # Update links in all files (both wikilink and markdown formats)
                    m_slug_link_re = re.compile(
                        rf"\[\[[^\]]*{re.escape(m_slug)}[^\]]*\]\]"
                        rf"|\[[^\]]*\]\({re.escape(m_slug)}\.md\)"
                    )
                    replacement = self._md_link(keep_slug, {keep_slug: keep_title})
                    for fpath in self._iter_topic_files():
                        try:
                            with open(fpath, encoding="utf-8") as f:
                                fc = f.read()
                            updated = m_slug_link_re.sub(replacement, fc)
                            if updated != fc:
                                with open(fpath, "w", encoding="utf-8") as f:
                                    f.write(updated)
                        except Exception:
                            pass

                    os.remove(merge_path)
                    merged_count += 1
                    logger.info("DreamWikiBackend: merged '%s' → '%s'", m_slug, keep_slug)

                    for doc_id, info in tracker.items():
                        topics = info.get("topics", []) if isinstance(info, dict) else info
                        if m_slug in topics:
                            if isinstance(info, dict):
                                info["topics"] = [keep_slug if s == m_slug else s for s in topics]
                            else:
                                tracker[doc_id] = [keep_slug if s == m_slug else s for s in topics]
                except Exception as exc:
                    logger.warning("DreamWikiBackend: consolidation error for %s: %s", m_slug, exc)

        # Add missing links
        catalog = self._get_wiki_catalog()
        for ml in data.get("missing_links", []):
            from_slug = ml.get("from", "")
            to_slug = ml.get("to", "")
            if not from_slug or not to_slug:
                continue
            fpath = self._find_topic_path(from_slug)
            if fpath and os.path.exists(fpath) and to_slug in catalog:
                try:
                    with open(fpath, encoding="utf-8") as f:
                        fc = f.read()
                    if not self._has_link(fc, to_slug):
                        link = self._md_link(to_slug, catalog)
                        if "## Related" in fc:
                            fc += f"\n- {link}"
                        else:
                            fc += f"\n\n## Related\n- {link}"
                        with open(fpath, "w", encoding="utf-8") as f:
                            f.write(fc)
                except Exception:
                    pass

        if merged_count:
            self._save_tracker()
            self._rebuild_index()
            self._append_log("lint", f"{merged_count} pages merged")
            logger.info("DreamWikiBackend: consolidated %d page(s)", merged_count)

    # ── RagBackend interface ──────────────────────────────────────────────────

    async def insert(self, text: str, doc_id: str) -> None:
        if self._llm_busy and self._llm_busy():
            raise RuntimeError("LLM busy — deferring dream wiki indexing")

        os.makedirs(self._topics_dir, exist_ok=True)
        tracker = self._load_tracker()

        wiki_index = self._read_index()
        catalog = self._get_wiki_catalog()

        date_str = doc_id.rsplit("_", 1)[-1] if "_" in doc_id else doc_id

        actions = await self._llm_analyze(text, wiki_index)

        slugs: list[str] = []
        new_slugs: list[str] = []

        for act in actions:
            action = act.get("action", "create")
            entity_type = act.get("type", "topic")
            raw_slug = act.get("slug", "misc")
            title = act.get("title", raw_slug)
            content = act.get("content", "")
            tags = act.get("tags", [])
            links = act.get("links", [])

            if not content:
                continue

            slug = _slugify(raw_slug)

            if slug not in catalog:
                similar = _find_similar_slug(slug, list(catalog.keys()))
                if similar:
                    slug = similar

            is_new = slug not in catalog
            slugs.append(slug)

            await self._update_page(
                slug, title, content, date_str, action, tags=tags, links=links,
                entity_type=entity_type,
            )

            if is_new:
                new_slugs.append(slug)
                catalog[slug] = title

        tracker[doc_id] = {"dreamed_at": _now_iso(), "topics": slugs}
        self._indexed.add(doc_id)
        self._save_tracker()

        self._rebuild_index()
        self._append_log("ingest", f"{doc_id} → {len(slugs)} Seiten", slugs)

        logger.info(
            "DreamWikiBackend: processed %s → %d pages (%d new): %s",
            doc_id, len(slugs), len(new_slugs), slugs,
        )

        if new_slugs:
            try:
                await self.consolidate()
            except Exception as exc:
                logger.warning("DreamWikiBackend: consolidation failed: %s", exc)

    async def wait_for_indexed(
        self, doc_id: str, timeout: int = 120, poll_interval: float = 5.0,
    ) -> bool:
        return doc_id in self._indexed

    async def query(self, question: str) -> str:
        if not os.path.isdir(self._topics_dir):
            return "Noch keine Dokumente indiziert."

        from pawlia.workspace_search import WorkspaceSearch

        workspace_root = os.path.dirname(self._wiki_dir)
        hits = WorkspaceSearch(
            workspace_root,
            config={
                "top_k": 6,
                "min_score": 0.35,
                "include_root_files": False,
                "exclude_dirs": ["memory", "skills"],
            },
        ).search(question)

        if not hits:
            return "Keine relevanten Informationen gefunden."

        results: list[str] = []
        total = 0
        for hit in hits:
            part = f"{hit.section_ref} ({hit.path})\n{hit.snippet}"
            if total + len(part) > self._MAX_RESULT_CHARS:
                break
            results.append(part)
            total += len(part)

        if not results:
            return "Keine relevanten Informationen gefunden."
        return "\n\n---\n\n".join(results)
