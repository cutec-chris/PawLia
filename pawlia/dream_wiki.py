"""Dream Wiki backend — Karpathy's LLM Wiki pattern for PawLia.

Instead of just indexing raw documents, the LLM incrementally builds
a persistent, structured wiki with cross-references, YAML frontmatter,
and ``[[wikilinks]]``.  The wiki is a compounding artifact: knowledge is
compiled once and kept current, not re-derived on every query.

Storage layout inside *index_path*:
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
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from typing import Callable, Optional

import yaml

logger = logging.getLogger("pawlia.dream_wiki")

# Common stop words (DE + EN) excluded from keyword matching.
_STOP_WORDS = frozenset(
    "der die das und oder ein eine ist war hat haben was wie wer wo wann "
    "warum ich du wir sie er es mit von zu für auf in an bei nach über "
    "unter vor hinter zwischen nicht auch noch schon nur aber denn wenn "
    "dass weil als ob the a an and or is was are were has have what how "
    "who where when why i you we they he she it with from to for on at "
    "by about do did not also".split()
)


def _slugify(name: str) -> str:
    """Convert a topic name to a filesystem-safe slug."""
    slug = name.lower().strip()
    for old, new in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        slug = slug.replace(old, new)
    slug = unicodedata.normalize("NFKD", slug)
    slug = slug.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    return slug[:80] or "misc"


def _find_similar_slug(
    new_slug: str, existing_slugs: list[str], threshold: float = 0.7
) -> Optional[str]:
    """Return the most similar existing slug if similarity >= threshold."""
    best_slug: Optional[str] = None
    best_ratio = 0.0
    for existing in existing_slugs:
        ratio = SequenceMatcher(None, new_slug, existing).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_slug = existing
    return best_slug if best_ratio >= threshold else None


async def _llm_call(cfg: dict, system_prompt: str, user_prompt: str) -> str:
    """Make a single LLM call and return the stripped content string."""
    import urllib.request

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

    return re.sub(r"<think.*?</think >", "", content, flags=re.DOTALL).strip()


def _parse_json_array(content: str) -> list:
    """Try to parse a JSON array from LLM output, tolerating wrapping."""
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*\]", content, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    return []


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
    ):
        self._index_path = index_path
        self._cfg = cfg
        self._llm_busy = llm_busy_check
        self._wiki_dir = os.path.join(index_path, "wiki")
        self._topics_dir = os.path.join(self._wiki_dir, "topics")
        self._tracker_path = os.path.join(self._wiki_dir, "dreamed_files.json")
        self._indexed: set[str] = set()
        self._tracker: Optional[dict] = None

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

    def _get_wiki_catalog(self) -> dict[str, str]:
        """Return {slug: title} for all existing wiki pages."""
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
                           tags: list[str] | None = None) -> str:
        fm = {
            "slug": slug,
            "title": title,
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
        lines = ["# Wiki Index\n"]
        for slug, title in sorted(catalog.items()):
            lines.append(f"- [[{slug}]] — {title}")
        lines.append(f"\n> {len(catalog)} Seiten")
        path = os.path.join(self._wiki_dir, "index.md")
        os.makedirs(self._wiki_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _append_log(self, entry_type: str, detail: str, slugs: list[str] | None = None) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        slug_list = f" | Seiten: {', '.join(slugs)}" if slugs else ""
        line = f"\n## [{timestamp}] {entry_type} | {detail}{slug_list}\n"
        path = os.path.join(self._wiki_dir, "log.md")
        os.makedirs(self._wiki_dir, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

    # ── LLM analysis ──────────────────────────────────────────────────────────

    async def _llm_analyze(self, text: str, wiki_index: str) -> list[dict]:
        """Analyze a chat log and return wiki actions (create/update pages)."""
        from pawlia.prompt_utils import load_system_prompt

        system_prompt = load_system_prompt("dream/analyze.md")
        user_prompt = (
            f"## Aktueller Wiki-Index\n{wiki_index}\n\n"
            f"## Gesprächsprotokoll\n{text}"
        )

        content = await _llm_call(self._cfg, system_prompt, user_prompt)
        actions = _parse_json_array(content)

        if not actions:
            logger.warning("DreamWikiBackend: could not parse LLM JSON, using raw")
            return [{"action": "create", "slug": "misc",
                      "title": "Verschiedenes", "content": content,
                      "tags": [], "links": []}]
        return actions

    # ── Page management ───────────────────────────────────────────────────────

    async def _update_page(self, slug: str, title: str, content: str,
                           date_str: str, action: str,
                           tags: list[str] | None = None,
                           links: list[str] | None = None) -> None:
        filepath = os.path.join(self._topics_dir, f"{slug}.md")
        os.makedirs(self._topics_dir, exist_ok=True)

        link_section = ""
        if links:
            link_lines = [f"- [[{l}]]" for l in links if l != slug]
            if link_lines:
                link_section = "\n\n## Verwandte Seiten\n" + "\n".join(link_lines)

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
                if "## Verwandte Seiten" in existing:
                    for l in links:
                        if l != slug and f"[[{l}]]" not in existing:
                            existing += f"\n- [[{l}]]"
                else:
                    existing += link_section
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(existing + section)
        else:
            frontmatter = self._build_frontmatter(slug, title, date_str, tags)
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
            filepath = os.path.join(self._topics_dir, f"{slug}.md")
            try:
                with open(filepath, encoding="utf-8") as f:
                    content = f.read()
                summary = content[:500]
                page_summaries.append(f"### {slug}: {title}\n{summary}\n")
            except Exception:
                continue

        system_prompt = load_system_prompt("dream/consolidate.md")
        user_prompt = (
            f"## Wiki-Seiten ({len(catalog)} insgesamt)\n\n"
            + "\n".join(page_summaries)
        )

        content = await _llm_call(self._cfg, system_prompt, user_prompt)

        # Try parsing as a single JSON object first
        data = {}
        arrays = _parse_json_array(content)
        if arrays:
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

            keep_path = os.path.join(self._topics_dir, f"{keep_slug}.md")
            if not os.path.exists(keep_path):
                logger.warning("DreamWikiBackend: consolidate keep target missing: %s", keep_slug)
                continue

            for m_slug in merge_slugs:
                merge_path = os.path.join(self._topics_dir, f"{m_slug}.md")
                if not os.path.exists(merge_path):
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

                    with open(keep_path, "a", encoding="utf-8") as f:
                        f.write(f"\n\n---\n*Merged from [[{m_slug}]]*\n\n{body}")

                    # Update wikilinks in all files
                    for fname in os.listdir(self._topics_dir):
                        if not fname.endswith(".md"):
                            continue
                        fpath = os.path.join(self._topics_dir, fname)
                        try:
                            with open(fpath, encoding="utf-8") as f:
                                fc = f.read()
                            updated = fc.replace(f"[[{m_slug}]]", f"[[{keep_slug}]]")
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
        for ml in data.get("missing_links", []):
            from_slug = ml.get("from", "")
            to_slug = ml.get("to", "")
            if not from_slug or not to_slug:
                continue
            fpath = os.path.join(self._topics_dir, f"{from_slug}.md")
            if os.path.exists(fpath) and to_slug in self._get_wiki_catalog():
                try:
                    with open(fpath, encoding="utf-8") as f:
                        fc = f.read()
                    if f"[[{to_slug}]]" not in fc:
                        if "## Verwandte Seiten" in fc:
                            fc += f"\n- [[{to_slug}]]"
                        else:
                            fc += f"\n\n## Verwandte Seiten\n- [[{to_slug}]]"
                        with open(fpath, "w", encoding="utf-8") as f:
                            f.write(fc)
                except Exception:
                    pass

        if merged_count:
            self._save_tracker()
            self._rebuild_index()
            self._append_log("lint", f"{merged_count} Seiten zusammengeführt")
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
                slug, title, content, date_str, action, tags=tags, links=links
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

        topic_files = sorted(
            f for f in os.listdir(self._topics_dir) if f.endswith(".md")
        )
        if not topic_files:
            return "Noch keine Dokumente indiziert."

        query_words = set(re.split(r"\W+", question.lower())) - _STOP_WORDS - {""}

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
            score += sum(2 for w in query_words if w in name_part)
            score += sum(3 for w in query_words if f"[[{w}" in content.lower())
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
