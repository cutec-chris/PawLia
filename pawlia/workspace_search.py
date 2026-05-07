"""On-the-fly BM25 search over workspace markdown files.

Scans workspace/*.md files (excluding raw chat logs) on every call —
no persistent index, no embeddings. Results are cached on the Session
after the first search so subsequent turns reuse them without re-scanning.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)$", re.MULTILINE)
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)

_DEFAULT_TOP_K = 5
_DEFAULT_MIN_SCORE = 0.5  # fraction of best hit (0–1 normalized)
_DEFAULT_SNIPPET_CHARS = 150
_DEFAULT_EXCLUDE_DIRS = {"memory"}
_IDENTITY_FILES = frozenset(
    {"bootstrap.md", "identity.md", "user.md", "soul.md", "memory.md"}
)


@dataclass
class SearchHit:
    path: str        # relative workspace path, e.g. "wiki/topics/foo.md"
    heading: str     # section heading ("" for file preamble)
    snippet: str     # first ~150 chars of section body
    score: float     # normalised 0–1 (1.0 = best hit in this query)
    wikilinks: List[str] = field(default_factory=list)

    @property
    def wikilink_ref(self) -> str:
        """Return wikilink form for this file, e.g. [[foo]] or [[research/proj/hash]]."""
        path_no_ext = self.path[: -len(".md")] if self.path.endswith(".md") else self.path
        if path_no_ext.startswith("wiki/topics/"):
            return f"[[{path_no_ext[len('wiki/topics/'):]}]]"
        return f"[[{path_no_ext}]]"


@dataclass
class _Section:
    path: str
    heading: str
    body: str


class WorkspaceSearch:
    def __init__(self, workspace_root: str, config: Optional[dict] = None):
        self.workspace_root = workspace_root
        cfg = config or {}
        self.enabled: bool = bool(cfg.get("enabled", True))
        self.top_k: int = int(cfg.get("top_k", _DEFAULT_TOP_K))
        self.min_score: float = float(cfg.get("min_score", _DEFAULT_MIN_SCORE))
        self.snippet_chars: int = int(cfg.get("snippet_chars", _DEFAULT_SNIPPET_CHARS))
        self.exclude_dirs: frozenset = frozenset(
            cfg.get("exclude_dirs", _DEFAULT_EXCLUDE_DIRS)
        )
        self.include_root_files: bool = bool(cfg.get("include_root_files", True))

    def search(self, query: str) -> List[SearchHit]:
        if not self.enabled or not query.strip():
            return []
        sections = self._collect_sections()
        if not sections:
            return []
        return self._bm25_search(query, sections)

    # ------------------------------------------------------------------
    # File collection
    # ------------------------------------------------------------------

    def _iter_md_files(self):
        """Yield workspace-relative paths for all .md files in scope."""
        root = self.workspace_root
        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = os.path.relpath(dirpath, root)
            is_root = rel_dir == "."

            dirnames[:] = sorted(
                d for d in dirnames
                if not d.startswith(".") and d not in self.exclude_dirs
            )

            for fname in sorted(filenames):
                if not fname.endswith(".md"):
                    continue
                if is_root:
                    if not self.include_root_files:
                        continue
                    if fname in _IDENTITY_FILES:
                        continue
                rel = fname if is_root else os.path.join(rel_dir, fname)
                yield rel.replace(os.sep, "/")

    def _collect_sections(self) -> List[_Section]:
        sections: List[_Section] = []
        for rel_path in self._iter_md_files():
            abs_path = os.path.join(self.workspace_root, rel_path)
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                    raw = fh.read()
            except OSError:
                continue
            content = _FRONTMATTER_RE.sub("", raw).strip()
            sections.extend(self._split_sections(content, rel_path))
        return sections

    @staticmethod
    def _split_sections(content: str, path: str) -> List["_Section"]:
        """Split markdown content into heading-delimited sections."""
        lines = content.splitlines()
        sections: List[_Section] = []
        current_heading = ""
        current_lines: List[str] = []

        def flush() -> None:
            body = "\n".join(current_lines).strip()
            if body or current_heading:
                sections.append(_Section(path=path, heading=current_heading, body=body))

        for line in lines:
            m = re.match(r"^#{1,6}\s+(.+?)$", line)
            if m:
                flush()
                current_heading = m.group(1).strip()
                current_lines = []
            else:
                current_lines.append(line)
        flush()
        return sections

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _bm25_search(self, query: str, sections: List[_Section]) -> List[SearchHit]:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return self._fallback_tf_search(query, sections)

        corpus = [
            self._tokenize(f"{s.heading} {s.body}") for s in sections
        ]
        bm25 = BM25Okapi(corpus)
        raw_scores = bm25.get_scores(self._tokenize(query))
        max_raw = max(raw_scores) if len(raw_scores) else 0.0

        if max_raw <= 0:
            return []

        hits: List[SearchHit] = []
        for i, raw in enumerate(raw_scores):
            norm = raw / max_raw
            if norm < self.min_score:
                continue
            s = sections[i]
            snippet = self._make_snippet(s.body)
            hits.append(SearchHit(
                path=s.path,
                heading=s.heading,
                snippet=snippet,
                score=norm,
                wikilinks=_WIKILINK_RE.findall(snippet),
            ))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[: self.top_k]

    def _fallback_tf_search(self, query: str, sections: List[_Section]) -> List[SearchHit]:
        """Simple term-frequency fallback when rank_bm25 is unavailable."""
        query_tokens = set(self._tokenize(query))
        scored: List[tuple] = []
        for s in sections:
            tokens = self._tokenize(f"{s.heading} {s.body}")
            if not tokens:
                continue
            score = sum(1 for t in tokens if t in query_tokens) / len(tokens)
            if score > 0:
                scored.append((score, s))

        if not scored:
            return []

        max_score = max(s for s, _ in scored)
        hits: List[SearchHit] = []
        for raw, s in scored:
            norm = raw / max_score
            if norm < self.min_score:
                continue
            snippet = self._make_snippet(s.body)
            hits.append(SearchHit(
                path=s.path,
                heading=s.heading,
                snippet=snippet,
                score=norm,
                wikilinks=_WIKILINK_RE.findall(snippet),
            ))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[: self.top_k]

    def _make_snippet(self, body: str) -> str:
        text = body.strip()
        if len(text) <= self.snippet_chars:
            return text
        return text[: self.snippet_chars].rstrip() + "…"
