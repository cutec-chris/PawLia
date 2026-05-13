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
_HEADING_LINE_RE = re.compile(r"^#{1,6}\s+(.+?)$")
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_PARAGRAPH_SPLIT_RE = re.compile(r"\n{2,}")
_WORD_RE = re.compile(r"\w+")

_SUBSTANTIVE_MIN_WORDS = 4
_PARAGRAPH_SPLIT_THRESHOLD = 300  # split headingless sections longer than this into paragraphs
_QUESTION_STARTERS = frozenset({
    # German
    "was", "wie", "warum", "weshalb", "wann", "wo", "wer", "welche", "welcher",
    "welches", "erkläre", "erklaere", "beschreibe", "erkläre", "erkläre",
    "erzähl", "zeig", "hilf", "kannst", "könntest", "könntest", "magst",
    "schreib", "erstelle", "mach", "analysiere", "vergleiche", "erkläre",
    # English
    "what", "how", "why", "when", "where", "who", "which", "explain",
    "describe", "tell", "show", "help", "can", "could", "would", "write",
    "create", "make", "analyze", "compare",
})
_TOPIC_SHIFT_THRESHOLD = 0.15  # overlap fraction below which we treat it as a new topic

_DEFAULT_TOP_K = 5
_DEFAULT_MIN_SCORE = 0.5  # fraction of best hit (0–1 normalized)
_DEFAULT_SNIPPET_CHARS = 150
_DEFAULT_EXCLUDE_DIRS = {"memory", "skills"}
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
        """Split markdown content into heading-delimited sections.

        Headingless sections longer than _PARAGRAPH_SPLIT_THRESHOLD are further
        split into paragraphs so individual facts rank well under BM25.
        """
        lines = content.splitlines()
        sections: List[_Section] = []
        current_heading = ""
        current_lines: List[str] = []

        def flush() -> None:
            body = "\n".join(current_lines).strip()
            if not body and not current_heading:
                return
            if not current_heading and len(body) > _PARAGRAPH_SPLIT_THRESHOLD:
                # Flat file: split into paragraph chunks so BM25 scores individual facts
                for para in _PARAGRAPH_SPLIT_RE.split(body):
                    para = para.strip()
                    if para:
                        sections.append(_Section(path=path, heading="", body=para))
            else:
                sections.append(_Section(path=path, heading=current_heading, body=body))

        for line in lines:
            m = _HEADING_LINE_RE.match(line)
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
        return _WORD_RE.findall(text.lower())

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

        query_tokens = set(self._tokenize(query))
        hits: List[SearchHit] = []
        for i, raw in enumerate(raw_scores):
            norm = raw / max_raw
            if norm < self.min_score:
                continue
            s = sections[i]
            snippet = self._make_snippet(s.body, query_tokens)
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

        query_tokens = set(self._tokenize(query))
        max_score = max(s for s, _ in scored)
        hits: List[SearchHit] = []
        for raw, s in scored:
            norm = raw / max_score
            if norm < self.min_score:
                continue
            snippet = self._make_snippet(s.body, query_tokens)
            hits.append(SearchHit(
                path=s.path,
                heading=s.heading,
                snippet=snippet,
                score=norm,
                wikilinks=_WIKILINK_RE.findall(snippet),
            ))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[: self.top_k]

    # ------------------------------------------------------------------
    # Public helpers (used by ChatAgent for trigger decisions)
    # ------------------------------------------------------------------

    @staticmethod
    def is_substantive(text: str) -> bool:
        """Return True if *text* looks like a real question/topic rather than small talk.

        Triggers workspace search and topic-shift detection.
        """
        words = text.split()
        if len(words) >= _SUBSTANTIVE_MIN_WORDS:
            return True
        # Short but starts with a question/action word → still substantive
        if words and words[0].lower().rstrip("?!.,") in _QUESTION_STARTERS:
            return True
        return False

    @staticmethod
    def is_topic_shift(new_text: str, recent_exchanges: list) -> bool:
        """Return True if *new_text* introduces a significantly different topic.

        Compares content-bearing tokens (length >= 4) in *new_text* against the
        last few exchanges. Low overlap → topic shift.
        """
        if not recent_exchanges:
            return True

        def _content_tokens(text: str) -> set:
            return {t for t in re.findall(r"\w+", text.lower()) if len(t) >= 4}

        recent_tokens: set = set()
        for exc in recent_exchanges[-3:]:
            user_text = exc[0] if isinstance(exc, (list, tuple)) else ""
            bot_text = exc[1] if isinstance(exc, (list, tuple)) and len(exc) > 1 else ""
            recent_tokens |= _content_tokens(user_text + " " + bot_text)

        new_tokens = _content_tokens(new_text)
        if not new_tokens:
            return False
        overlap = len(new_tokens & recent_tokens) / len(new_tokens)
        return overlap < _TOPIC_SHIFT_THRESHOLD

    @staticmethod
    def make_topic_heading(text: str, max_chars: int = 70) -> str:
        """Generate a short heading from the first sentence of *text*."""
        # Take first sentence or line
        for sep in (".", "!", "?", "\n"):
            idx = text.find(sep)
            if 10 < idx < max_chars:
                return text[: idx + 1].strip()
        first = text.strip()
        if len(first) > max_chars:
            cut = first[:max_chars].rsplit(" ", 1)[0]
            return cut + "…"
        return first

    def _make_snippet(self, body: str, query_tokens: Optional[set] = None) -> str:
        """Return the most query-relevant part of *body* up to snippet_chars.

        If query_tokens are given, scores each line by token overlap and returns
        the best-matching line (or falls back to the start of the body).
        """
        text = body.strip()
        if len(text) <= self.snippet_chars:
            return text
        if query_tokens:
            scored_lines = [
                (l.strip(), set(_WORD_RE.findall(l.lower())))
                for l in text.splitlines() if l.strip()
            ]
            if scored_lines:
                best, best_tokens = max(
                    scored_lines, key=lambda lt: len(query_tokens & lt[1])
                )
                if query_tokens & best_tokens:
                    if len(best) <= self.snippet_chars:
                        return best
                    return best[: self.snippet_chars].rstrip() + "…"
        return text[: self.snippet_chars].rstrip() + "…"
