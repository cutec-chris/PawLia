"""On-the-fly BM25 search over workspace markdown files.

Scans workspace/*.md files (excluding raw chat logs) on every substantive
message — no persistent index, no embeddings.
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

_DEFAULT_TOP_K = 3
_DEFAULT_MIN_SCORE = 0.7  # fraction of best hit (0–1 normalized)
_DEFAULT_MIN_RAW_SCORE = 0.5  # adjusted score below which even the best hit isn't a real match
_DEFAULT_SNIPPET_CHARS = 100
_DEFAULT_EXCLUDE_DIRS = {"memory", "skills"}

# Stopwords stripped from BM25 queries. Without this, German filler words like
# "warum", "vielen", "ja", "hast", "du", "die" can BM25-match unrelated wiki
# entries that happen to contain those tokens, producing context-relevant hits
# for what is actually small talk or follow-up speech. Stripped from queries
# only — the corpus is left intact so explicit topic searches still work.
_QUERY_STOPWORDS = frozenset({
    # German function words / fillers
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem",
    "einer", "eines", "und", "oder", "aber", "doch", "nicht", "nichts", "kein",
    "keine", "keinen", "keinem", "keiner", "ist", "sind", "war", "waren",
    "sein", "seine", "seinen", "habe", "hat", "hast", "haben", "hatte",
    "hatten", "wird", "werden", "wurde", "wurden", "ich", "du", "er", "sie",
    "es", "wir", "ihr", "mich", "mir", "dich", "dir", "uns", "euch", "ihn",
    "ihm", "ihnen", "mein", "meine", "dein", "deine", "ihre", "ihrer", "ihres",
    "auf", "in", "im", "an", "am", "zu", "zum", "zur", "von", "vom", "mit",
    "bei", "nach", "vor", "über", "ueber", "unter", "durch", "für", "fuer",
    "ohne", "gegen", "aus", "bis", "ab", "als", "wie", "wenn", "weil",
    "dass", "ob", "so", "noch", "auch", "nur", "schon", "mal", "ja", "nein",
    "doch", "wirklich", "echt", "vielen", "dank", "danke", "bitte", "okay",
    "ok", "hä", "hae", "hmm", "naja", "halt", "eben", "sehr", "ganz", "viel",
    "viele", "mehr", "weniger", "etwas", "irgendwas", "irgendwie", "irgendwo",
    "warum", "weshalb", "wieso", "wer", "was", "wo", "wann", "welcher",
    "welche", "welches", "wem", "wen",
    # English function words
    "the", "a", "an", "and", "or", "but", "not", "no", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "of", "to", "in",
    "on", "at", "by", "with", "from", "as", "if", "then", "so", "than",
    "that", "this", "these", "those", "for", "about", "into", "through",
    "during", "before", "after", "above", "below", "between", "yes", "thanks",
    "thank", "please", "okay", "what", "where", "when", "why", "how", "who",
    "which", "whom", "really", "very", "much", "many", "some", "any",
})
_MIN_QUERY_CONTENT_TOKENS = 1  # need ≥1 non-stopword token after filtering
_IDENTITY_FILES = frozenset(
    {"bootstrap.md", "identity.md", "user.md", "soul.md", "memory.md"}
)


@dataclass
class SearchHit:
    path: str        # relative workspace path, e.g. "wiki/topics/foo.md"
    heading: str     # section heading ("" for file preamble)
    section_ref: str # Obsidian-compatible section ref, e.g. [[foo#Bar]]
    page_ref: str    # Obsidian-compatible page ref, e.g. [[foo]]
    snippet: str     # first ~150 chars of section body
    score: float     # normalised 0–1 (1.0 = best hit in this query)
    wikilinks: List[str] = field(default_factory=list)

    @property
    def wikilink_ref(self) -> str:
        """Return wikilink form for this file, e.g. [[foo]] or [[research/proj/hash]]."""
        return self.page_ref


@dataclass
class _Section:
    path: str
    heading: str
    body: str
    page_ref: str
    section_ref: str
    wikilinks: List[str]


def _path_to_page_ref(path: str) -> str:
    """Return an Obsidian-compatible page ref for a markdown path."""
    path_no_ext = path[: -len(".md")] if path.endswith(".md") else path
    if path_no_ext.startswith("wiki/topics/"):
        return f"[[{path_no_ext[len('wiki/topics/'):]}]]"
    return f"[[{path_no_ext}]]"


def _path_to_section_ref(path: str, heading: str) -> str:
    """Return an Obsidian-compatible section ref for a markdown path."""
    page_ref = _path_to_page_ref(path)
    if not heading:
        return page_ref
    return page_ref[:-2] + f"#{heading}]]"


def _wikilink_page_target(link: str) -> str:
    """Collapse [[page#section|label]] to its page target for graph scoring."""
    target = link.split("|", 1)[0].strip()
    return target.split("#", 1)[0].strip()


class WorkspaceSearch:
    def __init__(self, workspace_root: str, config: Optional[dict] = None):
        self.workspace_root = workspace_root
        cfg = config or {}
        self.enabled: bool = bool(cfg.get("enabled", True))
        self.top_k: int = int(cfg.get("top_k", _DEFAULT_TOP_K))
        self.min_score: float = float(cfg.get("min_score", _DEFAULT_MIN_SCORE))
        self.min_raw_score: float = float(cfg.get("min_raw_score", _DEFAULT_MIN_RAW_SCORE))
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
            page_ref = _path_to_page_ref(path)
            if not current_heading and len(body) > _PARAGRAPH_SPLIT_THRESHOLD:
                # Flat file: split into paragraph chunks so BM25 scores individual facts
                for para in _PARAGRAPH_SPLIT_RE.split(body):
                    para = para.strip()
                    if para:
                        sections.append(_Section(
                            path=path,
                            heading="",
                            body=para,
                            page_ref=page_ref,
                            section_ref=page_ref,
                            wikilinks=_WIKILINK_RE.findall(para),
                        ))
            else:
                sections.append(_Section(
                    path=path,
                    heading=current_heading,
                    body=body,
                    page_ref=page_ref,
                    section_ref=_path_to_section_ref(path, current_heading),
                    wikilinks=_WIKILINK_RE.findall(body),
                ))

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

    @classmethod
    def _tokenize_query(cls, text: str) -> List[str]:
        """Tokenize a query, stripping stopwords and 1-2 character tokens.

        Single letters and very short tokens contribute pure noise to BM25
        because they appear in almost every document, so we drop them along
        with explicit stopwords.
        """
        return [
            t for t in cls._tokenize(text)
            if len(t) >= 3 and t not in _QUERY_STOPWORDS
        ]

    def _bm25_search(self, query: str, sections: List[_Section]) -> List[SearchHit]:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return self._fallback_tf_search(query, sections)

        query_tokens_list = self._tokenize_query(query)
        if len(query_tokens_list) < _MIN_QUERY_CONTENT_TOKENS:
            return []

        corpus = [
            self._tokenize(f"{s.heading} {s.body}") for s in sections
        ]
        bm25 = BM25Okapi(corpus)
        raw_scores = bm25.get_scores(query_tokens_list)

        query_tokens = set(query_tokens_list)
        incoming_links = self._incoming_link_counts(sections)
        adjusted_scores = [
            self._adjust_score(raw, sections[i], query_tokens, incoming_links)
            for i, raw in enumerate(raw_scores)
        ]
        max_adjusted = max(adjusted_scores) if adjusted_scores else 0.0

        if max_adjusted < self.min_raw_score:
            return []

        hits: List[SearchHit] = []
        for i, score in enumerate(adjusted_scores):
            norm = score / max_adjusted
            if norm < self.min_score:
                continue
            s = sections[i]
            snippet = self._make_snippet(s.body, query_tokens)
            hits.append(SearchHit(
                path=s.path,
                heading=s.heading,
                section_ref=s.section_ref,
                page_ref=s.page_ref,
                snippet=snippet,
                score=norm,
                wikilinks=s.wikilinks,
            ))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[: self.top_k]

    def _fallback_tf_search(self, query: str, sections: List[_Section]) -> List[SearchHit]:
        """Simple term-frequency fallback when rank_bm25 is unavailable."""
        query_tokens = set(self._tokenize_query(query))
        if len(query_tokens) < _MIN_QUERY_CONTENT_TOKENS:
            return []
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
        incoming_links = self._incoming_link_counts(sections)
        adjusted = [
            (self._adjust_score(raw, s, query_tokens, incoming_links), s)
            for raw, s in scored
        ]
        max_score = max(score for score, _ in adjusted)
        hits: List[SearchHit] = []
        for raw, s in adjusted:
            norm = raw / max_score
            if norm < self.min_score:
                continue
            snippet = self._make_snippet(s.body, query_tokens)
            hits.append(SearchHit(
                path=s.path,
                heading=s.heading,
                section_ref=s.section_ref,
                page_ref=s.page_ref,
                snippet=snippet,
                score=norm,
                wikilinks=s.wikilinks,
            ))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[: self.top_k]

    @staticmethod
    def _incoming_link_counts(sections: List[_Section]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for section in sections:
            source = section.page_ref[2:-2]
            targets = {
                _wikilink_page_target(link)
                for link in section.wikilinks
            }
            for target in targets:
                if not target or target == source:
                    continue
                counts[target] = counts.get(target, 0) + 1
        return counts

    @staticmethod
    def _adjust_score(
        raw_score: float,
        section: _Section,
        query_tokens: set[str],
        incoming_links: dict[str, int],
    ) -> float:
        heading_tokens = set(_WORD_RE.findall(section.heading.lower()))
        page_tokens = set(_WORD_RE.findall(section.page_ref.lower()))
        related_boost = 0.15 if section.heading.lower() == "related" else 0.0
        heading_boost = 0.6 * len(query_tokens & heading_tokens)
        page_boost = 0.35 * len(query_tokens & page_tokens)
        link_boost = min(0.2, 0.05 * incoming_links.get(section.page_ref[2:-2], 0))
        return raw_score + heading_boost + page_boost + related_boost + link_boost

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
