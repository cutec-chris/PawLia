import asyncio

from pawlia.dream_wiki import DreamWikiBackend, _SENTINEL, _parse_json_array


def test_dream_wiki_writes_pages_into_type_subfolders(tmp_path):
    index_path = tmp_path / "session" / "user" / "memory_index"
    wiki_dir = tmp_path / "session" / "user" / "workspace" / "wiki"
    index_path.mkdir(parents=True)

    backend = DreamWikiBackend(str(index_path), {}, wiki_dir=str(wiki_dir))

    asyncio.run(
        backend._update_page(
            "max-mustermann",
            "Max Mustermann",
            "Ein Kollege aus dem Ops-Team.",
            "2026-05-18",
            "create",
            entity_type="person",
        )
    )

    assert (wiki_dir / "topics" / "person" / "max-mustermann.md").exists()


def test_dream_wiki_moves_flat_pages_into_type_subfolders(tmp_path):
    index_path = tmp_path / "session" / "user" / "memory_index"
    wiki_dir = tmp_path / "session" / "user" / "workspace" / "wiki"
    topics_dir = wiki_dir / "topics"
    topics_dir.mkdir(parents=True)
    index_path.mkdir(parents=True)
    flat = topics_dir / "balu.md"
    flat.write_text(
        "---\ntype: object\ntitle: Balu\n---\n\n# Balu\n\nHund.\n",
        encoding="utf-8",
    )

    backend = DreamWikiBackend(str(index_path), {}, wiki_dir=str(wiki_dir))

    catalog = backend._get_wiki_catalog()

    assert catalog["balu"] == "Balu"
    assert not flat.exists()
    assert (topics_dir / "object" / "balu.md").exists()


def test_dream_wiki_links_use_typed_obsidian_paths(tmp_path):
    index_path = tmp_path / "session" / "user" / "memory_index"
    wiki_dir = tmp_path / "session" / "user" / "workspace" / "wiki"
    index_path.mkdir(parents=True)

    backend = DreamWikiBackend(str(index_path), {}, wiki_dir=str(wiki_dir))
    asyncio.run(
        backend._update_page(
            "balu",
            "Balu",
            "Hund.",
            "2026-05-18",
            "create",
            entity_type="object",
        )
    )

    assert backend._md_link("balu", {"balu": "Balu"}) == "[[object/balu|Balu]]"


# ---------------------------------------------------------------------------
# _parse_json_array — audit problem #6
# ---------------------------------------------------------------------------
def test_parse_json_array_accepts_bare_array():
    result = _parse_json_array('[{"action": "create", "slug": "x"}]')
    assert result is not _SENTINEL
    assert result[0]["slug"] == "x"


def test_parse_json_array_strips_json_code_fence():
    content = '```json\n[{"action": "create", "slug": "y"}]\n```'
    result = _parse_json_array(content)
    assert result is not _SENTINEL
    assert result[0]["slug"] == "y"


def test_parse_json_array_strips_german_preamble():
    content = 'Hier ist das JSON-Array:\n[{"action": "create", "slug": "z"}]'
    result = _parse_json_array(content)
    assert result is not _SENTINEL
    assert result[0]["slug"] == "z"


def test_parse_json_array_strips_english_preamble():
    content = 'Here is the JSON:\n[{"action": "create", "slug": "a"}]'
    result = _parse_json_array(content)
    assert result is not _SENTINEL
    assert result[0]["slug"] == "a"


def test_parse_json_array_wraps_single_object():
    """If the model returns an object instead of an array, wrap it."""
    content = '{"action": "create", "slug": "b"}'
    result = _parse_json_array(content)
    assert result is not _SENTINEL
    assert isinstance(result, list)
    assert result[0]["slug"] == "b"


def test_parse_json_array_returns_sentinel_for_prose():
    content = "Hallo Chris! 👋 Hier ist dein Bericht."
    result = _parse_json_array(content)
    assert result is _SENTINEL


def test_parse_json_array_handles_empty_string():
    assert _parse_json_array("") is _SENTINEL
    assert _parse_json_array("   \n  ") is _SENTINEL


def test_parse_json_array_handles_malformed_json():
    result = _parse_json_array("[{broken json")
    assert result is _SENTINEL


def test_parse_json_array_extracts_array_embedded_in_prose():
    content = (
        "Here are the extracted topics:\n\n"
        "```\n"
        '[{"action": "create", "slug": "c"}, {"action": "update", "slug": "d"}]\n'
        "```\n\n"
        "Let me know if you want to add more."
    )
    result = _parse_json_array(content)
    assert result is not _SENTINEL
    assert len(result) == 2
    assert result[0]["slug"] == "c"
    assert result[1]["slug"] == "d"


# ---------------------------------------------------------------------------
# rag_llm_call — json_mode passes the right hint to the provider
#
# These share the ``fake_urlopen`` fixture (see conftest) instead of each
# redefining a urlopen double — they only differ in cfg, json_mode and the
# canned response.
# ---------------------------------------------------------------------------
from pawlia.utils import PAWLIA_USER_AGENT, rag_llm_call

_OLLAMA_CFG = {"rag_provider": "ollama", "rag_model": "qwen3.5:latest",
               "embedding_host": "http://localhost:11434"}
_OPENAI_CFG = {"rag_provider": "openai", "rag_model": "x",
               "rag_base_url": "http://api.example/v1"}


def test_rag_llm_call_includes_format_json_for_ollama_when_json_mode(fake_urlopen):
    """When json_mode=True and provider=ollama, the payload must include
    'format: json' so the model is steered into JSON output."""
    captured = fake_urlopen({"message": {"content": "[]"}})
    asyncio.run(rag_llm_call(_OLLAMA_CFG, "sys", "user", json_mode=True))
    assert captured["body"].get("format") == "json"
    assert captured["url"].endswith("/api/chat")


def test_rag_llm_call_includes_response_format_for_openai_compat_when_json_mode(fake_urlopen):
    captured = fake_urlopen({"choices": [{"message": {"content": "{}"}}]})
    asyncio.run(rag_llm_call(_OPENAI_CFG, "sys", "user", json_mode=True))
    assert captured["body"].get("response_format") == {"type": "json_object"}
    assert captured["url"].endswith("/chat/completions")


def test_rag_llm_call_omits_json_hints_when_disabled(fake_urlopen):
    captured = fake_urlopen({"message": {"content": ""}})
    asyncio.run(rag_llm_call(_OLLAMA_CFG, "sys", "user"))  # no json_mode
    assert "format" not in captured["body"]
    assert "response_format" not in captured["body"]


def test_rag_llm_call_array_mode_omits_response_format_for_openai_compat(fake_urlopen):
    """OpenAI-compatible json_object mode forbids a top-level array, so for
    json_mode='array' we must NOT send response_format (it would steer the
    model into an object and collapse the actions list)."""
    captured = fake_urlopen({"choices": [{"message": {"content": "[]"}}]})
    asyncio.run(rag_llm_call(_OPENAI_CFG, "sys", "user", json_mode="array"))
    assert "response_format" not in captured["body"]


def test_rag_llm_call_array_mode_still_sets_format_json_for_ollama(fake_urlopen):
    """Ollama's format:json does not constrain the top-level type, so any
    truthy json_mode (including 'array') should still set it."""
    captured = fake_urlopen({"message": {"content": "[]"}})
    asyncio.run(rag_llm_call(_OLLAMA_CFG, "sys", "user", json_mode="array"))
    assert captured["body"].get("format") == "json"


def test_rag_llm_call_sends_pawlia_user_agent(fake_urlopen):
    """Every provider request must carry an explicit User-Agent — some
    providers (Groq behind Cloudflare) 403 the default Python-urllib UA."""
    captured = fake_urlopen({"choices": [{"message": {"content": "{}"}}]})
    asyncio.run(rag_llm_call(_OPENAI_CFG, "sys", "user"))
    assert captured["ua"] == PAWLIA_USER_AGENT


# ---------------------------------------------------------------------------
# DreamWikiBackend — pure page/link/index helpers (no LLM, no network)
# ---------------------------------------------------------------------------
import pytest


@pytest.fixture
def backend(tmp_path):
    """A backend over an empty tmp wiki. Pass cfg overrides via .with_cfg()."""
    index_path = tmp_path / "session" / "user" / "memory_index"
    wiki_dir = tmp_path / "session" / "user" / "workspace" / "wiki"
    index_path.mkdir(parents=True)

    def _make(cfg=None):
        return DreamWikiBackend(str(index_path), cfg or {}, wiki_dir=str(wiki_dir))

    _make.wiki_dir = wiki_dir
    _make.index_path = index_path
    return _make


def _seed_page(backend_maker, slug, *, entity_type="topic", title=None, body="Body."):
    """Write a typed page via the backend and return its path."""
    b = backend_maker()
    asyncio.run(b._update_page(slug, title or slug, body, "2026-05-18", "create",
                               entity_type=entity_type))
    return b


# ---- _parse_frontmatter (static) ------------------------------------------
def test_parse_frontmatter_reads_a_yaml_block():
    fm = DreamWikiBackend._parse_frontmatter("---\ntitle: X\ntype: person\n---\n\nbody")
    assert fm == {"title": "X", "type": "person"}


def test_parse_frontmatter_returns_none_without_a_block():
    assert DreamWikiBackend._parse_frontmatter("# just a heading\n") is None


def test_parse_frontmatter_returns_none_on_unterminated_block():
    assert DreamWikiBackend._parse_frontmatter("---\ntitle: X\nno closing") is None


# ---- _build_frontmatter ----------------------------------------------------
def test_build_frontmatter_roundtrips_through_parse(backend):
    b = backend()
    raw = b._build_frontmatter("balu", "Balu", "2026-05-18", tags=["pet"],
                               entity_type="object")
    fm = DreamWikiBackend._parse_frontmatter(raw + "\n\nbody")
    assert fm["slug"] == "balu"
    assert fm["title"] == "Balu"
    assert fm["type"] == "object"
    assert fm["created"] == fm["updated"] == "2026-05-18"
    assert fm["tags"] == ["pet"]


# ---- topic path resolution -------------------------------------------------
def test_find_topic_path_resolves_typed_then_flat(backend):
    b = _seed_page(backend, "max", entity_type="person")
    found = b._find_topic_path("max", "person")
    assert found and found.endswith("topics/person/max.md")


def test_find_topic_path_falls_back_to_search_when_type_unknown(backend):
    b = _seed_page(backend, "max", entity_type="person")
    # Caller doesn't know the type — still finds it by walking topics/.
    assert b._find_topic_path("max") is not None


def test_find_topic_path_returns_none_for_unknown_slug(backend):
    assert backend()._find_topic_path("ghost") is None


def test_slug_target_is_type_relative(backend):
    b = _seed_page(backend, "max", entity_type="person")
    assert b._slug_target("max") == "person/max"


def test_slug_target_returns_bare_slug_when_missing(backend):
    assert backend()._slug_target("ghost") == "ghost"


# ---- link rendering --------------------------------------------------------
def test_md_link_uses_wikilink_format_by_default(backend):
    b = _seed_page(backend, "balu", entity_type="object", title="Balu")
    assert b._md_link("balu") == "[[object/balu|Balu]]"


def test_md_link_uses_markdown_format_when_configured(backend):
    maker = backend
    _seed_page(maker, "balu", entity_type="object", title="Balu")
    b = maker({"wiki_link_format": "markdown"})
    assert b._md_link("balu") == "[Balu](object/balu.md)"


def test_index_link_is_relative_to_topics_dir_in_markdown_mode(backend):
    maker = backend
    _seed_page(maker, "balu", entity_type="object", title="Balu")
    b = maker({"wiki_link_format": "markdown"})
    assert b._index_link("balu", "Balu") == "[Balu](topics/object/balu.md)"


@pytest.mark.parametrize("text,slug,expected", [
    ("see [[object/balu|Balu]] here", "balu", True),
    ("see [[balu]] here", "balu", True),
    ("see [[balu#Section]] here", "balu", True),
    ("see [Balu](object/balu.md) here", "balu", True),
    ("see [Balu](balu.md) here", "balu", True),
    ("no link to balu at all", "balu", False),
    ("a link to [[balu-the-bear]]", "balu", False),
])
def test_has_link_detects_every_link_form(backend, text, slug, expected):
    assert backend()._has_link(text, slug) is expected


# ---- prompt adaptation -----------------------------------------------------
def test_adapt_prompt_rewrites_markdown_instructions_for_wikilink_mode(backend):
    b = backend()  # default = wikilink
    prompt = "Links use standard Markdown format: `[Title](slug.md)`"
    adapted = b._adapt_prompt_for_link_format(prompt)
    assert "Obsidian wikilink format" in adapted
    assert "Markdown format" not in adapted


def test_adapt_prompt_is_a_noop_in_markdown_mode(backend):
    b = backend({"wiki_link_format": "markdown"})
    prompt = "Links use standard Markdown format: `[Title](slug.md)`"
    assert b._adapt_prompt_for_link_format(prompt) == prompt


# ---- tracker persistence ---------------------------------------------------
def test_tracker_roundtrips_through_disk(backend):
    b = backend()
    b._tracker = {"docA": "hashA"}
    b._save_tracker()

    fresh = backend()
    assert fresh._load_tracker() == {"docA": "hashA"}
    assert fresh._indexed == {"docA"}


def test_tracker_defaults_to_empty_when_absent(backend):
    assert backend()._load_tracker() == {}


# ---- catalog + index -------------------------------------------------------
def test_catalog_prefers_frontmatter_title_over_slug(backend):
    b = _seed_page(backend, "max-mustermann", entity_type="person",
                   title="Max Mustermann")
    assert b._get_wiki_catalog()["max-mustermann"] == "Max Mustermann"


def test_rebuild_index_groups_pages_by_type_and_counts_them(backend):
    maker = backend
    _seed_page(maker, "max", entity_type="person", title="Max")
    _seed_page(maker, "berlin", entity_type="place", title="Berlin")
    b = maker()
    b._rebuild_index()

    index = (maker.wiki_dir / "index.md").read_text(encoding="utf-8")
    assert "## Person" in index
    assert "## Place" in index
    assert "2 pages" in index
    # Person section comes before Place per _TYPE_ORDER.
    assert index.index("## Person") < index.index("## Place")


def test_append_log_accumulates_entries(backend):
    b = backend()
    b._append_log("create", "added a page", slugs=["max"])
    b._append_log("update", "edited it", slugs=["max"])

    log = (backend.wiki_dir / "log.md").read_text(encoding="utf-8")
    assert "create | added a page" in log
    assert "update | edited it" in log
    assert "pages: max" in log
