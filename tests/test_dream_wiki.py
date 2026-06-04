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
# ---------------------------------------------------------------------------
def test_rag_llm_call_includes_format_json_for_ollama_when_json_mode(monkeypatch):
    """When json_mode=True and provider=ollama, the payload must include
    'format: json' so the model is steered into JSON output."""
    import asyncio
    import json as _json
    import urllib.request

    from pawlia.utils import rag_llm_call

    captured = {}

    def _fake_urlopen(req, timeout=None):
        body = req.data.decode() if req.data else ""
        captured["body"] = _json.loads(body)
        captured["url"] = req.full_url

        class _Resp:
            def __init__(self, payload):
                self._payload = payload

            def read(self):
                return _json.dumps(self._payload).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _Resp({"message": {"content": "[]"}})

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    cfg = {"rag_provider": "ollama", "rag_model": "qwen3.5:latest",
           "embedding_host": "http://localhost:11434"}

    asyncio.run(rag_llm_call(cfg, "sys", "user", json_mode=True))

    assert captured["body"].get("format") == "json"
    assert captured["url"].endswith("/api/chat")


def test_rag_llm_call_includes_response_format_for_openai_compat_when_json_mode(monkeypatch):
    import asyncio
    import json as _json
    import urllib.request

    from pawlia.utils import rag_llm_call

    captured = {}

    def _fake_urlopen(req, timeout=None):
        body = req.data.decode() if req.data else ""
        captured["body"] = _json.loads(body)
        captured["url"] = req.full_url

        class _Resp:
            def __init__(self, payload):
                self._payload = payload

            def read(self):
                return _json.dumps(self._payload).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _Resp({"choices": [{"message": {"content": "{}"}}]})

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    cfg = {"rag_provider": "openai", "rag_model": "x",
           "rag_base_url": "http://api.example/v1"}

    asyncio.run(rag_llm_call(cfg, "sys", "user", json_mode=True))

    assert captured["body"].get("response_format") == {"type": "json_object"}
    assert captured["url"].endswith("/chat/completions")


def test_rag_llm_call_omits_json_hints_when_disabled(monkeypatch):
    import asyncio
    import json as _json
    import urllib.request

    from pawlia.utils import rag_llm_call

    captured = {}

    def _fake_urlopen(req, timeout=None):
        body = req.data.decode() if req.data else ""
        captured["body"] = _json.loads(body)

        class _Resp:
            def read(self):
                return b'{"message":{"content":""}}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    cfg = {"rag_provider": "ollama", "embedding_host": "http://localhost:11434"}
    asyncio.run(rag_llm_call(cfg, "sys", "user"))  # no json_mode

    assert "format" not in captured["body"]
    assert "response_format" not in captured["body"]


def test_rag_llm_call_array_mode_omits_response_format_for_openai_compat(monkeypatch):
    """OpenAI-compatible json_object mode forbids a top-level array, so for
    json_mode='array' we must NOT send response_format (it would steer the
    model into an object and collapse the actions list)."""
    import asyncio
    import json as _json
    import urllib.request

    from pawlia.utils import rag_llm_call

    captured = {}

    def _fake_urlopen(req, timeout=None):
        captured["body"] = _json.loads(req.data.decode()) if req.data else {}

        class _Resp:
            def read(self):
                return b'{"choices":[{"message":{"content":"[]"}}]}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    cfg = {"rag_provider": "openai", "rag_model": "x",
           "rag_base_url": "http://api.example/v1"}
    asyncio.run(rag_llm_call(cfg, "sys", "user", json_mode="array"))

    assert "response_format" not in captured["body"]


def test_rag_llm_call_array_mode_still_sets_format_json_for_ollama(monkeypatch):
    """Ollama's format:json does not constrain the top-level type, so any
    truthy json_mode (including 'array') should still set it."""
    import asyncio
    import json as _json
    import urllib.request

    from pawlia.utils import rag_llm_call

    captured = {}

    def _fake_urlopen(req, timeout=None):
        captured["body"] = _json.loads(req.data.decode()) if req.data else {}

        class _Resp:
            def read(self):
                return b'{"message":{"content":"[]"}}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    cfg = {"rag_provider": "ollama", "embedding_host": "http://localhost:11434"}
    asyncio.run(rag_llm_call(cfg, "sys", "user", json_mode="array"))

    assert captured["body"].get("format") == "json"


def test_rag_llm_call_sends_pawlia_user_agent(monkeypatch):
    """Every provider request must carry an explicit User-Agent — some
    providers (Groq behind Cloudflare) 403 the default Python-urllib UA."""
    import asyncio
    import json as _json
    import urllib.request

    from pawlia.utils import PAWLIA_USER_AGENT, rag_llm_call

    seen = {}

    def _fake_urlopen(req, timeout=None):
        seen["ua"] = req.get_header("User-agent")

        class _Resp:
            def read(self):
                return b'{"choices":[{"message":{"content":"{}"}}]}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    cfg = {"rag_provider": "openai", "rag_model": "x",
           "rag_base_url": "http://api.example/v1"}
    asyncio.run(rag_llm_call(cfg, "sys", "user"))

    assert seen["ua"] == PAWLIA_USER_AGENT
