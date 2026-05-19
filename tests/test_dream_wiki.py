import asyncio

from pawlia.dream_wiki import DreamWikiBackend


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
