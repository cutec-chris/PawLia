import textwrap

from pawlia.memory import _format_workspace_refs
from pawlia.workspace_search import WorkspaceSearch


def test_workspace_search_returns_obsidian_section_refs(tmp_path):
    workspace = tmp_path / "workspace"
    topics = workspace / "wiki" / "topics" / "topic"
    topics.mkdir(parents=True)
    (topics / "homelab.md").write_text(
        textwrap.dedent(
            """
            # Homelab

            ## Proxmox Setup

            Proxmox runs on the server with 64GB RAM.

            ## Related

            - [[backup-strategy]]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (topics / "backup-strategy.md").write_text(
        textwrap.dedent(
            """
            # Backup Strategy

            ## Borg

            Borg backups run nightly.
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    hits = WorkspaceSearch(str(workspace)).search("How is the Proxmox setup configured?")

    assert hits
    assert hits[0].page_ref == "[[topic/homelab]]"
    assert hits[0].section_ref == "[[topic/homelab#Proxmox Setup]]"
    assert hits[0].wikilink_ref == "[[topic/homelab]]"


def test_workspace_refs_format_reads_page_but_shows_section(tmp_path):
    workspace = tmp_path / "workspace"
    topics = workspace / "wiki" / "topics" / "topic"
    topics.mkdir(parents=True)
    (topics / "homelab.md").write_text(
        textwrap.dedent(
            """
            # Homelab

            ## Proxmox Setup

            Proxmox runs on the server with 64GB RAM.
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    hits = WorkspaceSearch(str(workspace)).search("Tell me about the Proxmox setup")
    rendered = _format_workspace_refs(hits, user_query="Tell me about the Proxmox setup")

    assert "[[topic/homelab#Proxmox Setup]]" in rendered
    assert '`files read-section --filename "[[topic/homelab]]" --section "Proxmox Setup"`' in rendered


def test_workspace_search_uses_path_style_refs_for_typed_wiki_pages(tmp_path):
    workspace = tmp_path / "workspace"
    people = workspace / "wiki" / "topics" / "person"
    people.mkdir(parents=True)
    (people / "max.md").write_text(
        textwrap.dedent(
            """
            # Max

            ## Relationship

            Max is a colleague from the ops team.
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    hits = WorkspaceSearch(str(workspace)).search("Who is Max from ops?")

    assert hits
    assert hits[0].page_ref == "[[person/max]]"
    assert hits[0].section_ref.startswith("[[person/max#")


def test_workspace_search_keeps_non_wiki_paths_obsidian_compatible(tmp_path):
    workspace = tmp_path / "workspace"
    research = workspace / "research" / "ops"
    research.mkdir(parents=True)
    (research / "README.md").write_text(
        textwrap.dedent(
            """
            # Ops Notes

            ## Maintenance Window

            The maintenance window is every Sunday night.
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    hits = WorkspaceSearch(str(workspace)).search("When is the maintenance window?")

    assert hits
    assert hits[0].page_ref == "[[research/ops/README]]"
    assert hits[0].section_ref == "[[research/ops/README#Maintenance Window]]"
