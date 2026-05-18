import json
import os
import subprocess
import sys


_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "skills",
    "files",
    "scripts",
    "files.py",
)


def _run(tmp_path, *args):
    env = os.environ.copy()
    env["PAWLIA_USER_ID"] = "u1"
    env["PAWLIA_SESSION_DIR"] = str(tmp_path)
    proc = subprocess.run(
        [sys.executable, _SCRIPT, *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(proc.stdout)


def test_read_accepts_wikilink_section_anchor(tmp_path):
    file_path = tmp_path / "u1" / "workspace" / "wiki" / "topics" / "topic"
    file_path.mkdir(parents=True)
    (file_path / "homelab.md").write_text(
        "# Homelab\n\n## Proxmox Setup\n\nServer with 64GB RAM.\n\n## Backup\n\nNightly Borg.\n",
        encoding="utf-8",
    )

    result = _run(tmp_path, "read", "--filename", "[[topic/homelab#Proxmox Setup]]")

    assert result["success"] is True
    assert result["section"] == "Proxmox Setup"
    assert "64GB RAM" in result["content"]
    assert "Nightly Borg" not in result["content"]


def test_read_section_accepts_embedded_anchor(tmp_path):
    file_path = tmp_path / "u1" / "workspace" / "research" / "proj"
    file_path.mkdir(parents=True)
    (file_path / "industrial-forge.md").write_text(
        "# Industrial Forge\n\n## Fuel\n\n1 Gasoline burns for 15m.\n\n## Notes\n\nHuge structure.\n",
        encoding="utf-8",
    )

    result = _run(
        tmp_path,
        "read-section",
        "--filename",
        "[[research/proj/industrial-forge#Fuel]]",
        "--section",
        "ignored",
    )

    assert result["success"] is True
    assert result["section"] == "Fuel"
    assert "15m" in result["content"]
    assert "Huge structure" not in result["content"]


def test_list_is_paginated_by_default(tmp_path):
    workspace = tmp_path / "u1" / "workspace"
    workspace.mkdir(parents=True)
    for idx in range(205):
        (workspace / f"file_{idx:03d}.txt").write_text("x", encoding="utf-8")

    result = _run(tmp_path, "list")

    assert result["success"] is True
    assert result["count"] == 200
    assert result["total_count"] == 205
    assert result["has_more"] is True
    assert result["next_offset"] == 200

    page2 = _run(tmp_path, "list", "--offset", "200", "--limit", "200")

    assert page2["success"] is True
    assert page2["count"] == 5
    assert page2["has_more"] is False
