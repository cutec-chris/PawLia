#!/usr/bin/env python3
"""Workspace git remote setup — SSH key management and remote configuration."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SESSION_DIR = os.environ.get("PAWLIA_SESSION_DIR", "/app/session")
SSH_DIR = Path(SESSION_DIR) / ".ssh"
KEY_PATH = SSH_DIR / "workspace_ed25519"
KNOWN_HOSTS = SSH_DIR / "known_hosts"
SSH_CMD = f"ssh -i {KEY_PATH} -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile={KNOWN_HOSTS}"


def _out(data: dict):
    print(json.dumps(data, ensure_ascii=False))


def cmd_keygen(_args):
    SSH_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    pub = Path(str(KEY_PATH) + ".pub")
    if KEY_PATH.exists() and pub.exists():
        _out({"success": True, "public_key": pub.read_text().strip(),
              "key_path": str(KEY_PATH), "already_existed": True})
        return
    r = subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-f", str(KEY_PATH), "-N", "", "-C", "pawlia-workspace"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        _out({"success": False, "error": r.stderr.strip()})
        return
    _out({"success": True, "public_key": pub.read_text().strip(),
          "key_path": str(KEY_PATH), "already_existed": False})


def cmd_list_workspaces(_args):
    workspaces = []
    for ws in Path(SESSION_DIR).glob("*/workspace"):
        if not (ws / ".git").is_dir():
            continue
        r = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=str(ws), capture_output=True, text=True,
        )
        remote = r.stdout.strip() if r.returncode == 0 else ""
        r2 = subprocess.run(
            ["git", "config", "--get", "core.sshCommand"],
            cwd=str(ws), capture_output=True, text=True,
        )
        ssh_configured = r2.returncode == 0 and "workspace_ed25519" in r2.stdout
        workspaces.append({"path": str(ws), "remote_url": remote,
                           "ssh_configured": ssh_configured})
    _out({"success": True, "workspaces": workspaces})


def cmd_configure(args):
    ws = Path(args.workspace)
    if not (ws / ".git").is_dir():
        _out({"success": False, "error": f"Not a git repo: {args.workspace}"})
        return

    if args.remote_url:
        r = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=str(ws), capture_output=True, text=True,
        )
        if r.returncode == 0:
            subprocess.run(["git", "remote", "set-url", "origin", args.remote_url],
                           cwd=str(ws), capture_output=True)
        else:
            subprocess.run(["git", "remote", "add", "origin", args.remote_url],
                           cwd=str(ws), capture_output=True)

    r = subprocess.run(
        ["git", "config", "core.sshCommand", SSH_CMD],
        cwd=str(ws), capture_output=True, text=True,
    )
    if r.returncode != 0:
        _out({"success": False, "error": f"core.sshCommand: {r.stderr.strip()}"})
        return

    r = subprocess.run(
        ["git", "ls-remote", "--exit-code", "origin"],
        cwd=str(ws), capture_output=True, text=True,
    )
    if r.returncode == 0:
        _out({"success": True, "message": "SSH configured and remote connection verified."})
    else:
        _out({"success": False,
              "error": (r.stderr or r.stdout).strip(),
              "hint": "Key not yet accepted on remote, or wrong URL."})


def cmd_test(args):
    ws = Path(args.workspace)
    if not (ws / ".git").is_dir():
        _out({"success": False, "error": f"Not a git repo: {args.workspace}"})
        return
    r = subprocess.run(
        ["git", "push", "--dry-run", "origin"],
        cwd=str(ws), capture_output=True, text=True,
    )
    output = (r.stderr or r.stdout).strip()
    _out({"success": r.returncode == 0, "output": output,
          "returncode": r.returncode})


def cmd_create_job(args):
    """Register a push job in the room's automations/jobs.json."""
    jobs_path = Path(args.workspace).parent / "automations" / "jobs.json"
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        jobs = json.loads(jobs_path.read_text()) if jobs_path.exists() else []
    except (json.JSONDecodeError, OSError):
        jobs = []

    job_id = "job-workspace-git-push"
    existing = next((j for j in jobs if j.get("id") == job_id), None)
    if existing:
        _out({"success": True, "created": False,
              "message": "Push job already exists.", "job": existing})
        return

    job = {
        "id": job_id,
        "name": "Workspace Git Push",
        "schedule": args.schedule or "03:00",
        "instruction": "Führe einen git push für den Workspace durch",
        "notify": False,
        "enabled": True,
    }
    jobs.append(job)
    jobs_path.write_text(json.dumps(jobs, indent=2, ensure_ascii=False))
    _out({"success": True, "created": True, "job": job})


COMMANDS = {
    "keygen": cmd_keygen,
    "list-workspaces": cmd_list_workspaces,
    "configure": cmd_configure,
    "test": cmd_test,
    "create-job": cmd_create_job,
}

parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest="cmd")

sub.add_parser("keygen")
sub.add_parser("list-workspaces")

p_cfg = sub.add_parser("configure")
p_cfg.add_argument("--workspace", required=True)
p_cfg.add_argument("--remote-url")

p_test = sub.add_parser("test")
p_test.add_argument("--workspace", required=True)

p_job = sub.add_parser("create-job")
p_job.add_argument("--workspace", required=True)
p_job.add_argument("--schedule", default="03:00")

args = parser.parse_args()
if not args.cmd or args.cmd not in COMMANDS:
    print(json.dumps({"success": False, "error": f"Unknown command. Use: {', '.join(COMMANDS)}"}))
    sys.exit(1)

COMMANDS[args.cmd](args)
