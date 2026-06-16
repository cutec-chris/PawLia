#!/usr/bin/env python3
"""workspace-git setup script — keygen, configure, sync (remote wins), push, jobs."""
import argparse, json, os, subprocess, sys
from pathlib import Path

SSH_CMD = "ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"


def _out(d):
    print(json.dumps(d, ensure_ascii=False))


def _git(ws, *args):
    """Run a git command, return (returncode, stdout, stderr)."""
    r = subprocess.run(["git", *args], cwd=str(ws),
                       capture_output=True, text=True)
    return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()


def _has_uncommitted(ws):
    rc, out, _ = _git(ws, "status", "--porcelain")
    return rc == 0 and bool(out.strip())


def cmd_keygen(args):
    ws = Path(args.workspace)
    ws.mkdir(parents=True, exist_ok=True)
    keydir = ws / ".ssh"
    keydir.mkdir(exist_ok=True)
    keypath = keydir / "id_ed25519"
    if keypath.exists():
        pub = keypath.with_suffix(".pub").read_text().strip()
        _out({"success": True, "created": False, "public_key": pub,
              "message": "Key already exists."})
        return
    r = subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-f", str(keypath), "-N", "",
         "-C", "pawlia-workspace-git"],
        capture_output=True, text=True)
    if r.returncode != 0:
        _out({"success": False, "error": r.stderr.strip()})
        return
    pub = keypath.with_suffix(".pub").read_text().strip()
    _out({"success": True, "created": True, "public_key": pub})


def cmd_list_workspaces(args):
    session = Path(os.environ.get("PAWLIA_SESSION_DIR", "/app/session"))
    user = os.environ.get("PAWLIA_USER_ID", "")
    base = session / user / "workspace"
    dirs = []
    if base.is_dir():
        for d in sorted(base.iterdir()):
            if d.is_dir() and (d / ".git").is_dir():
                r = subprocess.run(
                    ["git", "remote", "get-url", "origin"],
                    cwd=str(d), capture_output=True, text=True)
                url = r.stdout.strip() if r.returncode == 0 else None
                dirs.append({"path": str(d), "remote": url})
    _out({"success": True, "workspaces": dirs})


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


def cmd_sync(args):
    """Sync workspace from remote — the remote ALWAYS wins on divergence.

    Flow:
      1. Commit uncommitted local changes (so nothing is silently destroyed
         by the hard reset; they remain recoverable via the reflog).
      2. git fetch origin
      3. If local HEAD == origin/HEAD -> up to date
      4. merge --ff-only (keeps non-conflicting local history)
      5. otherwise reset --hard origin/HEAD (remote wins)
    """
    ws = Path(args.workspace)
    if not (ws / ".git").is_dir():
        _out({"success": False, "error": f"Not a git repo: {args.workspace}"})
        return

    committed = False
    # Step 1: Commit uncommitted changes so they survive a hard reset (reflog)
    if _has_uncommitted(ws):
        _git(ws, "add", "-A")
        rc, out, err = _git(ws, "commit", "-m", "pre-sync")
        committed = rc == 0

    # Step 2: fetch
    rc, out, err = _git(ws, "fetch", "origin")
    if rc != 0:
        _out({"success": False,
              "error": f"git fetch failed: {err or out}",
              "step": "fetch",
              "committed": committed})
        return

    # Step 3: already up to date?
    _, local_head, _ = _git(ws, "rev-parse", "HEAD")
    _, remote_head, rerr = _git(ws, "rev-parse", "origin/HEAD")
    if remote_head and local_head == remote_head:
        _out({"success": True, "up_to_date": True, "committed": committed})
        return

    # Step 4: try fast-forward (preserves non-conflicting local history)
    rc, out, err = _git(ws, "merge", "--ff-only", "origin/HEAD")
    if rc == 0:
        _out({"success": True, "fast_forwarded": True,
              "committed": committed, "output": out or err})
        return

    # Step 5: divergent or blocked — remote wins, hard reset
    rc, out, err = _git(ws, "reset", "--hard", "origin/HEAD")
    if rc == 0:
        _out({"success": True, "reset_hard": True, "committed": committed,
              "message": "Local diverged from remote — reset --hard origin/HEAD (remote wins)."})
        return
    _out({"success": False,
          "error": f"git reset --hard failed: {err or out}",
          "step": "reset",
          "committed": committed})


def cmd_push(args):
    """Stage all, commit with auto-message, push."""
    ws = Path(args.workspace)
    if not (ws / ".git").is_dir():
        _out({"success": False, "error": f"Not a git repo: {args.workspace}"})
        return

    # Check if there's anything to commit
    if _has_uncommitted(ws):
        _git(ws, "add", "-A")
        rc, out, err = _git(ws, "commit", "-m",
                            f"auto-sync {subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ']).decode().strip()}")
        if rc != 0 and "nothing to commit" not in (err + out).lower():
            _out({"success": False,
                  "error": f"git commit failed: {err or out}",
                  "step": "commit"})
            return

    rc, out, err = _git(ws, "push", "origin")
    if rc != 0:
        # Maybe we need to pull first
        if "non-fast-forward" in (err + out).lower() or "rejected" in (err + out).lower():
            # Try sync first, then push again
            _out({"success": False,
                  "error": "Push rejected (non-fast-forward). Run sync first.",
                  "step": "push",
                  "hint": "Use the sync command to pull remote changes, then push again."})
        else:
            _out({"success": False,
                  "error": f"git push failed: {err or out}",
                  "step": "push"})
        return

    _out({"success": True, "pushed": True, "output": out or err})


def cmd_status(args):
    """Show git status and recent log for a workspace."""
    ws = Path(args.workspace)
    if not (ws / ".git").is_dir():
        _out({"success": False, "error": f"Not a git repo: {args.workspace}"})
        return

    _, status_out, _ = _git(ws, "status", "--short", "--branch")
    _, log_out, _ = _git(ws, "log", "--oneline", "-5")
    _, stash_out, _ = _git(ws, "stash", "list")

    _out({
        "success": True,
        "status": status_out,
        "recent_log": log_out,
        "stash_list": stash_out,
    })


_JOB_DEFAULTS = {
    "push": {
        "id": "job-workspace-git-push",
        "name": "Workspace Git Push",
        "schedule": "*/30 * * * *",
        "instruction": "Führe einen git push für den Workspace durch. Nutze den Befehl: {scripts_dir}/setup.py push --workspace <workspace-path>",
    },
    "pull": {
        "id": "job-workspace-git-pull",
        "name": "Workspace Git Pull",
        "schedule": "*/15 * * * *",
        "instruction": (
            "Synchronisiere den Workspace vom Remote — das REMOTE gewinnt immer. "
            "Nutze den Befehl: {scripts_dir}/setup.py sync --workspace <workspace-path>"
        ),
    },
}


def cmd_create_job(args):
    job_type = getattr(args, "type", "push") or "push"
    if job_type not in _JOB_DEFAULTS:
        _out({"success": False, "error": f"Unknown type '{job_type}'. Use: push, pull"})
        return

    jobs_path = Path(args.workspace).parent / "automations" / "jobs.json"
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        jobs = json.loads(jobs_path.read_text()) if jobs_path.exists() else []
    except (json.JSONDecodeError, OSError):
        jobs = []

    defaults = _JOB_DEFAULTS[job_type]
    job_id = defaults["id"]
    existing = next((j for j in jobs if j.get("id") == job_id), None)
    if existing:
        # Update the instruction in case it changed
        existing["instruction"] = defaults["instruction"]
        jobs_path.write_text(json.dumps(jobs, indent=2, ensure_ascii=False))
        _out({"success": True, "created": False,
              "message": f"{job_type.capitalize()} job already exists (instruction updated).", "job": existing})
        return

    job = {
        "id": job_id,
        "name": defaults["name"],
        "schedule": args.schedule or defaults["schedule"],
        "instruction": defaults["instruction"],
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
    "sync": cmd_sync,
    "push": cmd_push,
    "status": cmd_status,
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

p_sync = sub.add_parser("sync")
p_sync.add_argument("--workspace", required=True)

p_push = sub.add_parser("push")
p_push.add_argument("--workspace", required=True)

p_status = sub.add_parser("status")
p_status.add_argument("--workspace", required=True)

p_job = sub.add_parser("create-job")
p_job.add_argument("--workspace", required=True)
p_job.add_argument("--type", default="push", choices=["push", "pull"])
p_job.add_argument("--schedule")

args = parser.parse_args()
if not args.cmd or args.cmd not in COMMANDS:
    print(json.dumps({"success": False, "error": f"Unknown command. Use: {', '.join(COMMANDS)}"}))
    sys.exit(1)

COMMANDS[args.cmd](args)
