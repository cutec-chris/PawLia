---
name: workspace-git
description: >
  Sets up SSH-based git push for the Pawlia workspace, manages the workspace
  git remote, and creates automation jobs for regular pushes.
  Use when the user wants to: sync the workspace to an external git repo,
  set up git push, configure a git remote for the workspace, fix git push
  errors (SSH key, host key, authentication), check push status.
  Triggers on phrases like "workspace git", "git push einrichten", "ssh key
  für git", "workspace remote", "git sync", "push workspace".
license: MIT
metadata:
  author: Christian Ulrich
  version: "1.0"
  max_tool_turns: 20
---

# Workspace Git

Sets up authenticated git push for the Pawlia workspace and registers an
automation job that keeps it synced.

## Workflow

### Step 1 — Check current state

```
python <scripts_dir>/setup.py list-workspaces
```

Report what workspaces exist, which have a remote configured, and whether
SSH is already set up. If everything is already configured and working, skip
ahead to Step 4 (test).

### Step 2 — Generate SSH key

```
python <scripts_dir>/setup.py keygen
```

Output includes `public_key`. Show it to the user and ask them to add it
as a deploy key (read+write) to the remote repository. Wait for the user
to confirm before continuing.

### Step 3 — Configure remote

Once the user confirms the key is added:

```
python <scripts_dir>/setup.py configure \
  --workspace <path> \
  [--remote-url <git-ssh-url>]
```

Use the workspace path from Step 1. Include `--remote-url` only if the
remote is not yet configured or the user provided a new URL.

The script sets `core.sshCommand` in the workspace `.git/config` so all
subsequent git operations (including the automatic push in `workspace_git.py`)
use the session-persisted key. It then runs `git ls-remote` to verify the
connection.

If the test fails: check the error. Common causes:
- Key not yet added on the remote → ask the user to double-check
- Wrong URL format (must be SSH, e.g. `git@host:user/repo.git`) → ask for correction
- Host unreachable → report and stop

### Step 4 — Verify push

```
python <scripts_dir>/setup.py test --workspace <path>
```

Run a `git push --dry-run`. Report success or the full error output.

### Step 5 — Create automation job

After a successful test:

```
python <scripts_dir>/setup.py create-job \
  --workspace <path> \
  --schedule 03:00
```

This registers a daily push job in the room's `automations/jobs.json`.
If the job already exists, skip silently.

## Output format

All script commands output JSON:
```json
{"success": true, ...}
{"success": false, "error": "...", "hint": "..."}
```

## Example output

Ich habe deinen SSH-Key generiert:

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... pawlia-workspace
```

Füge diesen Key als Deploy-Key (mit Schreibzugriff) in dein Git-Repo ein,
dann sag mir Bescheid und ich richte den Push ein.

---

*(Nach Bestätigung)*

Super — Remote konfiguriert und Verbindung erfolgreich getestet. Ich habe
außerdem einen täglichen Push-Job um 03:00 Uhr angelegt, damit der Workspace
automatisch synchron bleibt.

## Error handling

| Error | Recovery |
|-------|----------|
| `Host key verification failed` | SSH key not yet in known_hosts — `configure` uses `accept-new`, retrying after key add should fix it |
| `Permission denied (publickey)` | Key not added to remote or wrong key — show key again, ask user to re-check |
| `Repository not found` | Wrong URL — ask user for correct SSH URL |
| `Not a git repo` | Run `list-workspaces` again, pick correct path |
