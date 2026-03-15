---
name: files
description: Read, write, list, and delete files in the user's personal workspace. Use for creating notes, saving text, reading previously saved files, writing workspace config files like IDENTITY.md, soul.md, USER.md, and deleting files like bootstrap.md.
license: MIT
metadata:
  author: Christian Ulrich
  version: "1.1"
---

# File Workspace

Manages files inside the user's sandboxed workspace (`session/{user_id}/workspace/`).
Path traversal outside the workspace is blocked by the script.

## IMPORTANT

Always use the **bash tool** to run the commands below.
Always pass `--user-id` and `--session-dir` from the context.

---

## List files

```
python <scripts_dir>/files.py list --user-id <user_id> --session-dir "<session_dir>"
```

## Read a file

```
python <scripts_dir>/files.py read --user-id <user_id> --session-dir "<session_dir>" --filename "<name>"
```

## Write a file

**Option A — short content (single line or simple text):** use `--content`:

```
python <scripts_dir>/files.py write --user-id <user_id> --session-dir "<session_dir>" --filename "<name>" --content "content here"
```

**Option B — multiline content:** pipe content via stdin using a heredoc (works on all platforms):

```
python <scripts_dir>/files.py write --user-id <user_id> --session-dir "<session_dir>" --filename "<name>" << 'EOF'
line 1
line 2
line 3
EOF
```

Subdirectories are supported in filenames (e.g. `notes/today.txt`).

## Delete a file

```
python <scripts_dir>/files.py delete --user-id <user_id> --session-dir "<session_dir>" --filename "<name>"
```

## Output

All commands output JSON. On success: `{"success": true, ...}`. On error: `{"success": false, "error": "..."}`.
Report results naturally to the user.
