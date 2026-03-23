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
The `--user-id` and `--session-dir` arguments are automatically provided via environment variables — do NOT pass them manually.

---

## List files

```
python <scripts_dir>/files.py list
```

## Read a file

```
python <scripts_dir>/files.py read --filename "<name>"
```

## Write a file

**Option A — short content (single line or simple text):** use `--content`:

```
python <scripts_dir>/files.py write --filename "<name>" --content "content here"
```

**Option B — multiline content:** pipe content via stdin using a heredoc (works on all platforms):

```
python <scripts_dir>/files.py write --filename "<name>" << 'EOF'
line 1
line 2
line 3
EOF
```

Subdirectories are supported in filenames (e.g. `notes/today.txt`).

## Delete a file

```
python <scripts_dir>/files.py delete --filename "<name>"
```

## Output

All commands output JSON. On success: `{"success": true, ...}`. On error: `{"success": false, "error": "..."}`.
Report results naturally to the user.

## Verification after write

After a `write` command, the response includes `"content_written"` — the content that was actually read back from disk.
**Always compare `content_written` against what you intended to write.** If they differ, report the discrepancy to the user and rewrite the file.
