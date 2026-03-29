---
name: files
description: Read, write, list, and delete files in the user's personal workspace. Use for creating notes, saving text, reading previously saved files, writing workspace config files like identity.md, soul.md, user.md, and deleting files like bootstrap.md. All filenames are automatically lowercased.
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

Content is passed via the `CONTENT` environment variable (set automatically by the workflow executor):

```
python <scripts_dir>/files.py write --filename "<name>"
```

For direct CLI use, content can also be passed via `--content` (single line only) or stdin.

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
