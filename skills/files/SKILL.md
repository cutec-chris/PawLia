---
name: files
description: Read, write, edit, list, grep, and delete files in the user's personal workspace. Use for creating notes, saving text, reading previously saved files, making targeted edits to existing files, searching across the workspace, writing workspace config files like identity.md, soul.md, user.md, and deleting files like bootstrap.md. All filenames are automatically lowercased.
license: MIT
metadata:
  author: Christian Ulrich
  version: "1.2"
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

Returns every file in the workspace **recursively**, with paths relative to the workspace root (e.g. `notes/today.md`).

## Read a file

```
python <scripts_dir>/files.py read --filename "<name>"
```

Optional line-range arguments for large files:

```
python <scripts_dir>/files.py read --filename "<name>" --offset 0 --limit 100
```

`--offset` is 0-based and counts lines. The response includes `total_lines` so you can decide whether to fetch more.

## Write a file

Content is passed via the `CONTENT` environment variable (set automatically by the workflow executor):

```
python <scripts_dir>/files.py write --filename "<name>"
```

For direct CLI use, content can also be passed via `--content` (single line only) or stdin.

Subdirectories are supported in filenames (e.g. `notes/today.txt`).

## Edit a file (targeted change)

For small, targeted modifications to an existing file, prefer `edit` over rewriting the whole file with `write`. Pass the exact text to replace via `OLD_STRING` and the replacement via `NEW_STRING` (env vars set automatically by the workflow executor).

```
python <scripts_dir>/files.py edit --filename "<name>"
```

Rules:
- `OLD_STRING` must appear **exactly once** in the file. If it occurs multiple times, the command fails — include more surrounding context (lines above/below) to make it unique, or pass `--replace-all` to replace every occurrence.
- `OLD_STRING` must match the file content **byte for byte**, including whitespace and indentation. If unsure, run `read` first.
- `OLD_STRING` and `NEW_STRING` must differ.

The response includes `content_after` — verify the change matches your intent.

## Grep across the workspace

```
python <scripts_dir>/files.py grep --pattern "<regex>"
```

Restrict to a single file:

```
python <scripts_dir>/files.py grep --pattern "<regex>" --filename "<name>"
```

Returns a list of `{filename, line, text}` matches. Use this to locate where a string lives before editing.

## Delete a file

```
python <scripts_dir>/files.py delete --filename "<name>"
```

## Output

All commands output JSON. On success: `{"success": true, ...}`. On error: `{"success": false, "error": "..."}`.
Report results naturally to the user.

## Verification after write/edit

After a `write`, the response includes `content_written`; after an `edit`, it includes `content_after`. **Always compare against what you intended.** If they differ, report the discrepancy to the user and try again. If `list` does not show a file you just wrote, re-read the file by name — `list` is recursive and any subdirectory file should appear with its full relative path.
