---
name: files
description: Write, edit, delete, outline, and read-section files in the user's personal workspace. For simple reads, listing, or searching use the direct tools read_file, list_files, or grep_files instead — they are faster and do not require spawning a skill runner. Use this skill for creating notes, saving text, making targeted edits, writing workspace config files like identity.md, soul.md, user.md, and deleting files like bootstrap.md. Do NOT use for reading or editing skill scripts or SKILL.md files — delegate the whole task to the skill-creator skill instead. Filenames are resolved case-insensitively — if a file exists under a different casing it is used as-is; new files keep the casing you provide.
license: MIT
metadata:
  author: Christian Ulrich
  version: "1.3"
  trust: internal
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

Returns files in the workspace **recursively**, with paths relative to the workspace root (e.g. `notes/today.md`).

Default behaviour: returns at most 200 files so the result stays compact for the model. If `has_more: true` appears in the response, continue with:

```
python <scripts_dir>/files.py list --offset 200 --limit 200
```

## Read a file

```
python <scripts_dir>/files.py read --filename "<name>"
```

`--filename` also accepts wikilink syntax — e.g. `[[topic/ios-development]]` resolves to `wiki/topics/topic/ios-development.md`, `[[person/max-mustermann]]` resolves to `wiki/topics/person/max-mustermann.md`, and `[[research/proj/abc]]` resolves as a workspace-relative path. Section anchors are supported too: `[[topic/ios-development#Tooling]]` can be passed directly to `read` and will return that section. The same wikilink syntax works for `outline`, `read-section`, and `delete`.

**Default behaviour**: returns the first 150 lines. If `has_more: true` appears in the response, use `--offset` to continue reading.

Read in blocks:

```
python <scripts_dir>/files.py read --filename "<name>" --offset 0 --limit 150
python <scripts_dir>/files.py read --filename "<name>" --offset 150 --limit 150
```

`--offset` is 0-based and counts lines. The response includes `total_lines` and `next_offset` when more lines exist.

**Query-based reading** (preferred when looking for specific content): returns only the lines that match the query plus a few lines of context. Non-matching sections are replaced with `[... N lines skipped ...]` so the result stays compact.

```
python <scripts_dir>/files.py read --filename "<name>" --query "trauben wein flasche"
```

Use this instead of reading the whole file when you know what you're looking for.

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

## BM25 workspace search

```
python <scripts_dir>/files.py search --query "<search terms>"
```

Search across all workspace markdown files using BM25 (keyword relevance). Returns files with their headings and a snippet — the model can gauge relevance without reading every file.

```
python <scripts_dir>/files.py search --query "Ordnungsamt Biederitz" --limit 5
```

Returns `{filename, heading, headings (list), snippet, score}` for each hit. Use this before `files read` or `files read-section` to find the right file first. Prefer this over `grep` when you don't know the exact filename.

## Outline a markdown file

```
python <scripts_dir>/files.py outline --filename "<name>"
```

Returns the file's heading structure as `[{level, title, line}, ...]` (ATX `#` headings only; headings inside fenced code blocks are skipped). Use this to orient yourself in a long file before reading specific parts.

## Read a section by heading

```
python <scripts_dir>/files.py read-section --filename "<name>" --section "<heading title>"
```

Returns content from the matching heading line up to (but not including) the next heading at the same or higher level. Pass the heading **title only**, without leading `#`s. Match is exact, with case-insensitive fallback. If multiple headings share the same title the call fails — use `read --offset --limit` based on `outline` line numbers in that case.

## Delete a file

```
python <scripts_dir>/files.py delete --filename "<name>"
```

## Output

All commands output JSON. On success: `{"success": true, ...}`. On error: `{"success": false, "error": "..."}`.
Report results naturally to the user.

## Verification after write/edit

After a `write`, the response includes `content_written`; after an `edit`, it includes `content_after`. **Always compare against what you intended.** If they differ, report the discrepancy to the user and try again. If `list` does not show a file you just wrote, re-read the file by name — `list` is recursive and any subdirectory file should appear with its full relative path.
