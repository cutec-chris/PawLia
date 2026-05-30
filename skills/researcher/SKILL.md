---
name: researcher
description: >
  Collect web sources into named research projects and answer questions from
  them. Scrapes URLs (recursive crawl, PDFs, YouTube transcripts) into a
  project, then answers questions grounded in the gathered sources via
  semantic/keyword search. Use for "research X", building a sourced dossier on
  a topic, or querying previously gathered material — as opposed to a one-shot
  web search (perplexica/searxng). Commands: create, list, add, query, delete,
  rename (syntax in the skill instructions).
license: MIT
metadata:
  author: Christian Ulrich
  version: "2.1"
  trust: internal
  optional_config:
    - embedding_provider        # enables semantic search (ollama or openai-compat)
    - embedding_model           # e.g. bge-m3:latest
    - embedding_dim             # embedding vector dimension
    - embedding_host            # e.g. http://localhost:11434
    - embedding_base_url        # for openai-compatible providers
    - embedding_api_key         # for openai-compatible providers
    - rag_embedding_timeout     # embedding request timeout in seconds (default: 120)
---

# Researcher Skill

## How to use

The query contains a researcher command. Run it via the Bash tool.
The user ID is automatically provided via the `PAWLIA_USER_ID` environment variable — do NOT pass it manually.

```
python <scripts_dir>/researcher.py <command> [args...]
```

### Commands

| Command | Bash call | Description |
|---------|-----------|-------------|
| `create <name> <desc>` | `python <scripts_dir>/researcher.py create "<name>" "<description>"` | Create a new research project |
| `list` | `python <scripts_dir>/researcher.py list` | List all projects |
| `add <project> <url> [depth]` | `python <scripts_dir>/researcher.py add "<project>" "<url>" [depth]` | Scrape URL and save to workspace (depth for recursive crawling) |
| `query <project> <question>` | `python <scripts_dir>/researcher.py query "<project>" "<question>"` | Search the project's documents |
| `delete <project>` | `python <scripts_dir>/researcher.py delete "<project>"` | Delete a project |
| `rename <old> <new>` | `python <scripts_dir>/researcher.py rename "<old>" "<new>"` | Rename a project |

## Storage

Documents are saved as markdown under:
```
$PAWLIA_SESSION_DIR/{user_id}/research/{project}/
```

This lives **beside** the workspace, not inside it, so scraped sources never
leak into the workspace listing, BM25 search, git push or the DreamWiki.

No RAG backend or DreamWiki is involved. The embed index (`.index/`) is built
lazily on the first `query` call and invalidated automatically after each `add`.

## Search

- **With embedding config**: semantic search via cosine similarity on bge-m3 (or configured model)
- **Without embedding config**: keyword search fallback

## Step-by-step instructions

1. Parse the query to identify the command and arguments.
2. Run the command using the **Bash** tool.
3. Return the output to the user.

## Error handling

If the script exits with an error, report: "Research error: <error message from stderr>"
