---
name: memory
description: >
  Long-term memory — search everything from past conversations.
  ALWAYS use this skill when the user: asks what was said before,
  references earlier conversations, wants to know or remember something
  from the past, asks "what did I/we/you say about...", "do you remember...",
  "have we talked about...", "what was that thing about...",
  or any question that requires knowledge from previous days/sessions.
  Also use when YOU are unsure whether a topic was discussed before.
  Commands: "index" to reindex, "status" to check index state,
  "dream" to trigger wiki consolidation, "lint" for wiki health check.
license: MIT
metadata:
  author: Christian Ulrich
  version: "2.0"
  requires_config:
    - rag_model              # LLM used for wiki analysis
    - embedding_host         # API host (used for LLM calls too)
  optional_config:
    - rag_provider           # defaults to embedding_provider / ollama
    - rag_numctx             # LLM context window (default: 4096)
    - rag_timeout            # LLM timeout in seconds (default: 600)
    - rag_backend            # backend: markdown (default) | lightrag | simple | mem0
    # Embedding settings — only needed for lightrag/simple/mem0 backends:
    - embedding_provider
    - embedding_model
    - embedding_dim
    - rag_embedding_timeout
    - rag_max_async_llm
    - rag_max_async_embedding
---

# Memory Skill

## How to use

The query is either a search question or a management command. Run via the Bash tool.
The user ID is automatically provided via the `PAWLIA_USER_ID` environment variable — do NOT pass it manually.

```
python <scripts_dir>/memory.py search "<question>"
python <scripts_dir>/memory.py index
python <scripts_dir>/memory.py dream
python <scripts_dir>/memory.py lint
python <scripts_dir>/memory.py status
```

### Commands

| Command | Bash call | Description |
|---------|-----------|-------------|
| search | `python <scripts_dir>/memory.py search "<question>"` | Semantic search across all past conversations |
| index | `python <scripts_dir>/memory.py index` | Index any new/updated daily chat logs |
| dream | `python <scripts_dir>/memory.py dream` | Trigger Dream Wiki: process chat logs into structured wiki pages |
| lint | `python <scripts_dir>/memory.py lint` | Wiki health check: merge overlapping pages, fix missing links |
| status | `python <scripts_dir>/memory.py status` | Show how many days are indexed |

## Step-by-step instructions

1. Determine if the query is a command (`index`, `status`, `dream`, `lint`) or a search question.
2. For search questions, use the `search` command with the user's question.
3. Run the command using the **Bash** tool.
4. Return the result — for searches, present the relevant information naturally.

## Important

- The `index` command is also run automatically on every `search` to pick up new logs.
- Search results contain relevant excerpts from past conversations — use them to answer the user's question.
- The `dream` command processes unprocessed chat logs into a structured wiki with cross-references (`[[wikilinks]]`).
- The Dream Wiki runs automatically when the user has been idle for 30 minutes (configurable via `dream_idle_minutes`).
