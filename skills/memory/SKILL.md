---
name: memory
description: >
  Long-term memory powered by LightRAG. Indexes daily conversation logs
  and allows semantic search across all past conversations.
  Use when the user asks about something from past conversations,
  wants to recall what was discussed days/weeks ago, or says things like
  "we talked about...", "remember when...", "what did I say about...".
  The query should be a natural language question about past conversations.
  Special commands: "index" to manually trigger indexing of new chat logs,
  "status" to show indexing status.
license: MIT
metadata:
  author: Christian Ulrich
  version: "1.0"
  requires_config:
    - embedding_provider
    - embedding_model
    - embedding_dim
    - embedding_host
  optional_config:
    - rag_provider           # defaults to embedding_provider
    - rag_model              # LLM for RAG entity extraction (default: qwen3.5:latest)
    - rag_numctx             # LLM context window (default: 4096)
    - rag_timeout            # LLM timeout in seconds (default: 600)
    - rag_embedding_timeout  # embedding timeout in seconds (default: 120)
    - rag_max_async_llm      # max parallel LLM requests (default: 2)
    - rag_max_async_embedding # max parallel embedding requests (default: 4)
---

# Memory Skill

## How to use

The query is either a search question or a management command. Run via the Bash tool:

```
python <scripts_dir>/memory.py <user_id> search "<question>"
python <scripts_dir>/memory.py <user_id> index
python <scripts_dir>/memory.py <user_id> status
```

### Commands

| Command | Bash call | Description |
|---------|-----------|-------------|
| search | `python <scripts_dir>/memory.py <user_id> search "<question>"` | Semantic search across all past conversations |
| index | `python <scripts_dir>/memory.py <user_id> index` | Index any new/updated daily chat logs |
| status | `python <scripts_dir>/memory.py <user_id> status` | Show how many days are indexed |

## Step-by-step instructions

1. Determine if the query is a command (`index`, `status`) or a search question.
2. For search questions, use the `search` command with the user's question.
3. Replace `<scripts_dir>` and `<user_id>` with actual values.
4. Run the command using the **Bash** tool.
5. Return the result — for searches, present the relevant information naturally.

## Important

- The `index` command is also run automatically on every `search` to pick up new logs.
- Search results contain relevant excerpts from past conversations — use them to answer the user's question.
