---
name: researcher
description: >
  Create and manage research projects with their own knowledge base.
  Each project has its own RAG instance for indexing and querying documents.
  Use when the user wants to research a topic, add URLs/documents to a project,
  or query an existing research project.
  The query MUST be a command:
  "create <name> <description>" to create a new project,
  "list" to list all projects,
  "add <project> <url> [depth]" to scrape and index a URL (depth for recursive, default 1),
  "query <project> <question>" to query the project's knowledge base,
  "delete <project>" to delete a project,
  "rename <old> <new>" to rename a project.
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

# Researcher Skill

## How to use

The query contains a researcher command. Run it via the Bash tool:

```
python <scripts_dir>/researcher.py <user_id> <command> [args...]
```

### Commands

| Command | Bash call | Description |
|---------|-----------|-------------|
| `create <name> <desc>` | `python <scripts_dir>/researcher.py <user_id> create "<name>" "<description>"` | Create a new research project |
| `list` | `python <scripts_dir>/researcher.py <user_id> list` | List all projects |
| `add <project> <url> [depth]` | `python <scripts_dir>/researcher.py <user_id> add "<project>" "<url>" [depth]` | Scrape URL and index it (depth for recursive crawling) |
| `query <project> <question>` | `python <scripts_dir>/researcher.py <user_id> query "<project>" "<question>"` | Query the project knowledge base |
| `delete <project>` | `python <scripts_dir>/researcher.py <user_id> delete "<project>"` | Delete a project |
| `rename <old> <new>` | `python <scripts_dir>/researcher.py <user_id> rename "<old>" "<new>"` | Rename a project |

## Step-by-step instructions

1. Parse the query to identify the command and arguments.
2. Replace `<scripts_dir>` with the actual scripts directory path.
3. Replace `<user_id>` with the actual user ID.
4. Run the command using the **Bash** tool.
5. Return the output to the user.

## Error handling

If the script exits with an error, report: "Research error: <error message from stderr>"
