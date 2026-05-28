---
name: perplexica
description: Perform AI-powered web searches using a Perplexica instance. Returns a synthesized answer with cited sources. Use when the user asks for current information, research topics, news, or complex questions requiring web search with summarization.
license: MIT
metadata:
  author: Christian Ulrich
  version: "1.0"
  trust: external
  compatibility: Requires Perplexica instance configuration
  requires_config:
    - url
---

# Perplexica AI Search

## Instructions

1. Run the search script with the provided arguments:
   ```
   python <scripts_dir>/search.py --query "<query>" --focus <focus_mode>
   ```
   The script reads the Perplexica URL from `skill-config.perplexica.url`. You may pass `--url "<url>"` only if an explicit override is needed.
   Optional: set `skill-config.perplexica.chat_model_provider` and `skill-config.perplexica.chat_model` to override the LLM used by Perplexica (e.g. `ollama` + `llama3.1`).
   Valid values for `--focus`: `webSearch` (default), `academicSearch`, `youtubeSearch`, `redditSearch`, `wolframAlphaSearch`. Omit `--focus` if unsure — it defaults to `webSearch`.
2. The script outputs a JSON object with `answer` (string) and `sources` (array of objects with `title`, `url`, `snippet` fields)
3. Return the answer followed by the sources

## Output format

Return results like this:
```
<answer>

Sources:
1. **<title>** — <url>
```

Return only the answer and sources, no additional commentary.

## Error handling

If the script exits with an error, report: "Search failed: <error message from stderr>"
