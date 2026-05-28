---
name: perplexica
description: Perform AI-powered web searches using a Perplexica or Vane instance. Returns a synthesized answer with cited sources. Use when the user asks for current information, research topics, news, or complex questions requiring web search with summarization.
license: MIT
metadata:
  author: Christian Ulrich
  version: "1.1"
  trust: external
  compatibility: Requires Perplexica or Vane instance configuration
  requires_config:
    - url
---

# Perplexica / Vane AI Search

## Instructions

1. Run the search script with the provided arguments:
   ```
   python <scripts_dir>/search.py --query "<query>" --focus <focus_mode>
   ```
   The script reads the instance URL from `skill-config.perplexica.url`. You may pass `--url "<url>"` only if an explicit override is needed.
   
   **Optional model configuration**: By default, the skill automatically picks the first available chat and embedding model from the instance. You only need to set these if you want to override the default:
   - `skill-config.perplexica.chat_model_provider` (provider display name, e.g. `Groq`)
   - `skill-config.perplexica.chat_model` (model key, e.g. `llama-3.1-8b-instant`)
   - `skill-config.perplexica.embedding_model_provider` + `skill-config.perplexica.embedding_model`
   
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
