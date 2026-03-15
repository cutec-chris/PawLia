---
name: searxng
description: Perform web searches using a SearXNG instance. Use when the user asks for web search results, current information, news, or wants to find online resources.
license: MIT
metadata:
  author: Christian Ulrich
  version: "2.0"
  compatibility: Requires SearXNG instance configuration
  requires_config:
    - url
---

# SearXNG Web Search

## Instructions

1. Run the search script with the provided arguments:
   ```
   python <scripts_dir>/search.py --query "<query>" --limit <limit> --url "<url>" --timeout <timeout>
   ```
2. Parse the JSON output (array of objects with `title`, `url`, `content` fields)
3. Return the results as a structured list

## Output format

Return results like this:
```
1. **<title>**
   <url>
   <content>
```

Return only the search results, no additional commentary.

## Error handling

If the script exits with an error, report: "Search failed: <error message from stderr>"
