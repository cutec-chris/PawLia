# Browser Skill

A Lynx-style headless browser for AI agents. Fetches websites and renders them
as interactive markdown where every clickable/fillable element gets a bracketed
identifier. The agent can then reference those IDs to navigate and interact with
pages across multiple calls.

## How it works

```
browser.py open <url>           → fetch page, render as markdown with element IDs
browser.py click <ID>           → follow link or click button
browser.py fill <ID> <value>    → fill an input field
browser.py submit <FORM_ID>     → submit a form with current field values
browser.py show                 → re-render the current page
browser.py back                 → go back in history
```

## Element ID scheme

| Prefix | Element type                         |
|--------|--------------------------------------|
| `L#`   | Hyperlinks                           |
| `B#`   | Buttons and submit inputs            |
| `I#`   | Text/email/password/number inputs    |
| `S#`   | Select dropdowns                     |
| `T#`   | Textarea fields                      |
| `F#`   | Forms                                |

### Example rendered output

```
# Example Domain

This domain is for use in illustrative examples. [L1]

[ FORM F1: GET https://example.com/search ]
  [I1] ________ (text: q)*
  [B1] Search
[ END FORM F1 ]

## Links

- About [L2]
- Contact Us [L3]
```

### Example session

```bash
# Open a page
python browser.py open https://example.com/login

# Output:
# [ FORM F1: POST https://example.com/login ]
#   [I1] ________ (text: username)*
#   [I2] ________ (password: password)*
#   [B1] Log In
# [ END FORM F1 ]

# Fill fields and submit
python browser.py fill I1 myuser
python browser.py fill I2 mypassword
python browser.py submit F1

# Now the page after login is rendered
```

## Session state

State is persisted in `~/.pawlia_browser.json`. This includes:
- Current URL
- Cookie jar
- Navigation history (last 10 URLs)
- Element map from the last rendered page
- Current form field values

Because state is file-based, multiple calls to the script within the same
conversation share state automatically.

## Stealth / anti-bot

The script sends realistic Chrome headers including `User-Agent`, `sec-ch-ua`,
`Sec-Fetch-*`, `Accept-Language`, and `DNT`. Cookies are preserved across calls.
This passes basic bot detection on most sites.

For sites with advanced fingerprinting (Cloudflare, etc.) this approach will
not be sufficient — those require a full browser with JavaScript execution.

## Dependencies

- Python 3.x (stdlib only for HTML parsing)
- `requests` (for HTTP)

Both are available in the pawlia `.venv`.

## Implementation plan

- [x] README (this file)
- [x] SKILL.md — LLM instructions
- [x] scripts/browser.py — core implementation
  - [x] Stealth HTTP session (Chrome headers, cookie persistence)
  - [x] HTML → Markdown renderer with element IDs
  - [x] Session state (load/save `~/.pawlia_browser.json`)
  - [x] `open` command
  - [x] `click` command (links + buttons)
  - [x] `fill` command
  - [x] `submit` command (GET + POST form encoding)
  - [x] `show` command
  - [x] `back` command
  - [x] Redirect following
  - [x] Encoding detection (gzip, charset sniffing)
