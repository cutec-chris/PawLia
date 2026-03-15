---
name: browser
description: >
  Headless browser for visiting websites and interacting with them.
  IMPORTANT: The query MUST be a browser command in one of these exact formats:
  "open <url>" to visit a page,
  "click <ID>" to follow a link or button (e.g. "click L3", "click B1"),
  "fill <ID> <value>" to fill a field (e.g. "fill I1 Berlin"),
  "submit <FORM_ID>" to submit a form (e.g. "submit F1"),
  "show" to re-display the current page,
  "back" to navigate back.
  Do NOT pass natural language or search terms as the query — pass the command directly.
  Example: to open bahn.de use query="open https://bahn.de"
license: MIT
metadata:
  author: Christian Ulrich
  version: "1.0"
---

# Browser Skill

## How to use

The query contains a browser command. Run it via the Bash tool:

```
python <scripts_dir>/browser.py <query>
```

### Example commands

| Query | What it does |
|-------|--------------|
| `open https://bahn.de` | Fetch and render the page |
| `open bahn.de` | Same (https:// added automatically) |
| `show` | Re-render current page |
| `back` | Go back |
| `click L3` | Follow link L3 |
| `click B1` | Click button B1 (submits its form) |
| `fill I1 Berlin` | Set input I1 to "Berlin" |
| `fill I2 2024-12-01` | Set input I2 to a date |
| `submit F1` | Submit form F1 with filled values |

## Step-by-step instructions

1. Parse the query:
   - If the query looks like a URL (starts with `http`, `https`, or a domain like `bahn.de`) → prepend `open`: use `open <query>`
   - If the query starts with a known command (`open`, `click`, `fill`, `submit`, `show`, `back`) → use as-is
2. Replace `<scripts_dir>` with the actual scripts directory path.
3. Run the command using the **Bash** tool: `python <scripts_dir>/browser.py <command>`
4. Return the full output — it contains the rendered page and element list.

## Multi-step example

To fill and submit a form across multiple calls:
- Call 1: `open https://example.com/login`
- Call 2: `fill I1 myusername`
- Call 3: `fill I2 mysecretpassword`
- Call 4: `submit F1`

State is automatically saved between calls.

## Element IDs (shown in page output)

- `L#` = links (clickable)
- `B#` = buttons / submit inputs
- `I#` = text/email/password inputs and checkboxes
- `S#` = select dropdowns
- `T#` = textarea fields
- `F#` = forms

## Error handling — SELF-REPAIR

When a command fails, DO NOT report the error to the user. Instead, recover:

| Error | Recovery action |
|-------|----------------|
| `No element [X] on current page` | Run `show` to see available elements, then retry with correct ID |
| `Element [X] is not clickable` | Check element type with `show`, use `fill`/`submit` instead |
| `Element [X] is not fillable` | Check element type, maybe it's a button — use `click` |
| `No active page` | Run `open <url>` to load the page first |
| `Connection error` | Try the URL again, or try an alternative URL |
| `No form [X]` | Run `show` to find the correct form ID |
| `No history to go back to` | Use `open` to navigate directly |

General recovery strategy:
1. After ANY error, run `show` to see the current page state and available elements.
2. Compare what you expected with what's actually there.
3. Adjust your approach (different element ID, different command, etc.).
4. Only give up after 2-3 recovery attempts.
