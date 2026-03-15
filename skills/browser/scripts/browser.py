#!/usr/bin/env python3
"""Headless browser skill — fetch and interact with websites as markdown.

Commands:
  open <url>           Fetch and render a URL
  click <ID>           Follow link L# or click button B#
  fill <ID> <value>    Set value of input I#, select S#, or textarea T#
  submit <FORM_ID>     Submit form F# with current field values
  show                 Re-render current page
  back                 Navigate back in history
"""

import sys
import json
import os
import re
import urllib.parse
from html.parser import HTMLParser
import requests
import urllib3

# Ensure stdout is UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# Windows Python 3.14 has broken system root store for some CAs.
# Suppress the warning — this is intentional for a browsing tool.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
_VERIFY = False

SESSION_FILE = os.path.join(os.path.expanduser("~"), ".pawlia_browser.json")

STEALTH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8,"
        "application/signed-exchange;v=b3;q=0.7"
    ),
    "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate",  # no br — requests can't decode Brotli
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "sec-ch-ua": '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "Cache-Control": "max-age=0",
}

SKIP_TAGS = frozenset(
    {"script", "style", "noscript", "svg", "math", "iframe", "object", "embed", "template"}
)


def _attr(attrs, key, default=""):
    for k, v in attrs:
        if k == key:
            return v or default
    return default


def _has_attr(attrs, key):
    return any(k == key for k, _ in attrs)


# ---------------------------------------------------------------------------
# HTML → Markdown renderer
# ---------------------------------------------------------------------------

class Renderer(HTMLParser):
    """Converts HTML to annotated markdown, assigning IDs to interactive elements."""

    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url

        # Output
        self._lines: list[str] = []
        self._buf: list[str] = []  # current inline buffer
        self._last_blank = True

        # Skip depth (script/style/etc.)
        self._skip_depth = 0
        self._head_depth = 0

        # Element counters
        self._cnt = {c: 0 for c in "LBISTF"}
        # Element map: id_str → dict
        self.elements: dict[str, dict] = {}

        # Inline formatting depth
        self._strong = 0
        self._em = 0
        self._code_inline = 0
        self._pre = 0
        self._del = 0

        # Links stack: list of (href, id_str | None)
        self._links: list[tuple] = []

        # List stack: list of ('ul'/'ol', counter)
        self._list_stack: list[tuple] = []

        # Headings
        self._heading = 0

        # Title
        self.title = ""
        self._in_title = False

        # Tables
        self._table_depth = 0
        self._in_thead = False
        self._in_tr = False
        self._cells: list[str] = []
        self._cell_buf: list[str] = []
        self._in_cell = False
        self._is_header_row = False

        # Forms
        self.forms: dict[str, dict] = {}   # F1 → {action, method, fields:{name:value}, selects:{name:value}}
        self._cur_form: str | None = None

        # Select
        self._cur_select: str | None = None
        self._cur_select_name = ""
        self._select_opts: list[tuple] = []  # (value, label, selected)
        self._in_opt = False
        self._opt_value = ""
        self._opt_selected = False
        self._opt_buf: list[str] = []

        # Textarea
        self._cur_textarea: str | None = None

        # Labels (for= → label text)
        self._labels: dict[str, str] = {}
        self._in_label = False
        self._label_for = ""
        self._label_buf: list[str] = []

    # -- ID allocation -------------------------------------------------------

    def _next(self, prefix: str) -> str:
        self._cnt[prefix] += 1
        return f"{prefix}{self._cnt[prefix]}"

    # -- Output helpers -------------------------------------------------------

    def _flush_buf(self) -> str:
        text = "".join(self._buf).strip()
        self._buf = []
        return text

    def _emit(self, line: str):
        s = line.rstrip()
        if s == "":
            if not self._last_blank:
                self._lines.append("")
                self._last_blank = True
        else:
            self._lines.append(s)
            self._last_blank = False

    def _block_break(self):
        """Flush inline buffer and ensure we're at a block boundary."""
        text = self._flush_buf()
        if text:
            indent = "  " * len(self._list_stack)
            for part in text.split("\n"):
                part = part.strip()
                if part:
                    self._emit(indent + part)
        self._emit("")

    def _inline_text(self) -> str:
        return "".join(self._buf)

    # -- Main parser callbacks ------------------------------------------------

    def handle_starttag(self, tag, attrs):
        # Head section (skip everything except title)
        if tag == "head":
            self._head_depth += 1
            return
        if self._head_depth > 0 and tag != "title":
            if tag in SKIP_TAGS:
                self._skip_depth += 1
            return

        # Skip tags
        if tag in SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return

        # Title
        if tag == "title":
            self._in_title = True
            return

        # Label
        if tag == "label":
            self._in_label = True
            self._label_for = _attr(attrs, "for")
            self._label_buf = []
            return

        # --- Headings ---
        m = re.match(r"^h([1-6])$", tag)
        if m:
            self._block_break()
            self._heading = int(m.group(1))
            return

        # --- Block layout elements ---
        if tag in (
            "p", "div", "section", "article", "aside", "main",
            "header", "footer", "nav", "figure", "figcaption",
            "details", "summary", "address", "fieldset",
        ):
            self._block_break()
            return

        if tag == "legend":
            self._block_break()
            self._buf.append("**")
            return

        if tag == "br":
            self._buf.append("\n")
            return

        if tag == "hr":
            self._block_break()
            self._emit("---")
            return

        # --- Pre / code ---
        if tag == "pre":
            self._block_break()
            self._pre += 1
            self._emit("```")
            return

        if tag == "code":
            self._code_inline += 1
            if self._pre == 0:
                self._buf.append("`")
            return

        # --- Inline formatting ---
        if tag in ("strong", "b"):
            self._strong += 1
            self._buf.append("**")
            return

        if tag in ("em", "i"):
            self._em += 1
            self._buf.append("_")
            return

        if tag in ("del", "s", "strike"):
            self._del += 1
            self._buf.append("~~")
            return

        # --- Links ---
        if tag == "a":
            href = _attr(attrs, "href")
            if href and not href.startswith("javascript") and href != "#":
                href = urllib.parse.urljoin(self.base_url, href)
                lid = self._next("L")
                self.elements[lid] = {"type": "link", "href": href, "text": ""}
                self._links.append((href, lid))
            else:
                self._links.append((None, None))
            return

        # --- Images ---
        if tag == "img":
            alt = _attr(attrs, "alt")
            src = _attr(attrs, "src")
            if alt:
                self._buf.append(f"[{alt}]")
            elif src:
                self._buf.append("[img]")
            return

        # --- Lists ---
        if tag == "ul":
            self._block_break()
            self._list_stack.append(("ul", 0))
            return

        if tag == "ol":
            self._block_break()
            start = int(_attr(attrs, "start") or "1")
            self._list_stack.append(("ol", start - 1))
            return

        if tag == "li":
            text = self._flush_buf()
            if text:
                self._emit(text)
            if self._list_stack:
                kind, n = self._list_stack[-1]
                indent = "  " * (len(self._list_stack) - 1)
                if kind == "ul":
                    self._buf.append(f"{indent}- ")
                else:
                    n += 1
                    self._list_stack[-1] = (kind, n)
                    self._buf.append(f"{indent}{n}. ")
            return

        if tag == "dl":
            self._block_break()
            return
        if tag == "dt":
            self._block_break()
            self._buf.append("**")
            return
        if tag == "dd":
            self._block_break()
            self._buf.append("  ")
            return

        # --- Blockquote ---
        if tag == "blockquote":
            self._block_break()
            self._buf.append("> ")
            return

        # --- Tables ---
        if tag == "table":
            self._block_break()
            self._table_depth += 1
            return

        if tag == "thead":
            self._in_thead = True
            return

        if tag in ("tbody", "tfoot"):
            return

        if tag == "tr":
            self._flush_buf()
            self._in_tr = True
            self._cells = []
            self._is_header_row = self._in_thead
            return

        if tag in ("td", "th"):
            self._in_cell = True
            self._cell_buf = []
            if tag == "th":
                self._is_header_row = True
            return

        # --- Forms ---
        if tag == "form":
            self._block_break()
            fid = self._next("F")
            action = urllib.parse.urljoin(self.base_url, _attr(attrs, "action") or "")
            method = (_attr(attrs, "method") or "GET").upper()
            enctype = _attr(attrs, "enctype") or "application/x-www-form-urlencoded"
            self.forms[fid] = {
                "action": action,
                "method": method,
                "enctype": enctype,
                "fields": {},
            }
            self._cur_form = fid
            self._emit(f"[ FORM {fid}: {method} {action} ]")
            self.elements[fid] = {"type": "form", "action": action, "method": method}
            return

        # --- Input ---
        if tag == "input":
            itype = (_attr(attrs, "type") or "text").lower()
            name = _attr(attrs, "name")
            value = _attr(attrs, "value")
            placeholder = _attr(attrs, "placeholder")
            required = _has_attr(attrs, "required")
            checked = _has_attr(attrs, "checked")
            req = "*" if required else ""

            # Hidden: store in form, don't render
            if itype == "hidden":
                if self._cur_form and name:
                    self.forms[self._cur_form]["fields"][name] = value
                return

            label = self._labels.get(_attr(attrs, "id"), "")
            hint = label or placeholder or name

            # Submit / button / reset / image → B prefix
            if itype in ("submit", "button", "reset", "image"):
                bid = self._next("B")
                display = value or hint or itype.capitalize()
                self.elements[bid] = {
                    "type": "button",
                    "input_type": itype,
                    "name": name,
                    "value": value,
                    "form": self._cur_form,
                }
                self._flush_buf()
                self._emit(f"  [{bid}] {display}")
                return

            # Checkbox
            if itype == "checkbox":
                iid = self._next("I")
                box = "[x]" if checked else "[ ]"
                self.elements[iid] = {
                    "type": "input", "input_type": itype,
                    "name": name, "value": "on" if checked else "", "form": self._cur_form,
                }
                if self._cur_form and name:
                    self.forms[self._cur_form]["fields"][name] = "on" if checked else ""
                self._flush_buf()
                self._emit(f"  [{iid}] {box} {hint}{req}")
                return

            # Radio
            if itype == "radio":
                iid = self._next("I")
                dot = "(o)" if checked else "( )"
                self.elements[iid] = {
                    "type": "input", "input_type": itype,
                    "name": name, "value": value, "form": self._cur_form,
                }
                self._flush_buf()
                self._emit(f"  [{iid}] {dot} {hint} = {value}")
                return

            # Regular text-like input
            iid = self._next("I")
            display_val = value or "________"
            self.elements[iid] = {
                "type": "input", "input_type": itype,
                "name": name, "value": value, "form": self._cur_form,
            }
            if self._cur_form and name:
                self.forms[self._cur_form]["fields"][name] = value
            self._flush_buf()
            self._emit(f"  [{iid}] {display_val} ({itype}: {hint}){req}")
            return

        # --- Textarea ---
        if tag == "textarea":
            tid = self._next("T")
            name = _attr(attrs, "name")
            placeholder = _attr(attrs, "placeholder")
            required = _has_attr(attrs, "required")
            req = "*" if required else ""
            hint = placeholder or name
            self.elements[tid] = {
                "type": "textarea", "name": name, "value": "", "form": self._cur_form,
            }
            if self._cur_form and name:
                self.forms[self._cur_form]["fields"][name] = ""
            self._cur_textarea = tid
            self._flush_buf()
            self._emit(f"  [{tid}] ________ (textarea: {hint}){req}")
            return

        # --- Select ---
        if tag == "select":
            sid = self._next("S")
            name = _attr(attrs, "name")
            label = self._labels.get(_attr(attrs, "id"), "")
            hint = label or name
            self.elements[sid] = {
                "type": "select", "name": name, "value": "", "form": self._cur_form,
                "options": [],
            }
            if self._cur_form and name:
                self.forms[self._cur_form]["fields"][name] = ""
            self._cur_select = sid
            self._cur_select_name = name
            self._select_opts = []
            self._flush_buf()
            self._emit(f"  [{sid}] Select: {hint}")
            return

        # --- Option ---
        if tag == "option" and self._cur_select:
            self._in_opt = True
            self._opt_value = _attr(attrs, "value")
            self._opt_selected = _has_attr(attrs, "selected")
            self._opt_buf = []
            return

        # --- Button element ---
        if tag == "button":
            btype = (_attr(attrs, "type") or "submit").lower()
            name = _attr(attrs, "name")
            value = _attr(attrs, "value")
            bid = self._next("B")
            self.elements[bid] = {
                "type": "button", "button_type": btype,
                "name": name, "value": value, "form": self._cur_form,
            }
            self._flush_buf()
            # Text content will be captured in handle_data; we mark start here
            self._buf.append(f"\x00BTN:{bid}\x00")
            return

        # Span / generic inline — ignore tag, pass text through
        # (div already handled above as block)

    def handle_endtag(self, tag):
        if tag == "head":
            self._head_depth = max(0, self._head_depth - 1)
            return

        if tag in SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth > 0:
            return

        if tag == "title":
            self._in_title = False
            return

        if tag == "label":
            self._in_label = False
            label_text = "".join(self._label_buf).strip()
            if self._label_for and label_text:
                self._labels[self._label_for] = label_text
            return

        # Headings
        m = re.match(r"^h([1-6])$", tag)
        if m:
            text = self._flush_buf()
            if text:
                level = int(m.group(1))
                self._emit("#" * level + " " + text)
            self._heading = 0
            self._emit("")
            return

        # Block layout
        if tag in (
            "p", "div", "section", "article", "aside", "main",
            "header", "footer", "nav", "figure", "figcaption",
            "details", "summary", "address", "fieldset",
        ):
            self._block_break()
            return

        if tag == "legend":
            self._buf.append("**")
            self._block_break()
            return

        if tag == "pre":
            text = self._flush_buf()
            if text:
                self._emit(text)
            self._emit("```")
            self._pre = max(0, self._pre - 1)
            self._emit("")
            return

        if tag == "code":
            self._code_inline = max(0, self._code_inline - 1)
            if self._pre == 0:
                self._buf.append("`")
            return

        if tag in ("strong", "b"):
            self._buf.append("**")
            self._strong = max(0, self._strong - 1)
            return

        if tag in ("em", "i"):
            self._buf.append("_")
            self._em = max(0, self._em - 1)
            return

        if tag in ("del", "s", "strike"):
            self._buf.append("~~")
            self._del = max(0, self._del - 1)
            return

        if tag == "a":
            if self._links:
                href, lid = self._links.pop()
                if lid:
                    # Annotate: append [Ln] after the link text
                    self._buf.append(f" [{lid}]")
                    # Update element text
                    inline = self._inline_text()
                    if lid in self.elements:
                        self.elements[lid]["text"] = inline.strip()
            return

        if tag in ("ul", "ol"):
            if self._list_stack:
                self._list_stack.pop()
            self._block_break()
            return

        if tag == "li":
            text = self._flush_buf()
            if text:
                self._emit(text)
            return

        if tag == "dl":
            self._block_break()
            return

        if tag == "dt":
            self._buf.append("**")
            self._block_break()
            return

        if tag == "dd":
            self._block_break()
            return

        if tag == "blockquote":
            self._block_break()
            return

        # Table
        if tag == "table":
            self._table_depth = max(0, self._table_depth - 1)
            self._emit("")
            return

        if tag == "thead":
            self._in_thead = False
            return

        if tag in ("td", "th"):
            self._in_cell = False
            cell_text = "".join(self._cell_buf).strip()
            self._cells.append(cell_text)
            self._cell_buf = []
            return

        if tag == "tr":
            if self._cells:
                row = " | ".join(self._cells)
                self._emit(f"| {row} |")
                if self._is_header_row:
                    sep = " | ".join("---" for _ in self._cells)
                    self._emit(f"| {sep} |")
                    self._is_header_row = False
            self._cells = []
            self._in_tr = False
            return

        # Form
        if tag == "form":
            self._block_break()
            if self._cur_form:
                self._emit(f"[ END FORM {self._cur_form} ]")
                self._emit("")
            self._cur_form = None
            return

        # Select
        if tag == "select":
            if self._cur_select:
                sid = self._cur_select
                # Store options in element
                self.elements[sid]["options"] = self._select_opts
                # Build display: "opt1 | [opt2*] | opt3"
                parts = []
                selected_val = ""
                for v, lbl, sel in self._select_opts:
                    if sel:
                        parts.append(f"[{lbl}*]")
                        selected_val = v
                    else:
                        parts.append(lbl)
                opts_display = " | ".join(parts) if parts else "(empty)"
                # Replace the placeholder line with full option list
                # Find the last emitted line for this select and update it
                for i in range(len(self._lines) - 1, -1, -1):
                    if f"[{sid}] Select:" in self._lines[i]:
                        self._lines[i] += f"  ({opts_display})"
                        break
                # Set default value
                if self._cur_form and self._cur_select_name:
                    self.forms[self._cur_form]["fields"][self._cur_select_name] = selected_val
                    self.elements[sid]["value"] = selected_val
            self._cur_select = None
            self._select_opts = []
            return

        if tag == "option":
            if self._in_opt and self._cur_select:
                label = "".join(self._opt_buf).strip()
                value = self._opt_value if self._opt_value else label
                self._select_opts.append((value, label, self._opt_selected))
            self._in_opt = False
            self._opt_buf = []
            return

        if tag == "textarea":
            self._cur_textarea = None
            return

        if tag == "button":
            # Extract button text from buffer, replacing the marker
            raw = "".join(self._buf)
            m = re.search(r"\x00BTN:(\w+)\x00(.*)", raw, re.DOTALL)
            if m:
                bid = m.group(1)
                btn_text = m.group(2).strip()
                self._buf = list(raw[: raw.index("\x00BTN:")])
                display = btn_text or self.elements.get(bid, {}).get("value", "Button")
                self._emit(f"  [{bid}] {display}")
            return

    def handle_data(self, data):
        if self._skip_depth > 0:
            return

        if self._in_title:
            self.title += data
            return

        if self._in_label:
            self._label_buf.append(data)
            return

        if self._in_opt:
            self._opt_buf.append(data)
            return

        if self._in_cell:
            self._cell_buf.append(data)
            return

        if self._pre > 0:
            self._buf.append(data)
            return

        # Collapse whitespace for normal text
        text = re.sub(r"[\t\r\n ]+", " ", data)
        self._buf.append(text)

    def get_output(self) -> str:
        # Flush any remaining buffer
        remaining = self._flush_buf()
        if remaining:
            self._emit(remaining)

        # Clean up: remove leading/trailing blanks, collapse runs of blanks
        lines = self._lines
        result = []
        prev_blank = True
        for line in lines:
            is_blank = line.strip() == ""
            if is_blank and prev_blank:
                continue
            result.append(line)
            prev_blank = is_blank

        return "\n".join(result).strip()


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def load_session() -> dict:
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "url": None,
        "html": None,
        "title": None,
        "cookies": {},
        "history": [],
        "elements": {},
        "forms": {},
        "rendered": None,
    }


def save_session(session: dict):
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def cookies_to_dict(jar) -> dict:
    """Serialize a RequestsCookieJar to a plain dict, keeping last value on name collision."""
    result = {}
    for cookie in jar:
        result[cookie.name] = cookie.value
    return result


def make_http_session(cookies: dict) -> requests.Session:
    s = requests.Session()
    s.headers.update(STEALTH_HEADERS)
    s.cookies.update(cookies)
    return s


def fetch(url: str, session_data: dict, method="GET", data=None, files=None) -> requests.Response:
    http = make_http_session(session_data.get("cookies", {}))
    referer = session_data.get("url")
    if referer:
        http.headers["Referer"] = referer
        http.headers["Sec-Fetch-Site"] = "same-origin"

    if method == "POST":
        resp = http.post(url, data=data, files=files, allow_redirects=True, timeout=20, verify=_VERIFY)
    else:
        resp = http.get(url, allow_redirects=True, timeout=20, verify=_VERIFY)

    resp.raise_for_status()
    return resp, http


def get_html(resp: requests.Response) -> str:
    """Extract HTML text from response, handling encoding correctly."""
    # Check content type — skip non-HTML
    ct = resp.headers.get("Content-Type", "")
    if ct and "html" not in ct and "xml" not in ct and "text" not in ct:
        return f"<html><body><pre>[Binary content: {ct}]</pre></body></html>"

    # requests guesses latin-1 for HTTP/1.0 responses without charset header.
    # Sniff real charset from HTML meta tags in that case.
    enc = resp.encoding or "utf-8"
    if enc.lower() in ("iso-8859-1", "latin-1", "windows-1252"):
        m = re.search(
            rb'charset=["\']?\s*([A-Za-z0-9_-]+)',
            resp.content[:4096],
            re.IGNORECASE,
        )
        if m:
            enc = m.group(1).decode("ascii", errors="replace")
        else:
            enc = "utf-8"

    resp.encoding = enc
    return resp.text


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def render_html(html: str, url: str) -> tuple[str, dict, dict, str]:
    """Render HTML to markdown. Returns (markdown, elements, forms, title)."""
    r = Renderer(url)
    r.feed(html)
    return r.get_output(), r.elements, r.forms, r.title.strip()


def format_output(url: str, title: str, markdown: str, elements: dict, forms: dict) -> str:
    header = f"[PAGE: {url}"
    if title:
        header += f" | {title}"
    header += "]"

    # Element summary
    summary_lines = ["---", "Elements on this page:"]
    for eid, el in elements.items():
        etype = el.get("type", "?")
        if etype == "link":
            summary_lines.append(f"  {eid}  {el.get('href', '')}")
        elif etype == "form":
            summary_lines.append(f"  {eid}  {el.get('method','')} {el.get('action','')}")
        elif etype == "input":
            summary_lines.append(
                f"  {eid}  ({el.get('input_type','text')}) name={el.get('name','')}"
                + (f"  value={el['value']!r}" if el.get("value") else "")
            )
        elif etype == "select":
            opts = [lbl for _, lbl, _ in el.get("options", [])]
            summary_lines.append(f"  {eid}  select name={el.get('name','')}  [{', '.join(opts[:5])}]")
        elif etype == "textarea":
            summary_lines.append(f"  {eid}  textarea name={el.get('name','')}")
        elif etype == "button":
            summary_lines.append(
                f"  {eid}  button  form={el.get('form','')}"
            )

    return "\n\n".join([header, markdown, "\n".join(summary_lines)])


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_open(url: str):
    session = load_session()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        resp, http = fetch(url, session)
    except requests.RequestException as e:
        sys.exit(f"Connection error: {e}")

    final_url = resp.url
    html = get_html(resp)
    cookies = cookies_to_dict(http.cookies)

    # Save history
    history = session.get("history", [])
    if session.get("url"):
        history.append(session["url"])
    history = history[-10:]

    md, elements, forms, title = render_html(html, final_url)

    session.update({
        "url": final_url,
        "html": html,
        "title": title,
        "cookies": cookies,
        "history": history,
        "elements": elements,
        "forms": forms,
        "rendered": md,
    })
    save_session(session)
    print(format_output(final_url, title, md, elements, forms))


def cmd_show():
    session = load_session()
    if not session.get("url"):
        sys.exit("No active page. Use: browser.py open <url>")
    html = session.get("html", "")
    md, elements, forms, title = render_html(html, session["url"])
    print(format_output(session["url"], title, md, elements, forms))


def cmd_back():
    session = load_session()
    history = session.get("history", [])
    if not history:
        sys.exit("No history to go back to.")
    url = history.pop()
    session["history"] = history
    save_session(session)
    # Re-use open logic on the previous URL
    cmd_open_with_session(url, session)


def cmd_click(eid: str):
    session = load_session()
    if not session.get("url"):
        sys.exit("No active page. Use: browser.py open <url>")

    elements = session.get("elements", {})
    forms = session.get("forms", {})

    if eid not in elements:
        sys.exit(f"No element [{eid}] on current page.")

    el = elements[eid]
    etype = el.get("type")

    if etype == "link":
        cmd_open(el["href"])
        return

    if etype == "button":
        form_id = el.get("form")
        if form_id and form_id in forms:
            _submit_form(form_id, forms[form_id], session)
            return
        # Button without a form — try href if available
        href = el.get("href")
        if href:
            cmd_open(href)
        else:
            sys.exit(f"Button [{eid}] has no associated form or link.")
        return

    sys.exit(f"Element [{eid}] is not clickable (type={etype}).")


def cmd_fill(eid: str, value: str):
    session = load_session()
    elements = session.get("elements", {})
    forms = session.get("forms", {})

    if eid not in elements:
        sys.exit(f"No element [{eid}] on current page.")

    el = elements[eid]
    etype = el.get("type")

    if etype not in ("input", "textarea", "select"):
        sys.exit(f"Element [{eid}] is not fillable (type={etype}).")

    # Update value in elements
    session["elements"][eid]["value"] = value

    # Update in form fields
    name = el.get("name")
    form_id = el.get("form")
    if form_id and name and form_id in forms:
        session["forms"][form_id]["fields"][name] = value

    save_session(session)
    print(f"Set [{eid}] = {value!r}")

    # Show current form state if there is one
    if form_id and form_id in forms:
        f = forms[form_id]
        print(f"\nForm {form_id} fields:")
        for k, v in f["fields"].items():
            print(f"  {k} = {v!r}")


def cmd_submit(form_id: str):
    session = load_session()
    forms = session.get("forms", {})

    if form_id not in forms:
        sys.exit(f"No form [{form_id}] on current page.")

    _submit_form(form_id, forms[form_id], session)


def _submit_form(form_id: str, form: dict, session: dict):
    action = form.get("action") or session.get("url", "")
    method = form.get("method", "GET").upper()
    fields = form.get("fields", {})
    # Filter out empty hidden fields but keep filled ones
    data = {k: v for k, v in fields.items() if v is not None}

    if method == "GET":
        # Append as query string
        parts = urllib.parse.urlsplit(action)
        qs = urllib.parse.urlencode(data)
        new_url = urllib.parse.urlunsplit(
            (parts.scheme, parts.netloc, parts.path, qs, "")
        )
        cmd_open(new_url)
    else:
        # POST
        try:
            resp, http = fetch(action, session, method="POST", data=data)
        except requests.RequestException as e:
            sys.exit(f"Connection error: {e}")

        final_url = resp.url
        html = get_html(resp)
        cookies = cookies_to_dict(http.cookies)

        history = session.get("history", [])
        if session.get("url"):
            history.append(session["url"])
        history = history[-10:]

        md, elements, forms_new, title = render_html(html, final_url)
        session.update({
            "url": final_url,
            "html": html,
            "title": title,
            "cookies": cookies,
            "history": history,
            "elements": elements,
            "forms": forms_new,
            "rendered": md,
        })
        save_session(session)
        print(format_output(final_url, title, md, elements, forms_new))


def cmd_open_with_session(url: str, session: dict):
    """Internal: open URL re-using existing session (for back navigation)."""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        resp, http = fetch(url, session)
    except requests.RequestException as e:
        sys.exit(f"Connection error: {e}")

    final_url = resp.url
    html = get_html(resp)
    cookies = cookies_to_dict(http.cookies)

    md, elements, forms, title = render_html(html, final_url)
    session.update({
        "url": final_url,
        "html": html,
        "title": title,
        "cookies": cookies,
        "elements": elements,
        "forms": forms,
        "rendered": md,
    })
    save_session(session)
    print(format_output(final_url, title, md, elements, forms))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    # Handle single-argument form: browser.py "open https://example.com"
    # (happens when the LLM passes the whole command as one quoted string)
    if len(args) == 1 and " " in args[0]:
        args = args[0].split(None, 3)

    # Auto-prepend "open" if the first arg looks like a URL but not a command
    known_cmds = {"open", "show", "back", "click", "fill", "submit"}
    if args and args[0].lower() not in known_cmds:
        first = args[0]
        if first.startswith(("http://", "https://")) or (
            "." in first and not first.startswith("-")
        ):
            args = ["open"] + args



    cmd = args[0].lower()

    if cmd == "open":
        if len(args) < 2:
            sys.exit("Usage: browser.py open <url>")
        cmd_open(args[1])

    elif cmd == "show":
        cmd_show()

    elif cmd == "back":
        cmd_back()

    elif cmd == "click":
        if len(args) < 2:
            sys.exit("Usage: browser.py click <ID>")
        eid = args[1].upper().lstrip("[").rstrip("]")
        cmd_click(eid)

    elif cmd == "fill":
        if len(args) < 3:
            sys.exit("Usage: browser.py fill <ID> <value>")
        eid = args[1].upper().lstrip("[").rstrip("]")
        value = " ".join(args[2:])
        cmd_fill(eid, value)

    elif cmd == "submit":
        if len(args) < 2:
            sys.exit("Usage: browser.py submit <FORM_ID>")
        fid = args[1].upper().lstrip("[").rstrip("]")
        cmd_submit(fid)

    else:
        sys.exit(f"Unknown command: {cmd!r}\nRun without arguments for usage.")


if __name__ == "__main__":
    main()
