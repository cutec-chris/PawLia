"""Opencode daemon: long-lived ``opencode serve`` process + session cache.

Why a daemon at all
-------------------
The previous design spawned a fresh ``opencode run`` subprocess per
``creator.py implement|fix`` call (coding.py:_run_opencode). That works for
one-shot tasks but breaks the "addressable skill" property the rest of the
system promises: every follow-up question ("und was ist mit X?", "schick den
diff") starts a brand-new agent with no memory of the previous run, so the
model has to re-read every file from scratch.

The fix is a persistent opencode server per skill path. The server keeps an
OpenAPI 3.1 endpoint at ``http://127.0.0.1:<port>`` and lets us create sessions
and stream messages into them over HTTP. A single opencode session is a full
agent loop with tool calls, file edits, and reasoning — exactly what the
"sub-agent LLM" the SkillRunner used to spin up was, but durable.

Lifecycle
---------
- One daemon per (project_dir, skill_path) tuple, kept in a class-level cache.
- Started on first use; the subprocess is reaped automatically when the
  PawLia process exits (we register an ``atexit`` hook as a safety net).
- Health is re-checked on every call; if the server died or the port is wrong
  we restart once and retry.
- Sessions are keyed by ``(skill_path, user_id)`` and reused within a TTL
  (default 30 min) so a follow-up question continues the same conversation.

Auth
----
If ``OPENCODE_SERVER_PASSWORD`` is set, the daemon is started with that env
var and every request carries HTTP basic auth. The username is
``OPENCODE_SERVER_USERNAME`` (default ``opencode``).

Concurrency
-----------
``start()`` and the per-send lock make it safe to call from multiple
asyncio tasks in the same process. We never call into a half-started
server — if startup is in progress, senders wait on an event.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from base64 import b64encode
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Per-task timeouts. The synchronous /message endpoint blocks until the agent
# finishes the turn; large file edits routinely take 60–120 s on a small
# model, plus the 5 s grace for slow providers.
_DEFAULT_SEND_TIMEOUT = 600
_HEALTH_TIMEOUT = 5
_STARTUP_GRACE = 10.0     # s to wait for the port to come up
_SESSION_TTL = 30 * 60    # re-use a session for 30 min after last use


@dataclass
class _Session:
    id: str
    title: str
    last_used: float
    turns: int = 0


@dataclass
class OpencodeDaemon:
    """A long-lived ``opencode serve`` instance scoped to a skill directory.

    Attributes
    ----------
    skill_path:
        Absolute path to the skill the daemon is working on. The daemon's
        ``cwd`` is the parent (so file edits land in the right place) and
        every created session is "pinned" to it via opencode's own project
        detection.
    base_url:
        ``http://host:port`` of the running server.
    proc:
        The subprocess.Popen handle; ``None`` when we are reusing an external
        server (e.g. OPENCODE_SERVER_URL is set).
    sessions:
        Map of ``(user_id) → _Session`` so the same agent loop continues
        across SkillRunner invocations for that user.
    """

    skill_path: str
    base_url: str
    proc: Optional[subprocess.Popen] = None
    sessions: Dict[str, _Session] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _session_ttl: float = _SESSION_TTL

    # -- HTTP plumbing -----------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        timeout: float = _DEFAULT_SEND_TIMEOUT,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        data: Optional[bytes] = None
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        pw = os.environ.get("OPENCODE_SERVER_PASSWORD")
        if pw:
            user = os.environ.get("OPENCODE_SERVER_USERNAME", "opencode")
            token = b64encode(f"{user}:{pw}".encode("utf-8")).decode("ascii")
            headers["Authorization"] = f"Basic {token}"

        if body is not None:
            data = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"opencode {method} {path} -> HTTP {exc.code}: {err_body[:500]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"opencode {method} {path} -> {exc.reason}"
            ) from exc

        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8", errors="replace"))
        except ValueError as exc:
            raise RuntimeError(
                f"opencode {method} {path}: invalid JSON in response"
            ) from exc

    def health(self) -> bool:
        try:
            r = self._request("GET", "/global/health", timeout=_HEALTH_TIMEOUT)
            return bool(r.get("healthy"))
        except Exception as exc:
            logger.debug("opencode health check failed: %s", exc)
            return False

    # -- Session management ------------------------------------------------

    def get_or_create_session(
        self, user_id: str, title: str = "pawlia-task"
    ) -> Tuple[str, bool]:
        """Return ``(session_id, was_created)``.

        Reuses a session whose ``last_used`` is within ``_session_ttl``;
        otherwise creates a fresh one. ``was_created`` lets the caller add
        extra context (e.g. SKILL.md body) on the first message of a
        session without paying that cost on every follow-up.
        """
        key = user_id or "_anon"
        now = time.time()
        with self._lock:
            existing = self.sessions.get(key)
            if existing and (now - existing.last_used) < self._session_ttl:
                existing.last_used = now
                return existing.id, False
            if existing:
                # Expired — drop it, opencode keeps the old session around
                # for inspection but we won't continue it.
                self.sessions.pop(key, None)

        created = self._request(
            "POST", "/session", body={"title": title}, timeout=30
        )
        sid = created.get("id")
        if not sid:
            raise RuntimeError(f"opencode create session: no id in {created!r}")
        with self._lock:
            self.sessions[key] = _Session(id=sid, title=title, last_used=now)
        return sid, True

    def touch_session(self, user_id: str) -> None:
        key = user_id or "_anon"
        with self._lock:
            s = self.sessions.get(key)
            if s:
                s.last_used = time.time()
                s.turns += 1

    # -- Send / receive ----------------------------------------------------

    def send_message(
        self,
        session_id: str,
        text: str,
        system: Optional[str] = None,
        agent: Optional[str] = None,
        model: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Send a user turn and wait for the assistant response.

        Returns the full response envelope (info + parts). The caller is
        responsible for picking out the bits it cares about — the daemon
        does not interpret the model output.
        """
        parts: List[Dict[str, Any]] = [{"type": "text", "text": text}]
        body: Dict[str, Any] = {"parts": parts}
        if system:
            body["system"] = system
        if agent:
            body["agent"] = agent
        if model:
            body["model"] = model

        return self._request(
            "POST",
            f"/session/{session_id}/message",
            body=body,
            timeout=_DEFAULT_SEND_TIMEOUT,
        )

    def abort_session(self, session_id: str) -> None:
        """Abort a running turn. Used when the caller times out."""
        try:
            self._request("POST", f"/session/{session_id}/abort", timeout=5)
        except Exception as exc:
            logger.debug("abort_session(%s) failed: %s", session_id, exc)

    def shutdown(self) -> None:
        if not self.proc:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None


# ---------------------------------------------------------------------------
# Module-level registry: one daemon per (project_dir, skill_path) pair.
# ---------------------------------------------------------------------------

_REGISTRY: Dict[Tuple[str, str], OpencodeDaemon] = {}
_REGISTRY_LOCK = threading.Lock()
_REGISTERED_ATEXIT = False


def _pick_port(preferred: Optional[int]) -> int:
    """Pick a port for ``opencode serve`` to bind to.

    Honours the explicit override from config / env when given; otherwise
    asks the kernel for a free port (port 0 + bind + read assigned port).
    Returning 0 would be the most general choice but opencode insists on
    a concrete number at start, so we resolve it ourselves.
    """
    if preferred:
        return int(preferred)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(base_url: str, deadline: float) -> bool:
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"{base_url}/global/health", timeout=_HEALTH_TIMEOUT
            ) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
                if payload.get("healthy"):
                    return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


def _start_daemon(skill_path: str, port: int) -> OpencodeDaemon:
    """Spawn a fresh ``opencode serve`` and wait until it answers /health."""
    binary = shutil.which("opencode")
    if not binary:
        raise RuntimeError(
            "opencode CLI not found on PATH — install with `npm i -g opencode-ai`"
        )

    env = os.environ.copy()
    pw = os.environ.get("OPENCODE_SERVER_PASSWORD")
    if pw:
        env["OPENCODE_SERVER_PASSWORD"] = pw

    log_path = Path(skill_path) / ".opencode-serve.log"
    log_fh = open(log_path, "ab")
    proc = subprocess.Popen(
        [binary, "serve", "--port", str(port), "--hostname", "127.0.0.1"],
        cwd=skill_path,
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # don't kill us when PawLia dies
    )

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + _STARTUP_GRACE
    if not _wait_for_health(base_url, deadline):
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
        raise RuntimeError(
            f"opencode serve on port {port} did not become healthy within "
            f"{int(_STARTUP_GRACE)}s — see {log_path}"
        )

    daemon = OpencodeDaemon(
        skill_path=skill_path,
        base_url=base_url,
        proc=proc,
    )
    logger.info("opencode daemon ready: %s (skill=%s)", base_url, skill_path)
    return daemon


def get_daemon(
    skill_path: str,
    *,
    user_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> OpencodeDaemon:
    """Return the daemon for ``skill_path``, starting it on first use.

    Configuration lookup order (first non-empty wins):
    1. ``config["coding"]["opencode_daemon"]`` dict (host, port, password, ttl)
    2. Environment: ``OPENCODE_SERVER_URL``, ``OPENCODE_SERVER_PORT``,
       ``OPENCODE_SERVER_PASSWORD``, ``OPENCODE_SERVER_USERNAME``
    3. Auto: spawn a fresh ``opencode serve`` on a random port.

    When ``OPENCODE_SERVER_URL`` is set the daemon runs in "remote" mode —
    it only talks to that URL, never spawns a subprocess. That lets ops
    point PawLia at a shared opencode instance instead of one-per-process.
    """
    cfg = (config or {}).get("coding", {}).get("opencode_daemon") or {}
    external_url = (
        os.environ.get("OPENCODE_SERVER_URL")
        or cfg.get("url")
    )
    preferred_port = (
        int(os.environ.get("OPENCODE_SERVER_PORT", 0)) or cfg.get("port")
    )

    if external_url:
        daemon = OpencodeDaemon(skill_path=skill_path, base_url=external_url.rstrip("/"))
        if not daemon.health():
            raise RuntimeError(
                f"External opencode server at {external_url} is not healthy"
            )
        return daemon

    key = (skill_path, user_id or "")
    with _REGISTRY_LOCK:
        existing = _REGISTRY.get(key)
        if existing and existing.health():
            return existing
        if existing:
            # Died — clean up before restarting.
            existing.shutdown()
            _REGISTRY.pop(key, None)

        port = _pick_port(preferred_port)
        daemon = _start_daemon(skill_path, port)
        _REGISTRY[key] = daemon

    _ensure_atexit()
    return daemon


def _ensure_atexit() -> None:
    global _REGISTERED_ATEXIT
    if _REGISTERED_ATEXIT:
        return
    _REGISTERED_ATEXIT = True

    def _shutdown_all() -> None:
        with _REGISTRY_LOCK:
            daemons = list(_REGISTRY.values())
            _REGISTRY.clear()
        for d in daemons:
            try:
                d.shutdown()
            except Exception:
                pass

    atexit.register(_shutdown_all)


# ---------------------------------------------------------------------------
# High-level helpers used by coding.py
# ---------------------------------------------------------------------------


_TEXT_PART_RE = re.compile(r"<text>\s*(.+?)\s*</text>", re.DOTALL)
_TOOL_FILE_KEYS = ("filePath", "path", "filepath", "file_path")


def extract_assistant_text(response: Dict[str, Any]) -> str:
    """Concatenate every ``text`` part of the assistant message."""
    out: List[str] = []
    for part in (response.get("parts") or []):
        if part.get("type") == "text":
            txt = part.get("text")
            if isinstance(txt, str) and txt.strip():
                out.append(txt)
    return "\n".join(out).strip()


def extract_edited_files(response: Dict[str, Any]) -> List[str]:
    """Collect every file path mentioned in any ``tool``/``tool_use`` part."""
    out: List[str] = []
    seen = set()
    for part in (response.get("parts") or []):
        if part.get("type") not in ("tool", "tool_use"):
            continue
        inp = part.get("input") or {}
        for key in _TOOL_FILE_KEYS:
            val = inp.get(key)
            if isinstance(val, str) and val and val not in seen:
                seen.add(val)
                out.append(val)
                break
    return out


def run_task(
    skill_path: str,
    task_prompt: str,
    *,
    user_id: str = "_anon",
    config: Optional[Dict[str, Any]] = None,
    session_title: str = "pawlia-task",
    system: Optional[str] = None,
    agent: Optional[str] = None,
    follow_up: Optional[str] = None,
) -> Dict[str, Any]:
    """One-shot wrapper used by ``coding._run_opencode``.

    Creates (or reuses) a session, sends ``task_prompt`` (and the optional
    ``follow_up`` continuation), and returns a result dict shaped like the
    other backends (``ok``, ``backend``, ``output``, ``files_modified``).

    The session lives on after the call so the next run_task() against the
    same ``user_id`` continues the conversation. ``follow_up`` is for the
    case where the caller already has a pending question for the same
    session — usually unused.
    """
    skill_abs = os.path.abspath(skill_path)
    daemon = get_daemon(skill_abs, user_id=user_id, config=config)

    session_id, _created = daemon.get_or_create_session(user_id, title=session_title)
    response = daemon.send_message(session_id, task_prompt, system=system, agent=agent)
    daemon.touch_session(user_id)

    if follow_up:
        response2 = daemon.send_message(session_id, follow_up, system=system, agent=agent)
        daemon.touch_session(user_id)
        text2 = extract_assistant_text(response2)
        text1 = extract_assistant_text(response)
        text = (text1 + "\n" + text2).strip() if text2 else text1
        edited = sorted(set(extract_edited_files(response)) | set(extract_edited_files(response2)))
    else:
        text = extract_assistant_text(response)
        edited = extract_edited_files(response)

    return {
        "ok": True,
        "backend": "opencode-daemon",
        "output": text[-3000:] if text else "",
        "files_modified": edited,
        "session_id": session_id,
    }
