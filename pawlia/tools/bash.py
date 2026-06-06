"""Bash tool - executes shell commands."""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional

from pawlia.tools.base import Tool

logger = logging.getLogger(__name__)

# Emitted once if a skill command runs without the filesystem sandbox because
# bubblewrap is unavailable, so the missing protection is visible in the logs.
_sandbox_warned = False


# Tools commonly missing in the PawLia container (alpine + busybox) plus
# GNU-grep flags that BusyBox grep doesn't support. Surfaced as a hint to
# the LLM so it can recover in one step instead of trying every variant.
_MISSING_TOOL_HINTS = {
    "curl": (
        "curl is not installed in this sandbox. "
        "Use 'wget <url> -O <file>' or python -c "
        "'import urllib.request; urllib.request.urlretrieve(\"<url>\", \"<file>\")'."
    ),
    "wget": (
        "wget is not installed in this sandbox. "
        "Use python -c 'import urllib.request; urllib.request.urlretrieve(...)'."
    ),
    "bash": (
        "bash is not installed; only /bin/sh (busybox ash) is available. "
        "Avoid bashisms like '[[ ]]', arrays, '{a,b}' brace expansion, "
        "'source', process substitution. Use POSIX sh syntax."
    ),
    "ssh-keygen": (
        "ssh-keygen is not installed. Use python to generate keys, e.g. "
        "python -c 'from cryptography.hazmat.primitives.asymmetric import rsa; ...'."
    ),
    "tar": (
        "tar is missing or limited in busybox. Use python's tarfile module instead."
    ),
}

_GNU_GREP_HINT = (
    "grep in this sandbox is BusyBox and does not support GNU-only flags "
    "(--include, --exclude, --exclude-dir, -P/--perl-regexp). "
    "Use 'find <dir> -name <glob> -exec grep -E <pattern> {{}} +' to filter "
    "paths, or python -c 'import re; ...' for PCRE / multiline / lookbehind."
)


def _command_uses_gnu_grep(cmd: str) -> bool:
    """True if the command passes GNU-grep-only flags to grep."""
    for flag in ("--include", "--exclude", "--exclude-dir",
                 "--perl-regexp"):
        if re.search(rf"(?:^|[\s|;&<>])grep[^\n|;&]*\s{re.escape(flag)}\b", cmd):
            return True
    # -P is short, only match when preceded by whitespace or at start.
    if re.search(r"(?:^|[\s|;&<>])grep[^\n|;&]*\s-P\b", cmd):
        return True
    return False


def _missing_tool_hint(error_msg: str) -> Optional[str]:
    """If the error indicates a missing executable, return a hint; else None.

    Catches patterns like 'sh: curl: not found', 'bash: foo: command not
    found', and the German busybox variant '...: Kommando nicht gefunden'.
    The inner tool name (group 2) is preferred over the shell prefix
    (group 1) so 'sh: curl: not found' resolves to the curl hint, not sh.
    """
    if not error_msg:
        return None
    m = re.search(
        r"(?:^|[\s;|])([A-Za-z_][\w.-]*): (?:([A-Za-z_][\w.-]*): )?"
        r"(?:not found|command not found|Kommando nicht gefunden)\b",
        error_msg,
    )
    if m:
        tool = m.group(2) or m.group(1)
        return _MISSING_TOOL_HINTS.get(tool)
    return None


class BashTool(Tool):
    name = "bash"
    description = "Execute a shell command or script. Use to run skill scripts from the scripts/ directory."

    def parameters(self) -> Dict[str, Any]:
        return {
            "command": {
                "type": "string",
                "description": "Shell command to execute",
                "minLength": 1,
            }
        }

    def required_parameters(self) -> list[str]:
        return ["command"]

    def normalize_args(self, args: Any) -> Dict[str, Any]:
        normalized = super().normalize_args(args)
        if "command" in normalized:
            return normalized
        for alias in ("cmd", "script"):
            value = normalized.get(alias)
            if isinstance(value, str):
                normalized["command"] = value
                break
        return normalized

    @staticmethod
    def _validate_cwd(cwd: Optional[str], context: Optional[Dict[str, Any]]) -> Optional[str]:
        """Ensure cwd stays within session_dir or project tree."""
        if not cwd:
            return None  # empty/None → no restriction; subprocess uses process cwd
        real_cwd = os.path.realpath(cwd)
        # __file__ is pawlia/tools/bash.py → go up 3 levels to project root
        pkg_dir = os.path.realpath(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
        )
        allowed = [pkg_dir]
        if context and context.get("session_dir"):
            allowed.append(os.path.realpath(context["session_dir"]))
        if any(real_cwd.startswith(base + os.sep) or real_cwd == base
               for base in allowed):
            return cwd
        return None

    def execute(self, args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        cmd = args.get("command", "")
        if not cmd:
            return "Error: No command provided."

        # Pre-check: GNU-grep flags that BusyBox grep does not support. The
        # command would fail with a cryptic "unrecognized option" message,
        # so we surface the limitation upfront.
        if _command_uses_gnu_grep(cmd):
            return _GNU_GREP_HINT

        cwd = context.get("cwd") if context else None
        cwd = self._validate_cwd(cwd, context)

        timeout = context.get("timeout", 120) if context else 120

        # Inject context as environment variables so skill scripts can read
        # them without the LLM having to construct --user-id / --session-dir
        # arguments (prevents hallucination of these values).
        env = os.environ.copy()
        if context:
            if context.get("user_id"):
                env["PAWLIA_USER_ID"] = context["user_id"]
            if context.get("session_dir"):
                env["PAWLIA_SESSION_DIR"] = context["session_dir"]
            if context.get("config_path"):
                env["PAWLIA_CONFIG_PATH"] = context["config_path"]
            if context.get("skill_config"):
                env["PAWLIA_SKILL_CONFIG"] = json.dumps(
                    context["skill_config"], ensure_ascii=False
                )
            # Extra env vars from workflow executor (e.g. multiline content)
            for k, v in context.get("env_extra", {}).items():
                env[k] = v

        run_kwargs: Dict[str, Any] = dict(
            capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
            cwd=cwd,
            env=env,
        )

        def _fmt(r: subprocess.CompletedProcess) -> str:
            out = r.stdout.strip()
            err = r.stderr.strip()
            if r.returncode != 0:
                msg = err or out
                hint = _missing_tool_hint(msg)
                if hint:
                    return f"Error (exit {r.returncode}): {msg}\nHint: {hint}"
                return f"Error (exit {r.returncode}): {msg}"
            return out or "(no output)"

        def _to_powershell(command: str) -> str:
            stripped = command.strip()
            if stripped == "pwd":
                return "(Get-Location).Path"
            if stripped == "true":
                return "exit 0"
            if stripped == "false":
                return "exit 1"
            m = re.fullmatch(r"sleep\s+(\d+)", stripped)
            if m:
                return f"Start-Sleep -Seconds {m.group(1)}"
            m = re.fullmatch(r"echo\s+(.+?)\s+>&2;\s*exit\s+(\d+)", stripped)
            if m:
                text = m.group(1).strip().strip('"').replace("'", "''")
                return f"[Console]::Error.WriteLine('{text}'); exit {m.group(2)}"
            return command

        shells = [["bash", "-c", cmd], ["sh", "-c", cmd]]
        if sys.platform == "win32":
            shells.append(["powershell", "-Command", _to_powershell(cmd)])
            shells.append(None)  # sentinel for cmd.exe fallback

        # Filesystem write-sandbox: skill commands may only write under the
        # per-user session dir or /tmp. When bubblewrap is available we run the
        # command inside it with a read-only root so out-of-bounds writes are
        # rejected by the kernel. Resolve the shell up front (bwrap would mask
        # the bash→sh FileNotFoundError fallback otherwise).
        sandbox_on = bool(
            context
            and context.get("session_dir")
            and context.get("sandbox", True)
        )
        if sandbox_on:
            from pawlia.sandbox import bwrap_available, wrap_argv, writable_roots

            if bwrap_available():
                writable = writable_roots(
                    context.get("session_dir"), context.get("user_id")
                )
                shell_bin = shutil.which("bash") or shutil.which("sh")
                if shell_bin:
                    shells = [wrap_argv([shell_bin, "-c", cmd], writable)]
            else:
                global _sandbox_warned
                if not _sandbox_warned:
                    _sandbox_warned = True
                    logger.warning(
                        "Filesystem sandbox unavailable (bubblewrap missing or "
                        "user namespaces disabled) — skill commands run "
                        "without write isolation."
                    )

        for shell in shells:
            try:
                if shell is None:
                    return _fmt(subprocess.run(cmd, shell=True, **run_kwargs))
                return _fmt(subprocess.run(shell, **run_kwargs))
            except FileNotFoundError as e:
                # A missing executable: try the next shell alternative.
                # A missing cwd that passed validation: report clearly.
                if cwd and e.filename == cwd:
                    return f"Error: Working directory does not exist: {cwd}"
                continue
            except subprocess.TimeoutExpired:
                return f"Error: Command timed out ({timeout}s)"
            except Exception as e:
                return f"Error: {e}"

        return "Error: No shell available."
