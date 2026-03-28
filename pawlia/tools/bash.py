"""Bash tool - executes shell commands."""

import os
import re
import subprocess
import sys
from typing import Any, Dict, Optional

from pawlia.tools.base import Tool


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

    def execute(self, args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        cmd = args.get("command", "")
        if not cmd:
            return "Error: No command provided."

        cwd = context.get("cwd") if context else None

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
                return f"Error (exit {r.returncode}): {err or out}"
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

        for shell in shells:
            try:
                if shell is None:
                    return _fmt(subprocess.run(cmd, shell=True, **run_kwargs))
                return _fmt(subprocess.run(shell, **run_kwargs))
            except FileNotFoundError:
                continue
            except subprocess.TimeoutExpired:
                return f"Error: Command timed out ({timeout}s)"
            except Exception as e:
                return f"Error: {e}"

        return "Error: No shell available."
