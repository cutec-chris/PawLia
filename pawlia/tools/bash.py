"""Bash tool - executes shell commands."""

import os
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
            }
        }

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

        shells = [["bash", "-c", cmd], ["sh", "-c", cmd]]
        if sys.platform == "win32":
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
