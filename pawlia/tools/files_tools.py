"""Direct file tools for ChatAgent — no SkillRunner overhead."""

import json
import os
import subprocess
from typing import Any, Dict, List, Optional

from pawlia.tools.base import Tool


def _find_files_script() -> str:
    """Return the absolute path to skills/files/scripts/files.py."""
    # __file__ is pawlia/tools/files_tools.py
    # Go up 3 levels: pawlia/tools/ -> pawlia/ -> project_root/
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(project_root, "skills", "files", "scripts", "files.py")


def _run_files_subcommand(
    subcommand: str,
    args: Dict[str, Any],
    context: Optional[Dict[str, Any]],
) -> str:
    """Run files.py with the given subcommand and arguments."""
    user_id = context.get("user_id", "default") if context else "default"
    session_dir = context.get("session_dir", ".") if context else "."

    cmd = [
        "python", _find_files_script(),
        subcommand,
        "--user-id", user_id,
        "--session-dir", session_dir,
    ]

    # Add subcommand-specific args
    if subcommand == "read":
        filename = args.get("filename", "")
        if not filename:
            return json.dumps({"success": False, "error": "Missing filename."})
        cmd.extend(["--filename", filename])
        offset = args.get("offset")
        if offset is not None:
            cmd.extend(["--offset", str(offset)])
        limit = args.get("limit")
        if limit is not None:
            cmd.extend(["--limit", str(limit)])
        query = args.get("query")
        if query is not None:
            cmd.extend(["--query", query])

    elif subcommand == "list":
        offset = args.get("offset")
        if offset is not None:
            cmd.extend(["--offset", str(offset)])
        limit = args.get("limit")
        if limit is not None:
            cmd.extend(["--limit", str(limit)])

    elif subcommand == "grep":
        pattern = args.get("pattern", "")
        if not pattern:
            return json.dumps({"success": False, "error": "Missing pattern."})
        cmd.extend(["--pattern", pattern])
        filename = args.get("filename")
        if filename:
            cmd.extend(["--filename", filename])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip() or "unknown error"
            return json.dumps({"success": False, "error": err})
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return json.dumps({"success": False, "error": "Command timed out (30s)"})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


class ReadFileTool(Tool):
    """Read a file from the user's workspace."""

    name = "read_file"
    description = (
        "Read the contents of a file in the workspace. "
        "Useful for inspecting documents, notes, code, or any text file. "
        "Supports line offset and limit for large files."
    )
    trust = "internal"

    def parameters(self) -> Dict[str, Any]:
        return {
            "filename": {
                "type": "string",
                "description": (
                    "Path to the file relative to the workspace root. "
                    "Supports wikilinks like [[topic/name]]."
                ),
                "minLength": 1,
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start reading from (0-based). Defaults to 0.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read. Defaults to 150.",
            },
            "query": {
                "type": "string",
                "description": "Return only lines matching this query plus a few lines of context. Use this instead of reading the whole file when looking for specific content.",
            },
        }

    def required_parameters(self) -> List[str]:
        return ["filename"]

    def execute(self, args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        return _run_files_subcommand("read", args, context)


class ListFilesTool(Tool):
    """List files in the user's workspace."""

    name = "list_files"
    description = (
        "List files and directories in the workspace. "
        "Useful to discover what files exist before reading them."
    )
    trust = "internal"

    def parameters(self) -> Dict[str, Any]:
        return {
            "offset": {
                "type": "integer",
                "description": "0-based file offset for pagination. Defaults to 0.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum files to return (default 200).",
            },
        }

    def execute(self, args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        return _run_files_subcommand("list", args, context)


class GrepFilesTool(Tool):
    """Search file contents in the workspace."""

    name = "grep_files"
    description = (
        "Search for a pattern across files in the workspace using regex. "
        "Useful to find mentions of a topic, name, or keyword in notes and documents."
    )
    trust = "internal"

    def parameters(self) -> Dict[str, Any]:
        return {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for.",
                "minLength": 1,
            },
            "filename": {
                "type": "string",
                "description": "Optional: restrict search to a single file.",
            },
        }

    def required_parameters(self) -> List[str]:
        return ["pattern"]

    def execute(self, args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        return _run_files_subcommand("grep", args, context)
