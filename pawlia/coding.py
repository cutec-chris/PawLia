"""Coding backends for skill script generation and debugging.

Supports three backends, selected via config or auto-detected:
  - aider:    `aider --message ... --yes` (CLI, stable API)
  - opencode: `opencode run --dir ... --format json` (CLI)
  - llm:      direct LLM call via LLMFactory (always available)

Config (config.yaml):
  coding:
    backend: auto          # auto | aider | opencode | llm

  Per-skill override via skill-config:
    skill-config:
      skill-creator:
        coding_backend: aider    # overrides global setting
"""

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _references_dir() -> Path:
    return _PROJECT_ROOT / "skills" / "skill-creator" / "references"


def _build_task_prompt(
    skill_md_body: str,
    task: str,
    existing_files: Dict[str, str],
    references: Dict[str, str],
    mode: str,
    error_output: str = "",
    failing_command: str = "",
) -> str:
    """Build the task prompt for any backend."""
    parts = [f"# Task\n{task}\n"]

    if mode == "fix":
        parts.append(f"## Error\n{error_output}\n")
        if failing_command:
            parts.append(f"## Failing command\n```\n{failing_command}\n```\n")

    parts.append("## SKILL.md instructions\n```")
    parts.append(skill_md_body)
    parts.append("```\n")

    if existing_files:
        parts.append("## Existing files\n")
        for fname, content in existing_files.items():
            parts.append(f"### {fname}\n```")
            parts.append(content)
            parts.append("```\n")

    if references:
        parts.append("## References\n")
        for rname, content in references.items():
            parts.append(f"### {rname}\n```")
            parts.append(content[:2000])
            parts.append("```\n")

    if mode == "implement":
        parts.append(
            "## Rules\n"
            "- Scripts must output JSON with `success` (bool) + result fields\n"
            "- Exit 0 on success, non-zero on failure\n"
            "- Read config from PAWLIA_SKILL_CONFIG env var, credentials from CRED_* env vars\n"
            "- Scripts receive PAWLIA_SESSION_DIR and PAWLIA_USER_ID env vars\n"
            "- Propagate real upstream errors (status code + body), never generic messages\n"
            "- Do NOT pre-format user-facing text — return structured data only\n"
        )
    elif mode == "fix":
        parts.append(
            "## Rules\n"
            "- Fix the root cause, not the symptom\n"
            "- Preserve the existing CLI interface (argparse args)\n"
            "- Keep the same JSON output format\n"
            "- Add error handling for the specific failure mode\n"
        )

    return "\n".join(parts)


def _collect_skill_files(skill_path: Path) -> Dict[str, str]:
    """Collect existing script files from a skill directory."""
    files = {}
    scripts_dir = skill_path / "scripts"
    if scripts_dir.is_dir():
        for f in scripts_dir.iterdir():
            if f.is_file() and not f.name.startswith("."):
                try:
                    files[f"scripts/{f.name}"] = f.read_text(encoding="utf-8")
                except OSError:
                    pass
    for name in ("harness.sh", "harness.py", "harness.mjs"):
        p = skill_path / name
        if p.is_file():
            try:
                files[name] = p.read_text(encoding="utf-8")
            except OSError:
                pass
    return files


def _collect_references(skill_path: Path) -> Dict[str, str]:
    """Collect reference files from the skill and from skill-creator."""
    refs = {}
    refs_dir = skill_path / "references"
    if refs_dir.is_dir():
        for f in refs_dir.iterdir():
            if f.is_file() and f.suffix == ".md":
                try:
                    refs[f.name] = f.read_text(encoding="utf-8")
                except OSError:
                    pass
    creator_refs = _references_dir()
    if creator_refs.is_dir():
        for name in ("patterns.md", "design-principles.md"):
            p = creator_refs / name
            if p.is_file():
                try:
                    refs[f"creator/{name}"] = p.read_text(encoding="utf-8")
                except OSError:
                    pass
    return refs


def _parse_skill_md(skill_path: Path) -> Optional[str]:
    """Extract SKILL.md body (after frontmatter)."""
    skill_md = skill_path / "SKILL.md"
    if not skill_md.is_file():
        return None
    content = skill_md.read_text(encoding="utf-8")
    parts = content.split("---", 2)
    return parts[2].strip() if len(parts) >= 3 else content.strip()


# ── Backend detection ──────────────────────────────────────────────────

def detect_backend(config: Dict[str, Any]) -> str:
    """Detect coding backend: 'aider' | 'opencode' | 'llm'.

    Priority: skill-config override > global config > auto-detect.
    """
    skill_config = (config.get("skill-config") or {}).get("skill-creator", {})
    configured = (
        skill_config.get("coding_backend")
        or (config.get("coding") or {}).get("backend")
        or "auto"
    )

    if configured != "auto":
        return configured

    # opencode first: it runs a full agentic coding loop with its own turn
    # management, which is the point of delegating to it instead of the
    # single-shot llm backend. aider is the secondary CLI fallback.
    if shutil.which("opencode"):
        return "opencode"
    if shutil.which("aider"):
        return "aider"
    return "llm"


# Install commands for the optional CLI coding backends. opencode ships as an
# npm package; aider installs into the active Python environment. Both run
# without root when the npm prefix / venv is user-writable (the prod image
# bakes them in at build time, so this is mainly a dev/runtime convenience).
_BACKEND_INSTALL = {
    "opencode": ["npm", "install", "-g", "opencode-ai"],
    "aider": [sys.executable, "-m", "pip", "install", "--quiet", "aider-chat"],
}


def backend_available(backend: str) -> bool:
    """True if the CLI for *backend* is on PATH (llm needs no binary)."""
    if backend in ("llm", "auto"):
        return True
    return shutil.which(backend) is not None


def install_backend(backend: str) -> Dict[str, Any]:
    """Install the CLI for *backend* (opencode|aider). Best-effort.

    Returns ``{"ok": bool, "backend", "already"|"output"|"error"}``. Idempotent:
    a backend already on PATH returns ``ok=True, already=True`` without running.
    """
    if backend not in _BACKEND_INSTALL:
        return {"ok": False, "backend": backend,
                "error": f"No installer for backend '{backend}' (opencode|aider only)."}
    if backend_available(backend):
        return {"ok": True, "backend": backend, "already": True}

    cmd = _BACKEND_INSTALL[backend]
    if backend == "opencode" and not shutil.which("npm"):
        return {"ok": False, "backend": backend,
                "error": "npm not found — cannot install opencode. Install Node.js/npm first."}
    logger.info("Installing coding backend %s: %s", backend, " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return {"ok": False, "backend": backend, "error": "install timed out after 600s"}
    except Exception as exc:
        return {"ok": False, "backend": backend, "error": str(exc)}
    ok = proc.returncode == 0 and backend_available(backend)
    return {
        "ok": ok,
        "backend": backend,
        "output": (proc.stdout or "")[-1500:],
        "error": "" if ok else (proc.stderr or proc.stdout or "")[-1500:],
    }


# ── Aider backend ─────────────────────────────────────────────────────

def _run_backend(
    backend: str,
    cmd: list[str],
    cwd: str | None = None,
    files_modified: list[str] | None = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Run a coding backend subprocess and return a standardised result dict."""
    logger.info("Running %s: %s", backend, " ".join(cmd[:6]) + "...")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=cwd,
            env=env,
        )
        return {
            "ok": proc.returncode == 0,
            "backend": backend,
            "output": (proc.stdout or "")[-3000:],
            "error": (proc.stderr or "")[-1500:] if proc.returncode != 0 else "",
            "files_modified": files_modified or [],
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "backend": backend, "error": f"{backend} timed out after 300s"}
    except Exception as e:
        return {"ok": False, "backend": backend, "error": str(e)}


def _run_aider(
    skill_path: Path,
    task_prompt: str,
    existing_files: Dict[str, str],
) -> Dict[str, Any]:
    """Run aider in non-interactive --yes mode."""
    cmd = [
        "aider",
        "--message", task_prompt,
        "--yes",
        "--no-auto-commits",
        "--no-dirty-commits",
        "--no-gitignore",
    ]

    conventions = _references_dir() / "patterns.md"
    if conventions.is_file():
        cmd.extend(["--read", str(conventions)])

    for fname in existing_files:
        fpath = skill_path / fname
        if fpath.is_file():
            cmd.append(str(fpath))

    return _run_backend("aider", cmd, cwd=str(skill_path), files_modified=list(existing_files.keys()))


def _run_opencode(
    skill_path: Path,
    task_prompt: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run opencode in non-interactive mode with its own model configuration.

    opencode is treated as a self-contained CLI: it picks its own model and
    authenticates against its own configured providers (``opencode auth``,
    ``~/.config/opencode``, project ``opencode.json``). PawLia does not pass
    ``--model`` and does not forward any provider API key — coupling PawLia's
    ``coder`` chain to opencode was a footgun, since providers opencode does
    not ship natively caused it to fall back to ``opencode/big-pickle``
    silently. Users who want a specific model set it in opencode's own config.

    Uses ``--format json`` so opencode emits a stream of JSON events on stdout
    (one per line) instead of the default formatted output. JSON mode also lets
    us extract the files opencode actually edited from ``tool_use`` events, and
    lets the model reply in clean text we can return to callers.

    In opencode-1.x, passing ``--file`` together with a positional message
    argument confuses the CLI parser and the message is interpreted as a file
    path (``File not found: <msg>``). We work around this by inlining the
    contents of ``SKILL.md`` and the patterns reference into the task prompt
    itself, so the model gets the same context without needing ``--file``.
    """
    cmd = [
        "opencode", "run",
        "--format", "json",
        "--dir", str(skill_path),
    ]

    context_blocks: list[str] = []
    skill_md = skill_path / "SKILL.md"
    if skill_md.is_file():
        try:
            context_blocks.append(
                f"=== {skill_md.name} ===\n{skill_md.read_text(encoding='utf-8', errors='replace')}"
            )
        except OSError:
            pass

    conventions = _references_dir() / "patterns.md"
    if conventions.is_file():
        try:
            context_blocks.append(
                f"=== {conventions.name} ===\n{conventions.read_text(encoding='utf-8', errors='replace')}"
            )
        except OSError:
            pass

    prompt = task_prompt
    if context_blocks:
        prompt = (
            "The following context files describe the conventions and the skill "
            "you are working on. Treat them as authoritative.\n\n"
            + "\n\n".join(context_blocks)
            + "\n\n--- task ---\n"
            + task_prompt
        )
    cmd.append(prompt)

    result = _run_backend("opencode", cmd)

    # Parse JSON event stream for any tools/files opencode actually edited,
    # so callers can verify the skill files were touched.
    edited: list[str] = []
    text_parts: list[str] = []
    for line in (result.get("output") or "").splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            evt = json.loads(line)
        except ValueError:
            continue
        evt_type = evt.get("type")
        part = evt.get("part") or {}
        if evt_type == "text":
            t = part.get("text")
            if isinstance(t, str) and t.strip():
                text_parts.append(t)
        elif evt_type == "tool_use":
            inp = part.get("input") or {}
            for key in ("filePath", "path", "filepath", "file_path"):
                val = inp.get(key)
                if isinstance(val, str) and val:
                    edited.append(val)
                    break
    if edited:
        result["files_modified"] = sorted(set(edited))
    if text_parts:
        # Concatenate assistant text replies so callers see the model's answer.
        result["output"] = "\n".join(text_parts)[-3000:]
    return result


# ── LLM backend (fallback) ───────────────────────────────────────────

def _run_llm(
    skill_path: Path,
    task_prompt: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Direct LLM call for code generation."""
    from pawlia.llm import LLMFactory

    factory = LLMFactory(config)
    llm = factory.get("coder")

    system = (
        "You are a code generator for PawLia skills. "
        "Output ONLY file contents. For each file, output:\n"
        "```<filename>\n<content>\n```\n"
        "Generate complete, working code. No explanations."
    )

    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=task_prompt),
    ]

    logger.info("Running LLM backend for code generation...")
    try:
        response = llm.invoke(messages)
        content = response.content or ""
    except Exception as e:
        return {"ok": False, "backend": "llm", "error": str(e)}

    files_written = _extract_and_write_files(content, skill_path)

    return {
        "ok": len(files_written) > 0,
        "backend": "llm",
        "output": content[-2000:],
        "files_written": files_written,
    }


def _extract_and_write_files(content: str, skill_path: Path) -> List[str]:
    """Extract ```filename blocks from LLM output and write them."""
    import re

    written = []
    pattern = re.compile(r"```(\S+)\n(.*?)```", re.DOTALL)
    for match in pattern.finditer(content):
        filename = match.group(1).strip()
        file_content = match.group(2)

        if not filename or filename in ("python", "bash", "sh", "javascript",
                                         "json", "yaml", "text", "md"):
            continue

        if "/" in filename:
            target = skill_path / filename
        else:
            target = skill_path / "scripts" / filename

        if target.parent.is_dir() or target.parent.name == "scripts":
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                target.write_text(file_content, encoding="utf-8")
                written.append(str(target.relative_to(skill_path)))
                logger.info("Wrote %s", target)
            except OSError as e:
                logger.warning("Failed to write %s: %s", target, e)

    return written


# ── Public API ─────────────────────────────────────────────────────────

def run_implement(
    skill_path: Path,
    task: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Implement scripts for a skill using the configured coding backend."""
    skill_md_body = _parse_skill_md(skill_path)
    existing_files = _collect_skill_files(skill_path)
    references = _collect_references(skill_path)

    task_prompt = _build_task_prompt(
        skill_md_body=skill_md_body or "",
        task=task,
        existing_files=existing_files,
        references=references,
        mode="implement",
    )

    backend = detect_backend(config)
    logger.info("Implementing via %s backend: %s", backend, task[:100])

    if backend == "aider":
        return _run_aider(skill_path, task_prompt, existing_files)
    elif backend == "opencode":
        return _run_opencode(skill_path, task_prompt, config)
    else:
        return _run_llm(skill_path, task_prompt, config)


def run_fix(
    skill_path: Path,
    error: str,
    command: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Fix a broken skill using the configured coding backend."""
    skill_md_body = _parse_skill_md(skill_path)
    existing_files = _collect_skill_files(skill_path)
    references = _collect_references(skill_path)

    task = (
        f"Fix the following error in this skill.\n"
        f"Failing command: {command}\n"
        f"Error output: {error}"
    )

    task_prompt = _build_task_prompt(
        skill_md_body=skill_md_body or "",
        task=task,
        existing_files=existing_files,
        references=references,
        mode="fix",
        error_output=error,
        failing_command=command,
    )

    backend = detect_backend(config)
    logger.info("Fixing via %s backend: %s", backend, error[:100])

    if backend == "aider":
        return _run_aider(skill_path, task_prompt, existing_files)
    elif backend == "opencode":
        return _run_opencode(skill_path, task_prompt, config)
    else:
        return _run_llm(skill_path, task_prompt, config)
