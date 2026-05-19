#!/usr/bin/env python3
"""PawLia skill creator — init, validate, list, and package skills.

Creates skills in the user's workspace via PAWLIA_SESSION_DIR.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

# Project root (for finding bundled skills to validate/list against)
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # thalia/
BUNDLED_DIR = PROJECT_ROOT / "skills"


def _workspace_skills_dir() -> Path:
    """Return the user's workspace skills directory from env."""
    session_dir = os.environ.get("PAWLIA_SESSION_DIR")
    user_id = os.environ.get("PAWLIA_USER_ID")
    if not session_dir or not user_id:
        return None
    return Path(session_dir) / user_id / "workspace" / "skills"


# ── Templates ──────────────────────────────────────────────────────────

SKILL_MD_TEMPLATE = '''\
---
name: {name}
description: >
  {description}
license: MIT
metadata:
  author: Christian Ulrich
  version: "1.0"{config_field}
{credentials_field}---

# {title}

## Instructions

1. Parse the query to extract the required parameters.
2. Run the script:
   ```
   python <scripts_dir>/{script_name} <args>
   ```
   Do not pass deployment config such as URLs or timeouts as CLI args unless
   the user explicitly asks for an override. Scripts read those values from
   `PAWLIA_SKILL_CONFIG`.
3. Parse the JSON output (`success`, plus result fields or `error`).
4. Return a clean, formatted result to the user.

## Output format

Describe the exact shape you want the sub-agent to return. Example:

```
Result: <value>
```

The script itself returns JSON:
```json
{{"success": true, "result": "..."}}
```

On error:
```json
{{"success": false, "error": "error message"}}
```

## Error handling

| Error | Recovery action |
|-------|-----------------|
| Connection error | Retry once, then report to user |
| Invalid input | Tell user what format is expected |
| Missing credential | Tell user which `CRED_*` env var is missing |
'''

SCRIPT_TEMPLATES = {
    "python": '''\
#!/usr/bin/env python3
"""{title} skill script."""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="{title} skill")
    parser.add_argument("--param", help="Example parameter")
    args = parser.parse_args()

    # Read credentials (if declared in SKILL.md's requires_credentials):
    #   api_key = os.environ.get("CRED_MY_API_KEY")
    #
    # Read PawLia runtime env:
    #   user_id = os.environ.get("PAWLIA_USER_ID")
    #   session_dir = os.environ.get("PAWLIA_SESSION_DIR")
    #   skill_config = json.loads(os.environ.get("PAWLIA_SKILL_CONFIG", "{}"))

    try:
        # TODO: implement your skill logic here
        result = {{"success": True, "message": "Not implemented yet"}}
    except Exception as e:
        result = {{"success": False, "error": str(e)}}
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
''',
    "node": '''\
#!/usr/bin/env node
// {title} skill script.

const args = process.argv.slice(2);

// TODO: implement your skill logic here
console.log(JSON.stringify({{ success: true, message: "Not implemented yet" }}));
''',
    "bash": '''\
#!/usr/bin/env bash
set -euo pipefail

# {title} skill script.
# Usage: {script_name} [--param value]

PARAM="${{1:-}}"

# TODO: implement your skill logic here
echo '{{"success": true, "message": "Not implemented yet"}}'
''',
}


# ── Helpers ────────────────────────────────────────────────────────────

def _validate_name(name: str):
    """Ensure skill name is lowercase-with-hyphens."""
    if not re.match(r'^[a-z][a-z0-9-]{0,62}$', name):
        return None
    return name


def _title(name: str) -> str:
    return name.replace("-", " ").title()


def _script_name(name: str, script_type: str) -> str:
    ext = {"python": ".py", "node": ".mjs", "bash": ".sh"}
    return name + ext.get(script_type, ".py")


def _find_skill(name: str) -> Path:
    """Find a skill: workspace first, then bundled."""
    ws = _workspace_skills_dir()
    if ws:
        candidate = ws / name
        if (candidate / "SKILL.md").is_file():
            return candidate
    candidate = BUNDLED_DIR / name
    if (candidate / "SKILL.md").is_file():
        return candidate
    return None


def _parse_frontmatter(skill_md: Path):
    """Parse YAML frontmatter from a SKILL.md file."""
    import yaml
    with open(skill_md, encoding="utf-8") as f:
        content = f.read()
    parts = content.split("---", 2)
    if len(parts) < 3:
        return None, None, content
    meta = yaml.safe_load(parts[1]) or {}
    body = parts[2].strip()
    return meta, body, content


# ── Commands ───────────────────────────────────────────────────────────

def cmd_init(args):
    """Initialize a new skill in the user's workspace."""
    name = _validate_name(args.name)
    if not name:
        print(json.dumps({
            "success": False,
            "error": f"Invalid skill name '{args.name}'. Use lowercase letters, digits, and hyphens, max 63 chars.",
        }))
        sys.exit(1)

    # Resolve target directory — always workspace
    ws = _workspace_skills_dir()
    if not ws:
        print(json.dumps({
            "success": False,
            "error": "PAWLIA_SESSION_DIR / PAWLIA_USER_ID not set — this script must run as a PawLia skill.",
        }))
        sys.exit(1)

    target = ws / name

    if target.exists():
        print(json.dumps({
            "success": False,
            "error": f"Skill '{name}' already exists at {target}",
        }))
        sys.exit(1)

    description = args.description or f"The {name} skill."
    title = _title(name)

    # Build credentials field (TOP-LEVEL — sibling to metadata)
    cred_keys = []
    if args.credentials:
        cred_keys = [c.strip() for c in args.credentials.split(",") if c.strip()]
    if cred_keys:
        cred_yaml = "requires_credentials:\n" + "\n".join(f"  - {k}" for k in cred_keys) + "\n"
    else:
        cred_yaml = ""

    # Build config field (NESTED under metadata — loader checks metadata.requires_config)
    config_keys = []
    if args.config:
        config_keys = [c.strip() for c in args.config.split(",") if c.strip()]
    if config_keys:
        config_yaml = "\n  requires_config:\n" + "\n".join(f"    - {k}" for k in config_keys)
    else:
        config_yaml = ""

    # Parse resources
    resources = []
    if args.resources:
        resources = [r.strip() for r in args.resources.split(",")]

    script_type = args.script or "python"

    # Create directory
    target.mkdir(parents=True, exist_ok=True)

    sname = _script_name(name, script_type)

    # Write SKILL.md
    skill_md = SKILL_MD_TEMPLATE.format(
        name=name,
        description=description,
        title=title,
        script_name=sname,
        credentials_field=cred_yaml,
        config_field=config_yaml,
    )
    (target / "SKILL.md").write_text(skill_md, encoding="utf-8")
    created = [str(target / "SKILL.md")]

    # Create resource directories and files
    for resource in resources:
        res_dir = target / resource
        res_dir.mkdir(exist_ok=True)

        if resource == "scripts" and not args.no_script:
            script_path = res_dir / sname
            script_content = SCRIPT_TEMPLATES[script_type].format(
                title=title,
                script_name=sname,
            )
            script_path.write_text(script_content, encoding="utf-8")
            script_path.chmod(0o755)
            created.append(str(script_path))
        elif resource == "references":
            ref_file = res_dir / "guide.md"
            ref_file.write_text(f"# {title} Reference\n\nTODO: Add reference documentation.\n", encoding="utf-8")
            created.append(str(ref_file))
        elif resource == "assets":
            (res_dir / ".gitkeep").write_text("", encoding="utf-8")

    print(json.dumps({
        "success": True,
        "name": name,
        "path": str(target),
        "files_created": created,
        "resources": resources,
    }, ensure_ascii=False))


def cmd_validate(args):
    """Validate a skill's SKILL.md and directory structure."""
    name = args.name
    skill_path = _find_skill(name)

    if not skill_path:
        print(json.dumps({
            "success": False,
            "error": f"Skill '{name}' not found (checked workspace + bundled)",
        }))
        sys.exit(1)

    issues = []
    warnings = []

    meta, body, content = _parse_frontmatter(skill_path / "SKILL.md")

    if meta is None:
        issues.append("Missing or malformed YAML frontmatter (must be --- / YAML / --- / markdown)")
        print(json.dumps({
            "success": False, "name": name, "path": str(skill_path),
            "issues": issues, "warnings": warnings,
        }))
        sys.exit(1)

    # Required fields
    if not meta.get("name"):
        issues.append("Missing required field: 'name'")
    elif meta["name"] != name:
        issues.append(f"Name mismatch: frontmatter says '{meta['name']}', directory is '{name}'")

    if not meta.get("description"):
        issues.append("Missing required field: 'description'")
    elif len(meta.get("description", "")) < 20:
        warnings.append("Description is very short (<20 chars) — may not trigger reliably")

    # Placement check: requires_config must be NESTED under metadata (not top-level)
    if "requires_config" in meta and "requires_config" not in (meta.get("metadata") or {}):
        issues.append(
            "'requires_config' is at top-level but the loader reads it from "
            "metadata.requires_config — move it under the 'metadata:' block"
        )

    # Placement check: requires_credentials must be TOP-LEVEL (not under metadata)
    if "requires_credentials" not in meta and (meta.get("metadata") or {}).get("requires_credentials"):
        issues.append(
            "'requires_credentials' is nested under metadata but the loader reads it "
            "from the top level — move it out of the 'metadata:' block"
        )

    # Instruction body
    if not body:
        issues.append("No instruction body — sub-agent needs instructions")
    elif len(body.split("\n")) < 5:
        warnings.append("Instruction body is very short (<5 lines)")

    # scripts/ checks
    scripts_dir = skill_path / "scripts"
    if scripts_dir.is_dir():
        if "<scripts_dir>" not in body:
            issues.append("Has scripts/ but SKILL.md doesn't use <scripts_dir> placeholder")
        if not [f for f in scripts_dir.iterdir() if f.is_file()]:
            warnings.append("scripts/ directory is empty")

    # Unused resource dirs
    for res_type in ["references", "assets"]:
        res_dir = skill_path / res_type
        if res_dir.is_dir():
            files = [f for f in res_dir.iterdir() if f.is_file() and f.name != ".gitkeep"]
            if files and res_type not in body.lower():
                warnings.append(f"Has {res_type}/ but SKILL.md doesn't reference it")

    # Example output section
    if "## Example output" not in body:
        issues.append(
            "Missing '## Example output' section — the sub-agent needs a concrete sample "
            "of the expected user-facing output to format responses correctly. "
            "Add a ## Example output section with 5-20 lines showing exactly what good "
            "output looks like, and annotate critical elements (links, exact phrases) "
            "with '← keep' so the sub-agent knows not to change them."
        )

    # Line count
    line_count = len(body.split("\n"))
    if line_count > 500:
        warnings.append(f"SKILL.md body is {line_count} lines (max recommended: 500)")

    # Extraneous files
    allowed = {"SKILL.md", "scripts", "references", "assets"}
    for item in skill_path.iterdir():
        if item.name not in allowed and not item.name.startswith("."):
            warnings.append(f"Extraneous file/dir: {item.name}")

    if issues:
        print(json.dumps({
            "success": False, "name": name, "path": str(skill_path),
            "issues": issues, "warnings": warnings,
        }))
    else:
        print(json.dumps({
            "success": True, "name": name, "path": str(skill_path),
            "issues": [], "warnings": warnings,
        }))


def cmd_list(args):
    """List all discoverable skills (workspace + bundled)."""
    skills = []

    def scan(directory: Path, source: str):
        if not directory.is_dir():
            return
        for child in sorted(directory.iterdir()):
            if not child.is_dir():
                continue
            skill_md = child / "SKILL.md"
            if not skill_md.is_file():
                continue
            try:
                meta, body, _ = _parse_frontmatter(skill_md)
                desc = (meta.get("description") or "").strip().split("\n")[0][:120] if meta else "(parse error)"
                skills.append({
                    "name": meta.get("name", child.name) if meta else child.name,
                    "description": desc,
                    "version": (meta.get("metadata") or {}).get("version", "?") if meta else "?",
                    "source": source,
                    "path": str(child),
                    "has_scripts": (child / "scripts").is_dir(),
                    "has_references": (child / "references").is_dir(),
                    "has_assets": (child / "assets").is_dir(),
                })
            except Exception:
                skills.append({
                    "name": child.name, "description": "(parse error)",
                    "version": "?", "source": source, "path": str(child),
                    "has_scripts": False, "has_references": False, "has_assets": False,
                })

    # Workspace skills first
    ws = _workspace_skills_dir()
    if ws:
        scan(ws, "workspace")

    # Bundled skills
    scan(BUNDLED_DIR, "bundled")

    print(json.dumps({"success": True, "skills": skills}, ensure_ascii=False))


def cmd_package(args):
    """Package a skill into a .skill zip file."""
    name = args.name
    skill_path = _find_skill(name)

    if not skill_path:
        print(json.dumps({
            "success": False,
            "error": f"Skill '{name}' not found",
        }))
        sys.exit(1)

    # Quick validate
    meta, body, _ = _parse_frontmatter(skill_path / "SKILL.md")
    if not meta or not meta.get("name") or not meta.get("description"):
        print(json.dumps({
            "success": False,
            "error": f"Skill '{name}' failed validation — fix SKILL.md first",
        }))
        sys.exit(1)

    # No symlinks
    for root, dirs, files in os.walk(skill_path):
        for f in files:
            fpath = Path(root) / f
            if fpath.is_symlink():
                print(json.dumps({
                    "success": False,
                    "error": f"Symlink at {fpath} — remove before packaging",
                }))
                sys.exit(1)

    # Create zip
    output_dir = Path(args.output) if args.output else skill_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{name}.skill"

    with zipfile.ZipFile(output_file, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(skill_path):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for f in files:
                if f.startswith("."):
                    continue
                fpath = Path(root) / f
                arcname = fpath.relative_to(skill_path.parent)
                zf.write(fpath, arcname)

    print(json.dumps({
        "success": True,
        "name": name,
        "package": str(output_file),
        "size_bytes": output_file.stat().st_size,
    }, ensure_ascii=False))


def _find_harness(skill_path: Path):
    """Locate the skill's harness. Returns (path, interpreter_argv) or None."""
    candidates = [
        ("harness.sh", ["sh"]),
        ("harness.py", [sys.executable]),
        ("harness.mjs", ["node"]),
        ("harness.js", ["node"]),
    ]
    for filename, interp in candidates:
        p = skill_path / filename
        if p.is_file():
            return p, interp + [str(p)]
    return None


def _build_cred_env(meta: dict):
    """Load CRED_* env vars for credentials declared in requires_credentials.

    Mirrors skill_runner._load_credentials — same key normalization so the
    harness sees exactly what the skill sees at runtime.
    Returns (env_dict, missing_keys).
    """
    required = meta.get("requires_credentials") or []
    if not required:
        return {}, []
    session_dir = os.environ.get("PAWLIA_SESSION_DIR")
    user_id = os.environ.get("PAWLIA_USER_ID")
    if not session_dir or not user_id:
        return {}, list(required)
    cred_path = Path(session_dir) / user_id / ".credentials.json"
    if not cred_path.is_file():
        return {}, list(required)
    try:
        with open(cred_path, encoding="utf-8") as f:
            all_creds = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}, list(required)
    env = {}
    missing = []
    for key in required:
        if key in all_creds:
            env_key = "CRED_" + re.sub(r"[^A-Za-z0-9]", "_", key).upper()
            env[env_key] = str(all_creds[key])
        else:
            missing.append(key)
    return env, missing


def _build_skill_config_env(name: str, meta: dict):
    """Load skill-config.<name> for harness runs.

    The harness should see the same PAWLIA_SKILL_CONFIG env var that the real
    SkillRunner/BashTool injects at runtime.
    Returns (env_dict, missing_keys).
    """
    config = {}
    config_path = os.environ.get("PAWLIA_CONFIG_PATH")
    candidates = [Path(config_path)] if config_path else []
    candidates.extend([PROJECT_ROOT / "config.yaml", PROJECT_ROOT / "config.yml"])
    for candidate in candidates:
        if not candidate or not candidate.is_file():
            continue
        try:
            with open(candidate, encoding="utf-8") as f:
                root = yaml.safe_load(f) or {}
            config = (root.get("skill-config") or {}).get(name, {}) or {}
            break
        except (OSError, yaml.YAMLError):
            continue

    required = (meta.get("metadata") or {}).get("requires_config") or []
    missing = [key for key in required if key not in config]
    return {"PAWLIA_SKILL_CONFIG": json.dumps(config, ensure_ascii=False)}, missing


def cmd_test(args):
    """Run the skill's harness with production-equivalent env.

    Loads credentials from .credentials.json, injects them as CRED_* env
    vars, injects skill-config.<name> as PAWLIA_SKILL_CONFIG, keeps
    PAWLIA_SESSION_DIR/PAWLIA_USER_ID, and runs the harness script. Returns
    full stdout/stderr — no truncation, no wrapping.
    """
    name = args.name
    skill_path = _find_skill(name)
    if not skill_path:
        print(json.dumps({
            "success": False,
            "error": f"Skill '{name}' not found (checked workspace + bundled)",
        }))
        sys.exit(1)

    meta, _body, _content = _parse_frontmatter(skill_path / "SKILL.md")
    if meta is None:
        print(json.dumps({
            "success": False, "name": name,
            "error": "Malformed SKILL.md — cannot parse frontmatter",
        }))
        sys.exit(1)

    found = _find_harness(skill_path)
    if not found:
        print(json.dumps({
            "success": False, "name": name,
            "error": f"No harness found in {skill_path}",
            "hint": "Add harness.sh, harness.py, or harness.mjs at the skill root. "
                    "See skill-creator SKILL.md § Harness for the contract.",
        }))
        sys.exit(1)

    harness_path, cmd = found
    cred_env, missing = _build_cred_env(meta)
    config_env, missing_config = _build_skill_config_env(name, meta)
    env = {**os.environ, **cred_env, **config_env}

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=args.timeout, env=env, check=False, cwd=str(skill_path),
        )
        stdout, stderr, exit_code, timed_out = proc.stdout, proc.stderr, proc.returncode, False
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        exit_code = None
        timed_out = True

    print(json.dumps({
        "success": (not timed_out) and exit_code == 0,
        "name": name,
        "harness": str(harness_path),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "missing_credentials": missing,
        "missing_config": missing_config,
        "stdout": stdout,
        "stderr": stderr,
    }, ensure_ascii=False))


def cmd_compile(args):
    """Compile a skill's SKILL.md into workflow.yaml via the pawlia compiler.

    The compiler is an LLM call — it can fail if no compiler model is configured
    or if the SKILL.md is malformed. A skill still runs without workflow.yaml
    (fallback to tool-call/command mode), so compilation is optional but
    recommended after substantive edits.
    """
    name = args.name
    skill_path = _find_skill(name)

    if not skill_path:
        print(json.dumps({
            "success": False,
            "error": f"Skill '{name}' not found (checked workspace + bundled)",
        }))
        sys.exit(1)

    # Compiler resolves skills by walking the parent dir, so pass the parent
    skills_dir = str(skill_path.parent)

    cmd = [
        sys.executable, "-m", "pawlia.skills.compiler",
        "--skill", name,
        "--skills-dir", skills_dir,
    ]
    if args.force:
        cmd.append("--force")

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, check=False,
        )
    except subprocess.TimeoutExpired:
        print(json.dumps({
            "success": False,
            "error": "Compiler timed out after 300s",
        }))
        sys.exit(1)
    except FileNotFoundError as e:
        print(json.dumps({
            "success": False,
            "error": f"Could not invoke compiler: {e}",
        }))
        sys.exit(1)

    workflow_path = skill_path / "workflow.yaml"
    workflow_exists = workflow_path.is_file()

    if proc.returncode != 0 or not workflow_exists:
        print(json.dumps({
            "success": False,
            "name": name,
            "error": "Compilation failed",
            "stderr": (proc.stderr or "")[-1500:],
            "stdout": (proc.stdout or "")[-500:],
            "workflow_written": workflow_exists,
            "hint": "Skill still works in fallback mode without workflow.yaml. "
                    "Check the compiler model config (agents.compiler in config.yaml).",
        }))
        sys.exit(1)

    print(json.dumps({
        "success": True,
        "name": name,
        "workflow": str(workflow_path),
        "size_bytes": workflow_path.stat().st_size,
    }, ensure_ascii=False))


# ── Implement / Fix (coding backends) ──────────────────────────────────

def _load_root_config() -> dict:
    """Load the root config.yaml for coding backend selection."""
    config_path = os.environ.get("PAWLIA_CONFIG_PATH")
    candidates = [Path(config_path)] if config_path else []
    candidates.extend([PROJECT_ROOT / "config.yaml", PROJECT_ROOT / "config.yml"])
    for candidate in candidates:
        if not candidate or not candidate.is_file():
            continue
        try:
            with open(candidate, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            continue
    return {}


def cmd_implement(args):
    """Implement skill scripts using a coding backend (aider/opencode/llm)."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from pawlia.coding import run_implement

    name = args.name
    skill_path = _find_skill(name)
    if not skill_path:
        print(json.dumps({
            "success": False,
            "error": f"Skill '{name}' not found (checked workspace + bundled)",
        }))
        sys.exit(1)

    config = _load_root_config()
    task = args.task or "Implement all scripts and a harness for this skill"
    result = run_implement(skill_path, task, config)

    print(json.dumps({
        "success": result.get("ok", False),
        "name": name,
        "backend": result.get("backend", "unknown"),
        "files_written": result.get("files_written", []),
        "files_modified": result.get("files_modified", []),
        "output": (result.get("output") or "")[-1500:],
        "error": result.get("error", ""),
    }, ensure_ascii=False))

    if not result.get("ok"):
        sys.exit(1)


def cmd_fix(args):
    """Fix a broken skill script using a coding backend."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from pawlia.coding import run_fix

    name = args.name
    skill_path = _find_skill(name)
    if not skill_path:
        print(json.dumps({
            "success": False,
            "error": f"Skill '{name}' not found (checked workspace + bundled)",
        }))
        sys.exit(1)

    config = _load_root_config()
    error = args.error or ""
    command = args.failed_cmd or ""
    result = run_fix(skill_path, error, command, config)

    print(json.dumps({
        "success": result.get("ok", False),
        "name": name,
        "backend": result.get("backend", "unknown"),
        "files_written": result.get("files_written", []),
        "files_modified": result.get("files_modified", []),
        "output": (result.get("output") or "")[-1500:],
        "error": result.get("error", ""),
    }, ensure_ascii=False))

    if not result.get("ok"):
        sys.exit(1)


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PawLia skill creator")
    sub = parser.add_subparsers(dest="command")

    # init
    p_init = sub.add_parser("init", help="Create a new skill in the user's workspace")
    p_init.add_argument("--name", required=True, help="Skill name (lowercase, hyphens)")
    p_init.add_argument("--description", help="One-line description")
    p_init.add_argument("--resources", help="Comma-separated: scripts,references,assets")
    p_init.add_argument("--credentials", help="Comma-separated credential key names (top-level requires_credentials, e.g. api_key,token)")
    p_init.add_argument("--config", help="Comma-separated config key names (nested under metadata.requires_config, e.g. url,timeout)")
    p_init.add_argument("--script", choices=["python", "node", "bash"], default="python")
    p_init.add_argument("--no-script", action="store_true", help="Skip script template")

    # validate
    p_validate = sub.add_parser("validate", help="Validate a skill's SKILL.md and structure")
    p_validate.add_argument("--name", required=True, help="Skill name to validate")

    # list
    sub.add_parser("list", help="List all skills (workspace + bundled)")

    # package
    p_package = sub.add_parser("package", help="Package skill into .skill zip")
    p_package.add_argument("--name", required=True, help="Skill name to package")
    p_package.add_argument("--output", help="Output directory")

    # compile
    p_compile = sub.add_parser("compile", help="Compile SKILL.md into workflow.yaml (LLM-driven)")
    p_compile.add_argument("--name", required=True, help="Skill name to compile")
    p_compile.add_argument("--force", action="store_true", help="Re-compile even if version matches")

    # test
    p_test = sub.add_parser("test", help="Run the skill's harness end-to-end")
    p_test.add_argument("--name", required=True, help="Skill name to test")
    p_test.add_argument("--timeout", type=int, default=60, help="Harness timeout in seconds (default: 60)")

    # implement
    p_impl = sub.add_parser("implement", help="Implement skill scripts via coding backend (aider/opencode/llm)")
    p_impl.add_argument("--name", required=True, help="Skill name")
    p_impl.add_argument("--task", help="Task description (default: implement all scripts)")

    # fix
    p_fix = sub.add_parser("fix", help="Fix a broken skill script via coding backend")
    p_fix.add_argument("--name", required=True, help="Skill name")
    p_fix.add_argument("--error", help="Error output from the failing command")
    p_fix.add_argument("--failed-cmd", help="The command that failed")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "package":
        cmd_package(args)
    elif args.command == "compile":
        cmd_compile(args)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "implement":
        cmd_implement(args)
    elif args.command == "fix":
        cmd_fix(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
