#!/usr/bin/env python3
"""PawLia skill creator — init, validate, list, and package skills.

Creates skills in the user's workspace via PAWLIA_SESSION_DIR.
"""

import argparse
import json
import os
import re
import sys
import zipfile
from pathlib import Path

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
  version: "1.0"
{credentials_field}
---

# {title}

## Instructions

1. Parse the query to extract the required parameters.
2. Run the script:
   ```
   python <scripts_dir>/{script_name} <args>
   ```
3. Return the output to the user.

## Output format

The script returns JSON:
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
'''

SCRIPT_TEMPLATES = {
    "python": '''\
#!/usr/bin/env python3
"""{title} skill script."""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="{title} skill")
    parser.add_argument("--param", help="Example parameter")
    args = parser.parse_args()

    # TODO: implement your skill logic here
    result = {{"success": True, "message": "Not implemented yet"}}
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

    # Build credentials field for frontmatter
    cred_keys = []
    if args.credentials:
        cred_keys = [c.strip() for c in args.credentials.split(",")]
    if cred_keys:
        cred_yaml = "requires_credentials:\n" + "\n".join(f"  - {k}" for k in cred_keys)
    else:
        cred_yaml = ""

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


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PawLia skill creator")
    sub = parser.add_subparsers(dest="command")

    # init
    p_init = sub.add_parser("init", help="Create a new skill in the user's workspace")
    p_init.add_argument("--name", required=True, help="Skill name (lowercase, hyphens)")
    p_init.add_argument("--description", help="One-line description")
    p_init.add_argument("--resources", help="Comma-separated: scripts,references,assets")
    p_init.add_argument("--credentials", help="Comma-separated credential key names (e.g. api_key,token)")
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

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "package":
        cmd_package(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
