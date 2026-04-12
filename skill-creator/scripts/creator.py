#!/usr/bin/env python3
"""Skill creator — scaffold, validate, and list PawLia skills."""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Base skills directories
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # thalia/
BUNDLED_DIR = PROJECT_ROOT / "skills"
USER_DIR = BUNDLED_DIR / "user"

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
---

# {title} Skill

## How to use

Describe how to invoke this skill.  The sub-agent runs commands via the Bash tool:

```
python <scripts_dir>/{script_name} <args>
```

## Step-by-step instructions

1. Parse the query for the required parameters.
2. Run the script:
   ```
   python <scripts_dir>/{script_name} --param "<value>"
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

def _skill_dir(name: str) -> Path:
    """Return the target directory for a user skill."""
    return USER_DIR / name


def _validate_name(name: str) -> str:
    """Ensure skill name is lowercase-with-hyphens."""
    if not re.match(r'^[a-z][a-z0-9-]{0,62}$', name):
        print(json.dumps({
            "success": False,
            "error": f"Invalid skill name '{name}'. Use lowercase letters, digits, and hyphens, max 63 chars.",
        }))
        sys.exit(1)
    return name


def _title(name: str) -> str:
    """Convert skill-name to Skill Name."""
    return name.replace("-", " ").title()


def _script_name(name: str, script_type: str) -> str:
    ext = {"python": ".py", "node": ".mjs", "bash": ".sh"}
    return name + ext.get(script_type, ".py")


# ── Commands ───────────────────────────────────────────────────────────

def cmd_create(args):
    """Scaffold a new user skill."""
    name = _validate_name(args.name)
    description = args.description
    script_type = args.script or "python"
    no_script = args.no_script

    target = _skill_dir(name)
    if target.exists():
        print(json.dumps({
            "success": False,
            "error": f"Skill '{name}' already exists at {target}",
        }))
        sys.exit(1)

    # Create directory structure
    target.mkdir(parents=True, exist_ok=True)

    title = _title(name)
    sname = _script_name(name, script_type)

    # Write SKILL.md
    skill_md = SKILL_MD_TEMPLATE.format(
        name=name,
        description=description,
        title=title,
        script_name=sname,
    )
    (target / "SKILL.md").write_text(skill_md, encoding="utf-8")

    created = [str(target / "SKILL.md")]

    # Write script
    if not no_script:
        scripts_dir = target / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        script_path = scripts_dir / sname
        script_content = SCRIPT_TEMPLATES[script_type].format(
            title=title,
            script_name=sname,
        )
        script_path.write_text(script_content, encoding="utf-8")
        script_path.chmod(0o755)
        created.append(str(script_path))

    print(json.dumps({
        "success": True,
        "name": name,
        "path": str(target),
        "files_created": created,
        "script_type": script_type if not no_script else None,
    }, ensure_ascii=False))


def cmd_list(args):
    """List all discoverable skills."""
    skills = []

    def scan(directory: Path, source: str):
        if not directory.is_dir():
            return
        for child in sorted(directory.iterdir()):
            skill_md = child / "SKILL.md" if child.is_dir() else None
            if skill_md and skill_md.is_file():
                # Quick parse frontmatter
                try:
                    with open(skill_md, encoding="utf-8") as f:
                        content = f.read()
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        import yaml
                        meta = yaml.safe_load(parts[1]) or {}
                        skills.append({
                            "name": meta.get("name", child.name),
                            "description": (meta.get("description") or "").strip().split("\n")[0][:120],
                            "source": source,
                            "path": str(child),
                            "has_scripts": (child / "scripts").is_dir(),
                        })
                except Exception:
                    skills.append({
                        "name": child.name,
                        "description": "(parse error)",
                        "source": source,
                        "path": str(child),
                        "has_scripts": False,
                    })

    scan(BUNDLED_DIR, "bundled")
    scan(USER_DIR, "user")

    print(json.dumps({"success": True, "skills": skills}, ensure_ascii=False))


def cmd_validate(args):
    """Validate a skill's SKILL.md."""
    name = args.name

    # Find the skill
    skill_path = None
    for base in [USER_DIR, BUNDLED_DIR]:
        candidate = base / name
        if (candidate / "SKILL.md").is_file():
            skill_path = candidate
            break

    if not skill_path:
        print(json.dumps({
            "success": False,
            "error": f"Skill '{name}' not found",
        }))
        sys.exit(1)

    issues = []
    skill_md = skill_path / "SKILL.md"

    # Read and parse
    with open(skill_md, encoding="utf-8") as f:
        content = f.read()

    # Check frontmatter exists
    if not content.startswith("---"):
        issues.append("Missing YAML frontmatter (must start with ---)")
    else:
        parts = content.split("---", 2)
        if len(parts) < 3:
            issues.append("Malformed frontmatter — expected --- / YAML / --- / markdown")

    # Parse YAML if possible
    meta = {}
    if len(parts) >= 3:
        try:
            import yaml
            meta = yaml.safe_load(parts[1]) or {}
        except Exception as e:
            issues.append(f"Invalid YAML in frontmatter: {e}")

    # Check required fields
    if not meta.get("name"):
        issues.append("Missing required field: 'name'")
    elif meta["name"] != name:
        issues.append(f"Name mismatch: frontmatter says '{meta['name']}', directory is '{name}'")

    if not meta.get("description"):
        issues.append("Missing required field: 'description'")

    # Check instructions body
    body = parts[2].strip() if len(parts) >= 3 else ""
    if not body:
        issues.append("No instruction body after frontmatter — sub-agent needs instructions")

    # Check for <scripts_dir> usage if scripts/ exists
    scripts_dir = skill_path / "scripts"
    if scripts_dir.is_dir():
        if "<scripts_dir>" not in body:
            issues.append("Has scripts/ directory but SKILL.md doesn't use <scripts_dir> placeholder")

    # Check scripts dir has at least one executable
    if scripts_dir.is_dir():
        scripts = [f for f in scripts_dir.iterdir() if f.is_file()]
        if not scripts:
            issues.append("scripts/ directory is empty")

    if issues:
        print(json.dumps({
            "success": False,
            "name": name,
            "path": str(skill_path),
            "issues": issues,
        }))
    else:
        print(json.dumps({
            "success": True,
            "name": name,
            "path": str(skill_path),
            "issues": [],
        }))


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PawLia skill creator")
    sub = parser.add_subparsers(dest="command")

    # create
    p_create = sub.add_parser("create", help="Scaffold a new skill")
    p_create.add_argument("--name", required=True, help="Skill name (lowercase, hyphens)")
    p_create.add_argument("--description", required=True, help="One-line description")
    p_create.add_argument("--script", choices=["python", "node", "bash"], default="python")
    p_create.add_argument("--no-script", action="store_true", help="Skip scripts/ directory")

    # list
    sub.add_parser("list", help="List all skills")

    # validate
    p_validate = sub.add_parser("validate", help="Validate a skill")
    p_validate.add_argument("--name", required=True, help="Skill name to validate")

    args = parser.parse_args()

    if args.command == "create":
        cmd_create(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "validate":
        cmd_validate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
