"""Tests for pawlia.skills.loader."""

import os
import tempfile

from pawlia.skills.loader import AgentSkill, SkillLoader, _parse_frontmatter


def _create_skill_dir(parent: str, name: str, description: str = "Test skill",
                      requires_config: list = None) -> str:
    """Create a minimal skill directory with a SKILL.md."""
    skill_dir = os.path.join(parent, name)
    os.makedirs(skill_dir, exist_ok=True)

    meta_lines = [f"name: {name}", f"description: {description}"]
    if requires_config:
        meta_lines.append("metadata:")
        meta_lines.append("  requires_config:")
        for key in requires_config:
            meta_lines.append(f"    - {key}")

    frontmatter = "\n".join(meta_lines)
    content = f"---\n{frontmatter}\n---\n\n# Instructions\n\nDo the thing.\n"

    with open(os.path.join(skill_dir, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write(content)

    return skill_dir


class TestAgentSkill:
    def test_load_from_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_skill_dir(tmpdir, "test_skill", "A test skill")
            metadata = _parse_frontmatter(
                os.path.join(tmpdir, "test_skill", "SKILL.md")
            )
            skill = AgentSkill(os.path.join(tmpdir, "test_skill"), metadata)

            assert skill.name == "test_skill"
            assert skill.description == "A test skill"
            assert "Do the thing" in skill.instructions

    def test_openai_spec(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_skill_dir(tmpdir, "my_skill")
            metadata = _parse_frontmatter(
                os.path.join(tmpdir, "my_skill", "SKILL.md")
            )
            skill = AgentSkill(os.path.join(tmpdir, "my_skill"), metadata)
            spec = skill.as_openai_spec()

            assert spec["function"]["name"] == "my_skill"
            assert "query" in spec["function"]["parameters"]["properties"]
            assert "query" in spec["function"]["parameters"]["required"]


class TestSkillLoader:
    def test_discover(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_skill_dir(tmpdir, "skill_a", "Skill A")
            _create_skill_dir(tmpdir, "skill_b", "Skill B")
            # Directory without SKILL.md should be ignored
            os.makedirs(os.path.join(tmpdir, "not_a_skill"))

            skills = SkillLoader.discover(tmpdir)

            assert len(skills) == 2
            assert "skill_a" in skills
            assert "skill_b" in skills

    def test_discover_nonexistent_dir(self):
        skills = SkillLoader.discover("/nonexistent/path")
        assert skills == {}

    def test_discover_with_required_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_skill_dir(tmpdir, "needs_url", requires_config=["url"])

            # Without config -> skill is skipped
            skills = SkillLoader.discover(tmpdir)
            assert len(skills) == 0

            # With config -> skill is loaded
            config = {"skill-config": {"needs_url": {"url": "http://localhost"}}}
            skills = SkillLoader.discover(tmpdir, config)
            assert "needs_url" in skills

    def test_discover_real_agentskills(self):
        """Test discovery on the actual skills directory."""
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        skills_dir = os.path.join(pkg_dir, "skills")
        if not os.path.isdir(skills_dir):
            return  # skip if no skills dir

        skills = SkillLoader.discover(skills_dir)
        # Should find at least the skills that don't require config
        assert isinstance(skills, dict)
        for name, skill in skills.items():
            assert skill.name == name
            assert skill.description
