"""Tests for SkillLoader."""


from agent.tools.skill_loader import SkillLoader


class TestSkillLoader:
    """Tests for SkillLoader."""

    def test_discover_skills_empty_dir(self, tmp_path):
        """Test discovering skills in empty directory."""
        loader = SkillLoader(str(tmp_path))
        summaries = loader.discover_skills()
        assert summaries == []

    def test_discover_skills_nonexistent_dir(self, tmp_path):
        """Test discovering skills when directory doesn't exist."""
        loader = SkillLoader(str(tmp_path / "nonexistent"))
        summaries = loader.discover_skills()
        assert summaries == []

    def test_discover_skills_with_valid_skill(self, tmp_path):
        """Test discovering a valid skill."""
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("""---
name: my-skill
description: A test skill
---

# My Skill

Instructions here.
""")

        loader = SkillLoader(str(tmp_path))
        summaries = loader.discover_skills()

        assert len(summaries) == 1
        assert summaries[0].name == "my-skill"
        assert summaries[0].description == "A test skill"

    def test_load_skill_not_found(self, tmp_path):
        """Test loading a skill that doesn't exist."""
        loader = SkillLoader(str(tmp_path))
        skill = loader.load_skill("nonexistent")
        assert skill is None

    def test_load_skill_success(self, tmp_path):
        """Test loading a valid skill."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        content = """---
name: test-skill
description: Test description
---

# Test Skill

Full instructions here.
"""
        skill_file.write_text(content)

        loader = SkillLoader(str(tmp_path))
        skill = loader.load_skill("test-skill")

        assert skill is not None
        assert skill.name == "test-skill"
        assert skill.description == "Test description"
        assert "Full instructions here" in skill.instructions

    def test_load_skill_caching(self, tmp_path):
        """Test that loaded skills are cached."""
        skill_dir = tmp_path / "cached-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("""---
name: cached-skill
description: Cached
---
Instructions
""")

        loader = SkillLoader(str(tmp_path))
        
        # Load twice
        skill1 = loader.load_skill("cached-skill")
        skill2 = loader.load_skill("cached-skill")

        # Should be same object from cache
        assert skill1 is skill2

    def test_parse_frontmatter_missing(self, tmp_path):
        """Test parsing file without frontmatter."""
        loader = SkillLoader(str(tmp_path))
        result = loader._parse_frontmatter("No frontmatter here")
        assert result == {}

    def test_parse_frontmatter_with_quotes(self, tmp_path):
        """Test parsing frontmatter with quoted values."""
        loader = SkillLoader(str(tmp_path))
        content = """---
name: "quoted-name"
description: 'single quoted'
---
Body
"""
        result = loader._parse_frontmatter(content)
        assert result["name"] == "quoted-name"
        assert result["description"] == "single quoted"
