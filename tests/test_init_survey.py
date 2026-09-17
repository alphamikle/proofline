"""Tests for the redesigned `pfl init` survey (scope-first, presets, menus)."""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from proofline.cli import (
    AGENT_MENU_ALIASES,
    GIT_HISTORY_PRESETS,
    INDEX_PRESETS,
    _is_git_repo,
    _project_slug,
    _prompt_choice,
    apply_git_history_preset,
    apply_index_preset,
    survey_config,
)
from proofline.config import default_config


class PromptChoiceTests(unittest.TestCase):
    def test_digit_selects_option(self):
        with patch("typer.prompt", return_value="2"):
            self.assertEqual(_prompt_choice("Q", ["fast", "medium", "all"], "all"), "medium")

    def test_name_selects_option(self):
        with patch("typer.prompt", return_value="medium"):
            self.assertEqual(_prompt_choice("Q", ["fast", "medium", "all"], "all"), "medium")

    def test_invalid_falls_back_to_default(self):
        with patch("typer.prompt", return_value="9"):
            self.assertEqual(_prompt_choice("Q", ["fast", "medium", "all"], "all"), "all")

    def test_empty_returns_default(self):
        with patch("typer.prompt", return_value=""):
            self.assertEqual(_prompt_choice("Q", ["fast", "medium", "all"], "all"), "all")


class PresetTests(unittest.TestCase):
    def test_git_presets_cover_all_params(self):
        for name, values in GIT_HISTORY_PRESETS.items():
            for key in ("enabled", "max_commits_per_repo", "metadata_days", "patch_hunks",
                        "current_blame", "rename_detection", "cochange_window_days"):
                self.assertIn(key, values, f"{name}.{key}")
        self.assertIsNone(GIT_HISTORY_PRESETS["all"]["max_commits_per_repo"])
        self.assertFalse(GIT_HISTORY_PRESETS["fast"]["patch_hunks"])
        self.assertFalse(GIT_HISTORY_PRESETS["fast"]["current_blame"])
        self.assertFalse(GIT_HISTORY_PRESETS["medium"]["current_blame"])

    def test_apply_git_preset_writes_full_map(self):
        cfg = {"git_history": {}}
        apply_git_history_preset(cfg, "medium")
        self.assertEqual(cfg["git_history_preset"], "medium")
        self.assertEqual(cfg["git_history"]["max_commits_per_repo"], 5000)
        self.assertTrue(cfg["git_history"]["patch_hunks"])

    def test_index_presets(self):
        self.assertFalse(INDEX_PRESETS["fast"]["embeddings_enabled"])
        self.assertTrue(INDEX_PRESETS["full"]["embeddings_enabled"])
        cfg: dict = {}
        apply_index_preset(cfg, "fast")
        self.assertEqual(cfg["index_preset"], "fast")
        self.assertFalse(cfg["indexing"]["embeddings"]["enabled"])
        self.assertFalse(cfg["retrieval"]["reranker"]["enabled"])
        self.assertTrue(cfg["indexing"]["lexical_fts"])

    def test_agent_aliases(self):
        self.assertEqual(AGENT_MENU_ALIASES["claude"], "anthropic")
        self.assertEqual(AGENT_MENU_ALIASES["codex"], "openai")


class OwnStateTests(unittest.TestCase):
    def test_effective_exclude_dirs_adds_proofline(self):
        from proofline.extractors.repo import effective_exclude_dirs

        self.assertIn(".proofline", effective_exclude_dirs({"repos": {"exclude_dirs": ["node_modules"]}}))
        # No duplicates when already configured.
        out = effective_exclude_dirs({"repos": {"exclude_dirs": [".proofline"]}})
        self.assertEqual(out.count(".proofline"), 1)

    def test_init_appends_gitignore_without_config(self):
        import tempfile
        from proofline.cli import ensure_own_gitignore

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()
            gi = root / ".gitignore"
            gi.write_text("*.log\n", encoding="utf-8")
            target = root / "proofline.yaml"
            target.write_text("x: 1\n", encoding="utf-8")
            ensure_own_gitignore(target)
            text = gi.read_text(encoding="utf-8")
            self.assertIn(".proofline/", text)
            self.assertIn("*.log", text)
            self.assertNotIn("proofline.yaml", text)
            # Idempotent: second run changes nothing.
            ensure_own_gitignore(target)
            self.assertEqual(gi.read_text(encoding="utf-8"), text)

    def test_watch_ignores_proofline_dir(self):
        from proofline.watch import should_ignore_path

        self.assertTrue(should_ignore_path(".proofline/kb.duckdb", set()))
        self.assertFalse(should_ignore_path("src/app.py", set()))


class ScopeTests(unittest.TestCase):
    def test_project_slug(self):
        self.assertEqual(_project_slug(Path("/x/detax/proofline.yaml")), "detax")
        self.assertEqual(_project_slug(Path("/x/my-proj/config.yaml")), "my_proj")

    def test_is_git_repo(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertFalse(_is_git_repo(root))
            (root / ".git").mkdir()
            self.assertTrue(_is_git_repo(root))
            self.assertTrue(_is_git_repo(root / "sub" / "dir"))

    def test_single_scope_in_git_repo_sets_dot_silently(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()
            cfg = default_config()
            with patch("proofline.cli._prompt_choice", side_effect=["single", "all", "full", "none"]), \
                 patch("typer.prompt", return_value=""), \
                 patch("typer.confirm", return_value=False):
                import os
                old = os.getcwd()
                os.chdir(root)
                try:
                    out = survey_config(cfg, root / "proofline.yaml")
                finally:
                    os.chdir(old)
            self.assertEqual(out["repos"]["root"], ".")
            self.assertFalse(out["repos"]["update_existing"])
            self.assertEqual(out["workspace"], "./.proofline")

    def test_multi_scope_requires_explicit_path(self):
        cfg = default_config()
        with patch("proofline.cli._prompt_choice", side_effect=["multi", "all", "full", "none"]), \
             patch("typer.prompt", return_value="/srv/repos"), \
             patch("typer.confirm", return_value=False):
            out = survey_config(cfg, Path("/tmp/x/proofline.yaml"))
        self.assertEqual(out["repos"]["root"], "/srv/repos")


class SurveyFlowTests(unittest.TestCase):
    def _run(self, choices, confirms, prompts, cwd):
        import os
        cfg = default_config()
        with patch("proofline.cli._prompt_choice", side_effect=choices), \
             patch("typer.prompt", side_effect=prompts), \
             patch("typer.confirm", side_effect=confirms):
            old = os.getcwd()
            os.chdir(cwd)
            try:
                return survey_config(cfg, Path(cwd) / "proofline.yaml")
            finally:
                os.chdir(old)

    def test_medium_sources_full_claude(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()
            out = self._run(
                ["single", "medium", "full", "claude"],
                [True, True, False, True, False, True, False],
                ["datadoghq.eu", "https://c.example", "sentence_transformers",
                 "Qwen-test", "auto", "proofline-neo4j-probe", "bolt://h:7687",
                 "u1", "p1", "db1", "sonnet", "MY_KEY"],
                str(root),
            )
            self.assertEqual(out["git_history_preset"], "medium")
            self.assertEqual(out["git_history"]["max_commits_per_repo"], 5000)
            self.assertFalse(out["git_history"]["current_blame"])
            self.assertTrue(out["datadog"]["enabled"])
            self.assertEqual(out["datadog"]["site"], "datadoghq.eu")
            self.assertFalse(out["bigquery"]["enabled"])
            self.assertEqual(out["confluence"]["base_url"], "https://c.example")
            self.assertEqual(out["graph_backend"]["container_name"], "proofline-neo4j-probe")
            self.assertEqual(out["graph_backend"]["username"], "u1")
            self.assertEqual(out["neo4j"]["username"], "u1")
            self.assertEqual(out["agent"]["provider"], "anthropic")
            self.assertEqual(out["agent"]["model"], "sonnet")

    def test_sources_gate_no_skips_all(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()
            out = self._run(
                ["single", "all", "full", "none"], [False, False],
                ["sentence_transformers", "Qwen/Qwen3-Embedding-0.6B", "auto"],
                str(root),
            )
            for key in ("datadog", "bigquery", "confluence", "jira"):
                self.assertFalse(out[key]["enabled"])
            self.assertIsNone(out["git_history"]["max_commits_per_repo"])

    def test_fast_index_disables_ml(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()
            out = self._run(["single", "fast", "fast", "none"], [False, False], [], str(root))
            self.assertFalse(out["indexing"]["embeddings"]["enabled"])
            self.assertFalse(out["retrieval"]["reranker"]["enabled"])
            self.assertTrue(out["indexing"]["lexical_fts"])


class ContainerNameTests(unittest.TestCase):
    def test_default_container_is_legacy(self):
        from proofline.repair import cgc_container_name

        self.assertEqual(cgc_container_name({}), "cgc-neo4j")

    def test_custom_container_flows_to_env(self):
        from proofline.repair import cgc_container_name, cgc_environment

        cfg = {"graph_backend": {"container_name": "proofline-neo4j-detax"}}
        self.assertEqual(cgc_container_name(cfg), "proofline-neo4j-detax")
        self.assertEqual(cgc_environment(cfg)["NEO4J_CONTAINER_NAME"], "proofline-neo4j-detax")


if __name__ == "__main__":
    unittest.main()
