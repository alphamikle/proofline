from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from proofline.config import CONFIG_SHAPE_VERSION, config_shape_diff, load_config, migrate_config_file, upgrade_config_file


class ConfigMigrationTests(unittest.TestCase):
    def test_migration_adds_new_sections_without_overwriting_existing_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "proofline.yaml"
            path.write_text(
                "# local note\nworkspace: ./old-data\n\nrepos:\n  # keep this comment\n  root: ./old-repos\n",
                encoding="utf-8",
            )

            cfg, added = migrate_config_file(path, quiet=True)
            text = path.read_text(encoding="utf-8")
            written = yaml.safe_load(text)

            self.assertEqual(cfg["workspace"], "./old-data")
            self.assertEqual(cfg["repos"]["root"], "./old-repos")
            self.assertEqual(written["workspace"], "./old-data")
            self.assertEqual(written["repos"]["root"], "./old-repos")
            self.assertEqual(written["config_version"], CONFIG_SHAPE_VERSION)
            self.assertIn("git_history", written)
            self.assertIn("graph_backend", written)
            self.assertIn("git_history", added)
            self.assertIn("# local note", text)
            self.assertIn("# keep this comment", text)

            _, added_again = migrate_config_file(path, quiet=True)
            self.assertEqual(added_again, [])

    def test_migration_inserts_missing_section_before_next_known_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "proofline.yaml"
            path.write_text(
                "config_version: 2\nworkspace: ./data\nrepos:\n  root: ./repos\n\ndatadog:\n  enabled: false\n",
                encoding="utf-8",
            )

            migrate_config_file(path, quiet=True)
            text = path.read_text(encoding="utf-8")

            self.assertLess(text.index("git_history:"), text.index("datadog:"))

    def test_load_config_migrates_and_normalizes_runtime_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "proofline.yaml"
            path.write_text("workspace: ./data\n", encoding="utf-8")

            cfg = load_config(path, quiet=True)

            self.assertIn("git_history", cfg)
            self.assertIsNone(cfg["git_history"]["max_commits_per_repo"])
            self.assertIn("duckdb_path", cfg["storage"])
            self.assertEqual(yaml.safe_load(path.read_text(encoding="utf-8"))["config_version"], CONFIG_SHAPE_VERSION)

    def test_upgrade_config_file_backs_up_and_writes_full_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "proofline.yaml"
            path.write_text(
                "config_version: 1\nworkspace: ./old-data\nrepos:\n  root: ./old-repos\nagent:\n  provider: none\n",
                encoding="utf-8",
            )

            cfg, backup, missing = upgrade_config_file(path, use_agent=False, quiet=True)
            written = yaml.safe_load(path.read_text(encoding="utf-8"))

            self.assertIsNotNone(backup)
            self.assertTrue(backup.exists())
            self.assertIn("config_version: 1", backup.read_text(encoding="utf-8"))
            self.assertEqual(written["config_version"], CONFIG_SHAPE_VERSION)
            self.assertEqual(written["workspace"], "./old-data")
            self.assertEqual(written["repos"]["root"], "./old-repos")
            self.assertIn("exclude_dirs", written["repos"])
            self.assertIn("repos.exclude_dirs", missing)
            self.assertEqual(cfg["repos"]["root"], "./old-repos")

            diff = config_shape_diff(path)
            self.assertFalse(diff["needs_migration"])


if __name__ == "__main__":
    unittest.main()
