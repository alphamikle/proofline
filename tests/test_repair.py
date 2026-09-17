from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from proofline.repair import cgc_environment, run_repair


class RepairTests(unittest.TestCase):
    def test_dry_run_can_plan_repair_without_existing_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "proofline.yaml"

            steps = run_repair(
                config_path=config,
                dry_run=True,
                skip_python_deps=True,
                skip_cgc=True,
                skip_bin_links=True,
            )

            self.assertFalse(config.exists())
            self.assertTrue(all(step["ok"] for step in steps))
            self.assertEqual(steps[0]["action"], "would_create")
            self.assertEqual(steps[0]["details"], str(config))

    def test_cgc_environment_uses_neo4j_config(self) -> None:
        env = cgc_environment(
            {
                "neo4j": {
                    "uri": "bolt://localhost:17687",
                    "username": "neo",
                    "password": "secret",
                    "database": "graph",
                    "http_port": 17474,
                }
            }
        )

        self.assertEqual(env["NEO4J_URI"], "bolt://localhost:17687")
        self.assertEqual(env["NEO4J_USER"], "neo")
        self.assertEqual(env["NEO4J_PASSWORD"], "secret")
        self.assertEqual(env["NEO4J_DATABASE"], "graph")
        self.assertEqual(env["NEO4J_BOLT_PORT"], "17687")
        self.assertEqual(env["NEO4J_HTTP_PORT"], "17474")


    def test_link_is_idempotent_when_already_correct(self) -> None:
        import os
        import sys
        import tempfile
        from unittest.mock import patch

        from proofline import repair as repair_mod

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            dst = Path(tmp) / "bin"
            src.mkdir()
            (src / "pfl").write_text("x", encoding="utf-8")
            (src / "proofline").write_text("x", encoding="utf-8")
            dst.mkdir(parents=True, exist_ok=True)
            os.symlink(src / "pfl", dst / "pfl")
            os.symlink(src / "proofline", dst / "proofline")
            steps: list = []
            with patch.object(sys, "executable", str(src / "python")):
                repair_mod.link_cli_binaries(bin_dir=str(dst), dry_run=False, steps=steps)
            self.assertTrue(all(s.ok for s in steps))
            self.assertIn("already linked", steps[0].details)
            self.assertTrue((dst / "pfl").is_symlink())


if __name__ == "__main__":
    unittest.main()
