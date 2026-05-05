from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from proofline import version


class VersionUpdateTests(unittest.TestCase):
    def test_latest_remote_semver_tag_uses_numeric_order(self) -> None:
        stdout = "\n".join(
            [
                "aaa refs/tags/0.1.2",
                "bbb refs/tags/1.5.7",
                "ccc refs/tags/1.10.0",
                "ddd refs/tags/v9.9.9",
                "eee refs/tags/2026.01.01",
            ]
        )
        result = SimpleNamespace(returncode=0, stdout=stdout)
        with patch("proofline.version.subprocess.run", return_value=result):
            self.assertEqual(version._latest_remote_semver_tag("repo", 0.1), "1.10.0")

    def test_update_check_compares_latest_tag_to_current_version(self) -> None:
        with (
            patch("proofline.version._git", return_value="git@github.com:alphamikle/proofline.git"),
            patch("proofline.version._metadata_version", return_value="0.1.1"),
            patch("proofline.version._latest_remote_semver_tag", return_value="0.1.2"),
        ):
            info = version.update_check()

        self.assertTrue(info["update_available"])
        self.assertEqual(info["current_version"], "0.1.1")
        self.assertEqual(info["latest_version"], "0.1.2")
        self.assertEqual(info["command"], "pfl upgrade")

    def test_update_check_treats_unknown_current_version_as_old(self) -> None:
        with (
            patch("proofline.version._git", return_value=""),
            patch("proofline.version._metadata_version", return_value=None),
            patch("proofline.version.__version__", ""),
            patch("proofline.version._latest_remote_semver_tag", return_value="0.0.1"),
        ):
            info = version.update_check()

        self.assertTrue(info["update_available"])
        self.assertIsNone(info["current_version"])
        self.assertEqual(info["latest_version"], "0.0.1")

    def test_update_check_does_not_trigger_for_same_version(self) -> None:
        with (
            patch("proofline.version._git", return_value=""),
            patch("proofline.version._metadata_version", return_value="1.10.0"),
            patch("proofline.version._latest_remote_semver_tag", return_value="1.10.0"),
        ):
            info = version.update_check()

        self.assertFalse(info["update_available"])


if __name__ == "__main__":
    unittest.main()
