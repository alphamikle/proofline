from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from proofline.extractors.git_history import extract_repo_git_history


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return result.stdout.strip()


def commit(repo: Path, message: str) -> None:
    git(repo, "add", ".")
    git(repo, "commit", "-m", message)


class GitHistoryTests(unittest.TestCase):
    def make_repo(self, root: Path) -> Path:
        repo = root / "repo"
        repo.mkdir()
        git(repo, "init")
        git(repo, "config", "user.email", "alice@example.com")
        git(repo, "config", "user.name", "Alice")
        return repo

    def test_extracts_full_history_when_max_commits_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = self.make_repo(Path(tmp))
            (repo / "README.md").write_text("one\n", encoding="utf-8")
            commit(repo, "PAY-100 initial docs")
            (repo / "README.md").write_text("one\ntwo\n", encoding="utf-8")
            commit(repo, "PAY-101 update docs")

            rows = extract_repo_git_history(repo, "repo", {"current_blame": False, "write_commit_graph": False, "patch_hunks": False})

            self.assertEqual(len(rows["git_commits"]), 2)
            self.assertEqual({c["subject"] for c in rows["git_commits"]}, {"PAY-100 initial docs", "PAY-101 update docs"})
            self.assertTrue(any(link["target"] == "PAY-100" for link in rows["git_detected_links"]))

    def test_extracts_semantic_events_for_contracts_and_migrations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = self.make_repo(Path(tmp))
            (repo / "openapi.yaml").write_text("openapi: 3.0.0\npaths: {}\n", encoding="utf-8")
            (repo / "migrations").mkdir()
            (repo / "migrations" / "001.sql").write_text("-- init\n", encoding="utf-8")
            commit(repo, "initial")
            (repo / "openapi.yaml").write_text(
                """
openapi: 3.0.0
paths:
  /payments:
    get:
      responses:
        '200':
          description: ok
""".strip()
                + "\n",
                encoding="utf-8",
            )
            (repo / "migrations" / "001.sql").write_text("ALTER TABLE payments DROP COLUMN legacy_status;\n", encoding="utf-8")
            commit(repo, "PAY-200 add payment endpoint and migration")

            rows = extract_repo_git_history(repo, "repo", {"current_blame": False, "write_commit_graph": False})
            changes = {r["change_type"] for r in rows["git_semantic_changes"]}

            self.assertIn("API_ENDPOINT_ADDED", changes)
            self.assertIn("DB_COLUMN_DROPPED", changes)
            self.assertTrue(any(r["breaking_risk"] == "breaking" for r in rows["git_semantic_changes"]))

    def test_builds_cochange_edges_from_same_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = self.make_repo(Path(tmp))
            (repo / "a.py").write_text("a = 1\n", encoding="utf-8")
            commit(repo, "initial")
            (repo / "a.py").write_text("a = 2\n", encoding="utf-8")
            (repo / "b.py").write_text("b = 2\n", encoding="utf-8")
            commit(repo, "PAY-300 cochange")

            rows = extract_repo_git_history(repo, "repo", {"current_blame": False, "write_commit_graph": False})
            edges = rows["git_cochange_edges"]

            self.assertTrue(edges)
            self.assertTrue(any(edge["same_commit_count"] >= 1 for edge in edges))
            self.assertTrue(any(edge["same_jira_count"] >= 1 for edge in edges))


if __name__ == "__main__":
    unittest.main()
