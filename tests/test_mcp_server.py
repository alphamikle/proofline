from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

pytest.importorskip("mcp")

from proofline.cli import app
from proofline.mcp_server import _read_file_slice, _resolve_under, create_server, validate_select_sql
from proofline.storage import KB


def write_config(tmp_path: Path) -> Path:
    cfg = {
        "workspace": str(tmp_path / "data"),
        "repos": {"root": str(tmp_path / "repos")},
        "storage": {
            "duckdb_path": str(tmp_path / "data" / "kb.duckdb"),
            "sqlite_fts_path": str(tmp_path / "data" / "indexes" / "code_fts.sqlite"),
        },
        "indexing": {"embeddings": {"enabled": False}},
        "retrieval": {"reranker": {"enabled": False}},
        "graph_backend": {"enabled": False},
        "datadog": {"enabled": False},
        "bigquery": {"enabled": False},
        "confluence": {"enabled": False, "output_dir": str(tmp_path / "data" / "raw" / "confluence")},
        "jira": {"enabled": False, "output_dir": str(tmp_path / "data" / "raw" / "jira")},
        "agent": {"provider": "none"},
    }
    path = tmp_path / "proofline.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_create_server(tmp_path: Path) -> None:
    server = create_server(str(write_config(tmp_path)))
    assert server is not None


@pytest.mark.parametrize("query", ["SELECT 1", " select * from repo_files ", "WITH x AS (SELECT 1 AS n) SELECT n FROM x"])
def test_validate_select_sql_accepts_read_only_queries(query: str) -> None:
    assert validate_select_sql(query).lower().lstrip().startswith(("select", "with"))


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM repo_files",
        "UPDATE repo_files SET repo_id = 'x'",
        "PRAGMA table_info(repo_files)",
        "SELECT 1; SELECT 2",
        "SELECT * FROM repo_files; DROP TABLE repo_files",
    ],
)
def test_validate_select_sql_rejects_unsafe_queries(query: str) -> None:
    with pytest.raises(ValueError):
        validate_select_sql(query)


def test_resolve_under_rejects_parent_traversal(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    with pytest.raises(ValueError):
        _resolve_under(root, "../outside.json")


def test_read_file_slice_uses_repo_files_allowlist(tmp_path: Path) -> None:
    file_path = tmp_path / "repos" / "demo" / "src" / "app.py"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")

    kb = KB(tmp_path / "kb.duckdb")
    try:
        kb.execute(
            """
            INSERT INTO repo_files
            (repo_id, path, rel_path, ext, size_bytes, kind, sha1, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ["demo", str(file_path), "src/app.py", ".py", file_path.stat().st_size, "code", "sha", "now"],
        )
        result = _read_file_slice(kb, "demo", "src/app.py", start_line=2, end_line=3)
    finally:
        kb.close()

    assert result["text"] == "two\nthree"
    assert result["start_line"] == 2
    assert result["end_line"] == 3


def test_cli_rejects_invalid_transport_without_starting_server(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["mcp", "--config", str(write_config(tmp_path)), "--transport", "tcp"])
    assert result.exit_code != 0
    assert "Unsupported transport" in result.output
