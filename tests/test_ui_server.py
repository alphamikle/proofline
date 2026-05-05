from __future__ import annotations

import sys
from pathlib import Path

import yaml

from proofline.config import default_config
from proofline.ui.jobs import JobManager
from proofline.ui.server import read_status


def write_config(tmp_path: Path) -> Path:
    cfg = default_config()
    cfg["workspace"] = str(tmp_path / "data")
    cfg["repos"]["root"] = str(tmp_path / "repos")
    cfg["storage"]["duckdb_path"] = str(tmp_path / "data" / "kb.duckdb")
    cfg["storage"]["sqlite_fts_path"] = str(tmp_path / "data" / "indexes" / "code_fts.sqlite")
    cfg["storage"]["vector_index_path"] = str(tmp_path / "data" / "indexes" / "code_vectors.faiss")
    cfg["storage"]["vector_meta_path"] = str(tmp_path / "data" / "indexes" / "code_vectors_meta.parquet")
    path = tmp_path / "proofline.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def test_ui_status_reads_config_and_database(tmp_path: Path) -> None:
    path = write_config(tmp_path)

    status = read_status(path)

    assert status["ok"] is True
    assert status["config_path"] == str(path)
    assert status["config"]["workspace"] == str(tmp_path / "data")
    assert "pipeline_runs" in status["table_counts"]
    assert status["recent_pipeline_runs"] == []


def test_job_manager_builds_whitelisted_stage_command(tmp_path: Path) -> None:
    path = write_config(tmp_path)
    manager = JobManager(path)

    command, kind, label = manager._command_from_payload({"kind": "stage", "stage": "repos"})

    assert kind == "stage"
    assert label == "stage repo_ingest"
    assert command[:4] == [sys.executable, "-m", "proofline", "--no-update-check"]
    assert command[-3:] == ["repo_ingest", "--config", str(path)]


def test_job_manager_rejects_unknown_stage(tmp_path: Path) -> None:
    path = write_config(tmp_path)
    manager = JobManager(path)

    try:
        manager._command_from_payload({"kind": "stage", "stage": "definitely-nope"})
    except ValueError as exc:
        assert "Unknown stage" in str(exc)
    else:
        raise AssertionError("Unknown stage was accepted")
