from __future__ import annotations

from rich.console import Console

from proofline.pipeline import runner


def test_selected_order_respects_stage_aliases() -> None:
    assert runner.selected_order("runtime", "graph") == [
        "datadog",
        "bigquery",
        "entity_resolution",
        "graph",
    ]


def test_pipeline_progress_lists_done_running_and_pending(monkeypatch) -> None:
    console = Console(record=True, width=100, color_system=None)
    monkeypatch.setattr(runner, "console", console)

    runner.print_pipeline_progress(
        ["repo_ingest", "git_history", "code_index"],
        {"repo_ingest": "ok", "git_history": "running", "code_index": "pending"},
    )

    output = console.export_text()
    assert "Pipeline stages" in output
    assert "repo_ingest" in output
    assert "git_history" in output
    assert "code_index" in output
    assert "ok" in output
    assert "running" in output
    assert "pending" in output
    assert "100%" in output
    assert "0%" in output
