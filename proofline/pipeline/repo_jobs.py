from __future__ import annotations

from typing import Any

import pandas as pd

from proofline.utils import now_iso


def max_workers(section: dict[str, Any], default: int = 1) -> int:
    value = section.get("max_workers", section.get("workers", default))
    try:
        return max(1, int(value or default))
    except Exception:
        return max(1, default)


def repo_stage_status(kb, stage: str, repo_id: str) -> pd.DataFrame:
    return kb.query_df(
        """
        SELECT *
        FROM pipeline_repo_status
        WHERE stage = ? AND repo_id = ?
        ORDER BY finished_at DESC NULLS LAST, started_at DESC
        LIMIT 1
        """,
        [stage, repo_id],
    )


def repo_stage_done(kb, stage: str, repo_id: str, fingerprint: str) -> bool:
    existing = repo_stage_status(kb, stage, repo_id)
    return (
        not existing.empty
        and str(existing.iloc[0].get("status") or "") == "ok"
        and str(existing.iloc[0].get("fingerprint") or "") == fingerprint
    )


def mark_repo_stage(
    kb,
    stage: str,
    repo_id: str,
    fingerprint: str,
    status: str,
    *,
    started_at: str | None = None,
    item_count: int = 0,
    details: str = "",
) -> None:
    started = started_at or now_iso()
    kb.execute("DELETE FROM pipeline_repo_status WHERE stage = ? AND repo_id = ?", [stage, repo_id])
    kb.append_df("pipeline_repo_status", pd.DataFrame([{
        "stage": stage,
        "repo_id": repo_id,
        "fingerprint": fingerprint,
        "status": status,
        "started_at": started,
        "finished_at": now_iso() if status != "running" else "",
        "item_count": int(item_count),
        "details": details,
    }]))
