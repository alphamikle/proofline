from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from proofline.pipeline.runner import STAGES, resolve_stage


MAX_LOG_LINES = 4000


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    id: str
    kind: str
    label: str
    command: list[str]
    status: str = "pending"
    created_at: str = field(default_factory=now_iso)
    started_at: str = ""
    finished_at: str = ""
    returncode: int | None = None
    logs: list[str] = field(default_factory=list)
    error: str = ""
    process: subprocess.Popen[str] | None = field(default=None, repr=False)

    def to_payload(self, *, include_logs: bool = False) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "command": self.command,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "returncode": self.returncode,
            "error": self.error,
            "log_line_count": len(self.logs),
        }
        if include_logs:
            payload["logs"] = self.logs[-MAX_LOG_LINES:]
        return payload


class JobManager:
    def __init__(self, config_path: str | Path):
        self.config_path = str(config_path)
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [job.to_payload() for job in sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)]

    def get_job(self, job_id: str, *, include_logs: bool = False) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_payload(include_logs=include_logs) if job else None

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        command, kind, label = self._command_from_payload(payload)
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, label=label, command=command)
        with self._lock:
            self._jobs[job.id] = job
        thread = threading.Thread(target=self._run, args=(job,), name=f"proofline-ui-job-{job.id}", daemon=True)
        thread.start()
        return job.to_payload(include_logs=True)

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            process = job.process
            if process is None or process.poll() is not None:
                return job.to_payload(include_logs=True)
            job.status = "cancelling"
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except Exception as exc:
            with self._lock:
                job.error = str(exc)
        return job.to_payload(include_logs=True)

    def _run(self, job: Job) -> None:
        with self._lock:
            job.status = "running"
            job.started_at = now_iso()
        try:
            popen_kwargs: dict[str, Any] = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "bufsize": 1,
            }
            if os.name == "posix":
                popen_kwargs["preexec_fn"] = os.setsid
            process = subprocess.Popen(job.command, **popen_kwargs)
            with self._lock:
                job.process = process
            assert process.stdout is not None
            for line in process.stdout:
                with self._lock:
                    job.logs.append(line.rstrip("\n"))
                    if len(job.logs) > MAX_LOG_LINES:
                        job.logs = job.logs[-MAX_LOG_LINES:]
            returncode = process.wait()
            with self._lock:
                job.returncode = returncode
                if job.status == "cancelling":
                    job.status = "cancelled"
                else:
                    job.status = "ok" if returncode == 0 else "error"
        except Exception as exc:
            with self._lock:
                job.status = "error"
                job.error = str(exc)
        finally:
            with self._lock:
                job.finished_at = now_iso()
                job.process = None

    def _command_from_payload(self, payload: dict[str, Any]) -> tuple[list[str], str, str]:
        kind = str(payload.get("kind") or "").strip().lower()
        base = [sys.executable, "-m", "proofline", "--no-update-check"]
        config_args = ["--config", self.config_path]

        if kind == "stage":
            stage = _validated_stage(str(payload.get("stage") or ""))
            return base + ["stage", stage] + config_args, kind, f"stage {stage}"
        if kind == "run":
            command = base + ["run"] + config_args
            from_stage = str(payload.get("from_stage") or "").strip()
            to_stage = str(payload.get("to_stage") or "").strip()
            if from_stage:
                command += ["--from", _validated_stage(from_stage)]
            if to_stage:
                command += ["--to", _validated_stage(to_stage)]
            label = "run"
            if from_stage or to_stage:
                label += f" {from_stage or 'start'}..{to_stage or 'end'}"
            return command, kind, label
        if kind == "sync":
            source = str(payload.get("source") or "all").strip().replace("_", "-")
            if source not in {"all", "repos", "repo", "docs", "confluence", "jira", "runtime", "datadog", "data", "bigquery"}:
                raise ValueError(f"Unknown sync source: {source}")
            return base + ["sync", source] + config_args, kind, f"sync {source}"
        if kind == "build":
            target = str(payload.get("target") or "all").strip().replace("_", "-")
            if target not in {"all", "history", "change-history", "blame", "code", "embeddings", "api", "code-graph", "static", "identity", "graph", "endpoints", "endpoint-map", "capabilities", "visualization", "visual"}:
                raise ValueError(f"Unknown build target: {target}")
            return base + ["build", target] + config_args, kind, f"build {target}"
        if kind == "publish":
            return base + ["publish"] + config_args, kind, "publish"
        if kind == "doctor":
            return base + ["doctor"] + config_args, kind, "doctor"
        raise ValueError(f"Unknown job kind: {kind}")


def _validated_stage(name: str) -> str:
    stage = resolve_stage(name)
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {name}")
    return stage

