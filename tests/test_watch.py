"""Tests for pfl watch (single-repo live reindex).

Covers: event coalescing, ignore rules, repo resolution, incremental
chunk refresh, embedding delta. No watchdog/observer threads here -
handler logic is tested via direct method calls with fake events.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from proofline.watch import (
    DebouncedQueue,
    WatchEvent,
    coalesce_events,
    describe_batch,
    filter_indexable,
    should_ignore_path,
    watch_config_from_dict,
    watch_defaults,
)
from proofline.watch.events import RepoEventHandler
from proofline.watch.repos import find_enclosing_repo
class IgnoreRulesTests(unittest.TestCase):
    def test_ignores_vcs_and_ide_noise(self):
        excludes = {".git", "node_modules"}
        for rel in [".git/index", "a/.DS_Store", "x.py~", "y.swp", "node_modules/a.py", ""]:
            self.assertTrue(should_ignore_path(rel, excludes), rel)

    def test_keeps_source(self):
        self.assertFalse(should_ignore_path("src/app.py", {".git"}))

    def test_filter_respects_extensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("x=1\n", encoding="utf-8")
            (root / "b.bin").write_text("x", encoding="utf-8")
            out = filter_indexable(["a.py", "b.bin", ".git/x"], exclude_dirs=set(),
                                   include_extensions={".py"}, repo_root=root)
            self.assertEqual(out, ["a.py"])


class CoalesceTests(unittest.TestCase):
    def test_create_modify_merge(self):
        batch = coalesce_events([WatchEvent("created", "a.py"), WatchEvent("modified", "a.py"),
                                 WatchEvent("modified", "a.py")])
        self.assertEqual(batch["modified"], ["a.py"])
        self.assertEqual(batch["deleted"], [])

    def test_modify_then_delete_wins_delete(self):
        batch = coalesce_events([WatchEvent("modified", "a.py"), WatchEvent("deleted", "a.py")])
        self.assertEqual(batch["modified"], [])
        self.assertEqual(batch["deleted"], ["a.py"])

    def test_delete_then_create_resurrects(self):
        batch = coalesce_events([WatchEvent("deleted", "a.py"), WatchEvent("created", "a.py")])
        self.assertEqual(batch["modified"], ["a.py"])
        self.assertEqual(batch["deleted"], [])

    def test_moved_maps_src_delete_dest_modified(self):
        batch = coalesce_events([WatchEvent("moved", src_rel_path="a.py", dest_rel_path="b.py")])
        self.assertEqual(batch["modified"], ["b.py"])
        self.assertEqual(batch["deleted"], ["a.py"])

    def test_describe(self):
        self.assertIn("2 modified", describe_batch({"modified": ["a", "b"], "deleted": []}))


class QueueTests(unittest.TestCase):
    def test_drain_empty_timeout(self):
        q = DebouncedQueue(debounce_seconds=0.1)
        self.assertEqual(q.drain(timeout=0.05), [])

    def test_put_drain_nowait(self):
        q = DebouncedQueue(debounce_seconds=0.1)
        q.put(WatchEvent("modified", "a.py"))
        out = q.drain_nowait()
        self.assertEqual([e.rel_path for e in out], ["a.py"])
        self.assertTrue(q.empty())
class RepoResolutionTests(unittest.TestCase):
    def test_find_enclosing_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()
            sub = root / "a" / "b"
            sub.mkdir(parents=True)
            self.assertEqual(find_enclosing_repo(sub), root.resolve())

    def test_no_repo_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(find_enclosing_repo(Path(tmp) / "x"))

    def test_watch_defaults_shape(self):
        d = watch_defaults()
        self.assertIn("repo_ingest", d["fast_lane"])
        self.assertIn("git_history", d["slow_lane"])

    def test_config_from_dict(self):
        cfg = {"repos": {"exclude_dirs": [".git"], "include_extensions": [".py"], "max_file_mb": 5},
               "watch": {"debounce_seconds": 1.0}}
        wc = watch_config_from_dict(cfg, Path("/tmp/repo"))
        self.assertEqual(wc.debounce_seconds, 1.0)
        self.assertIn(".git", wc.exclude_dirs)
class FakeEvent:
    def __init__(self, src_path, dest_path=None, is_directory=False):
        self.src_path = src_path
        self.dest_path = dest_path or src_path
        self.is_directory = is_directory


class HandlerTests(unittest.TestCase):
    def test_handler_pushes_rel_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("x\n", encoding="utf-8")
            q = DebouncedQueue(debounce_seconds=0.1)
            handler = RepoEventHandler(root, q, {".git"})
            handler.on_modified(FakeEvent(str(root / "a.py")))
            handler.on_modified(FakeEvent(str(root / ".git" / "index")))
            handler.on_modified(FakeEvent(str(root / "sub"), is_directory=True))
            out = q.drain_nowait()
            self.assertEqual([e.rel_path for e in out], ["a.py"])

    def test_handler_maps_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            q = DebouncedQueue(debounce_seconds=0.1)
            handler = RepoEventHandler(root, q, set())
            handler.on_moved(FakeEvent(str(root / "a.py"), str(root / "b.py")))
            out = q.drain_nowait()
            self.assertEqual(len(out), 1)
            self.assertEqual(out[0].kind, "moved")
            self.assertEqual(out[0].dest_rel_path, "b.py")
class DispatchTests(unittest.TestCase):
    def test_dispatch_routes_watchdog_event_types(self):
        import types

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            q = DebouncedQueue(debounce_seconds=0.1)
            handler = RepoEventHandler(root, q, set())
            for kind, path in [("created", "a.py"), ("modified", "b.py"), ("deleted", "c.py")]:
                event = types.SimpleNamespace(event_type=kind, src_path=str(root / path),
                                              dest_path=str(root / path), is_directory=False)
                handler.dispatch(event)
            moved = types.SimpleNamespace(event_type="moved", src_path=str(root / "x.py"),
                                          dest_path=str(root / "y.py"), is_directory=False)
            handler.dispatch(moved)
            out = q.drain_nowait()
            kinds = sorted(e.kind for e in out)
            self.assertEqual(kinds, ["created", "deleted", "modified", "moved"])

    def test_dispatch_ignores_directories(self):
        import types

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            q = DebouncedQueue(debounce_seconds=0.1)
            handler = RepoEventHandler(root, q, set())
            event = types.SimpleNamespace(event_type="modified", src_path=str(root / "sub"),
                                          dest_path=str(root / "sub"), is_directory=True)
            handler.dispatch(event)
            self.assertTrue(q.empty())


class FakeKB:
    """Minimal DuckDB stand-in for reindex/embedding tests."""

    def __init__(self):
        self.tables = {}
        self.executed = []

    def _rows(self, table):
        return self.tables.setdefault(table, [])

    def query_df(self, sql, params=None):
        import re

        table = None
        m = re.search(r"FROM\s+([A-Za-z_][\w]*)", sql)
        if m:
            table = m.group(1)
        rows = list(self._rows(table)) if table else []
        if params and table:
            # Only WHERE repo_id = ? / rel filters used in tests.
            if "repo_id = ?" in sql:
                rows = [r for r in rows if str(r.get("repo_id")) == str(params[0])]
        if "COUNT(*) AS n" in sql:
            return pd.DataFrame([{"n": len(rows)}])
        if "SELECT rel_path" in sql and "file_fingerprint" not in sql:
            return pd.DataFrame([{"rel_path": r.get("rel_path")} for r in rows])
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def execute(self, sql, params=None):
        import re

        self.executed.append(sql)
        m = re.search(r"(?:DELETE FROM|delete from)\s+([A-Za-z_][\w]*)", sql, re.I)
        if not m:
            return
        table = m.group(1)
        rows = self._rows(table)
        if params and "repo_id = ?" in sql and "rel_path = ?" in sql:
            self.tables[table] = [r for r in rows if not (
                str(r.get("repo_id")) == str(params[0]) and str(r.get("rel_path")) == str(params[1]))]
        elif params and "repo_id = ?" in sql:
            keep = [r for r in rows if str(r.get("repo_id")) != str(params[0])]
            if "entity_id" in sql:
                keep = rows  # ownership delete uses entity_id param; emulate below
            self.tables[table] = keep
        elif "entity_id = ?" in sql and params:
            self.tables[table] = [r for r in rows if str(r.get("entity_id")) != str(params[0])]
        elif "repo_id = ?" not in sql:
            self.tables[table] = []

    def append_df(self, table, df):
        if df is None or df.empty:
            return
        self._rows(table).extend(df.to_dict("records"))

    def close(self):
        pass


def make_cfg(tmp, **over):
    cfg = {
        "repos": {"root": str(tmp), "exclude_dirs": [".git"], "include_extensions": [".py", ".md"],
                  "max_file_mb": 5, "include_git_history_metadata": False},
        "storage": {"duckdb_path": str(Path(tmp) / "kb.duckdb"),
                    "sqlite_fts_path": str(Path(tmp) / "fts.sqlite"),
                    "vector_index_path": str(Path(tmp) / "vec.faiss"),
                    "vector_meta_path": str(Path(tmp) / "meta.parquet")},
        "indexing": {"lexical_fts": True,
                     "ast_chunking": {"enabled": True, "fallback_regex": True, "keep_file_windows": True},
                     "embeddings": {"enabled": False}},
        "git_history": {"enabled": False},
        "code_graph": {"enabled": False},
    }
    cfg.update(over)
    return cfg


class ReindexScopeTests(unittest.TestCase):
    def test_refresh_files_only_touches_watched_repo(self):
        from proofline.watch.reindex import refresh_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
            (root / "other.py").write_text("x = 1\n", encoding="utf-8")
            cfg = make_cfg(tmp)
            kb = FakeKB()
            kb.tables["code_chunks"] = [
                {"chunk_id": "keep", "repo_id": "other-repo", "rel_path": "other.py", "text": "old"}]
            kb.tables["code_index_file_status"] = [
                {"repo_id": "other-repo", "rel_path": "other.py",
                 "file_fingerprint": "fp", "chunk_count": 1}]
            out = refresh_files(kb, cfg, root, "demo", ["a.py"], [])
            self.assertEqual(out["repo_id"], "demo")
            self.assertGreaterEqual(out["files"], 1)
            # Other repo untouched.
            keep = [r for r in kb.tables["code_chunks"] if r.get("repo_id") == "other-repo"]
            self.assertEqual(len(keep), 1)
            demo_rows = [r for r in kb.tables["code_chunks"] if r.get("repo_id") == "demo"]
            self.assertTrue(demo_rows)

    def test_refresh_files_deletes_removed(self):
        from proofline.watch.reindex import refresh_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = make_cfg(tmp)
            kb = FakeKB()
            kb.tables["code_chunks"] = [
                {"chunk_id": "c1", "repo_id": "demo", "rel_path": "gone.py", "text": "x"}]
            kb.tables["code_index_file_status"] = [
                {"repo_id": "demo", "rel_path": "gone.py", "file_fingerprint": "fp", "chunk_count": 1}]
            kb.tables["repo_files"] = []
            out = refresh_files(kb, cfg, root, "demo", [], ["gone.py"])
            self.assertEqual(out["files"], 0)
            self.assertEqual([r for r in kb.tables["code_chunks"] if r.get("rel_path") == "gone.py"], [])

    def test_second_refresh_is_noop_via_fingerprint(self):
        from proofline.watch.reindex import refresh_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("x = 1\n", encoding="utf-8")
            cfg = make_cfg(tmp)
            kb = FakeKB()
            first = refresh_files(kb, cfg, root, "demo", ["a.py"], [])
            second = refresh_files(kb, cfg, root, "demo", ["a.py"], [])
            self.assertGreaterEqual(first["files"], 1)
            self.assertEqual(second["files"], 0)


if __name__ == "__main__":
    unittest.main()