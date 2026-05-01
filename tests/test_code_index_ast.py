from __future__ import annotations

import json
import unittest

import pandas as pd

from proofline.extractors.code_index import chunk_text
from proofline.extractors.embeddings import eligible_chunks


def cfg(**ast_overrides):
    ast = {
        "enabled": True,
        "source": "cgc",
        "fallback_regex": True,
        "keep_file_windows": True,
        "include_node_types": ["Function", "Class", "Method", "Module", "Struct", "Enum", "Interface"],
        "max_symbol_lines": 240,
        "symbol_window_lines": 160,
        "symbol_window_overlap": 30,
        "include_context_prefix": True,
        "dedupe_overlapping_chunks": True,
    }
    ast.update(ast_overrides)
    return {"repos": {"max_file_mb": 2, "include_extensions": []}, "indexing": {"ast_chunking": ast}}


class AstCodeChunkingTests(unittest.TestCase):
    def test_creates_ast_chunks_from_graph_symbols(self) -> None:
        text = "\n".join([
            "class Service:",
            "    def handle(self):",
            "        return 1",
            "    def helper(self):",
            "        return 2",
        ])
        chunks = chunk_text(
            "repo",
            "/repo/service.py",
            "service.py",
            text,
            cfg=cfg(keep_file_windows=False),
            graph_symbols=[
                {"symbol_id": "class-1", "repo_id": "repo", "node_type": "Class", "name": "Service", "line_start": 1, "line_end": 5, "language": "python", "signature": "class Service:", "source": "cgc"},
                {"symbol_id": "method-1", "repo_id": "repo", "node_type": "Method", "name": "handle", "line_start": 2, "line_end": 3, "language": "python", "signature": "def handle(self):", "source": "cgc"},
            ],
        )

        self.assertEqual([c["kind"] for c in chunks], ["ast_class", "ast_method", "symbol"])
        self.assertEqual(chunks[2]["symbol"], "helper")
        method = chunks[1]
        metadata = json.loads(method["metadata"])
        self.assertEqual(method["start_line"], 2)
        self.assertEqual(method["end_line"], 3)
        self.assertEqual(metadata["chunk_source"], "cgc")
        self.assertEqual(metadata["node_type"], "Method")
        self.assertEqual(metadata["parent_symbol"], "Service")
        self.assertEqual(metadata["graph_symbol_id"], "method-1")
        self.assertTrue(method["text"].startswith("File: service.py\nLanguage: python\nSymbol: Service.handle"))

    def test_splits_large_symbols_inside_ast_bounds(self) -> None:
        text = "\n".join(f"line {i}" for i in range(1, 11))
        chunks = chunk_text(
            "repo",
            "/repo/worker.py",
            "worker.py",
            text,
            cfg=cfg(keep_file_windows=False, max_symbol_lines=3, symbol_window_lines=4, symbol_window_overlap=1),
            graph_symbols=[
                {"symbol_id": "fn-1", "repo_id": "repo", "node_type": "Function", "name": "run", "line_start": 2, "line_end": 9, "language": "python", "signature": "def run():", "source": "cgc"},
            ],
        )

        self.assertEqual([c["kind"] for c in chunks], ["ast_large_symbol_window", "ast_large_symbol_window", "ast_large_symbol_window"])
        self.assertEqual([(c["start_line"], c["end_line"]) for c in chunks], [(2, 5), (5, 8), (8, 9)])
        self.assertTrue(all(2 <= c["start_line"] <= c["end_line"] <= 9 for c in chunks))

    def test_falls_back_to_regex_and_windows_when_graph_missing(self) -> None:
        text = "def foo():\n    return 1\n"
        chunks = chunk_text("repo", "/repo/a.py", "a.py", text, max_lines=2, overlap=1, cfg=cfg())

        self.assertIn("symbol", {c["kind"] for c in chunks})
        self.assertIn("file_window", {c["kind"] for c in chunks})
        self.assertNotIn("ast_function", {c["kind"] for c in chunks})

    def test_dedupes_duplicate_graph_symbols(self) -> None:
        symbol = {"symbol_id": "fn-1", "repo_id": "repo", "node_type": "Function", "name": "foo", "line_start": 1, "line_end": 2, "language": "python", "signature": "def foo():", "source": "cgc"}
        chunks = chunk_text(
            "repo",
            "/repo/a.py",
            "a.py",
            "def foo():\n    return 1\n",
            cfg=cfg(keep_file_windows=False),
            graph_symbols=[symbol, dict(symbol)],
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["kind"], "ast_function")

    def test_ast_disabled_uses_existing_regex_path(self) -> None:
        chunks = chunk_text(
            "repo",
            "/repo/a.py",
            "a.py",
            "def foo():\n    return 1\n",
            cfg=cfg(enabled=False, keep_file_windows=False),
            graph_symbols=[
                {"symbol_id": "fn-1", "repo_id": "repo", "node_type": "Function", "name": "foo", "line_start": 1, "line_end": 2, "language": "python", "signature": "def foo():", "source": "cgc"},
            ],
        )

        self.assertEqual([c["kind"] for c in chunks], ["symbol"])

    def test_embeddings_include_ast_kinds_when_ast_chunking_enabled(self) -> None:
        chunks = pd.DataFrame([
            {"chunk_id": "1", "repo_id": "repo", "rel_path": "a.py", "language": "python", "kind": "ast_function", "symbol": "foo", "start_line": 1, "text": "x" * 80},
            {"chunk_id": "2", "repo_id": "repo", "rel_path": "a.py", "language": "python", "kind": "symbol", "symbol": "bar", "start_line": 2, "text": "y" * 80},
        ])
        config = cfg()
        config["indexing"]["embeddings"] = {"include_kinds": ["symbol"], "min_text_chars": 1}

        eligible = eligible_chunks(chunks, config)

        self.assertEqual(set(eligible["kind"]), {"ast_function", "symbol"})


if __name__ == "__main__":
    unittest.main()
