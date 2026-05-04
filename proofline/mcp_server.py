from __future__ import annotations

import math
import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from mcp.server.fastmcp import FastMCP

from proofline.agent.tools import KBTools
from proofline.config import ensure_dirs, load_config
from proofline.storage import KB
from proofline.utils import json_dumps, safe_read_text


MAX_LIMIT = 1000
MAX_SQL_LIMIT = 5000
MAX_FILE_LINES = 2000
MAX_RAW_CHARS = 200_000
MAX_RAW_FILE_BYTES = 2_000_000
RAW_SOURCES = {"jira", "confluence", "datadog", "bigquery"}
ALLOWED_TRANSPORTS = {"stdio", "streamable-http", "sse"}
BLOCKED_SQL = re.compile(
    r"(?is)\b(copy|attach|install|load|export|pragma|create|drop|delete|update|insert|alter|detach|call|vacuum|"
    r"read_csv|read_json|read_parquet|read_text|csv_scan|json_scan|parquet_scan|sqlite_scan|glob|httpfs|"
    r"postgres_scan|mysql_scan|iceberg_scan|delta_scan)\b"
)
EXTERNAL_SCAN_RE = re.compile(r"(?is)\bfrom\s+['\"]")
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@contextmanager
def _runtime(config_path: str) -> Iterator[tuple[dict[str, Any], KB, KBTools]]:
    cfg = load_config(config_path)
    ensure_dirs(cfg)
    kb = KB(cfg["storage"]["duckdb_path"])
    tools = KBTools(kb, cfg["storage"].get("sqlite_fts_path"), cfg)
    try:
        yield cfg, kb, tools
    finally:
        kb.close()


def create_server(config_path: str) -> FastMCP:
    """Create the read-only Proofline MCP server for an existing local KB."""
    mcp = FastMCP("proofline")

    @mcp.tool()
    def proofline_status() -> dict[str, Any]:
        """Summarize the configured Proofline KB, table row counts, and recent pipeline runs."""
        with _runtime(config_path) as (cfg, kb, _tools):
            return _proofline_status(config_path, cfg, kb)

    @mcp.tool()
    def list_tables() -> dict[str, Any]:
        """List all DuckDB tables available for read-only inspection."""
        with _runtime(config_path) as (_cfg, kb, _tools):
            return _list_tables(kb)

    @mcp.tool()
    def get_table_schema(table_name: str) -> dict[str, Any]:
        """Return DuckDB column names and types for one table."""
        with _runtime(config_path) as (_cfg, kb, _tools):
            return _get_table_schema(kb, table_name)

    @mcp.tool()
    def describe_database() -> dict[str, Any]:
        """Return every table, its schema, and row count for planning SQL queries."""
        with _runtime(config_path) as (_cfg, kb, _tools):
            tables = _table_names(kb)
            return {
                "tables": [
                    {
                        **_get_table_schema(kb, table),
                        "row_count": _count_table(kb, table),
                    }
                    for table in tables
                ]
            }

    @mcp.tool()
    def corpus_overview() -> dict[str, Any]:
        """Return high-level corpus aggregates across repos, chunks, APIs, graph, and capabilities."""
        with _runtime(config_path) as (_cfg, _kb, tools):
            return _corpus_overview(tools)

    @mcp.tool()
    def sql_select(query: str, limit: int = 500, offset: int = 0) -> dict[str, Any]:
        """Run a safe read-only SELECT/WITH query, paginated with LIMIT and OFFSET."""
        with _runtime(config_path) as (_cfg, kb, _tools):
            safe = validate_select_sql(query)
            page_limit = _clamp_limit(limit, MAX_SQL_LIMIT)
            page_offset = _clamp_offset(offset)
            df = kb.query_df(
                f"SELECT * FROM ({safe}) AS pfl_subquery LIMIT ? OFFSET ?",
                [page_limit, page_offset],
            )
            rows = _records(df)
            return {"query": safe, "limit": page_limit, "offset": page_offset, "row_count": len(rows), "rows": rows}

    @mcp.tool()
    def sql_count(query: str) -> dict[str, Any]:
        """Count rows returned by a safe read-only SELECT/WITH query."""
        with _runtime(config_path) as (_cfg, kb, _tools):
            safe = validate_select_sql(query)
            df = kb.query_df(f"SELECT COUNT(*) AS row_count FROM ({safe}) AS pfl_subquery")
            return {"query": safe, "row_count": int(df.iloc[0]["row_count"]) if not df.empty else 0}

    @mcp.tool()
    def sample_table(table_name: str, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """Read a paginated sample of rows from one known DuckDB table."""
        with _runtime(config_path) as (_cfg, kb, _tools):
            table = _validate_table_name(kb, table_name)
            page_limit = _clamp_limit(limit)
            page_offset = _clamp_offset(offset)
            df = kb.query_df(f"SELECT * FROM {table} LIMIT ? OFFSET ?", [page_limit, page_offset])
            rows = _records(df)
            return {"table_name": table, "limit": page_limit, "offset": page_offset, "row_count": len(rows), "rows": rows}

    @mcp.tool()
    def list_repos(limit: int = 100, offset: int = 0) -> dict[str, Any]:
        """List indexed repositories and inventory metadata."""
        with _runtime(config_path) as (_cfg, kb, _tools):
            return _list_repos(kb, limit=limit, offset=offset)

    @mcp.tool()
    def list_repo_files(
        repo_id: str,
        kind: str | None = None,
        path_glob: str | None = None,
        limit: int = 500,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List indexed files for a repository, optionally filtered by kind and glob."""
        with _runtime(config_path) as (_cfg, kb, _tools):
            return _list_repo_files(kb, repo_id, kind=kind, path_glob=path_glob, limit=limit, offset=offset)

    @mcp.tool()
    def get_file_metadata(repo_id: str, rel_path: str) -> dict[str, Any]:
        """Return metadata for one indexed repo file."""
        with _runtime(config_path) as (_cfg, kb, _tools):
            row = _repo_file_row(kb, repo_id, rel_path)
            return {"found": bool(row), "file": row or {}}

    @mcp.tool()
    def read_file_slice(repo_id: str, rel_path: str, start_line: int = 1, end_line: int = 200) -> dict[str, Any]:
        """Read a line slice from a file indexed in repo_files for the given repo and relative path."""
        with _runtime(config_path) as (_cfg, kb, _tools):
            return _read_file_slice(kb, repo_id, rel_path, start_line=start_line, end_line=end_line)

    @mcp.tool()
    def search_code(query: str, repo_id: str | None = None, limit: int = 25) -> list[dict[str, Any]]:
        """Search code chunks using Proofline's combined FTS/vector retrieval."""
        with _runtime(config_path) as (_cfg, _kb, tools):
            return _clean(tools.search_code(query, repo_id=repo_id, limit=_clamp_limit(limit)))

    @mcp.tool()
    def search_code_fts(query: str, repo_id: str | None = None, limit: int = 25) -> list[dict[str, Any]]:
        """Search code chunks using the SQLite full-text index only."""
        with _runtime(config_path) as (_cfg, _kb, tools):
            return _clean(tools.search_code_fts(query, repo_id=repo_id, limit=_clamp_limit(limit)))

    @mcp.tool()
    def search_code_graph(query: str, repo_id: str | None = None, limit: int = 25) -> dict[str, Any]:
        """Search code graph symbols and nearby relationships by name, path, or signature."""
        with _runtime(config_path) as (_cfg, _kb, tools):
            return _clean(tools.search_code_graph(query, repo_id=repo_id, limit=_clamp_limit(limit)))

    @mcp.tool()
    def get_code_chunk(chunk_id: str) -> dict[str, Any]:
        """Return one indexed code chunk by chunk_id."""
        with _runtime(config_path) as (_cfg, kb, _tools):
            df = kb.query_df("SELECT * FROM code_chunks WHERE chunk_id = ? LIMIT 1", [chunk_id])
            rows = _records(df)
            return {"found": bool(rows), "chunk": rows[0] if rows else {}}

    @mcp.tool()
    def resolve_entity(name: str) -> dict[str, Any]:
        """Resolve a service, repo, alias, or Datadog name to a Proofline service identity."""
        with _runtime(config_path) as (_cfg, _kb, tools):
            return _clean(tools.resolve_project(name))

    @mcp.tool()
    def get_service_profile(service_id: str) -> dict[str, Any]:
        """Return service identity, API endpoints, and ownership evidence for a service."""
        with _runtime(config_path) as (_cfg, _kb, tools):
            return _clean(tools.get_service_profile(service_id))

    @mcp.tool()
    def get_dependencies(service_id: str, env: str = "prod", window_days: int = 30) -> list[dict[str, Any]]:
        """Return downstream runtime/static dependencies for a service."""
        with _runtime(config_path) as (_cfg, _kb, tools):
            return _clean(tools.get_service_dependencies(service_id, env=env, window_days=window_days)[:MAX_LIMIT])

    @mcp.tool()
    def get_dependents(service_id: str, env: str = "prod", window_days: int = 30) -> list[dict[str, Any]]:
        """Return upstream services/entities that depend on this service."""
        with _runtime(config_path) as (_cfg, _kb, tools):
            return _clean(tools.get_service_dependents(service_id, env=env, window_days=window_days)[:MAX_LIMIT])

    @mcp.tool()
    def get_endpoint_dependencies(service_id: str, env: str = "prod", window_days: int = 30) -> list[dict[str, Any]]:
        """Return endpoint-level downstream dependencies for a service."""
        with _runtime(config_path) as (_cfg, _kb, tools):
            return _clean(tools.get_endpoint_dependencies(service_id, env=env, window_days=window_days)[:MAX_LIMIT])

    @mcp.tool()
    def graph_neighborhood(node_id: str, limit: int = 100) -> dict[str, Any]:
        """Return local graph nodes and edges adjacent to a node_id such as service:foo or repo:bar."""
        with _runtime(config_path) as (_cfg, _kb, tools):
            return _clean(tools.get_graph_neighborhood(node_id, limit=_clamp_limit(limit)))

    @mcp.tool()
    def get_node(node_id: str) -> dict[str, Any]:
        """Return one graph node by node_id."""
        with _runtime(config_path) as (_cfg, kb, _tools):
            df = kb.query_df("SELECT * FROM nodes WHERE node_id = ? LIMIT 1", [node_id])
            rows = _records(df)
            return {"found": bool(rows), "node": rows[0] if rows else {}}

    @mcp.tool()
    def get_edges(
        from_node: str | None = None,
        to_node: str | None = None,
        edge_type: str | None = None,
        limit: int = 500,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List graph edges filtered by source node, target node, and/or edge type."""
        with _runtime(config_path) as (_cfg, kb, _tools):
            return _get_edges(kb, from_node=from_node, to_node=to_node, edge_type=edge_type, limit=limit, offset=offset)

    @mcp.tool()
    def get_evidence(evidence_ids: list[str]) -> list[dict[str, Any]]:
        """Return evidence rows for specific evidence_ids."""
        with _runtime(config_path) as (_cfg, _kb, tools):
            return _clean(tools.get_evidence(evidence_ids[:MAX_LIMIT]))

    @mcp.tool()
    def search_evidence(query: str, limit: int = 100) -> list[dict[str, Any]]:
        """Search evidence excerpts, refs, source systems, repo ids, and file paths."""
        with _runtime(config_path) as (_cfg, kb, _tools):
            return _search_evidence(kb, query, limit=limit)

    @mcp.tool()
    def list_api_endpoints(
        service_id: str | None = None,
        repo_id: str | None = None,
        limit: int = 500,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List indexed API endpoints, optionally filtered by service or repo."""
        with _runtime(config_path) as (_cfg, kb, _tools):
            return _list_api_endpoints(kb, service_id=service_id, repo_id=repo_id, limit=limit, offset=offset)

    @mcp.tool()
    def search_capabilities(query: str, limit: int = 50) -> list[dict[str, Any]]:
        """Search indexed data/API capabilities by capability name, fields, and provider."""
        with _runtime(config_path) as (_cfg, _kb, tools):
            return _clean(tools.search_capabilities(query, limit=_clamp_limit(limit)))

    @mcp.tool()
    def get_bq_usage(service_id: str, window_days: int = 30) -> list[dict[str, Any]]:
        """Return BigQuery table usage associated with a service identity."""
        with _runtime(config_path) as (_cfg, _kb, tools):
            return _clean(tools.get_bq_usage(service_id, window_days=window_days)[:MAX_LIMIT])

    @mcp.tool()
    def get_change_history(service_id: str, query: str = "", limit: int = 50) -> dict[str, Any]:
        """Return commits, semantic changes, owners, and co-change evidence for a service."""
        with _runtime(config_path) as (_cfg, _kb, tools):
            return _clean(tools.get_change_history(service_id, query=query, limit=_clamp_limit(limit)))

    @mcp.tool()
    def search_commits(
        query: str,
        repo_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Search git commits by subject, body, detected links, and optional repo."""
        with _runtime(config_path) as (_cfg, kb, _tools):
            return _search_commits(kb, query, repo_id=repo_id, limit=limit, offset=offset)

    @mcp.tool()
    def get_commit(commit_sha: str, repo_id: str | None = None) -> dict[str, Any]:
        """Return one commit plus changed files, hunks, detected links, and revert metadata."""
        with _runtime(config_path) as (_cfg, kb, _tools):
            return _get_commit(kb, commit_sha, repo_id=repo_id)

    @mcp.tool()
    def get_file_history(repo_id: str, rel_path: str, limit: int = 100) -> dict[str, Any]:
        """Return commit/change history and current blame rows for one indexed repo file."""
        with _runtime(config_path) as (_cfg, kb, _tools):
            return _get_file_history(kb, repo_id, rel_path, limit=limit)

    @mcp.tool()
    def list_raw_artifacts(source: str, limit: int = 500, offset: int = 0) -> dict[str, Any]:
        """List raw artifacts under configured Proofline raw directories for a source or all sources."""
        with _runtime(config_path) as (cfg, _kb, _tools):
            return _list_raw_artifacts(cfg, source, limit=limit, offset=offset)

    @mcp.tool()
    def read_raw_artifact(source: str, relative_path: str, max_chars: int = 20000) -> dict[str, Any]:
        """Read a capped raw artifact from a configured Proofline raw directory."""
        with _runtime(config_path) as (cfg, _kb, _tools):
            return _read_raw_artifact(cfg, source, relative_path, max_chars=max_chars)

    @mcp.tool()
    def search_raw_artifacts(source: str, query: str, limit: int = 50) -> dict[str, Any]:
        """Search text/JSON raw artifacts under configured Proofline raw directories."""
        with _runtime(config_path) as (cfg, _kb, _tools):
            return _search_raw_artifacts(cfg, source, query, limit=limit)

    @mcp.resource("proofline://status")
    def status_resource() -> str:
        """Proofline KB status as JSON."""
        with _runtime(config_path) as (cfg, kb, _tools):
            return json_dumps(_proofline_status(config_path, cfg, kb))

    @mcp.resource("proofline://schema")
    def schema_resource() -> str:
        """Proofline DuckDB schema as JSON."""
        with _runtime(config_path) as (_cfg, kb, _tools):
            tables = [_get_table_schema(kb, table) for table in _table_names(kb)]
            return json_dumps({"tables": tables})

    @mcp.resource("proofline://repos")
    def repos_resource() -> str:
        """Proofline repository inventory as JSON."""
        with _runtime(config_path) as (_cfg, kb, _tools):
            return json_dumps(_list_repos(kb, limit=500, offset=0))

    @mcp.resource("proofline://table/{table_name}/schema")
    def table_schema_resource(table_name: str) -> str:
        """One Proofline table schema as JSON."""
        with _runtime(config_path) as (_cfg, kb, _tools):
            return json_dumps(_get_table_schema(kb, table_name))

    @mcp.prompt()
    def answer_engineering_question(question: str) -> str:
        """Guide an LLM client through evidence-backed engineering question answering."""
        return (
            "Use the Proofline MCP tools to answer this engineering question: "
            f"{question}\n\n"
            "Start with proofline_status, describe_database or corpus_overview when scope is unclear. "
            "Use resolve_entity for project/service names, then combine SQL, code search, file slices, graph, "
            "evidence, runtime/data/API/history tools as needed. Cite concrete evidence ids, files, tables, "
            "commits, endpoints, and raw artifact paths. Separate static/code, runtime, data, ownership, and "
            "history evidence. Clearly call out unknowns and weak or missing source coverage."
        )

    @mcp.prompt()
    def impact_analysis(project: str, feature: str) -> str:
        """Guide impact analysis over dependencies, APIs, code, data, and history."""
        return (
            f"Assess impact for project/service {project} and feature {feature}. Resolve the project first, "
            "then inspect service profile, dependencies, dependents, endpoint dependencies, API endpoints, "
            "code search results, change history, and relevant evidence/raw artifacts. Group findings by "
            "direct code changes, downstream/upstream runtime impact, API contracts, data sources, ownership, "
            "and historical risk. Cite evidence and list unknowns."
        )

    @mcp.prompt()
    def dependency_report(project: str) -> str:
        """Guide a dependency report for a Proofline project/service."""
        return (
            f"Build a dependency report for {project}. Use resolve_entity, get_service_profile, "
            "get_dependencies, get_dependents, get_endpoint_dependencies, graph_neighborhood, and SQL "
            "aggregates over edges/runtime_service_edges/endpoint_dependency_map. Separate observed runtime "
            "dependencies from static/code/config evidence, cite evidence_refs, and call out stale or missing data."
        )

    @mcp.prompt()
    def data_source_recommendation(project: str, feature: str) -> str:
        """Guide data/API capability recommendations for a feature."""
        return (
            f"Recommend data sources or API capabilities for {project} and feature {feature}. Use "
            "search_capabilities, get_bq_usage, list_api_endpoints, search_code, graph tools, and evidence "
            "retrieval. Prefer sources with ownership, usage, documentation, and runtime/static corroboration. "
            "Cite capability ids, table names, endpoint ids, owners, and unknowns."
        )

    return mcp


def serve(config_path: str, transport: str = "stdio") -> None:
    """Run the Proofline MCP server over stdio, streamable-http, or sse."""
    if transport not in ALLOWED_TRANSPORTS:
        raise ValueError(f"Unsupported MCP transport: {transport}")
    create_server(config_path).run(transport=transport)


def validate_select_sql(query: str) -> str:
    sql = query.strip()
    if sql.endswith(";"):
        sql = sql[:-1].strip()
    if not sql:
        raise ValueError("SQL query is empty")
    if ";" in sql:
        raise ValueError("Multiple SQL statements are not allowed")
    if not re.match(r"(?is)^(select|with)\b", sql):
        raise ValueError("Only SELECT or WITH read-only queries are allowed")
    if BLOCKED_SQL.search(sql):
        raise ValueError("SQL contains a blocked keyword")
    if EXTERNAL_SCAN_RE.search(sql):
        raise ValueError("SQL external file scans are not allowed")
    return sql


def _proofline_status(config_path: str, cfg: dict[str, Any], kb: KB) -> dict[str, Any]:
    counts = {table: _count_table(kb, table) for table in _table_names(kb)}
    runs = _records(
        kb.query_df(
            """
            SELECT stage, started_at, finished_at, status, details
            FROM pipeline_runs
            ORDER BY finished_at DESC NULLS LAST, started_at DESC NULLS LAST
            LIMIT 25
            """
        )
    )
    return {
        "config_path": str(config_path),
        "duckdb_path": str(cfg["storage"]["duckdb_path"]),
        "sqlite_fts_path": str(cfg["storage"].get("sqlite_fts_path") or ""),
        "workspace": str(cfg.get("workspace") or ""),
        "table_counts": counts,
        "recent_pipeline_runs": runs,
    }


def _table_names(kb: KB) -> list[str]:
    df = kb.query_df(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """
    )
    return [str(x) for x in df["table_name"].tolist()] if not df.empty else []


def _list_tables(kb: KB) -> dict[str, Any]:
    return {"tables": _table_names(kb)}


def _get_table_schema(kb: KB, table_name: str) -> dict[str, Any]:
    table = _validate_table_name(kb, table_name)
    df = kb.query_df(
        """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'main' AND table_name = ?
        ORDER BY ordinal_position
        """,
        [table],
    )
    return {"table_name": table, "columns": _records(df)}


def _validate_table_name(kb: KB, table_name: str) -> str:
    table = table_name.strip()
    if not IDENTIFIER_RE.match(table):
        raise ValueError("Invalid table name")
    if table not in set(_table_names(kb)):
        raise ValueError(f"Unknown table: {table}")
    return table


def _count_table(kb: KB, table_name: str) -> int:
    table = _validate_table_name(kb, table_name)
    df = kb.query_df(f"SELECT COUNT(*) AS n FROM {table}")
    return int(df.iloc[0]["n"]) if not df.empty else 0


def _corpus_overview(tools: KBTools) -> dict[str, Any]:
    queries = {
        "counts": """
            SELECT 'repo_inventory' AS table_name, count(*) AS rows FROM repo_inventory
            UNION ALL SELECT 'repo_files', count(*) FROM repo_files
            UNION ALL SELECT 'code_chunks', count(*) FROM code_chunks
            UNION ALL SELECT 'code_embedding_index', count(*) FROM code_embedding_index
            UNION ALL SELECT 'code_graph_symbols', count(*) FROM code_graph_symbols
            UNION ALL SELECT 'code_graph_edges', count(*) FROM code_graph_edges
            UNION ALL SELECT 'api_endpoints', count(*) FROM api_endpoints
            UNION ALL SELECT 'endpoint_dependency_map', count(*) FROM endpoint_dependency_map
            UNION ALL SELECT 'data_capabilities', count(*) FROM data_capabilities
        """,
        "languages": """
            SELECT language, count(*) AS chunks, count(DISTINCT repo_id) AS repos
            FROM code_chunks
            WHERE language IS NOT NULL AND language <> ''
            GROUP BY language
            ORDER BY repos DESC, chunks DESC
            LIMIT 50
        """,
        "repo_chunk_counts": """
            SELECT repo_id, count(*) AS chunks, count(DISTINCT rel_path) AS files
            FROM code_chunks
            GROUP BY repo_id
            ORDER BY chunks DESC
            LIMIT 50
        """,
        "api_methods": """
            SELECT method, count(*) AS endpoints, count(DISTINCT service_id) AS services
            FROM api_endpoints
            GROUP BY method
            ORDER BY endpoints DESC
            LIMIT 50
        """,
        "code_graph_node_types": """
            SELECT node_type, count(*) AS symbols, count(DISTINCT repo_id) AS repos
            FROM code_graph_symbols
            GROUP BY node_type
            ORDER BY symbols DESC
        """,
        "code_graph_edge_types": """
            SELECT edge_type, count(*) AS edges, count(DISTINCT repo_id) AS repos
            FROM code_graph_edges
            GROUP BY edge_type
            ORDER BY edges DESC
        """,
    }
    return {name: _records(tools.kb.query_df(query)) for name, query in queries.items()}


def _list_repos(kb: KB, limit: int = 100, offset: int = 0) -> dict[str, Any]:
    page_limit = _clamp_limit(limit)
    page_offset = _clamp_offset(offset)
    df = kb.query_df(
        """
        SELECT *
        FROM repo_inventory
        ORDER BY repo_id
        LIMIT ? OFFSET ?
        """,
        [page_limit, page_offset],
    )
    rows = _records(df)
    return {"limit": page_limit, "offset": page_offset, "row_count": len(rows), "repos": rows}


def _list_repo_files(
    kb: KB,
    repo_id: str,
    *,
    kind: str | None = None,
    path_glob: str | None = None,
    limit: int = 500,
    offset: int = 0,
) -> dict[str, Any]:
    page_limit = _clamp_limit(limit)
    page_offset = _clamp_offset(offset)
    clauses = ["repo_id = ?"]
    params: list[Any] = [repo_id]
    if kind:
        clauses.append("kind = ?")
        params.append(kind)
    if path_glob:
        clauses.append("rel_path LIKE ?")
        params.append(_glob_to_like(path_glob))
    params.extend([page_limit, page_offset])
    df = kb.query_df(
        f"""
        SELECT *
        FROM repo_files
        WHERE {' AND '.join(clauses)}
        ORDER BY rel_path
        LIMIT ? OFFSET ?
        """,
        params,
    )
    rows = _records(df)
    return {"repo_id": repo_id, "limit": page_limit, "offset": page_offset, "row_count": len(rows), "files": rows}


def _repo_file_row(kb: KB, repo_id: str, rel_path: str) -> dict[str, Any] | None:
    df = kb.query_df(
        """
        SELECT *
        FROM repo_files
        WHERE repo_id = ? AND (rel_path = ? OR path = ?)
        ORDER BY CASE WHEN rel_path = ? THEN 0 ELSE 1 END
        LIMIT 1
        """,
        [repo_id, rel_path, rel_path, rel_path],
    )
    rows = _records(df)
    return rows[0] if rows else None


def _read_file_slice(kb: KB, repo_id: str, rel_path: str, *, start_line: int = 1, end_line: int = 200) -> dict[str, Any]:
    row = _repo_file_row(kb, repo_id, rel_path)
    if not row:
        raise ValueError("File is not indexed in repo_files")
    start = max(1, int(start_line))
    end = max(start, int(end_line))
    if end - start + 1 > MAX_FILE_LINES:
        end = start + MAX_FILE_LINES - 1
    path = Path(str(row.get("path") or ""))
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Indexed file is not readable: {path}")
    selected: list[str] = []
    total_lines = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            total_lines = line_no
            if start <= line_no <= end:
                selected.append(line.rstrip("\n"))
    return {
        "repo_id": repo_id,
        "rel_path": str(row.get("rel_path") or rel_path),
        "path": str(path),
        "start_line": start,
        "end_line": min(end, total_lines),
        "total_lines": total_lines,
        "text": "\n".join(selected),
    }


def _get_edges(
    kb: KB,
    *,
    from_node: str | None = None,
    to_node: str | None = None,
    edge_type: str | None = None,
    limit: int = 500,
    offset: int = 0,
) -> dict[str, Any]:
    clauses = ["1=1"]
    params: list[Any] = []
    if from_node:
        clauses.append("from_node = ?")
        params.append(from_node)
    if to_node:
        clauses.append("to_node = ?")
        params.append(to_node)
    if edge_type:
        clauses.append("edge_type = ?")
        params.append(edge_type)
    page_limit = _clamp_limit(limit)
    page_offset = _clamp_offset(offset)
    params.extend([page_limit, page_offset])
    df = kb.query_df(
        f"""
        SELECT *
        FROM edges
        WHERE {' AND '.join(clauses)}
        ORDER BY confidence DESC NULLS LAST, last_seen DESC NULLS LAST
        LIMIT ? OFFSET ?
        """,
        params,
    )
    rows = _records(df)
    return {"limit": page_limit, "offset": page_offset, "row_count": len(rows), "edges": rows}


def _search_evidence(kb: KB, query: str, limit: int = 100) -> list[dict[str, Any]]:
    terms = _terms(query)
    clauses: list[str] = []
    params: list[Any] = []
    for term in terms[:8]:
        like = f"%{term}%"
        clauses.append(
            "(lower(evidence_id) LIKE ? OR lower(source_system) LIKE ? OR lower(source_ref) LIKE ? "
            "OR lower(repo_id) LIKE ? OR lower(file_path) LIKE ? OR lower(raw_excerpt) LIKE ?)"
        )
        params.extend([like, like, like, like, like, like])
    where = " OR ".join(clauses) if clauses else "1=1"
    params.append(_clamp_limit(limit))
    return _records(
        kb.query_df(
            f"""
            SELECT *
            FROM evidence
            WHERE {where}
            ORDER BY confidence DESC NULLS LAST, observed_at DESC NULLS LAST
            LIMIT ?
            """,
            params,
        )
    )


def _list_api_endpoints(
    kb: KB,
    *,
    service_id: str | None = None,
    repo_id: str | None = None,
    limit: int = 500,
    offset: int = 0,
) -> dict[str, Any]:
    clauses = ["1=1"]
    params: list[Any] = []
    if service_id:
        clauses.append("service_id = ?")
        params.append(service_id.replace("service:", ""))
    if repo_id:
        clauses.append("repo_id = ?")
        params.append(repo_id)
    page_limit = _clamp_limit(limit)
    page_offset = _clamp_offset(offset)
    params.extend([page_limit, page_offset])
    rows = _records(
        kb.query_df(
            f"""
            SELECT *
            FROM api_endpoints
            WHERE {' AND '.join(clauses)}
            ORDER BY service_id, path, method
            LIMIT ? OFFSET ?
            """,
            params,
        )
    )
    return {"limit": page_limit, "offset": page_offset, "row_count": len(rows), "endpoints": rows}


def _search_commits(
    kb: KB,
    query: str,
    *,
    repo_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    clauses = ["1=1"]
    params: list[Any] = []
    if repo_id:
        clauses.append("repo_id = ?")
        params.append(repo_id)
    for term in _terms(query)[:8]:
        like = f"%{term}%"
        clauses.append("(lower(subject) LIKE ? OR lower(body) LIKE ? OR lower(detected_jira_keys) LIKE ? OR lower(detected_urls) LIKE ?)")
        params.extend([like, like, like, like])
    page_limit = _clamp_limit(limit)
    page_offset = _clamp_offset(offset)
    params.extend([page_limit, page_offset])
    rows = _records(
        kb.query_df(
            f"""
            SELECT repo_id, commit_sha, parent_shas, author_name, author_email, commit_time,
                   subject, body, is_merge, is_revert, is_hotfix, detected_jira_keys,
                   detected_urls, indexed_at
            FROM git_commits
            WHERE {' AND '.join(clauses)}
            ORDER BY commit_time DESC NULLS LAST
            LIMIT ? OFFSET ?
            """,
            params,
        )
    )
    return {"query": query, "limit": page_limit, "offset": page_offset, "row_count": len(rows), "commits": rows}


def _get_commit(kb: KB, commit_sha: str, *, repo_id: str | None = None) -> dict[str, Any]:
    clauses = ["commit_sha = ?"]
    params: list[Any] = [commit_sha]
    if repo_id:
        clauses.append("repo_id = ?")
        params.append(repo_id)
    commit_rows = _records(kb.query_df(f"SELECT * FROM git_commits WHERE {' AND '.join(clauses)} LIMIT 1", params))
    if not commit_rows:
        return {"found": False, "commit": {}, "file_changes": [], "patch_hunks": [], "detected_links": [], "reverts": []}
    repo = str(commit_rows[0].get("repo_id") or repo_id or "")
    params2 = [repo, commit_sha]
    return {
        "found": True,
        "commit": commit_rows[0],
        "file_changes": _records(kb.query_df("SELECT * FROM git_file_changes WHERE repo_id = ? AND commit_sha = ? LIMIT 1000", params2)),
        "patch_hunks": _records(kb.query_df("SELECT * FROM git_patch_hunks WHERE repo_id = ? AND commit_sha = ? LIMIT 500", params2)),
        "detected_links": _records(kb.query_df("SELECT * FROM git_detected_links WHERE repo_id = ? AND commit_sha = ? LIMIT 500", params2)),
        "reverts": _records(
            kb.query_df(
                "SELECT * FROM git_reverts WHERE repo_id = ? AND (revert_commit_sha = ? OR reverted_commit_sha = ?) LIMIT 50",
                [repo, commit_sha, commit_sha],
            )
        ),
    }


def _get_file_history(kb: KB, repo_id: str, rel_path: str, *, limit: int = 100) -> dict[str, Any]:
    if not _repo_file_row(kb, repo_id, rel_path):
        raise ValueError("File is not indexed in repo_files")
    page_limit = _clamp_limit(limit)
    changes = _records(
        kb.query_df(
            """
            SELECT c.repo_id, c.commit_sha, c.author_name, c.author_email, c.commit_time,
                   c.subject, c.is_revert, c.is_hotfix,
                   f.old_path, f.new_path, f.change_type, f.added_lines, f.deleted_lines,
                   f.is_rename, f.is_binary
            FROM git_file_changes f
            LEFT JOIN git_commits c ON c.repo_id = f.repo_id AND c.commit_sha = f.commit_sha
            WHERE f.repo_id = ? AND (f.new_path = ? OR f.old_path = ?)
            ORDER BY c.commit_time DESC NULLS LAST
            LIMIT ?
            """,
            [repo_id, rel_path, rel_path, page_limit],
        )
    )
    blame = _records(
        kb.query_df(
            """
            SELECT *
            FROM git_blame_current
            WHERE repo_id = ? AND file_path = ?
            ORDER BY line_start
            LIMIT 500
            """,
            [repo_id, rel_path],
        )
    )
    return {"repo_id": repo_id, "rel_path": rel_path, "limit": page_limit, "changes": changes, "current_blame": blame}


def _list_raw_artifacts(cfg: dict[str, Any], source: str, *, limit: int = 500, offset: int = 0) -> dict[str, Any]:
    page_limit = _clamp_limit(limit)
    page_offset = _clamp_offset(offset)
    sources = _raw_sources(source)
    rows: list[dict[str, Any]] = []
    scanned = 0
    wanted_end = page_offset + page_limit
    for src in sources:
        root = _raw_root(cfg, src)
        if not root.exists():
            continue
        for path in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: str(p)):
            if scanned >= wanted_end:
                break
            rel = path.relative_to(root).as_posix()
            stat = path.stat()
            if scanned >= page_offset:
                rows.append(
                    {
                        "source": src,
                        "relative_path": rel,
                        "size_bytes": int(stat.st_size),
                        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    }
                )
            scanned += 1
    return {"source": source, "limit": page_limit, "offset": page_offset, "row_count": len(rows), "artifacts": rows}


def _read_raw_artifact(cfg: dict[str, Any], source: str, relative_path: str, *, max_chars: int = 20000) -> dict[str, Any]:
    sources = _raw_sources(source)
    if len(sources) != 1:
        raise ValueError("read_raw_artifact requires a single source")
    root = _raw_root(cfg, sources[0])
    path = _resolve_under(root, relative_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Raw artifact not found: {relative_path}")
    char_limit = min(max(1, int(max_chars)), MAX_RAW_CHARS)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        text = f.read(char_limit + 1)
    return {
        "source": sources[0],
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": int(path.stat().st_size),
        "max_chars": char_limit,
        "truncated": len(text) > char_limit,
        "text": text[:char_limit],
    }


def _search_raw_artifacts(cfg: dict[str, Any], source: str, query: str, *, limit: int = 50) -> dict[str, Any]:
    terms = _terms(query)
    if not terms:
        raise ValueError("Raw artifact search query is empty")
    rows: list[dict[str, Any]] = []
    page_limit = _clamp_limit(limit)
    for src in _raw_sources(source):
        root = _raw_root(cfg, src)
        if not root.exists():
            continue
        for path in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: str(p)):
            if len(rows) >= page_limit:
                break
            if path.stat().st_size > MAX_RAW_FILE_BYTES:
                continue
            text = safe_read_text(path, max_bytes=MAX_RAW_FILE_BYTES)
            if not text:
                continue
            lower = text.lower()
            if not all(term in lower for term in terms[:6]):
                continue
            first = min((lower.find(term) for term in terms[:6] if lower.find(term) >= 0), default=0)
            start = max(0, first - 300)
            end = min(len(text), first + 700)
            rows.append({"source": src, "relative_path": path.relative_to(root).as_posix(), "snippet": text[start:end]})
    return {"source": source, "query": query, "limit": page_limit, "row_count": len(rows), "matches": rows}


def _raw_sources(source: str) -> list[str]:
    src = source.strip().lower()
    if src == "all":
        return sorted(RAW_SOURCES)
    if src not in RAW_SOURCES:
        raise ValueError(f"Unknown raw source: {source}")
    return [src]


def _raw_root(cfg: dict[str, Any], source: str) -> Path:
    configured = cfg.get(source, {}).get("output_dir") if isinstance(cfg.get(source), dict) else None
    return Path(str(configured or Path(str(cfg.get("workspace") or ".")) / "raw" / source)).expanduser().resolve()


def _resolve_under(root: Path, relative_path: str) -> Path:
    base = root.expanduser().resolve()
    target = (base / relative_path).resolve()
    if target != base and base not in target.parents:
        raise ValueError("Path escapes the configured directory")
    return target


def _glob_to_like(pattern: str) -> str:
    return pattern.replace("%", r"\%").replace("_", r"\_").replace("*", "%").replace("?", "_")


def _terms(query: str) -> list[str]:
    return [term for term in re.split(r"\W+", query.lower()) if len(term) > 1]


def _clamp_limit(limit: int, maximum: int = MAX_LIMIT) -> int:
    return min(max(1, int(limit)), maximum)


def _clamp_offset(offset: int) -> int:
    return max(0, int(offset))


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    return _clean(df.to_dict("records"))


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, tuple):
        return [_clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    try:
        if bool(pd.isna(value)):
            return None
    except Exception:
        pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item"):
        try:
            return _clean(value.item())
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return value
