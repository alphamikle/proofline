from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Optional

import duckdb
import pandas as pd


class KB:
    def __init__(self, duckdb_path: str | Path):
        self.path = Path(duckdb_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.path))
        self.con.execute("PRAGMA threads=8")
        self.ensure_schema()

    def close(self) -> None:
        self.con.close()

    def ensure_schema(self) -> None:
        ddl = [
            """
            CREATE TABLE IF NOT EXISTS repo_inventory (
              repo_id TEXT, repo_path TEXT, repo_url TEXT, default_branch TEXT,
              commit_sha TEXT, primary_language TEXT, languages TEXT,
              probable_type TEXT, size_mb DOUBLE, worktree_size_mb DOUBLE,
              last_commit_at TEXT, has_codeowners BOOLEAN, has_readme BOOLEAN,
              has_openapi BOOLEAN, has_proto BOOLEAN, has_graphql BOOLEAN,
              has_asyncapi BOOLEAN, has_dockerfile BOOLEAN, has_k8s BOOLEAN,
              has_helm BOOLEAN, has_terraform BOOLEAN, has_package_manifest BOOLEAN,
              indexed_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS repo_files (
              repo_id TEXT, path TEXT, rel_path TEXT, ext TEXT, size_bytes BIGINT,
              kind TEXT, sha1 TEXT, indexed_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS repo_git_history (
              repo_id TEXT, commit_sha TEXT, author_name TEXT, author_email TEXT,
              commit_time TEXT, subject TEXT, changed_files TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ownership (
              entity_id TEXT, entity_type TEXT, owner_team TEXT, owner_people TEXT,
              source TEXT, confidence DOUBLE, evidence_ref TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS code_chunks (
              chunk_id TEXT, repo_id TEXT, file_path TEXT, rel_path TEXT, language TEXT,
              kind TEXT, symbol TEXT, start_line INTEGER, end_line INTEGER,
              text TEXT, metadata TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS code_embedding_index (
              faiss_id BIGINT, chunk_id TEXT, repo_id TEXT, rel_path TEXT,
              language TEXT, kind TEXT, symbol TEXT, start_line INTEGER,
              end_line INTEGER, text_sha1 TEXT, model_name TEXT, vector_dim INTEGER,
              embedded_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS api_contracts (
              contract_id TEXT, service_id TEXT, repo_id TEXT, contract_type TEXT,
              source_file TEXT, docs_url TEXT, commit_sha TEXT, confidence DOUBLE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS api_endpoints (
              endpoint_id TEXT, service_id TEXT, repo_id TEXT, contract_id TEXT,
              method TEXT, path TEXT, operation_id TEXT, request_schema TEXT,
              response_schema TEXT, source_file TEXT, source TEXT, confidence DOUBLE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS static_edges (
              edge_id TEXT, from_entity TEXT, to_entity TEXT, edge_type TEXT,
              source TEXT, repo_id TEXT, file_path TEXT, line_start INTEGER,
              line_end INTEGER, raw_match TEXT, confidence DOUBLE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS datadog_services (
              datadog_service TEXT, env TEXT, raw TEXT, source TEXT, pulled_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS datadog_service_edges (
              from_service TEXT, to_service TEXT, env TEXT, window_days INTEGER,
              source TEXT, first_seen TEXT, last_seen TEXT, confidence DOUBLE, raw TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS datadog_spans (
              span_id TEXT, trace_id TEXT, parent_id TEXT, service TEXT, env TEXT,
              resource TEXT, operation TEXT, route TEXT, method TEXT, url TEXT,
              peer_service TEXT, host TEXT, db_name TEXT, messaging_destination TEXT,
              duration_ms DOUBLE, error BOOLEAN, timestamp TEXT, raw TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS datadog_logs (
              log_id TEXT, trace_id TEXT, span_id TEXT, service TEXT, env TEXT,
              route TEXT, method TEXT, status_code TEXT, url TEXT, host TEXT,
              peer_service TEXT, db_name TEXT, messaging_destination TEXT,
              duration_ms DOUBLE, timestamp TEXT, message TEXT, raw TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS runtime_service_edges (
              edge_id TEXT, from_service TEXT, to_entity TEXT, to_type TEXT,
              edge_type TEXT, env TEXT, source TEXT, window_days INTEGER,
              count BIGINT, p95_ms DOUBLE, error_rate DOUBLE, first_seen TEXT,
              last_seen TEXT, confidence DOUBLE, evidence_refs TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS runtime_endpoint_edges (
              edge_id TEXT, service_id TEXT, endpoint_key TEXT, method TEXT, path TEXT,
              downstream_entity TEXT, downstream_type TEXT, dependency_kind TEXT,
              env TEXT, source TEXT, window_days INTEGER, count BIGINT, p95_ms DOUBLE,
              error_rate DOUBLE, first_seen TEXT, last_seen TEXT, confidence DOUBLE,
              evidence_refs TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS bq_jobs (
              job_id TEXT, project_id TEXT, user_email TEXT, creation_time TEXT,
              query_hash TEXT, referenced_tables TEXT, destination_table TEXT,
              total_bytes_processed BIGINT, total_slot_ms BIGINT, raw TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS bq_table_usage (
              principal_email TEXT, service_account TEXT, referenced_table TEXT,
              destination_table TEXT, query_hash TEXT, job_count BIGINT,
              last_seen TEXT, total_bytes_processed BIGINT, source TEXT, confidence DOUBLE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS service_account_map (
              service_account TEXT, service_id TEXT, repo_id TEXT, source TEXT,
              confidence DOUBLE, evidence_ref TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS service_identity (
              service_id TEXT, display_name TEXT, repo_id TEXT, repo_path TEXT,
              datadog_service TEXT, owner_team TEXT, api_docs TEXT,
              confidence DOUBLE, evidence_refs TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS entity_aliases (
              canonical_id TEXT, alias TEXT, alias_type TEXT, source TEXT,
              confidence DOUBLE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS nodes (
              node_id TEXT, node_type TEXT, display_name TEXT, source TEXT,
              confidence DOUBLE, properties TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS edges (
              edge_id TEXT, from_node TEXT, to_node TEXT, edge_type TEXT, env TEXT,
              source TEXT, first_seen TEXT, last_seen TEXT, confidence DOUBLE,
              evidence_refs TEXT, properties TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS evidence (
              evidence_id TEXT, evidence_type TEXT, source_system TEXT, source_ref TEXT,
              repo_id TEXT, file_path TEXT, line_start INTEGER, line_end INTEGER,
              raw_excerpt TEXT, observed_at TEXT, confidence DOUBLE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS endpoint_dependency_map (
              service_id TEXT, endpoint_id TEXT, method TEXT, path TEXT,
              downstream_entity TEXT, downstream_type TEXT, dependency_kind TEXT,
              env TEXT, runtime_count_7d BIGINT, runtime_count_30d BIGINT,
              p95_ms DOUBLE, error_rate DOUBLE, static_evidence_count BIGINT,
              runtime_evidence_count BIGINT, sources TEXT, confidence DOUBLE,
              evidence_refs TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS unresolved_entities (
              entity TEXT, entity_type TEXT, reason TEXT, confidence DOUBLE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS compatibility_index (
              entity_id TEXT, entity_type TEXT, service_id TEXT, risk_kind TEXT,
              checklist TEXT, confidence DOUBLE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS data_capabilities (
              capability_id TEXT, provider_entity TEXT, capability_name TEXT,
              fields TEXT, access_method TEXT, docs_url TEXT, owner_team TEXT,
              usage_count_30d BIGINT, confidence DOUBLE, evidence_refs TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS code_graph_runs (
              repo_id TEXT, repo_path TEXT, status TEXT, started_at TEXT,
              finished_at TEXT, command TEXT, details TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS code_graph_symbols (
              symbol_id TEXT, repo_id TEXT, node_type TEXT, name TEXT,
              file_path TEXT, rel_path TEXT, line_start INTEGER,
              line_end INTEGER, language TEXT, signature TEXT,
              source TEXT, properties TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS code_graph_edges (
              edge_id TEXT, repo_id TEXT, from_symbol_id TEXT,
              to_symbol_id TEXT, edge_type TEXT, file_path TEXT,
              rel_path TEXT, line_start INTEGER, source TEXT,
              confidence DOUBLE, properties TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS neo4j_exports (
              exported_at TEXT, node_count BIGINT, edge_count BIGINT,
              evidence_count BIGINT, status TEXT, details TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS pipeline_runs (
              stage TEXT, started_at TEXT, finished_at TEXT, status TEXT, details TEXT
            )
            """,
        ]
        for statement in ddl:
            self.con.execute(statement)

    def replace_df(self, table: str, df: pd.DataFrame) -> None:
        self.con.execute(f"DELETE FROM {table}")
        if not df.empty:
            self.con.register("_df", df)
            self.con.execute(f"INSERT INTO {table} SELECT * FROM _df")
            self.con.unregister("_df")

    def append_df(self, table: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        self.con.register("_df", df)
        self.con.execute(f"INSERT INTO {table} SELECT * FROM _df")
        self.con.unregister("_df")

    def query_df(self, sql: str, params: Optional[list[Any]] = None) -> pd.DataFrame:
        if params is None:
            return self.con.execute(sql).fetchdf()
        return self.con.execute(sql, params).fetchdf()

    def execute(self, sql: str, params: Optional[list[Any]] = None) -> None:
        if params is None:
            self.con.execute(sql)
        else:
            self.con.execute(sql, params)
