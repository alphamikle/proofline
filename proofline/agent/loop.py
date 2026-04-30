from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from proofline.agent.providers import AgentProviderError, complete_with_agent
from proofline.agent.tools import KBTools
from proofline.utils import json_dumps, json_loads


MAX_ROWS = 500
MAX_ITERATIONS = 10
MAX_TOOL_CALLS_PER_ITERATION = 10
MAX_OBSERVATION_CHARS = 100_000


SYSTEM_PROMPT = f"""
You are an evidence-seeking engineering research agent over a local code knowledge base.

The user may ask any question in any language. Decide what data is needed,
request tool calls, inspect observations, iterate, and answer in the user's language.

Return ONLY valid JSON. Do not wrap it in markdown.

Action response schema:
{{
  "thought_summary": "short user-safe summary of what you are trying to learn",
  "actions": [
    {{"tool": "tool_name", "args": {{}}}}
  ],
  "final": null
}}

Final response schema:
{{
  "thought_summary": "short user-safe summary",
  "actions": [],
  "final": {{
    "answer": "final answer for the user",
    "evidence": [
      {{"source": "tool/table/source name", "summary": "what this evidence showed"}}
    ],
    "unknowns": ["important missing or weak evidence"]
  }}
}}

Available tools:
- sql_select(query): read-only DuckDB SQL. Only SELECT is allowed. Results are capped at {MAX_ROWS} rows.
- get_table_schema(table_name): columns for a DuckDB table.
- list_tables(): available DuckDB tables.
- list_repos(limit): repository inventory summary.
- corpus_overview(): broad aggregate evidence across the whole corpus.
- resolve_entity(name): resolve a service/repo/entity name.
- search_code(query, repo_id, limit): code/doc retrieval via FTS + vector search.
- search_code_graph(query, repo_id, limit): symbols and graph relationships matching names/paths/signatures.
- graph_neighborhood(node_id, limit): local graph neighborhood for service:/repo:/node ids.
- get_service_profile(service_id): service identity, endpoints, owners.
- get_dependencies(service_id, env, window_days): service dependency edges.
- get_endpoint_dependencies(service_id, env, window_days): endpoint dependency rows.
- search_capabilities(query, limit): API/data capability matches.

Guidance:
- Prefer concrete evidence from tools over assumptions.
- For corpus-wide questions, use corpus_overview and SQL aggregates before search_code.
- For service/repo questions, resolve the entity first.
- If a SQL query fails, correct it in the next iteration.
- Keep each iteration to at most {MAX_TOOL_CALLS_PER_ITERATION} tool calls.
- Stop once evidence is sufficient.
- Be explicit about missing evidence, disabled sources, or weak retrieval.
""".strip()


@dataclass
class ProgressReporter:
    mode: str = "human"  # human | jsonl | quiet
    events: List[Dict[str, Any]] = field(default_factory=list)

    def emit(self, event: str, **payload: Any) -> None:
        row = {"event": event, **payload}
        self.events.append(row)
        if self.mode == "quiet":
            return
        if self.mode == "jsonl":
            print(json_dumps(row), file=sys.stderr, flush=True)
            return
        print(self._human(row), file=sys.stderr, flush=True)

    def _human(self, row: Dict[str, Any]) -> str:
        event = row.get("event")
        if event == "model_wait":
            return f"[ask] waiting for model ({row.get('phase')})..."
        if event == "agent_attempt":
            model = row.get("model") or ""
            suffix = f" ({model})" if model else ""
            return f"[ask] agent {row.get('name')}{suffix}..."
        if event == "agent_success":
            return f"[ask] agent {row.get('name')} responded"
        if event == "agent_empty":
            return f"[ask] agent {row.get('name')} returned no text; trying fallback..."
        if event == "agent_error":
            return f"[ask] agent {row.get('name')} failed: {row.get('error')}; trying fallback..."
        if event == "plan":
            return f"[ask] iteration {row.get('iteration')}/{MAX_ITERATIONS}: {row.get('summary')}"
        if event == "tool_start":
            return f"[ask] -> {row.get('description') or row.get('tool')}"
        if event == "tool_end":
            bits = []
            if row.get("rows") is not None:
                bits.append(f"{row.get('rows')} rows")
            if row.get("chars") is not None:
                bits.append(f"{row.get('chars')} chars")
            suffix = f": {', '.join(bits)}" if bits else ""
            return f"[ask] <- {row.get('tool')}{suffix}"
        if event == "tool_error":
            return f"[ask] !! {row.get('tool')}: {row.get('error')}"
        if event == "final_start":
            return "[ask] finalizing..."
        if event == "invalid_model_response":
            return f"[ask] model returned invalid JSON; retrying ({row.get('error')})"
        return f"[ask] {event}"


def run_agentic_ask(
    question: str,
    tools: KBTools,
    cfg: Dict[str, Any],
    *,
    project: Optional[str] = None,
    env: Optional[str] = None,
    window_days: Optional[int] = None,
    agent_name: Optional[str] = None,
    quiet: bool = False,
    raw_trace: bool = False,
) -> Dict[str, Any]:
    reporter = ProgressReporter("quiet" if quiet else "jsonl" if raw_trace else "human")
    state: Dict[str, Any] = {
        "question": question,
        "hints": {
            "project": project,
            "env": env or cfg.get("agent", {}).get("default_env", "prod"),
            "window_days": window_days or int(cfg.get("agent", {}).get("default_window_days", 30)),
            "agent": agent_name or cfg.get("agent", {}).get("active"),
        },
        "observations": [],
    }

    final: Optional[Dict[str, Any]] = None
    for iteration in range(1, MAX_ITERATIONS + 1):
        reporter.emit("model_wait", phase=f"planning iteration {iteration}")
        decision = _call_agent(state, cfg, reporter, agent_name=agent_name)
        if decision.get("_invalid"):
            reporter.emit("invalid_model_response", error=decision.get("error"))
            state["observations"].append({
                "tool": "model_response_error",
                "ok": False,
                "error": decision.get("error"),
                "raw": str(decision.get("raw") or "")[:4000],
            })
            continue

        summary = str(decision.get("thought_summary") or "planning next retrieval step").strip()
        actions = decision.get("actions") or []
        final = decision.get("final") if isinstance(decision.get("final"), dict) else None
        if final:
            reporter.emit("final_start")
            break

        if not isinstance(actions, list) or not actions:
            state["observations"].append({
                "tool": "agent_control",
                "ok": False,
                "error": "No actions and no final answer were provided. Either call tools or return final.",
            })
            continue

        actions = actions[:MAX_TOOL_CALLS_PER_ITERATION]
        reporter.emit("plan", iteration=iteration, summary=summary, tool_calls=len(actions))
        for action in actions:
            observation = _execute_action(action, tools, state, reporter)
            state["observations"].append(_truncate_observation(observation))
        _trim_observations(state)

    if final is None:
        reporter.emit("model_wait", phase="final answer")
        final = _finalize_from_trace(state, cfg, reporter, agent_name=agent_name)
    return {
        "question": question,
        "final": final or {},
        "trace": reporter.events,
        "observations": state.get("observations", []),
    }


def _call_agent(
    state: Dict[str, Any],
    cfg: Dict[str, Any],
    reporter: ProgressReporter,
    *,
    agent_name: Optional[str] = None,
) -> Dict[str, Any]:
    prompt = "STATE JSON:\n" + json_dumps(_compact_for_prompt(state))
    try:
        raw = complete_with_agent(
            SYSTEM_PROMPT,
            prompt,
            cfg,
            agent_name=agent_name,
            on_event=lambda event, payload: reporter.emit(event, **payload),
        )
    except AgentProviderError:
        raise
    except Exception as e:
        return {"_invalid": True, "error": str(e), "raw": ""}
    if not raw:
        return {"_invalid": True, "error": "agent provider returned no text", "raw": ""}
    return _parse_json_response(raw)


def _finalize_from_trace(
    state: Dict[str, Any],
    cfg: Dict[str, Any],
    reporter: ProgressReporter,
    *,
    agent_name: Optional[str] = None,
) -> Dict[str, Any]:
    final_state = dict(state)
    final_state["instruction"] = "Return a final JSON response now. Do not call tools."
    prompt = "STATE JSON:\n" + json_dumps(_compact_for_prompt(final_state))
    try:
        raw = complete_with_agent(
            SYSTEM_PROMPT,
            prompt,
            cfg,
            agent_name=agent_name,
            on_event=lambda event, payload: reporter.emit(event, **payload),
        )
    except Exception as e:
        reporter.emit("tool_error", tool="finalize", error=str(e))
        return {"answer": f"Не удалось получить финальный ответ от модели: {e}", "evidence": [], "unknowns": []}
    parsed = _parse_json_response(raw or "")
    final = parsed.get("final") if isinstance(parsed, dict) else None
    if isinstance(final, dict):
        reporter.emit("final_start")
        return final
    return {
        "answer": str(raw or "Не удалось получить финальный ответ от модели.").strip(),
        "evidence": [],
        "unknowns": ["The model did not return the expected final JSON shape."],
    }


def _parse_json_response(raw: str) -> Dict[str, Any]:
    try:
        data = json_loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except Exception as e:
            return {"_invalid": True, "error": str(e), "raw": raw}
    return {"_invalid": True, "error": "no JSON object found", "raw": raw}


def _execute_action(action: Any, tools: KBTools, state: Dict[str, Any], reporter: ProgressReporter) -> Dict[str, Any]:
    if not isinstance(action, dict):
        return {"tool": "invalid_action", "ok": False, "error": "Action must be an object."}
    tool_name = str(action.get("tool") or "").strip()
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    reporter.emit("tool_start", tool=tool_name, description=_describe_action(tool_name, args))
    try:
        data = _dispatch_tool(tool_name, args, tools, state)
        chars = len(json_dumps(data))
        rows = _row_count(data)
        reporter.emit("tool_end", tool=tool_name, rows=rows, chars=chars)
        return {"tool": tool_name, "ok": True, "args": _safe_args(args), "data": data}
    except Exception as e:
        reporter.emit("tool_error", tool=tool_name, error=str(e))
        return {"tool": tool_name, "ok": False, "args": _safe_args(args), "error": str(e)}


def _dispatch_tool(tool_name: str, args: Dict[str, Any], tools: KBTools, state: Dict[str, Any]) -> Any:
    if tool_name == "sql_select":
        return _sql_select(tools, str(args.get("query") or ""))
    if tool_name == "get_table_schema":
        return _get_table_schema(tools, str(args.get("table_name") or ""))
    if tool_name == "list_tables":
        return _list_tables(tools)
    if tool_name == "list_repos":
        return _list_repos(tools, int(args.get("limit") or 100))
    if tool_name == "corpus_overview":
        return _corpus_overview(tools)
    if tool_name == "resolve_entity":
        return tools.resolve_project(str(args.get("name") or args.get("query") or ""))
    if tool_name == "search_code":
        return tools.search_code(str(args.get("query") or ""), repo_id=_optional_str(args.get("repo_id")), limit=min(int(args.get("limit") or 25), MAX_ROWS))
    if tool_name == "search_code_graph":
        return tools.search_code_graph(str(args.get("query") or ""), repo_id=_optional_str(args.get("repo_id")), limit=min(int(args.get("limit") or 25), MAX_ROWS))
    if tool_name == "graph_neighborhood":
        return tools.get_graph_neighborhood(str(args.get("node_id") or ""), limit=min(int(args.get("limit") or 100), MAX_ROWS))
    if tool_name == "get_service_profile":
        return tools.get_service_profile(str(args.get("service_id") or ""))
    if tool_name == "get_dependencies":
        return tools.get_service_dependencies(
            str(args.get("service_id") or ""),
            env=str(args.get("env") or state.get("hints", {}).get("env") or "prod"),
            window_days=int(args.get("window_days") or state.get("hints", {}).get("window_days") or 30),
        )[:MAX_ROWS]
    if tool_name == "get_endpoint_dependencies":
        return tools.get_endpoint_dependencies(
            str(args.get("service_id") or ""),
            env=str(args.get("env") or state.get("hints", {}).get("env") or "prod"),
            window_days=int(args.get("window_days") or state.get("hints", {}).get("window_days") or 30),
        )[:MAX_ROWS]
    if tool_name == "search_capabilities":
        return tools.search_capabilities(str(args.get("query") or ""), limit=min(int(args.get("limit") or 50), MAX_ROWS))
    raise ValueError(f"Unknown tool: {tool_name}")


def _sql_select(tools: KBTools, query: str) -> Dict[str, Any]:
    safe = _validate_select_sql(query)
    df = tools.kb.query_df(f"SELECT * FROM ({safe}) AS pfl_subquery LIMIT {MAX_ROWS}")
    return {"query": safe, "row_count": len(df), "rows": df.to_dict("records")}


def _validate_select_sql(query: str) -> str:
    sql = query.strip().rstrip(";").strip()
    if not sql:
        raise ValueError("SQL query is empty")
    if not re.match(r"(?is)^select\b", sql):
        raise ValueError("Only SELECT statements are allowed")
    blocked = re.compile(r"(?is)\b(copy|attach|install|load|export|pragma|create|drop|delete|update|insert|alter|detach|call)\b")
    if blocked.search(sql):
        raise ValueError("SQL contains a blocked keyword")
    if ";" in sql:
        raise ValueError("Multiple SQL statements are not allowed")
    return sql


def _get_table_schema(tools: KBTools, table_name: str) -> Dict[str, Any]:
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table_name):
        raise ValueError("Invalid table name")
    df = tools.kb.query_df(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'main' AND table_name = ?
        ORDER BY ordinal_position
        """,
        [table_name],
    )
    return {"table_name": table_name, "columns": df.to_dict("records")}


def _list_tables(tools: KBTools) -> Dict[str, Any]:
    df = tools.kb.query_df(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
        ORDER BY table_name
        """
    )
    return {"tables": df["table_name"].tolist() if not df.empty else []}


def _list_repos(tools: KBTools, limit: int = 100) -> Dict[str, Any]:
    df = tools.kb.query_df(
        f"""
        SELECT repo_id, primary_language, probable_type, size_mb, worktree_size_mb,
               has_readme, has_openapi, has_proto, has_graphql, has_dockerfile,
               has_k8s, has_terraform, has_package_manifest, last_commit_at
        FROM repo_inventory
        ORDER BY repo_id
        LIMIT {min(limit, MAX_ROWS)}
        """
    )
    return {"row_count": len(df), "repos": df.to_dict("records")}


def _corpus_overview(tools: KBTools) -> Dict[str, Any]:
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
        "top_dirs": """
            SELECT regexp_extract(rel_path, '^([^/]+)', 1) AS top_dir,
                   count(*) AS chunks,
                   count(DISTINCT repo_id) AS repos
            FROM code_chunks
            WHERE rel_path LIKE '%/%'
            GROUP BY top_dir
            ORDER BY repos DESC, chunks DESC
            LIMIT 80
        """,
        "manifest_files": """
            SELECT lower(regexp_extract(rel_path, '[^/]+$')) AS file_name,
                   count(DISTINCT repo_id) AS repos,
                   count(*) AS files
            FROM repo_files
            WHERE lower(rel_path) IN ('package.json','pubspec.yaml','pyproject.toml','requirements.txt','go.mod','pom.xml','build.gradle','gradle.properties','cargo.toml','package-lock.json','yarn.lock','pnpm-lock.yaml','dockerfile')
               OR lower(rel_path) LIKE '%/package.json'
               OR lower(rel_path) LIKE '%/pubspec.yaml'
               OR lower(rel_path) LIKE '%/pyproject.toml'
               OR lower(rel_path) LIKE '%/requirements.txt'
               OR lower(rel_path) LIKE '%/go.mod'
               OR lower(rel_path) LIKE '%/pom.xml'
               OR lower(rel_path) LIKE '%/build.gradle'
               OR lower(rel_path) LIKE '%/cargo.toml'
            GROUP BY file_name
            ORDER BY repos DESC, files DESC
            LIMIT 60
        """,
        "test_layouts": """
            SELECT repo_id, count(DISTINCT rel_path) AS test_files
            FROM repo_files
            WHERE lower(rel_path) LIKE '%test%' OR lower(rel_path) LIKE '%spec%'
            GROUP BY repo_id
            ORDER BY test_files DESC
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
        "api_methods": """
            SELECT method, count(*) AS endpoints, count(DISTINCT service_id) AS services
            FROM api_endpoints
            GROUP BY method
            ORDER BY endpoints DESC
            LIMIT 50
        """,
    }
    out: Dict[str, Any] = {}
    for name, query in queries.items():
        out[name] = tools.kb.query_df(query).to_dict("records")
    return out


def _describe_action(tool_name: str, args: Dict[str, Any]) -> str:
    if tool_name == "sql_select":
        query = " ".join(str(args.get("query") or "").split())
        return f"sql_select: {query[:120]}"
    if tool_name in {"search_code", "search_code_graph", "search_capabilities"}:
        return f"{tool_name}: {str(args.get('query') or '')[:120]}"
    if tool_name in {"resolve_entity", "get_service_profile", "get_dependencies", "get_endpoint_dependencies"}:
        return f"{tool_name}: {args.get('name') or args.get('service_id') or ''}"
    return tool_name


def _truncate_observation(observation: Dict[str, Any]) -> Dict[str, Any]:
    raw = json_dumps(observation)
    if len(raw) <= MAX_OBSERVATION_CHARS:
        return observation
    compact = {
        "tool": observation.get("tool"),
        "ok": observation.get("ok"),
        "args": observation.get("args"),
        "truncated": True,
        "data_preview": raw[:MAX_OBSERVATION_CHARS],
    }
    if observation.get("error"):
        compact["error"] = observation.get("error")
    return compact


def _trim_observations(state: Dict[str, Any]) -> None:
    observations = state.get("observations") or []
    while len(json_dumps(observations)) > MAX_OBSERVATION_CHARS and len(observations) > 1:
        observations.pop(0)
    state["observations"] = observations


def _compact_for_prompt(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        return str(value)[:1000]
    if isinstance(value, list):
        limit = 40
        items = [_compact_for_prompt(item, depth + 1) for item in value[:limit]]
        if len(value) > limit:
            items.append({"_truncated_items": len(value) - limit})
        return items
    if isinstance(value, dict):
        return {str(k): _compact_for_prompt(v, depth + 1) for k, v in value.items()}
    if isinstance(value, str):
        return value if len(value) <= 5000 else value[:5000] + "...[truncated]"
    return value


def _row_count(data: Any) -> Optional[int]:
    if isinstance(data, dict):
        if isinstance(data.get("row_count"), int):
            return data.get("row_count")
        for key in ("rows", "repos", "tables", "columns"):
            if isinstance(data.get(key), list):
                return len(data[key])
    if isinstance(data, list):
        return len(data)
    return None


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_args(args: Dict[str, Any]) -> Dict[str, Any]:
    return _compact_for_prompt(args)
