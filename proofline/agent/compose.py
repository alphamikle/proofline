from __future__ import annotations

from typing import Any, Dict, List

from proofline.agent.providers import AgentProviderError, complete_with_agent
from proofline.utils import json_dumps

SYSTEM_RULES = """
You are an engineering LLM agent operating over a local evidence-backed knowledge graph.
Do not add facts without evidence. If evidence is weak, use likely, inferred, or unknown.
Always separate runtime evidence, static evidence, code-graph evidence from CGC/Neo4j, BigQuery/data evidence, and ownership evidence.
Use code_graph and graph_neighborhood sections to reason about files, symbols, calls, imports, inheritance, and blast radius.
Answer in a structured engineering style with risks, unknowns, and recommendations.
""".strip()


def maybe_llm_answer(context: Dict[str, Any], cfg: Dict[str, Any]) -> str | None:
    agent_cfg = cfg.get("agent", {})
    max_chars = int(agent_cfg.get("max_context_chars") or 100000)
    context_json = json_dumps(_compact_context(context))
    if len(context_json) > max_chars:
        context_json = context_json[:max_chars] + "\n...[context truncated]"
    user_prompt = "CONTEXT JSON:\n" + context_json
    try:
        return complete_with_agent(SYSTEM_RULES, user_prompt, cfg)
    except AgentProviderError:
        raise
    except Exception:
        return None


def _compact_context(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        return str(value)[:500]
    if isinstance(value, list):
        limit = 25
        items = [_compact_context(item, depth + 1) for item in value[:limit]]
        if len(value) > limit:
            items.append({"_truncated_items": len(value) - limit})
        return items
    if isinstance(value, dict):
        return {str(k): _compact_context(v, depth + 1) for k, v in value.items()}
    if isinstance(value, str):
        return value if len(value) <= 2000 else value[:2000] + "...[truncated]"
    return value


def render_markdown(context: Dict[str, Any]) -> str:
    qtype = context.get("question_type")
    if qtype == "impact_analysis":
        return render_impact(context)
    if qtype == "data_source_recommendation":
        return render_data_source(context)
    if qtype == "dependency_report":
        return render_dependency_report(context)
    return render_generic(context)


def svc_name(ctx: Dict[str, Any]) -> str:
    svc = ((ctx.get("project") or {}).get("service") or {})
    return svc.get("service_id") or svc.get("display_name") or ctx.get("project_name") or "unknown"


def render_impact(ctx: Dict[str, Any]) -> str:
    out = []
    service = svc_name(ctx)
    out.append(f"# Preliminary impact analysis: `{ctx.get('feature_name', '')}` in `{service}`")
    out.append(
        "\n> This analysis is based on the current graph/context. "
        "If no concrete diff or branch was provided, this is feature-name-based analysis, "
        "not a final change review."
    )
    deps = ctx.get("dependencies", [])
    dependents = ctx.get("dependents", [])
    bq = ctx.get("bq_usage", [])
    code_hits = ctx.get("code_hits", [])
    code_graph = ctx.get("code_graph", {})
    caps = ctx.get("capabilities", [])
    out.append(
        f"\n## Summary\n"
        f"- Downstream dependencies: {len(deps)}\n"
        f"- Upstream/runtime dependents: {len(dependents)}\n"
        f"- BigQuery usage rows linked to service: {len(bq)}\n"
        f"- Code graph symbols: {len(code_graph.get('symbols', [])) if isinstance(code_graph, dict) else 0}\n"
        f"- Code graph relationships: {len(code_graph.get('relationships', [])) if isinstance(code_graph, dict) else 0}\n"
        f"- Code/doc hits for feature text: {len(code_hits)}\n"
        f"- Capability matches: {len(caps)}"
    )
    out.append("\n## Potentially affected upstream consumers")
    out.extend(render_edge_list(dependents[:30], direction="dependent"))
    out.append("\n## Downstream dependencies to consider")
    out.extend(render_edge_list(deps[:30], direction="dependency"))
    out.append("\n## Compatibility risk checklist")
    out.append("- API contract: removed endpoint, changed path/method, added required request field, removed or renamed response field, enum/type/nullability change.")
    out.append("- Events/messages: required field added, schema version or meaning changed, event renamed or removed.")
    out.append("- Data: BigQuery/table schema changes, output table contract changes, destructive migrations.")
    out.append("- Static clients: generated clients, strict deserialization, DTO/schema imports.")
    out.append("\n## Code/doc search hits")
    out.extend(render_code_hits(code_hits[:10]))
    out.append("\n## Code graph hits")
    out.extend(render_code_graph(ctx.get("code_graph", {})))
    out.append("\n## Unknowns")
    out.append("- No concrete diff was provided. For precise breaking-change analysis, pass a branch, PR, diff, or affected files.")
    out.append("- Low-confidence or static-only dependencies should be checked with the relevant owners.")
    return "\n".join(out)


def render_data_source(ctx: Dict[str, Any]) -> str:
    out = []
    service = svc_name(ctx)
    out.append(f"# Data-source recommendation for `{ctx.get('feature_name', '')}` in `{service}`")
    caps = ctx.get("capabilities", [])
    if not caps:
        out.append(
            "\nNo strong capability matches were found. Try a more specific entity or field name, "
            "or verify that API specs and BigQuery jobs were ingested."
        )
        return "\n".join(out)
    out.append("\n## Candidates")
    for i, c in enumerate(caps[:15], 1):
        out.append(f"{i}. `{c.get('provider_entity')}` - {c.get('capability_name')}")
        out.append(
            f"   - access: {c.get('access_method')}; docs/source: `{c.get('docs_url')}`; "
            f"owner: {c.get('owner_team') or 'unknown'}; confidence: {fmt(c.get('confidence'))}"
        )
        if c.get("provider_entity", "").startswith("bq_table:"):
            out.append("   - note: this BigQuery table may be an analytics or batch snapshot; verify freshness and source-of-truth status before using it in an online path.")
    out.append("\n## Suggested next step")
    out.append(
        "For the top API candidate, open docs_url/source_file and generate a request example from the request/response schema. "
        "If the top candidate is a BigQuery table, use it as a data-lineage clue rather than automatically treating it as an online source of truth."
    )
    return "\n".join(out)


def render_dependency_report(ctx: Dict[str, Any]) -> str:
    out = []
    service = svc_name(ctx)
    out.append(f"# Dependency report: `{service}`")
    profile = ctx.get("profile", {})
    endpoints = (profile.get("endpoints") or [])
    deps = ctx.get("dependencies", [])
    epdeps = ctx.get("endpoint_dependencies", [])
    code_graph = ctx.get("code_graph", {})
    out.append(
        f"\n## Summary\n"
        f"- Endpoints discovered: {len(endpoints)}\n"
        f"- Service-level dependencies: {len(deps)}\n"
        f"- Endpoint dependency rows: {len(epdeps)}\n"
        f"- Code graph symbols: {len(code_graph.get('symbols', [])) if isinstance(code_graph, dict) else 0}"
    )
    out.append("\n## Service-level dependencies")
    out.extend(render_edge_list(deps[:50], direction="dependency"))
    out.append("\n## Endpoint-level dependency map")
    grouped = {}
    for row in epdeps:
        key = f"{row.get('method') or ''} {row.get('path') or row.get('endpoint_id') or ''}".strip()
        grouped.setdefault(key, []).append(row)
    if not grouped:
        out.append(
            "No endpoint-level rows were found. Possible reasons: missing Datadog span/log route fields, "
            "missing API surface, or unresolved repo/service identity mapping."
        )
    for ep, rows in list(grouped.items())[:100]:
        out.append(f"\n### {ep}")
        for r in rows[:20]:
            out.append(
                f"- `{r.get('downstream_entity')}` ({r.get('downstream_type')}, {r.get('dependency_kind')}) - "
                f"confidence {fmt(r.get('confidence'))}, runtime30d={r.get('runtime_count_30d')}, "
                f"p95={r.get('p95_ms')}, err={r.get('error_rate')}; sources={r.get('sources')}"
            )
    out.append("\n## Simplification candidates")
    out.extend(render_simplification_candidates(epdeps, deps))
    out.append("\n## Code graph context")
    out.extend(render_code_graph(ctx.get("code_graph", {})))
    return "\n".join(out)


def render_generic(ctx: Dict[str, Any]) -> str:
    return "# Context pack\n\n```json\n" + json_dumps(ctx)[:200000] + "\n```"


def render_edge_list(edges: List[Dict[str, Any]], direction: str) -> List[str]:
    out = []
    if not edges:
        return ["- No data."]
    for e in edges:
        if "to_entity" in e:
            target = e.get("to_entity")
            src = e.get("from_service")
            out.append(
                f"- `{src}` -> `{target}` ({e.get('edge_type')}, env={e.get('env')}, source={e.get('source')}, "
                f"confidence={fmt(e.get('confidence'))}, count={e.get('count')}, p95={e.get('p95_ms')}, "
                f"error_rate={e.get('error_rate')})"
            )
        elif "from_node" in e:
            out.append(f"- `{e.get('from_node')}` -> `{e.get('to_node')}` ({e.get('edge_type')}, source={e.get('source')}, confidence={fmt(e.get('confidence'))})")
        else:
            out.append(f"- `{e}`")
    return out


def render_code_hits(hits: List[Dict[str, Any]]) -> List[str]:
    if not hits:
        return ["- No code/doc hits."]
    out = []
    for h in hits:
        txt = (h.get("text") or "").strip().replace("\n", " ")[:240]
        out.append(f"- `{h.get('repo_id')}/{h.get('rel_path')}` {h.get('kind')} {h.get('symbol') or ''}: {txt}")
    return out


def render_code_graph(graph: Dict[str, Any]) -> List[str]:
    if not isinstance(graph, dict):
        return ["- No code graph data."]
    symbols = graph.get("symbols") or []
    rels = graph.get("relationships") or []
    if not symbols and not rels:
        return ["- No code graph data."]
    out = []
    for s in symbols[:12]:
        out.append(f"- `{s.get('repo_id')}/{s.get('rel_path')}` {s.get('node_type')} `{s.get('name')}` line={s.get('line_start')}")
    if rels:
        out.append("- Relationships:")
        for r in rels[:20]:
            out.append(
                f"- `{r.get('from_name')}` -> `{r.get('to_name')}` ({r.get('edge_type')}, "
                f"`{r.get('repo_id')}/{r.get('rel_path')}`, line={r.get('line_start')})"
            )
    return out


def render_simplification_candidates(epdeps: List[Dict[str, Any]], deps: List[Dict[str, Any]]) -> List[str]:
    out = []
    for r in epdeps:
        conf = float(r.get("confidence") or 0)
        runtime = int(r.get("runtime_evidence_count") or 0)
        static = int(r.get("static_evidence_count") or 0)
        if static > 0 and runtime == 0:
            out.append(
                f"- Static-only possible dependency `{r.get('downstream_entity')}` on `{r.get('method')} {r.get('path')}`. "
                f"Candidate for verification/removal if no runtime evidence exists. Confidence: {fmt(conf)}"
            )
        elif r.get("p95_ms") and float(r.get("p95_ms") or 0) > 500:
            out.append(
                f"- High-latency dependency `{r.get('downstream_entity')}` on `{r.get('method')} {r.get('path')}` "
                f"p95={r.get('p95_ms')}. Candidate for cache/async/projection analysis."
            )
        elif r.get("error_rate") and float(r.get("error_rate") or 0) > 0.01:
            out.append(
                f"- High-error dependency `{r.get('downstream_entity')}` on `{r.get('method')} {r.get('path')}` "
                f"error_rate={r.get('error_rate')}. Candidate for resilience/fallback analysis."
            )
    return out[:30] or ["- No obvious candidates were found by the current heuristics. Check static-only/runtime-only rows and high-latency/high-error dependencies manually."]


def fmt(x: Any) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "?"
