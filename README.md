# corp-kb-poc

A local proof of concept for building an evidence-backed knowledge graph around a corporate codebase, Datadog runtime signals, and BigQuery metadata/lineage signals.

The core idea is not to pass all repositories and logs directly to an LLM. The pipeline converts available sources into local facts first:

```text
repos + code/docs/configs/API specs
+ Datadog services/dependencies/spans/logs
+ BigQuery jobs/table usage
-> kb.duckdb + FTS index + graph tables
-> ask.sh / local LLM context pack
```

## What this package builds

The pipeline is designed to run with a single command and produce:

- `repo_inventory`: classification for all repositories: service, library, frontend, job, infra, or unknown.
- `repo_files`: current-checkout file index, excluding `.git`, `node_modules`, build artifacts, and similar low-signal paths.
- `code_chunks` plus SQLite FTS: local full-text search over code, docs, configs, and API specs.
- `api_contracts` and `api_endpoints`: OpenAPI, Swagger, proto/gRPC, GraphQL, and static route extraction.
- `static_edges`: package, config, URL, host, topic, and BigQuery references.
- `datadog_service_edges`: Datadog APM service dependency graph.
- `datadog_spans` and `datadog_logs`: normalized runtime facts from spans and logs.
- `runtime_service_edges`: observed runtime service, data, topic, and host dependencies.
- `runtime_endpoint_edges`: endpoint/resource-level runtime dependencies when Datadog fields are available.
- `bq_jobs` and `bq_table_usage`: BigQuery job metadata and table usage from `INFORMATION_SCHEMA.JOBS_BY_ORGANIZATION`.
- `service_identity` and `entity_aliases`: repo to Datadog service to service account to canonical service identity mapping.
- `nodes`, `edges`, and `evidence`: local unified graph tables.
- `endpoint_dependency_map`: endpoint to downstream service/table/topic/host map.
- `data_capabilities`: candidates for selecting a source service or table for a feature.
- `compatibility_index`: entities where changes require compatibility checks.

## Installation

```bash
git clone <this-poc-repo>
cd corp-kb-poc
./scripts/bootstrap.sh
```

Manual Python-only setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml
```

Full setup also runs the CodeGraphContext installer from `/Users/alfa/dev/code/infra/cgc.sh`:

```bash
./scripts/bootstrap.sh
```

This installs local Qwen/SentenceTransformers, FAISS, Neo4j Python client, CodeGraphContext, the broad SCIP indexer stack, and starts the local Neo4j Docker container used by CGC. The SCIP stack includes Python, TypeScript/JavaScript, Go, Rust, Java/Scala/Kotlin, C/C++/CUDA, Ruby, C#/Visual Basic, Dart, and PHP where the current platform supports the indexer binaries. The runtime stages are still controlled through `config.yaml`, so you can keep Qwen embeddings, reranking, Neo4j export, or CodeGraphContext disabled while the tooling remains available.

## Input preparation

Put corporate repositories under `./repos`:

```text
repos/
  repo-a/.git
  repo-b/.git
  repo-c/.git
```

Or provide a repository URL file in `config.yaml`:

```yaml
repos:
  root: ./repos
  clone_urls_file: ./repo_urls.txt
  update_existing: false
```

Datadog credentials:

```bash
export DD_API_KEY=...
export DD_APP_KEY=...
export DD_SITE=datadoghq.com   # or datadoghq.eu / us3.datadoghq.com / us5.datadoghq.com / ap1 / ap2
```

BigQuery credentials:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
export GOOGLE_CLOUD_PROJECT=your-bq-project
```

For user credentials:

```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=your-bq-project
```

Atlassian credentials:

```bash
export ATLASSIAN_EMAIL=you@company.com
export ATLASSIAN_API_TOKEN=...
export CONFLUENCE_BASE_URL=https://your-company.atlassian.net/wiki
export JIRA_BASE_URL=https://your-company.atlassian.net
```

For Server/Data Center or OAuth-style setups, use:

```bash
export ATLASSIAN_BEARER_TOKEN=...
```

Mirror Confluence and Jira locally before indexing:

```bash
make confluence
make jira
```

Confluence stores pages/blogposts, storage/view HTML bodies, comments, attachment metadata, and attachment files under `data/raw/confluence`. Jira stores issues, comments, changelogs, worklogs, remote links, issue properties, attachments, and site metadata under `data/raw/jira`.

## Run the full pipeline

```bash
./run.sh
```

Equivalent command:

```bash
python3 -m corp_kb.pipeline.runner full --config config.yaml
```

Continue from a specific stage:

```bash
python3 -m corp_kb.pipeline.runner full --config config.yaml --from-stage datadog
python3 -m corp_kb.pipeline.runner stage graph --config config.yaml
```

Pipeline stages:

```text
repo_ingest
code_index
embeddings
api_surface
code_graph
static_edges
datadog
bigquery
entity_resolution
graph
endpoint_map
capabilities
neo4j_export
smoke
```

Qwen/FAISS semantic retrieval:

```yaml
indexing:
  embeddings:
    enabled: true
    model_name: Qwen/Qwen3-Embedding-0.6B
    max_chunks: 10000   # recommended for first dry run
retrieval:
  reranker:
    enabled: true
    model_name: Qwen/Qwen3-Reranker-0.6B
```

Build only the vector index:

```bash
./run.sh --from-stage embeddings --to-stage embeddings
```

Neo4j projection:

```yaml
neo4j:
  enabled: true
  uri: bolt://localhost:7687
  username: neo4j
  password: codegraphcontext
```

CodeGraphContext hook:

```yaml
code_graph:
  enabled: true
  command: "$HOME/.local/bin/cgc index {repo_path}"
```

## Quick smoke checks

```bash
python3 -m corp_kb.pipeline.runner stage smoke --config config.yaml
```

You can also inspect the database directly:

```bash
duckdb ./data/kb.duckdb
```

Example SQL checks:

```sql
SELECT probable_type, COUNT(*) FROM repo_inventory GROUP BY 1 ORDER BY 2 DESC;
SELECT * FROM service_identity ORDER BY confidence DESC LIMIT 20;
SELECT * FROM runtime_service_edges ORDER BY confidence DESC, count DESC NULLS LAST LIMIT 50;
SELECT * FROM endpoint_dependency_map ORDER BY confidence DESC LIMIT 50;
```

## Ask the agent

Impact analysis:

```bash
./ask.sh impact \
  --project checkout-api \
  --feature "add payment eligibility field"
```

Data-source recommendation:

```bash
./ask.sh data-source \
  --project checkout-api \
  --feature "show customer payment eligibility"
```

Dependency report:

```bash
./ask.sh dependency-report \
  --project checkout-api \
  --env prod \
  --window-days 30
```

Natural language:

```bash
./ask.sh "If I implement payment eligibility in project checkout-api, what can break?"
```

Raw context pack instead of a markdown answer:

```bash
./ask.sh impact --project checkout-api --feature "payment eligibility" --raw-context
```

## Connect an external LLM

By default, `agent.provider: none`, so `ask.sh` generates a deterministic markdown report from graph facts.

For a local model or any corporate-approved CLI, use `provider: cli`; the command receives the full prompt on stdin and should write the answer to stdout:

```yaml
agent:
  provider: cli
  command: "ollama run qwen2.5-coder:32b"
```

HTTP providers are also supported:

```yaml
agent:
  provider: openai              # OpenAI Responses API
  model: gpt-5.2
  api_key_env: OPENAI_API_KEY
```

```yaml
agent:
  provider: openai_compatible   # /v1/chat/completions, for local/corporate gateways
  model: qwen2.5-coder:32b
  base_url: http://localhost:11434/v1
  api_key_env: ""
```

```yaml
agent:
  provider: anthropic           # Anthropic Messages API
  model: claude-sonnet-4-6
  api_key_env: ANTHROPIC_API_KEY
```

```yaml
agent:
  provider: anthropic_compatible # /v1/messages, for Claude-compatible gateways
  model: your-corporate-model
  base_url: https://llm-gateway.example.com/v1
  api_key_env: ANTHROPIC_API_KEY
```

The script sends a compact JSON context pack and system rules to the LLM. The rules instruct the model to avoid unsupported facts and to separate runtime, static, BigQuery/data, and ownership evidence. `agent.enrichment.model` is reserved for offline enrichment stages: for example, summarizing Confluence pages, extracting Jira feature ownership, or normalizing service names before the answer-time agent runs.

## Output locations

```text
data/
  kb.duckdb                         # main local database
  indexes/code_fts.sqlite           # FTS index over chunks
  indexes/code_vectors.faiss        # FAISS vector index when embeddings are enabled
  indexes/code_vectors_meta.parquet # FAISS id -> chunk metadata
  raw/datadog/...                   # raw-ish Datadog extracts when enabled
  raw/bigquery/...                  # raw-ish BigQuery extracts when enabled
  raw/confluence/...                # mirrored Confluence content and files
  raw/jira/...                      # mirrored Jira issues and files
  reports/                          # reserved for future report exports
```

Main tables:

```text
repo_inventory
repo_files
code_chunks
code_embedding_index
api_contracts
api_endpoints
static_edges
datadog_services
datadog_service_edges
datadog_spans
datadog_logs
runtime_service_edges
runtime_endpoint_edges
bq_jobs
bq_table_usage
service_identity
entity_aliases
unresolved_entities
nodes
edges
evidence
endpoint_dependency_map
data_capabilities
compatibility_index
code_graph_runs
neo4j_exports
```

## Confidence interpretation

Approximate scale:

```text
0.20-0.35: weak textual/static inference
0.40-0.55: regex/config/package evidence
0.60-0.75: API/static route/client evidence
0.80-0.90: Datadog service dependency / BigQuery metadata evidence
0.95+: runtime spans/logs plus static/config/API corroboration
```

Important: `not observed in Datadog` does not mean `does not exist`. It only means that the dependency was not found in the available runtime window or fields.

## Current limitations

- Entity resolution is best-effort. `unresolved_entities` shows where repo, service, Datadog, or service-account names were not merged.
- Endpoint-level graph quality depends on Datadog fields such as `service`, `env`, `trace_id`, `span_id`, `http.route`, `resource`, `peer.service`, `db.name`, and `messaging.destination`.
- The BigQuery layer uses metadata and job history, not table contents.
- Static route extraction covers common frameworks with regex-based heuristics. A production-grade version should add tree-sitter, LSP, or SCIP adapters.
- Raw `.git/objects` are not embedded directly. The pipeline extracts git metadata through `git log`.

## Architecture principle

The LLM is not the source of truth. The sources of truth are:

```text
code/config/API specs
Datadog runtime observations
BigQuery job/table metadata
ownership/catalog/git evidence
```

The LLM receives a compact context pack and turns it into an engineering answer.
