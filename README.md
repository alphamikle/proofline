# proofline

Proofline builds a local, evidence-backed knowledge graph around corporate repositories, API specs, runtime signals, data lineage metadata, and documentation. The goal is to answer engineering questions from observed facts instead of pushing a pile of raw repos and logs directly into an LLM.

```text
repos + code/docs/configs/API specs
+ runtime signals
+ warehouse metadata
-> local facts, indexes, graph tables
-> proofline ask / impact / data-source / dependencies
```

## Install

Quick install or update:

```bash
curl -o- https://raw.githubusercontent.com/alphamikle/proofline/main/install.sh | bash
```

Or with `wget`:

```bash
wget -qO- https://raw.githubusercontent.com/alphamikle/proofline/main/install.sh | bash
```

The installer clones Proofline into `~/.proofline`, creates `~/.proofline/.venv`, installs the `proofline` and `pfl` CLIs, creates `~/.proofline/proofline.yaml` with local data paths under `~/.proofline`, and links the executables to `~/.local/bin`.

Useful installer environment variables:

```bash
PROOFLINE_DIR="$HOME/tools/proofline" \
PROOFLINE_BIN_DIR="$HOME/bin" \
PROOFLINE_REF="main" \
curl -o- https://raw.githubusercontent.com/alphamikle/proofline/main/install.sh | bash
```

The optional CodeGraphContext stack is intentionally skipped by default because it installs extra local tooling. Enable it explicitly:

```bash
PROOFLINE_INSTALL_CGC=1 curl -o- https://raw.githubusercontent.com/alphamikle/proofline/main/install.sh | bash
```

After installation:

```bash
proofline doctor --config "$HOME/.proofline/proofline.yaml"
pfl doctor --config "$HOME/.proofline/proofline.yaml"
```

If your shell cannot find `proofline` or `pfl`, add this to your shell profile:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Development checkout:

```bash
git clone <this-repo>
cd proofline
bash scripts/install.sh
```

When run from a checkout, `scripts/install.sh` installs the current working tree into `~/.proofline`. That is useful while developing or before changes have been pushed to GitHub.

Manual Python-only setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
proofline init
```

`proofline init` opens an interactive survey when run in a terminal, writes the selected core settings to `proofline.yaml`, and prepares local working directories. Use `proofline init --non-interactive` to write defaults without prompts.

When a newer `pfl` sees an older local config, it automatically inserts missing top-level sections before the next known section, preserving existing values and user comments. Proofline prints a short note when it updates the config and flags enabled integrations that still need credentials or URLs.

## Configure

Put repositories under `./repos`:

```text
repos/
  repo-a/.git
  repo-b/.git
  repo-c/.git
```

Or configure a clone URL file in `proofline.yaml`:

```yaml
repos:
  root: ./repos
  clone_urls_file: ./repo_urls.txt
  update_existing: true
```

Common credentials:

```bash
export DD_API_KEY=...
export DD_APP_KEY=...
export DD_SITE=datadoghq.com

export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
export GOOGLE_CLOUD_PROJECT=your-bq-project

export ATLASSIAN_EMAIL=you@company.com
export ATLASSIAN_API_TOKEN=...
export CONFLUENCE_BASE_URL=https://your-company.atlassian.net/wiki
export JIRA_BASE_URL=https://your-company.atlassian.net
```

For Server/Data Center or OAuth-style Atlassian setups:

```bash
export ATLASSIAN_BEARER_TOKEN=...
```

Check setup:

```bash
proofline doctor
```

## CLI

Main commands:

```bash
proofline init
proofline doctor
proofline sync
proofline sync repos
proofline sync docs
proofline sync runtime
proofline sync data
proofline build
proofline build code
proofline build graph
proofline build embeddings
proofline build capabilities
proofline publish
proofline run --from runtime --to graph
proofline stage smoke
proofline status
pfl status
```

`proofline` and `pfl` are equivalent. `sync` updates source facts. `build` derives local indexes and graph structures. `publish` sends the graph to the configured external graph backend.

Default config resolution:

```text
--config
PROOFLINE_CONFIG
./proofline.yaml
```

## Typical Workflow

Create or refresh local facts:

```bash
proofline sync repos
proofline sync docs
proofline sync runtime
proofline sync data
```

Build derived indexes and the local graph:

```bash
proofline build
```

Run the full pipeline:

```bash
proofline run
```

Continue from or stop at a step:

```bash
proofline run --from runtime
proofline run --from embeddings --to embeddings
proofline run --from runtime --to graph
```

Useful stage aliases:

```text
repos       -> repo_ingest
history     -> git_history
code        -> code_index
api         -> api_surface
runtime     -> datadog
data        -> bigquery
identity    -> entity_resolution
endpoints   -> endpoint_map
publish     -> external graph backend projection
```

## Ask Questions

Impact analysis:

```bash
proofline impact \
  --project checkout-api \
  --feature "add payment eligibility field"
```

Data-source recommendation:

```bash
proofline data-source \
  --project checkout-api \
  --feature "show customer payment eligibility"
```

Dependency report:

```bash
proofline dependencies \
  --project checkout-api \
  --env prod \
  --window-days 30
```

Natural language:

```bash
proofline ask "If I implement payment eligibility in project checkout-api, what can break?"
```

Raw context pack:

```bash
proofline impact --project checkout-api --feature "payment eligibility" --raw-context
```

Search:

```bash
proofline search "payment eligibility" --project checkout-api
```

## External Graph Backend

Graph publishing is configured through `graph_backend`:

```yaml
graph_backend:
  enabled: true
  provider: neo4j
  uri: bolt://localhost:7687
  username: neo4j
  password: codegraphcontext
  database: neo4j
```

Publish with:

```bash
proofline publish
```

## LLM Answers

By default, `agent.provider: none`, so Proofline generates deterministic markdown from graph facts.

For a local model or any corporate-approved CLI, use `provider: cli`; the command receives the full prompt on stdin and writes the answer to stdout:

```yaml
agent:
  provider: cli
  command: "ollama run qwen2.5-coder:32b"
```

HTTP providers are also supported:

```yaml
agent:
  provider: openai
  model: gpt-5.2
  api_key_env: OPENAI_API_KEY
```

```yaml
agent:
  provider: openai_compatible
  model: qwen2.5-coder:32b
  base_url: http://localhost:11434/v1
  api_key_env: ""
```

```yaml
agent:
  provider: anthropic
  model: claude-sonnet-4-6
  api_key_env: ANTHROPIC_API_KEY
```

## Local Outputs

```text
data/
  kb.duckdb
  indexes/code_fts.sqlite
  indexes/code_vectors.faiss
  indexes/code_vectors_meta.parquet
  raw/datadog/...
  raw/bigquery/...
  raw/confluence/...
  raw/jira/...
  reports/
```

Main tables:

```text
repo_inventory
repo_files
git_commits
git_file_changes
git_patch_hunks
git_detected_links
git_reverts
git_blame_current
git_semantic_changes
git_cochange_edges
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
graph_backend_exports
pipeline_runs
```

## Confidence

Approximate scale:

```text
0.20-0.35: weak textual/static inference
0.40-0.55: regex/config/package evidence
0.60-0.75: API/static route/client evidence
0.80-0.90: runtime dependency / warehouse metadata evidence
0.95+: runtime signals plus static/config/API corroboration
```

`not observed in runtime data` does not mean `does not exist`. It only means the dependency was not found in the available window or fields.

## Principle

The LLM is not the source of truth. The sources of truth are code, configs, API specs, runtime observations, warehouse metadata, ownership/catalog data, and git evidence. The LLM receives a compact context pack and turns it into an engineering answer.
