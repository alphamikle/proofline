PYTHON ?= .venv/bin/python
CONFIG ?= config.yaml

.PHONY: bootstrap full repo code embeddings api code-graph static datadog bigquery confluence jira graph endpoint capabilities neo4j smoke
bootstrap:
	./scripts/bootstrap.sh
full:
	./run.sh
repo:
	$(PYTHON) -m corp_kb.pipeline.runner stage repo_ingest --config $(CONFIG)
code:
	$(PYTHON) -m corp_kb.pipeline.runner stage code_index --config $(CONFIG)
embeddings:
	$(PYTHON) -m corp_kb.pipeline.runner stage embeddings --config $(CONFIG)
api:
	$(PYTHON) -m corp_kb.pipeline.runner stage api_surface --config $(CONFIG)
code-graph:
	$(PYTHON) -m corp_kb.pipeline.runner stage code_graph --config $(CONFIG)
static:
	$(PYTHON) -m corp_kb.pipeline.runner stage static_edges --config $(CONFIG)
datadog:
	$(PYTHON) -m corp_kb.pipeline.runner stage datadog --config $(CONFIG)
bigquery:
	$(PYTHON) -m corp_kb.pipeline.runner stage bigquery --config $(CONFIG)
confluence:
	$(PYTHON) scripts/download_confluence.py --config $(CONFIG)
jira:
	$(PYTHON) scripts/download_jira.py --config $(CONFIG)
graph:
	$(PYTHON) -m corp_kb.pipeline.runner stage graph --config $(CONFIG)
endpoint:
	$(PYTHON) -m corp_kb.pipeline.runner stage endpoint_map --config $(CONFIG)
capabilities:
	$(PYTHON) -m corp_kb.pipeline.runner stage capabilities --config $(CONFIG)
neo4j:
	$(PYTHON) -m corp_kb.pipeline.runner stage neo4j_export --config $(CONFIG)
smoke:
	$(PYTHON) -m corp_kb.pipeline.runner stage smoke --config $(CONFIG)
