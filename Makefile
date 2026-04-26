.PHONY: bootstrap full repo code api static datadog bigquery graph endpoint capabilities smoke
bootstrap:
	./scripts/bootstrap.sh
full:
	./run.sh
repo:
	python3 -m corp_kb.pipeline.runner stage repo_ingest --config $${CONFIG:-config.yaml}
code:
	python3 -m corp_kb.pipeline.runner stage code_index --config $${CONFIG:-config.yaml}
api:
	python3 -m corp_kb.pipeline.runner stage api_surface --config $${CONFIG:-config.yaml}
static:
	python3 -m corp_kb.pipeline.runner stage static_edges --config $${CONFIG:-config.yaml}
datadog:
	python3 -m corp_kb.pipeline.runner stage datadog --config $${CONFIG:-config.yaml}
bigquery:
	python3 -m corp_kb.pipeline.runner stage bigquery --config $${CONFIG:-config.yaml}
graph:
	python3 -m corp_kb.pipeline.runner stage graph --config $${CONFIG:-config.yaml}
endpoint:
	python3 -m corp_kb.pipeline.runner stage endpoint_map --config $${CONFIG:-config.yaml}
capabilities:
	python3 -m corp_kb.pipeline.runner stage capabilities --config $${CONFIG:-config.yaml}
smoke:
	python3 -m corp_kb.pipeline.runner stage smoke --config $${CONFIG:-config.yaml}
