PYTHON ?= .venv/bin/python
CONFIG ?= proofline.yaml

.PHONY: bootstrap full repo history code embeddings api code-graph static datadog bigquery confluence jira graph endpoint capabilities publish smoke
bootstrap:
	$(PYTHON) -m proofline bootstrap
full:
	$(PYTHON) -m proofline run --config $(CONFIG)
repo:
	$(PYTHON) -m proofline stage repos --config $(CONFIG)
history:
	$(PYTHON) -m proofline stage history --config $(CONFIG)
code:
	$(PYTHON) -m proofline build code --config $(CONFIG)
embeddings:
	$(PYTHON) -m proofline build embeddings --config $(CONFIG)
api:
	$(PYTHON) -m proofline build api --config $(CONFIG)
code-graph:
	$(PYTHON) -m proofline build code-graph --config $(CONFIG)
static:
	$(PYTHON) -m proofline build static --config $(CONFIG)
datadog:
	$(PYTHON) -m proofline sync runtime --config $(CONFIG)
bigquery:
	$(PYTHON) -m proofline sync data --config $(CONFIG)
confluence:
	$(PYTHON) scripts/download_confluence.py --config $(CONFIG)
jira:
	$(PYTHON) scripts/download_jira.py --config $(CONFIG)
graph:
	$(PYTHON) -m proofline build graph --config $(CONFIG)
endpoint:
	$(PYTHON) -m proofline build endpoints --config $(CONFIG)
capabilities:
	$(PYTHON) -m proofline build capabilities --config $(CONFIG)
publish:
	$(PYTHON) -m proofline publish --config $(CONFIG)
smoke:
	$(PYTHON) -m proofline stage smoke --config $(CONFIG)
