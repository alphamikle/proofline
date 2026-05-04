import pandas as pd

from proofline.visualization import build_visualization_payload


def test_visualization_payload_projects_service_runtime_to_repo_edges():
    repos = pd.DataFrame([
        {"repo_id": "orders-api", "repo_path": "/repos/orders-api", "primary_language": "python", "probable_type": "service", "size_mb": 10},
        {"repo_id": "payments-api", "repo_path": "/repos/payments-api", "primary_language": "go", "probable_type": "service", "size_mb": 12},
    ])
    services = pd.DataFrame([
        {"service_id": "orders", "display_name": "orders-api", "repo_id": "orders-api", "repo_path": "/repos/orders-api", "datadog_service": "orders", "owner_team": "", "confidence": 0.9},
        {"service_id": "payments", "display_name": "payments-api", "repo_id": "payments-api", "repo_path": "/repos/payments-api", "datadog_service": "payments", "owner_team": "", "confidence": 0.9},
    ])
    aliases = pd.DataFrame([
        {"canonical_id": "service:orders", "alias": "orders", "alias_type": "datadog_service", "source": "datadog", "confidence": 0.9},
        {"canonical_id": "service:payments", "alias": "payments", "alias_type": "datadog_service", "source": "datadog", "confidence": 0.9},
    ])
    runtime = pd.DataFrame([
        {"edge_id": "edge-1", "from_service": "orders", "to_entity": "payments", "edge_type": "CALLS", "source": "datadog", "count": 25, "confidence": 0.9},
    ])

    payload = build_visualization_payload(
        repo_inventory=repos,
        service_identity=services,
        entity_aliases=aliases,
        runtime_service_edges=runtime,
        runtime_endpoint_edges=pd.DataFrame(),
        static_edges=pd.DataFrame(),
        git_cochange_edges=pd.DataFrame(),
        bq_table_usage=pd.DataFrame(),
        nodes=pd.DataFrame(),
        edges=pd.DataFrame(),
    )

    repo_edges = payload["projections"]["repos"]["edges"]
    service_edges = payload["projections"]["services"]["edges"]

    assert any(e["source"] == "repo:orders-api" and e["target"] == "repo:payments-api" for e in repo_edges)
    assert any(e["source"] == "service:orders" and e["target"] == "service:payments" for e in service_edges)
