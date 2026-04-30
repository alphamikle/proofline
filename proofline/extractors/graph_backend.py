from __future__ import annotations

from typing import Any, Dict, Iterable, List

import pandas as pd

from proofline.utils import json_loads, now_iso


def _props(raw: Any) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        obj = json_loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _chunks(rows: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def publish_graph_backend(kb, cfg: Dict[str, Any]) -> pd.DataFrame:
    neo = cfg.get("neo4j", {})
    if not neo.get("enabled", False):
        return pd.DataFrame([{
            "exported_at": now_iso(), "node_count": 0, "edge_count": 0,
            "evidence_count": 0, "status": "disabled", "details": "",
        }])

    nodes = kb.query_df("SELECT * FROM nodes")
    edges = kb.query_df("SELECT * FROM edges")
    evidence = kb.query_df("SELECT * FROM evidence")
    batch_size = int(neo.get("batch_size", 1000))

    try:
        from neo4j import GraphDatabase
    except Exception as e:
        return _publish_error(nodes, edges, evidence, f"neo4j package is missing: {e}")

    driver = None
    try:
        driver = GraphDatabase.driver(
            neo.get("uri", "bolt://localhost:7687"),
            auth=(neo.get("username", "neo4j"), neo.get("password", "")),
        )
        database = neo.get("database", "neo4j")
        with driver.session(database=database) as session:
            if neo.get("clear_existing", True):
                session.run("MATCH (n:KBNode) DETACH DELETE n")
            session.run("CREATE CONSTRAINT kb_node_id IF NOT EXISTS FOR (n:KBNode) REQUIRE n.id IS UNIQUE")
            session.run("CREATE INDEX kb_node_type IF NOT EXISTS FOR (n:KBNode) ON (n.type)")
            session.run("CREATE INDEX kb_node_source IF NOT EXISTS FOR (n:KBNode) ON (n.source)")
            node_rows = []
            for _, n in nodes.iterrows():
                props = _props(n.get("properties"))
                props.update({
                    "id": str(n.get("node_id") or ""),
                    "type": str(n.get("node_type") or ""),
                    "display_name": str(n.get("display_name") or ""),
                    "source": str(n.get("source") or ""),
                    "confidence": float(n.get("confidence") or 0),
                })
                node_rows.append(props)
            for batch in _chunks(node_rows, batch_size):
                session.run(
                    """
                    UNWIND $rows AS row
                    MERGE (n:KBNode {id: row.id})
                    SET n += row
                    """,
                    rows=batch,
                )

            edge_rows = []
            for _, e in edges.iterrows():
                props = _props(e.get("properties"))
                props.update({
                    "id": str(e.get("edge_id") or ""),
                    "from": str(e.get("from_node") or ""),
                    "to": str(e.get("to_node") or ""),
                    "type": str(e.get("edge_type") or ""),
                    "env": str(e.get("env") or ""),
                    "source": str(e.get("source") or ""),
                    "confidence": float(e.get("confidence") or 0),
                    "first_seen": str(e.get("first_seen") or ""),
                    "last_seen": str(e.get("last_seen") or ""),
                    "evidence_refs": str(e.get("evidence_refs") or "[]"),
                })
                edge_rows.append(props)
            for batch in _chunks(edge_rows, batch_size):
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (a:KBNode {id: row.from})
                    MATCH (b:KBNode {id: row.to})
                    MERGE (a)-[r:KB_EDGE {id: row.id}]->(b)
                    SET r += row
                    """,
                    rows=batch,
                )
    except Exception as e:
        return _publish_error(nodes, edges, evidence, e)
    finally:
        if driver is not None:
            try:
                driver.close()
            except Exception:
                pass

    return pd.DataFrame([{
        "exported_at": now_iso(),
        "node_count": int(len(nodes)),
        "edge_count": int(len(edges)),
        "evidence_count": int(len(evidence)),
        "status": "ok",
        "details": "",
    }])


def _publish_error(nodes: pd.DataFrame, edges: pd.DataFrame, evidence: pd.DataFrame, details: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "exported_at": now_iso(),
        "node_count": int(len(nodes)),
        "edge_count": int(len(edges)),
        "evidence_count": int(len(evidence)),
        "status": "error",
        "details": str(details)[:4000],
    }])
