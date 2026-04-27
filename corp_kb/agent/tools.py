from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from corp_kb.storage import KB
from corp_kb.utils import normalize_name, json_dumps
from corp_kb.extractors.embeddings import load_sentence_transformer


class KBTools:
    def __init__(self, kb: KB, sqlite_fts_path: str | Path | None = None, cfg: Dict[str, Any] | None = None):
        self.kb = kb
        self.sqlite_fts_path = Path(sqlite_fts_path) if sqlite_fts_path else None
        self.cfg = cfg or {}
        self._embedding_model = None
        self._reranker = None
        self._vector_index = None
        self._vector_meta = None

    def resolve_project(self, name: str) -> Dict[str, Any]:
        q = name.strip()
        norm = normalize_name(q)
        # exact alias
        aliases = self.kb.query_df("SELECT * FROM entity_aliases")
        svc = pd.DataFrame()
        if not aliases.empty:
            hits = aliases[(aliases["alias"].str.lower() == q.lower()) | (aliases["alias"].apply(lambda x: normalize_name(str(x)) == norm))]
            if not hits.empty:
                sid = str(hits.iloc[0]["canonical_id"]).replace("service:", "")
                svc = self.kb.query_df("SELECT * FROM service_identity WHERE service_id = ?", [sid])
        if svc.empty:
            svc = self.kb.query_df(
                """
                SELECT * FROM service_identity
                WHERE lower(service_id) = lower(?) OR lower(display_name) = lower(?) OR lower(repo_id) = lower(?) OR lower(datadog_service) = lower(?)
                LIMIT 1
                """,
                [norm, q, q, q],
            )
        if svc.empty:
            svc = self.kb.query_df(
                """
                SELECT * FROM service_identity
                WHERE service_id ILIKE ? OR display_name ILIKE ? OR repo_id ILIKE ? OR datadog_service ILIKE ?
                ORDER BY confidence DESC LIMIT 5
                """,
                [f"%{norm}%", f"%{q}%", f"%{q}%", f"%{q}%"],
            )
        if svc.empty:
            return {"query": q, "found": False, "candidates": []}
        row = svc.iloc[0].to_dict()
        return {"query": q, "found": True, "service": row, "candidates": svc.head(5).to_dict("records")}

    def get_service_profile(self, service_id: str) -> Dict[str, Any]:
        sid = service_id.replace("service:", "")
        service = self.kb.query_df("SELECT * FROM service_identity WHERE service_id = ? LIMIT 1", [sid]).to_dict("records")
        endpoints = self.kb.query_df("SELECT method, path, source, source_file, confidence FROM api_endpoints WHERE service_id = ? ORDER BY method, path LIMIT 500", [sid]).to_dict("records")
        owners = self.kb.query_df("SELECT * FROM ownership WHERE entity_id = ? OR entity_id = ? LIMIT 20", [f"service:{sid}", f"repo:{sid}"]).to_dict("records")
        return {"service": service[0] if service else {"service_id": sid}, "endpoints": endpoints, "owners": owners}

    def get_service_dependencies(self, service_id: str, env: str = "prod", window_days: int = 30) -> List[Dict[str, Any]]:
        sid = service_id.replace("service:", "")
        names = [sid]
        si = self.kb.query_df("SELECT datadog_service, repo_id FROM service_identity WHERE service_id = ?", [sid])
        if not si.empty:
            for c in ["datadog_service", "repo_id"]:
                v = str(si.iloc[0].get(c) or "")
                if v:
                    names.append(v)
        rows = []
        for name in set(names):
            rows += self.kb.query_df(
                """
                SELECT * FROM runtime_service_edges
                WHERE from_service = ? AND (? = '' OR env = ? OR env = '')
                ORDER BY confidence DESC, count DESC NULLS LAST
                LIMIT 1000
                """,
                [name, env, env],
            ).to_dict("records")
        if not rows:
            rows = self.kb.query_df(
                """
                SELECT * FROM edges
                WHERE from_node IN (?, ?) AND edge_type IN ('OBSERVED_CALL','REFERENCES_URL','REFERENCES_HOST','USES_CONFIG_KEY','REFERENCES_BQ_TABLE','REFERENCES_TOPIC')
                ORDER BY confidence DESC LIMIT 1000
                """,
                [f"service:{sid}", f"repo:{sid}"],
            ).to_dict("records")
        return rows

    def get_service_dependents(self, service_id: str, env: str = "prod", window_days: int = 30) -> List[Dict[str, Any]]:
        sid = service_id.replace("service:", "")
        names = [sid]
        si = self.kb.query_df("SELECT datadog_service, repo_id FROM service_identity WHERE service_id = ?", [sid])
        if not si.empty:
            for c in ["datadog_service", "repo_id"]:
                v = str(si.iloc[0].get(c) or "")
                if v:
                    names.append(v)
        rows = []
        for name in set(names):
            rows += self.kb.query_df(
                """
                SELECT * FROM runtime_service_edges
                WHERE to_entity = ? AND (? = '' OR env = ? OR env = '')
                ORDER BY confidence DESC, count DESC NULLS LAST
                LIMIT 1000
                """,
                [name, env, env],
            ).to_dict("records")
        if not rows:
            rows = self.kb.query_df(
                """
                SELECT * FROM edges
                WHERE to_node IN (?, ?) ORDER BY confidence DESC LIMIT 1000
                """,
                [f"service:{sid}", sid],
            ).to_dict("records")
        return rows

    def get_endpoint_dependencies(self, service_id: str, env: str = "prod", window_days: int = 30) -> List[Dict[str, Any]]:
        sid = service_id.replace("service:", "")
        return self.kb.query_df(
            """
            SELECT * FROM endpoint_dependency_map
            WHERE service_id = ? AND (? = '' OR env = ? OR env = '')
            ORDER BY path, method, confidence DESC, runtime_count_30d DESC NULLS LAST
            LIMIT 5000
            """,
            [sid, env, env],
        ).to_dict("records")

    def get_bq_usage(self, service_id: str, window_days: int = 30) -> List[Dict[str, Any]]:
        sid = service_id.replace("service:", "")
        aliases = self.kb.query_df("SELECT alias FROM entity_aliases WHERE canonical_id = ? AND alias_type = 'service_account'", [f"service:{sid}"])
        accounts = aliases["alias"].tolist() if not aliases.empty else []
        if not accounts:
            return []
        placeholders = ",".join(["?"] * len(accounts))
        return self.kb.query_df(f"SELECT * FROM bq_table_usage WHERE service_account IN ({placeholders}) ORDER BY job_count DESC LIMIT 1000", accounts).to_dict("records")

    def search_capabilities(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2]
        if not terms:
            terms = [query.lower()]
        where = " OR ".join(["lower(capability_name) LIKE ? OR lower(fields) LIKE ? OR lower(provider_entity) LIKE ?" for _ in terms])
        params = []
        for t in terms:
            like = f"%{t}%"
            params.extend([like, like, like])
        return self.kb.query_df(f"SELECT * FROM data_capabilities WHERE {where} ORDER BY confidence DESC, usage_count_30d DESC NULLS LAST LIMIT {int(limit)}", params).to_dict("records")

    def search_code(self, query: str, repo_id: str | None = None, limit: int = 25) -> List[Dict[str, Any]]:
        retrieval = self.cfg.get("retrieval", {})
        fts_top_k = int(retrieval.get("fts_top_k", max(limit, 25)))
        vector_top_k = int(retrieval.get("vector_top_k", max(limit, 25)))
        hits = self.search_code_fts(query, repo_id=repo_id, limit=fts_top_k)
        hits += self.search_code_vector(query, repo_id=repo_id, limit=vector_top_k)
        merged = self.merge_hits(hits)
        rerank_top_k = int(retrieval.get("rerank_top_k", limit))
        merged = self.rerank_code_hits(query, merged[:max(rerank_top_k, limit)])
        return merged[:limit]

    def search_code_fts(self, query: str, repo_id: str | None = None, limit: int = 25) -> List[Dict[str, Any]]:
        if not self.sqlite_fts_path or not self.sqlite_fts_path.exists():
            return []
        con = sqlite3.connect(str(self.sqlite_fts_path))
        con.row_factory = sqlite3.Row
        q = query.replace('"', ' ')
        try:
            if repo_id:
                rows = con.execute(
                    "SELECT *, bm25(chunks) AS fts_rank FROM chunks WHERE chunks MATCH ? AND repo_id = ? ORDER BY fts_rank LIMIT ?",
                    (q, repo_id, limit),
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT *, bm25(chunks) AS fts_rank FROM chunks WHERE chunks MATCH ? ORDER BY fts_rank LIMIT ?",
                    (q, limit),
                ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["retrieval_sources"] = ["fts"]
                d["fts_score"] = 1.0 / (1.0 + abs(float(d.get("fts_rank") or 0.0)))
                d["score"] = d["fts_score"]
                out.append(d)
            return out
        except Exception:
            return []
        finally:
            con.close()

    def search_code_vector(self, query: str, repo_id: str | None = None, limit: int = 25) -> List[Dict[str, Any]]:
        index_path = Path(self.cfg.get("storage", {}).get("vector_index_path", ""))
        meta_path = Path(self.cfg.get("storage", {}).get("vector_meta_path", ""))
        if not index_path.exists() or not meta_path.exists():
            return []
        try:
            import numpy as np
        except Exception:
            return []
        try:
            model_name = str(self.cfg.get("indexing", {}).get("embeddings", {}).get("model_name") or "Qwen/Qwen3-Embedding-0.6B")
            if self._embedding_model is None:
                device = str(self.cfg.get("indexing", {}).get("embeddings", {}).get("device") or "cpu")
                self._embedding_model = load_sentence_transformer(model_name, device=device)
            qvec = self._embedding_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")
            import faiss
            if self._vector_index is None:
                self._vector_index = faiss.read_index(str(index_path))
            if self._vector_meta is None:
                self._vector_meta = pd.read_parquet(meta_path)
            overfetch = max(limit * (5 if repo_id else 2), limit)
            scores, ids = self._vector_index.search(np.asarray(qvec), overfetch)
            rows = []
            for faiss_id, score in zip(ids[0].tolist(), scores[0].tolist()):
                if faiss_id < 0:
                    continue
                meta = self._vector_meta[self._vector_meta["faiss_id"] == faiss_id]
                if meta.empty:
                    continue
                m = meta.iloc[0].to_dict()
                if repo_id and str(m.get("repo_id")) != repo_id:
                    continue
                rows.append({**m, "vector_score": float(score), "score": float(score), "retrieval_sources": ["vector"]})
                if len(rows) >= limit:
                    break
            if not rows:
                return []
            ids_by_chunk = [r["chunk_id"] for r in rows]
            placeholders = ",".join(["?"] * len(ids_by_chunk))
            chunks = self.kb.query_df(f"SELECT * FROM code_chunks WHERE chunk_id IN ({placeholders})", ids_by_chunk)
            chunk_by_id = {str(r["chunk_id"]): r for r in chunks.to_dict("records")}
            out = []
            for r in rows:
                chunk = chunk_by_id.get(str(r.get("chunk_id")), {})
                if not chunk:
                    continue
                chunk.update({k: v for k, v in r.items() if k not in chunk})
                out.append(chunk)
            return out
        except Exception:
            return []

    def merge_hits(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for h in hits:
            cid = str(h.get("chunk_id") or "")
            if not cid:
                continue
            cur = merged.get(cid)
            if not cur:
                merged[cid] = dict(h)
                continue
            cur_sources = set(cur.get("retrieval_sources") or [])
            cur_sources.update(h.get("retrieval_sources") or [])
            cur["retrieval_sources"] = sorted(cur_sources)
            for key in ["fts_score", "vector_score"]:
                if h.get(key) is not None:
                    cur[key] = max(float(cur.get(key) or 0), float(h.get(key) or 0))
            cur["score"] = max(float(cur.get("score") or 0), float(h.get("score") or 0))
        return sorted(merged.values(), key=lambda h: float(h.get("score") or 0), reverse=True)

    def rerank_code_hits(self, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        reranker_cfg = self.cfg.get("retrieval", {}).get("reranker", {})
        if not reranker_cfg.get("enabled", False) or not hits:
            return hits
        try:
            if self._reranker is None:
                model_name = str(reranker_cfg.get("model_name") or "Qwen/Qwen3-Reranker-0.6B")
                if "Qwen3-Reranker" in model_name:
                    self._reranker = Qwen3YesNoReranker(model_name, device=str(reranker_cfg.get("device") or "cpu"))
                else:
                    from sentence_transformers import CrossEncoder
                    self._reranker = CrossEncoder(model_name, device=str(reranker_cfg.get("device") or "cpu"), trust_remote_code=True)
            pairs = [[query, str(h.get("text") or "")[:12000]] for h in hits]
            scores = self._reranker.predict(pairs, batch_size=int(reranker_cfg.get("batch_size", 8)))
            for h, s in zip(hits, scores):
                h["rerank_score"] = float(s)
            return sorted(hits, key=lambda h: float(h.get("rerank_score") or 0), reverse=True)
        except Exception:
            return hits

    def get_evidence(self, evidence_ids: List[str]) -> List[Dict[str, Any]]:
        if not evidence_ids:
            return []
        placeholders = ",".join(["?"] * len(evidence_ids))
        return self.kb.query_df(f"SELECT * FROM evidence WHERE evidence_id IN ({placeholders})", evidence_ids).to_dict("records")

    def search_code_graph(self, query: str, repo_id: str | None = None, limit: int = 25) -> Dict[str, Any]:
        terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2]
        if not terms:
            terms = [query.lower()]
        clauses = []
        params: List[Any] = []
        for t in terms[:6]:
            like = f"%{t}%"
            clauses.append("(name ILIKE ? OR rel_path ILIKE ? OR file_path ILIKE ? OR signature ILIKE ?)")
            params.extend([like, like, like, like])
        where = "(" + " OR ".join(clauses) + ")" if clauses else "1=1"
        if repo_id:
            where += " AND repo_id = ?"
            params.append(repo_id)
        symbols = self.kb.query_df(
            f"""
            SELECT * FROM code_graph_symbols
            WHERE {where}
            ORDER BY
              CASE node_type
                WHEN 'Function' THEN 0
                WHEN 'Class' THEN 1
                WHEN 'Struct' THEN 2
                WHEN 'Enum' THEN 3
                WHEN 'File' THEN 4
                ELSE 5
              END,
              repo_id, rel_path, line_start NULLS LAST
            LIMIT {int(limit)}
            """,
            params,
        )
        if symbols.empty and repo_id:
            return self.search_code_graph(query, repo_id=None, limit=limit)
        edges: List[Dict[str, Any]] = []
        ids = [str(x) for x in symbols["symbol_id"].tolist()] if not symbols.empty else []
        if ids:
            placeholders = ",".join(["?"] * len(ids))
            edges = self.kb.query_df(
                f"""
                SELECT e.*, fs.name AS from_name, fs.node_type AS from_type, fs.rel_path AS from_rel_path,
                       ts.name AS to_name, ts.node_type AS to_type, ts.rel_path AS to_rel_path
                FROM code_graph_edges e
                LEFT JOIN code_graph_symbols fs ON fs.symbol_id = e.from_symbol_id
                LEFT JOIN code_graph_symbols ts ON ts.symbol_id = e.to_symbol_id
                WHERE e.from_symbol_id IN ({placeholders}) OR e.to_symbol_id IN ({placeholders})
                ORDER BY e.edge_type, e.repo_id, e.rel_path
                LIMIT {int(limit) * 8}
                """,
                ids + ids,
            ).to_dict("records")
        return {"symbols": symbols.to_dict("records"), "relationships": edges}

    def get_graph_neighborhood(self, node_id: str, limit: int = 100) -> Dict[str, Any]:
        neo = self._neo4j_neighborhood(node_id, limit)
        if neo is not None:
            return {"source": "neo4j", **neo}
        rows = self.kb.query_df(
            """
            SELECT * FROM edges
            WHERE from_node = ? OR to_node = ?
            ORDER BY confidence DESC
            LIMIT ?
            """,
            [node_id, node_id, int(limit)],
        )
        node_ids = set([node_id])
        for e in rows.to_dict("records"):
            node_ids.add(str(e.get("from_node") or ""))
            node_ids.add(str(e.get("to_node") or ""))
        nodes: List[Dict[str, Any]] = []
        if node_ids:
            placeholders = ",".join(["?"] * len(node_ids))
            nodes = self.kb.query_df(f"SELECT * FROM nodes WHERE node_id IN ({placeholders})", list(node_ids)).to_dict("records")
        return {"source": "duckdb", "center": node_id, "nodes": nodes, "edges": rows.to_dict("records")}

    def _neo4j_neighborhood(self, node_id: str, limit: int) -> Dict[str, Any] | None:
        neo = self.cfg.get("neo4j", {})
        if not neo.get("enabled", False):
            return None
        try:
            from neo4j import GraphDatabase
        except Exception:
            return None
        try:
            driver = GraphDatabase.driver(
                neo.get("uri", "bolt://localhost:7687"),
                auth=(neo.get("username", "neo4j"), neo.get("password", "")),
            )
            database = neo.get("database", "neo4j")
            with driver.session(database=database) as session:
                result = session.run(
                    """
                    MATCH (n:KBNode {id: $node_id})
                    OPTIONAL MATCH (n)-[r:KB_EDGE]-(m:KBNode)
                    RETURN n {.*} AS center,
                           collect({edge: r {.*}, neighbor: m {.*}})[0..$limit] AS items
                    """,
                    node_id=node_id,
                    limit=int(limit),
                )
                record = result.single()
            driver.close()
            if not record or not record.get("center"):
                return None
            items = [x for x in (record.get("items") or []) if x and x.get("edge")]
            return {
                "center": record.get("center"),
                "nodes": [x.get("neighbor") for x in items if x.get("neighbor")],
                "edges": [x.get("edge") for x in items if x.get("edge")],
            }
        except Exception:
            return None


class Qwen3YesNoReranker:
    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 4096):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        self.max_length = max_length
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            'Note that the answer can only be "yes" or "no".<|im_end|>\n'
            "<|im_start|>user\n"
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        self.instruction = "Given a code search query, retrieve relevant code or documentation passages that answer the query"

    def predict(self, pairs: List[List[str]], batch_size: int = 4) -> List[float]:
        scores: List[float] = []
        for start in range(0, len(pairs), max(1, batch_size)):
            batch = pairs[start:start + max(1, batch_size)]
            texts = [
                f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"
                for query, doc in batch
            ]
            inputs = self.tokenizer(
                texts,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
            )
            for i, input_ids in enumerate(inputs["input_ids"]):
                inputs["input_ids"][i] = self.prefix_tokens + input_ids + self.suffix_tokens
            inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with self.torch.no_grad():
                logits = self.model(**inputs).logits[:, -1, :]
                true_vector = logits[:, self.token_true_id]
                false_vector = logits[:, self.token_false_id]
                yn_logits = self.torch.stack([false_vector, true_vector], dim=1)
                batch_scores = self.torch.nn.functional.log_softmax(yn_logits, dim=1)[:, 1].exp().tolist()
            scores.extend(float(s) for s in batch_scores)
        return scores
