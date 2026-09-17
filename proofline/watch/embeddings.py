"""Embeddings for one repo: reuse cached vectors, embed only new chunks.

Same shard layout as batch (``{shard}.faiss`` + ``{shard}.parquet``), so
watch and batch runs stay interchangeable. FAISS has no delete: removal
compacts the index from survivor vectors via ``reconstruct``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from proofline.extractors.embeddings import (
    chunk_payload,
    chunks_source_fingerprint,
    eligible_chunks,
    embedding_batch_size,
    embedding_model_id,
    load_embedder,
    load_existing_repo_index,
    record_embedding_status,
    replace_repo_embedding_rows,
    repo_vector_paths,
    safe_write_faiss,
    safe_write_parquet,
    text_sha1,
    vector_shard_dir,
)
from proofline.utils import now_iso
from proofline.watch import EventCallback, emit


def _meta_row(crow, model_name, vector_dim, faiss_id, embedded_at=None):
    get = crow.get if hasattr(crow, "get") else (lambda k, d=None: crow[k] if k in crow else d)
    return {
        "faiss_id": faiss_id, "chunk_id": get("chunk_id"), "repo_id": get("repo_id"),
        "rel_path": get("rel_path"), "language": get("language"), "kind": get("kind"),
        "symbol": get("symbol"), "start_line": get("start_line"), "end_line": get("end_line"),
        "text_sha1": get("_text_sha1", get("text_sha1")), "model_name": model_name,
        "vector_dim": vector_dim, "embedded_at": embedded_at or now_iso(),
    }


def _survivor_vectors(index, kept):
    """Pull survivor vectors from FAISS via reconstruct; None when impossible."""
    if index is None or kept.empty:
        return None
    try:
        import numpy as np

        vecs = np.asarray([index.reconstruct(int(i)) for i in range(len(kept))], dtype="float32")
        return vecs if len(vecs) == len(kept) else None
    except Exception:
        return None
def refresh_embeddings(kb, cfg, repo_id, *, embedder=None, on_event=None):
    emb = cfg.get("indexing", {}).get("embeddings", {})
    if not emb.get("enabled", True):
        return {"repo_id": repo_id, "skipped": True, "reason": "embeddings disabled"}
    model_name = embedding_model_id(emb)
    max_chars = int(emb.get("max_text_chars", 4096))
    try:
        import faiss as faiss_module
    except Exception as e:
        raise RuntimeError("Embeddings need FAISS. Run: ./scripts/bootstrap.sh") from e
    shard_dir = vector_shard_dir(cfg)
    shard_dir.mkdir(parents=True, exist_ok=True)
    index_path, meta_path = repo_vector_paths(cfg, repo_id)
    chunks = eligible_chunks(
        kb.query_df("SELECT * FROM code_chunks WHERE repo_id = ? ORDER BY rel_path, start_line, kind", [repo_id]),
        cfg,
    ).copy()
    if not chunks.empty:
        chunks["_text_sha1"] = chunks.apply(lambda r: text_sha1(chunk_payload(r, max_chars)), axis=1)
    fingerprint = chunks_source_fingerprint(chunks)
    started = now_iso()
    if chunks.empty:
        replace_repo_embedding_rows(kb, repo_id, model_name, pd.DataFrame())
        for stale in (index_path, meta_path):
            try:
                stale.unlink()
            except OSError:
                pass
        record_embedding_status(
            kb, repo_id=repo_id, model_name=model_name, source_fingerprint=fingerprint,
            status="ok", chunk_count=0, vector_count=0, vector_dim=0,
            started_at=started, index_path=index_path, meta_path=meta_path, details="watch: no chunks")
        emit(on_event, "embeddings", {"repo_id": repo_id, "vectors": 0})
        return {"repo_id": repo_id, "vectors": 0, "cached": 0, "added": 0}
    index, existing = load_existing_repo_index(faiss_module, index_path, meta_path, chunks, model_name)
    want = set(zip(chunks["chunk_id"].astype(str), chunks["_text_sha1"].astype(str)))
    if existing.empty:
        kept = existing
    else:
        mask = existing.apply(lambda r: (str(r["chunk_id"]), str(r["text_sha1"])) in want, axis=1)
        kept = existing[mask].reset_index(drop=True)
    removed = len(existing) - len(kept)
    have = set(zip(existing["chunk_id"].astype(str), existing["text_sha1"].astype(str))) if not existing.empty else set()
    pending = chunks[~chunks.apply(lambda r: (r["chunk_id"], r["_text_sha1"]) in have, axis=1)].reset_index(drop=True)
    if pending.empty and not removed:
        replace_repo_embedding_rows(kb, repo_id, model_name, existing)
        dim = int(existing.iloc[0]["vector_dim"]) if not existing.empty else 0
        record_embedding_status(
            kb, repo_id=repo_id, model_name=model_name, source_fingerprint=fingerprint,
            status="ok", chunk_count=len(chunks), vector_count=len(existing), vector_dim=dim,
            started_at=started, index_path=index_path, meta_path=meta_path, details="watch: cached")
        emit(on_event, "embeddings", {"repo_id": repo_id, "vectors": len(existing), "cached": len(existing)})
        return {"repo_id": repo_id, "vectors": len(existing), "cached": len(existing), "added": 0}
    import numpy as np

    model = embedder if embedder is not None else load_embedder(emb)
    new_vecs = None
    if not pending.empty:
        texts = [chunk_payload(row, max_chars) for _, row in pending.iterrows()]
        new_vecs = np.asarray(
            model.encode(texts, batch_size=len(texts), convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False), dtype="float32")
        if new_vecs.ndim != 2 or new_vecs.shape[0] != len(pending):
            raise RuntimeError(f"Embedding model returned unexpected shape: {new_vecs.shape}")
    dim = int(new_vecs.shape[1]) if new_vecs is not None else int(kept.iloc[0]["vector_dim"])
    if not kept.empty and int(kept.iloc[0]["vector_dim"]) != dim:
        raise RuntimeError(f"Embedding dimension changed from {kept.iloc[0]['vector_dim']} to {dim}")
    fresh = faiss_module.IndexFlatIP(dim)
    rows: List[Dict[str, Any]] = []
    survivors = _survivor_vectors(index, kept) if removed else None
    if not kept.empty and (not removed or survivors is not None):
        if survivors is None and not removed:
            # No removal: existing index order == kept meta order; reuse it.
            fresh = index
            rows = kept.to_dict("records")
        elif survivors is not None:
            fresh.add(survivors)
            rows = kept.to_dict("records")
            for pos in range(len(rows)):
                rows[pos]["faiss_id"] = pos
        else:
            kept = kept.iloc[0:0]
    if new_vecs is not None:
        first_id = int(fresh.ntotal)
        fresh.add(new_vecs)
        embedded_at = now_iso()
        for offset, (_, crow) in enumerate(pending.iterrows()):
            rows.append(_meta_row(crow, model_name, dim, first_id + offset, embedded_at))
    meta = pd.DataFrame(rows)
    safe_write_faiss(faiss_module, fresh, index_path)
    safe_write_parquet(meta, meta_path)
    replace_repo_embedding_rows(kb, repo_id, model_name, meta)
    record_embedding_status(
        kb, repo_id=repo_id, model_name=model_name, source_fingerprint=fingerprint,
        status="ok", chunk_count=len(chunks), vector_count=len(meta), vector_dim=dim,
        started_at=started, index_path=index_path, meta_path=meta_path,
        details=f"watch: +{len(pending)} -{removed}")
    emit(on_event, "embeddings",
         {"repo_id": repo_id, "vectors": len(meta), "added": len(pending), "removed": removed})
    return {"repo_id": repo_id, "vectors": len(meta), "cached": len(kept),
            "added": len(pending), "removed": removed}
