from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from proofline.utils import now_iso


def text_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def chunk_payload(row: pd.Series, max_chars: int) -> str:
    header = "\n".join([
        f"repo: {row.get('repo_id') or ''}",
        f"path: {row.get('rel_path') or ''}",
        f"language: {row.get('language') or ''}",
        f"kind: {row.get('kind') or ''}",
        f"symbol: {row.get('symbol') or ''}",
        "",
    ])
    text = str(row.get("text") or "")
    return (header + text)[:max_chars]


def eligible_chunks(chunks: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    emb = cfg.get("indexing", {}).get("embeddings", {})
    if chunks is None or chunks.empty:
        return pd.DataFrame()
    df = chunks.copy()
    include_kinds = set(emb.get("include_kinds") or [])
    exclude_languages = set(emb.get("exclude_languages") or [])
    min_chars = int(emb.get("min_text_chars", 80))
    max_chunks = emb.get("max_chunks")
    if include_kinds:
        df = df[df["kind"].isin(include_kinds)]
    if exclude_languages:
        df = df[~df["language"].isin(exclude_languages)]
    df = df[df["text"].fillna("").str.len() >= min_chars]
    df = df.sort_values(["repo_id", "rel_path", "start_line", "kind"], kind="stable")
    if max_chunks:
        df = df.head(int(max_chunks))
    return df.reset_index(drop=True)


def resolve_device(device: str | None) -> str:
    requested = (device or "cpu").strip().lower()
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def load_sentence_transformer(model_name: str, device: str | None = None):
    # Keep PyTorch fallback enabled for macOS MPS: unsupported ops can fall back
    # to CPU instead of failing the whole embedding run.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "Embeddings are enabled, but required ML dependencies are missing. "
            "Run: ./scripts/bootstrap.sh"
        ) from e
    resolved_device = resolve_device(device)
    try:
        return SentenceTransformer(model_name, device=resolved_device, trust_remote_code=True)
    except TypeError:
        return SentenceTransformer(model_name, device=resolved_device, model_kwargs={"trust_remote_code": True})


def valid_existing_meta(meta: pd.DataFrame, chunks: pd.DataFrame, model_name: str) -> pd.DataFrame:
    if meta is None or meta.empty:
        return pd.DataFrame()
    required = {"faiss_id", "chunk_id", "text_sha1", "model_name"}
    if not required.issubset(meta.columns):
        return pd.DataFrame()
    current = set(zip(chunks["chunk_id"], chunks["_text_sha1"]))
    existing = meta[meta["model_name"] == model_name].copy()
    existing = existing[existing.apply(lambda r: (r["chunk_id"], r["text_sha1"]) in current, axis=1)]
    if existing.empty:
        return pd.DataFrame()
    existing = existing.sort_values("faiss_id", kind="stable").reset_index(drop=True)
    if existing["faiss_id"].tolist() != list(range(len(existing))):
        return pd.DataFrame()
    return existing


def safe_write_faiss(faiss_module: Any, index: Any, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    faiss_module.write_index(index, str(tmp))
    os.replace(tmp, path)


def safe_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def build_code_embeddings(kb, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    emb = cfg.get("indexing", {}).get("embeddings", {})
    if not emb.get("enabled", False):
        return pd.DataFrame(), "disabled"

    try:
        import numpy as np
    except Exception as e:
        raise RuntimeError(
            "Embeddings are enabled, but numpy is missing. "
            "Run: ./scripts/bootstrap.sh"
        ) from e

    index_path = Path(cfg["storage"]["vector_index_path"])
    meta_path = Path(cfg["storage"]["vector_meta_path"])
    rebuild = bool(emb.get("rebuild", False))
    model_name = str(emb.get("model_name") or "Qwen/Qwen3-Embedding-0.6B")
    device = str(emb.get("device") or "cpu")

    chunks = eligible_chunks(kb.query_df("SELECT * FROM code_chunks"), cfg)
    if chunks.empty:
        return pd.DataFrame(), "no eligible chunks"

    batch_size = max(1, int(emb.get("batch_size", 4)))
    max_chars = int(emb.get("max_text_chars", 4096))
    checkpoint_batches = max(1, int(emb.get("checkpoint_interval_batches", 10)))
    resolved_device = resolve_device(device)
    chunks = chunks.copy()
    chunks["_text_sha1"] = chunks.apply(lambda r: text_sha1(chunk_payload(r, max_chars)), axis=1)
    existing_meta = pd.DataFrame()
    can_resume = index_path.exists() and meta_path.exists() and not rebuild
    if can_resume:
        try:
            existing_meta = valid_existing_meta(pd.read_parquet(meta_path), chunks, model_name)
        except Exception:
            existing_meta = pd.DataFrame()
            can_resume = False
        if len(existing_meta) == len(chunks):
            return existing_meta, f"cached: {index_path}"
        if meta_path.exists() and existing_meta.empty:
            can_resume = False

    total_chunks = len(chunks)

    model = load_sentence_transformer(model_name, device=device)

    try:
        import faiss
    except Exception as e:
        raise RuntimeError(
            "Embeddings were computed, but FAISS is missing. "
            "Run: ./scripts/bootstrap.sh"
        ) from e

    index_path.parent.mkdir(parents=True, exist_ok=True)
    if can_resume and not existing_meta.empty:
        try:
            index = faiss.read_index(str(index_path))
        except Exception:
            index = None
            existing_meta = pd.DataFrame()
        if index is not None and int(index.ntotal) != len(existing_meta):
            existing_meta = pd.DataFrame()
            index = None
    else:
        index = None

    existing_keys = set(zip(existing_meta["chunk_id"], existing_meta["text_sha1"])) if not existing_meta.empty else set()
    pending = chunks[~chunks.apply(lambda r: (r["chunk_id"], r["_text_sha1"]) in existing_keys, axis=1)]
    pending = pending.reset_index(drop=True)
    if pending.empty:
        return existing_meta, f"cached: {index_path}"

    rows: List[Dict[str, Any]] = existing_meta.to_dict("records") if not existing_meta.empty else []
    vector_dim = int(existing_meta.iloc[0]["vector_dim"]) if not existing_meta.empty else 0

    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None
    total_batches = (len(pending) + batch_size - 1) // batch_size
    iterator = range(0, len(pending), batch_size)
    if tqdm:
        iterator = tqdm(iterator, total=total_batches, desc=f"Embedding chunks on {resolved_device}", unit="batch")

    batches_done = 0
    for start in iterator:
        batch = pending.iloc[start:start + batch_size]
        texts = [chunk_payload(row, max_chars) for _, row in batch.iterrows()]
        vectors = model.encode(
            texts,
            batch_size=len(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[0] != len(batch):
            raise RuntimeError(f"Embedding model returned unexpected shape: {vectors.shape}")
        if index is None:
            vector_dim = int(vectors.shape[1])
            index = faiss.IndexFlatIP(vector_dim)
        elif int(vectors.shape[1]) != vector_dim:
            raise RuntimeError(f"Embedding dimension changed from {vector_dim} to {vectors.shape[1]}")
        first_id = int(index.ntotal)
        index.add(vectors)
        embedded_at = now_iso()
        for offset, (_, row) in enumerate(batch.iterrows()):
            rows.append({
                "faiss_id": first_id + offset,
                "chunk_id": row.get("chunk_id"),
                "repo_id": row.get("repo_id"),
                "rel_path": row.get("rel_path"),
                "language": row.get("language"),
                "kind": row.get("kind"),
                "symbol": row.get("symbol"),
                "start_line": row.get("start_line"),
                "end_line": row.get("end_line"),
                "text_sha1": row.get("_text_sha1"),
                "model_name": model_name,
                "vector_dim": vector_dim,
                "embedded_at": embedded_at,
            })
        batches_done += 1
        if batches_done % checkpoint_batches == 0:
            meta = pd.DataFrame(rows)
            safe_write_faiss(faiss, index, index_path)
            safe_write_parquet(meta, meta_path)

    if index is None:
        return pd.DataFrame(), "no vectors built"
    meta = pd.DataFrame(rows)
    safe_write_faiss(faiss, index, index_path)
    safe_write_parquet(meta, meta_path)
    return meta, (
        f"built/resumed: {len(meta)}/{total_chunks} vectors, dim={vector_dim}, "
        f"device={resolved_device}, batch_size={batch_size}, max_text_chars={max_chars}"
    )
