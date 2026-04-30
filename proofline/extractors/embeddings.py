from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urljoin

import pandas as pd
import requests

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


def embedding_provider(emb: Dict[str, Any]) -> str:
    provider = str(emb.get("provider") or "sentence_transformers").strip().lower()
    if provider == "command":
        provider = "cli"
    return provider


def embedding_model_id(emb: Dict[str, Any]) -> str:
    configured = str(emb.get("model_id") or "").strip()
    if configured:
        return configured
    provider = embedding_provider(emb)
    model_name = str(emb.get("model_name") or "Qwen/Qwen3-Embedding-0.6B").strip()
    if provider == "sentence_transformers":
        return model_name
    suffix = ""
    if emb.get("dimensions") is not None:
        suffix += f":dim{int(emb['dimensions'])}"
    if provider in {"openai", "openai_compatible"}:
        base = embedding_base_url(emb, "https://api.openai.com/v1" if provider == "openai" else None) or ""
        if base:
            suffix += ":" + hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest()[:8]
    if model_name:
        return f"{provider}:{model_name}{suffix}"
    command = str(emb.get("command") or "")
    digest = hashlib.sha1(command.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{provider}:{digest}{suffix}"


def load_embedder(emb: Dict[str, Any]):
    provider = embedding_provider(emb)
    if provider == "sentence_transformers":
        model_name = str(emb.get("model_name") or "Qwen/Qwen3-Embedding-0.6B")
        return SentenceTransformersEmbedder(model_name, str(emb.get("device") or "cpu"))
    if provider in {"openai", "openai_compatible"}:
        return OpenAIEmbeddingProvider(emb, provider)
    if provider == "cli":
        return CLIEmbeddingProvider(emb)
    raise RuntimeError(f"Unsupported embeddings.provider: {provider}")


def normalize_vectors(vectors: Any):
    import numpy as np

    arr = np.asarray(vectors, dtype="float32")
    if arr.ndim != 2:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


class SentenceTransformersEmbedder:
    def __init__(self, model_name: str, device: str | None = None):
        self.model = load_sentence_transformer(model_name, device=device)

    def encode(
        self,
        texts: List[str],
        batch_size: int | None = None,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ):
        return self.model.encode(
            texts,
            batch_size=batch_size or len(texts),
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )


class OpenAIEmbeddingProvider:
    def __init__(self, emb: Dict[str, Any], provider: str):
        self.emb = emb
        self.provider = provider
        self.model_name = str(emb.get("model_name") or "")
        if not self.model_name:
            raise RuntimeError("embeddings.model_name is required for OpenAI-compatible embeddings")
        default_base = "https://api.openai.com/v1" if provider == "openai" else None
        self.base_url = embedding_base_url(emb, default_base)
        if not self.base_url:
            raise RuntimeError("embeddings.base_url or OPENAI_BASE_URL is required for embeddings.provider=openai_compatible")
        self.api_key = configured_env(emb, "api_key_env", "OPENAI_API_KEY")
        if provider == "openai" and not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for embeddings.provider=openai")
        self.timeout = int(emb.get("request_timeout_seconds") or 600)
        self.dimensions = emb.get("dimensions")
        self.normalize = bool(emb.get("normalize_embeddings", True))

    def encode(
        self,
        texts: List[str],
        batch_size: int | None = None,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ):
        payload: Dict[str, Any] = {"model": self.model_name, "input": texts}
        if self.dimensions is not None:
            payload["dimensions"] = int(self.dimensions)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            urljoin(self.base_url.rstrip("/") + "/", "embeddings"),
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        vectors = parse_embedding_response(response.json())
        if self.normalize and normalize_embeddings:
            return normalize_vectors(vectors)
        import numpy as np

        return np.asarray(vectors, dtype="float32")


class CLIEmbeddingProvider:
    def __init__(self, emb: Dict[str, Any]):
        self.emb = emb
        self.command = str(emb.get("command") or "").strip()
        if not self.command:
            raise RuntimeError("embeddings.command is required for embeddings.provider=cli")
        self.model_name = str(emb.get("model_name") or "")
        self.timeout = int(emb.get("request_timeout_seconds") or 600)
        self.normalize = bool(emb.get("normalize_embeddings", True))

    def encode(
        self,
        texts: List[str],
        batch_size: int | None = None,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ):
        payload = {"model": self.model_name, "input": texts}
        p = subprocess.run(
            self.command,
            input=json.dumps(payload),
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout,
        )
        if p.returncode != 0:
            raise RuntimeError(f"Embedding command failed: {p.stderr.strip() or p.returncode}")
        try:
            data = json.loads(p.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError("Embedding command did not return JSON") from e
        vectors = parse_embedding_response(data)
        if self.normalize and normalize_embeddings:
            return normalize_vectors(vectors)
        import numpy as np

        return np.asarray(vectors, dtype="float32")


def parse_embedding_response(data: Dict[str, Any]) -> List[List[float]]:
    if isinstance(data.get("embeddings"), list):
        return data["embeddings"]
    rows = data.get("data")
    if isinstance(rows, list):
        ordered = sorted(rows, key=lambda r: int(r.get("index", 0)) if isinstance(r, dict) else 0)
        vectors = [r.get("embedding") for r in ordered if isinstance(r, dict)]
        if vectors and all(isinstance(v, list) for v in vectors):
            return vectors
    if isinstance(data.get("embedding"), list):
        return [data["embedding"]]
    raise RuntimeError("Embedding response must contain data[].embedding or embeddings")


def embedding_base_url(emb: Dict[str, Any], default: str | None) -> str | None:
    env_value = configured_env(emb, "base_url_env", "OPENAI_BASE_URL")
    value = str(emb.get("base_url") or env_value or default or "").strip()
    return value or None


def configured_env(section: Dict[str, Any], key: str, default_name: str) -> str | None:
    configured = section.get(key)
    if configured == "":
        return None
    return env_value(str(configured or default_name))


def env_value(name: str | None) -> str | None:
    if not name:
        return None
    value = os.environ.get(str(name))
    return value.strip() if value else None


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


def vector_shard_dir(cfg: Dict[str, Any]) -> Path:
    configured = cfg.get("indexing", {}).get("embeddings", {}).get("shard_dir")
    if configured:
        return Path(str(configured))
    index_path = Path(cfg["storage"]["vector_index_path"])
    return index_path.with_name(f"{index_path.stem}_shards")


def repo_shard_stem(repo_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", repo_id).strip("._")[:80] or "repo"
    digest = hashlib.sha1(repo_id.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{safe}-{digest}"


def repo_vector_paths(cfg: Dict[str, Any], repo_id: str) -> Tuple[Path, Path]:
    shard_dir = vector_shard_dir(cfg)
    stem = repo_shard_stem(repo_id)
    return shard_dir / f"{stem}.faiss", shard_dir / f"{stem}.parquet"


def chunks_source_fingerprint(chunks: pd.DataFrame) -> str:
    if chunks is None or chunks.empty:
        return hashlib.sha1(b"").hexdigest()
    h = hashlib.sha1()
    for row in chunks.sort_values(["repo_id", "rel_path", "start_line", "kind"], kind="stable").itertuples(index=False):
        chunk_id = str(getattr(row, "chunk_id", "") or "")
        text_hash = str(getattr(row, "_text_sha1", "") or "")
        h.update(f"{chunk_id}\0{text_hash}\n".encode("utf-8", errors="ignore"))
    return h.hexdigest()


def load_existing_repo_index(faiss_module: Any, index_path: Path, meta_path: Path, chunks: pd.DataFrame, model_name: str) -> Tuple[Any, pd.DataFrame]:
    if not index_path.exists() or not meta_path.exists():
        return None, pd.DataFrame()
    try:
        existing_meta = valid_existing_meta(pd.read_parquet(meta_path), chunks, model_name)
        if existing_meta.empty:
            return None, pd.DataFrame()
        index = faiss_module.read_index(str(index_path))
        if int(index.ntotal) != len(existing_meta):
            return None, pd.DataFrame()
        return index, existing_meta
    except Exception:
        return None, pd.DataFrame()


def replace_repo_embedding_rows(kb, repo_id: str, model_name: str, meta: pd.DataFrame) -> None:
    kb.execute("DELETE FROM code_embedding_index WHERE repo_id = ? AND model_name = ?", [repo_id, model_name])
    if not meta.empty:
        kb.append_df("code_embedding_index", meta)


def record_embedding_status(
    kb,
    *,
    repo_id: str,
    model_name: str,
    source_fingerprint: str,
    status: str,
    chunk_count: int,
    vector_count: int,
    vector_dim: int,
    started_at: str,
    index_path: Path,
    meta_path: Path,
    details: str = "",
) -> None:
    kb.execute("DELETE FROM code_embedding_repo_status WHERE repo_id = ? AND model_name = ?", [repo_id, model_name])
    kb.append_df("code_embedding_repo_status", pd.DataFrame([{
        "repo_id": repo_id,
        "model_name": model_name,
        "source_fingerprint": source_fingerprint,
        "status": status,
        "chunk_count": chunk_count,
        "vector_count": vector_count,
        "vector_dim": vector_dim,
        "started_at": started_at,
        "finished_at": now_iso() if status != "running" else "",
        "index_path": str(index_path),
        "meta_path": str(meta_path),
        "details": details,
    }]))
    try:
        kb.execute("DELETE FROM pipeline_repo_status WHERE stage = 'embeddings' AND repo_id = ?", [repo_id])
        kb.append_df("pipeline_repo_status", pd.DataFrame([{
            "stage": "embeddings",
            "repo_id": repo_id,
            "fingerprint": source_fingerprint,
            "status": status,
            "started_at": started_at,
            "finished_at": now_iso() if status != "running" else "",
            "item_count": int(vector_count),
            "details": details,
        }]))
    except Exception:
        pass


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

    rebuild = bool(emb.get("rebuild", False))
    model_name = embedding_model_id(emb)
    provider = embedding_provider(emb)
    device = str(emb.get("device") or "cpu")
    batch_size = max(1, int(emb.get("batch_size", 4)))
    max_chars = int(emb.get("max_text_chars", 4096))
    checkpoint_batches = max(1, int(emb.get("checkpoint_interval_batches", 10)))
    resolved_device = resolve_device(device)

    try:
        import faiss
    except Exception as e:
        raise RuntimeError("Embeddings are enabled, but FAISS is missing. Run: ./scripts/bootstrap.sh") from e

    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    repos = kb.query_df("SELECT repo_id FROM repo_inventory ORDER BY repo_id")
    if repos.empty:
        repos = kb.query_df("SELECT DISTINCT repo_id FROM code_chunks ORDER BY repo_id")
    if repos.empty:
        return pd.DataFrame(), "no repositories"
    workers = max(1, int(emb.get("max_workers") or 1))
    if workers > 1 and len(repos) > 1:
        return build_code_embeddings_parallel(cfg, repos["repo_id"].fillna("").astype(str).tolist(), workers)

    shard_dir = vector_shard_dir(cfg)
    shard_dir.mkdir(parents=True, exist_ok=True)
    model = None
    total_vectors = 0
    total_chunks = 0
    skipped = 0
    processed = 0

    repo_iterator = repos["repo_id"].fillna("").astype(str).tolist()
    if tqdm:
        repo_iterator = tqdm(repo_iterator, total=len(repos), desc=f"Embedding repositories via {provider}", unit="repo", position=0)

    for repo_id in repo_iterator:
        if not repo_id:
            continue
        index_path, meta_path = repo_vector_paths(cfg, repo_id)
        chunks = eligible_chunks(
            kb.query_df("SELECT * FROM code_chunks WHERE repo_id = ? ORDER BY rel_path, start_line, kind", [repo_id]),
            cfg,
        )
        chunks = chunks.copy()
        if not chunks.empty:
            chunks["_text_sha1"] = chunks.apply(lambda r: text_sha1(chunk_payload(r, max_chars)), axis=1)
        source_fingerprint = chunks_source_fingerprint(chunks)
        started = now_iso()

        if chunks.empty:
            replace_repo_embedding_rows(kb, repo_id, model_name, pd.DataFrame())
            record_embedding_status(
                kb,
                repo_id=repo_id,
                model_name=model_name,
                source_fingerprint=source_fingerprint,
                status="ok",
                chunk_count=0,
                vector_count=0,
                vector_dim=0,
                started_at=started,
                index_path=index_path,
                meta_path=meta_path,
                details="no eligible chunks",
            )
            processed += 1
            continue

        if rebuild:
            for path in (index_path, meta_path):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

        index, existing_meta = load_existing_repo_index(faiss, index_path, meta_path, chunks, model_name) if not rebuild else (None, pd.DataFrame())
        if not existing_meta.empty and len(existing_meta) == len(chunks):
            replace_repo_embedding_rows(kb, repo_id, model_name, existing_meta)
            vector_dim = int(existing_meta.iloc[0]["vector_dim"]) if not existing_meta.empty else 0
            record_embedding_status(
                kb,
                repo_id=repo_id,
                model_name=model_name,
                source_fingerprint=source_fingerprint,
                status="ok",
                chunk_count=len(chunks),
                vector_count=len(existing_meta),
                vector_dim=vector_dim,
                started_at=started,
                index_path=index_path,
                meta_path=meta_path,
                details="cached",
            )
            total_vectors += len(existing_meta)
            total_chunks += len(chunks)
            skipped += 1
            continue

        existing_keys = set(zip(existing_meta["chunk_id"], existing_meta["text_sha1"])) if not existing_meta.empty else set()
        pending = chunks[~chunks.apply(lambda r: (r["chunk_id"], r["_text_sha1"]) in existing_keys, axis=1)]
        pending = pending.reset_index(drop=True)
        rows: List[Dict[str, Any]] = existing_meta.to_dict("records") if not existing_meta.empty else []
        vector_dim = int(existing_meta.iloc[0]["vector_dim"]) if not existing_meta.empty else 0
        replace_repo_embedding_rows(kb, repo_id, model_name, existing_meta)
        record_embedding_status(
            kb,
            repo_id=repo_id,
            model_name=model_name,
            source_fingerprint=source_fingerprint,
            status="running",
            chunk_count=len(chunks),
            vector_count=len(rows),
            vector_dim=vector_dim,
            started_at=started,
            index_path=index_path,
            meta_path=meta_path,
            details=f"pending={len(pending)}",
        )

        if model is None:
            model = load_embedder(emb)

        total_batches = (len(pending) + batch_size - 1) // batch_size
        batch_iterator = range(0, len(pending), batch_size)
        if tqdm:
            batch_iterator = tqdm(batch_iterator, total=total_batches, desc=f"Embedding {repo_id}", unit="batch", position=1, leave=False)

        batches_done = 0
        try:
            for start in batch_iterator:
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
                    replace_repo_embedding_rows(kb, repo_id, model_name, meta)
                    record_embedding_status(
                        kb,
                        repo_id=repo_id,
                        model_name=model_name,
                        source_fingerprint=source_fingerprint,
                        status="running",
                        chunk_count=len(chunks),
                        vector_count=len(meta),
                        vector_dim=vector_dim,
                        started_at=started,
                        index_path=index_path,
                        meta_path=meta_path,
                        details=f"pending={len(chunks) - len(meta)}",
                    )

            if index is None:
                meta = pd.DataFrame()
            else:
                meta = pd.DataFrame(rows)
                safe_write_faiss(faiss, index, index_path)
                safe_write_parquet(meta, meta_path)
            replace_repo_embedding_rows(kb, repo_id, model_name, meta)
            record_embedding_status(
                kb,
                repo_id=repo_id,
                model_name=model_name,
                source_fingerprint=source_fingerprint,
                status="ok",
                chunk_count=len(chunks),
                vector_count=len(meta),
                vector_dim=vector_dim,
                started_at=started,
                index_path=index_path,
                meta_path=meta_path,
            )
            total_vectors += len(meta)
            total_chunks += len(chunks)
            processed += 1
        except Exception as e:
            record_embedding_status(
                kb,
                repo_id=repo_id,
                model_name=model_name,
                source_fingerprint=source_fingerprint,
                status="error",
                chunk_count=len(chunks),
                vector_count=len(rows),
                vector_dim=vector_dim,
                started_at=started,
                index_path=index_path,
                meta_path=meta_path,
                details=str(e),
            )
            raise

    return pd.DataFrame(), (
        f"repo-sharded: vectors={total_vectors}/{total_chunks}, repos={len(repos)}, "
        f"processed={processed}, skipped={skipped}, provider={provider}, device={resolved_device}, "
        f"batch_size={batch_size}, max_text_chars={max_chars}, shard_dir={shard_dir}"
    )


def build_code_embeddings_parallel(cfg: Dict[str, Any], repo_ids: List[str], workers: int) -> Tuple[pd.DataFrame, str]:
    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None
    completed = 0
    skipped = 0
    vectors = 0
    chunks = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(build_code_embeddings_for_repo, cfg, repo_id) for repo_id in repo_ids if repo_id]
        iterator = as_completed(futures)
        if tqdm:
            iterator = tqdm(iterator, total=len(futures), desc="Embedding repositories", unit="repo", position=0)
        for future in iterator:
            result = future.result()
            completed += 1
            skipped += int(bool(result.get("skipped")))
            vectors += int(result.get("vectors") or 0)
            chunks += int(result.get("chunks") or 0)
    return pd.DataFrame(), f"repo-sharded parallel: vectors={vectors}/{chunks}, repos={completed}, skipped={skipped}, workers={workers}, shard_dir={vector_shard_dir(cfg)}"


def build_code_embeddings_for_repo(cfg: Dict[str, Any], repo_id: str) -> Dict[str, Any]:
    from proofline.storage import KB

    emb = cfg.get("indexing", {}).get("embeddings", {})
    import numpy as np
    import faiss

    kb = KB(cfg["storage"]["duckdb_path"])
    try:
        rebuild = bool(emb.get("rebuild", False))
        model_name = embedding_model_id(emb)
        batch_size = max(1, int(emb.get("batch_size", 4)))
        max_chars = int(emb.get("max_text_chars", 4096))
        checkpoint_batches = max(1, int(emb.get("checkpoint_interval_batches", 10)))
        index_path, meta_path = repo_vector_paths(cfg, repo_id)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        chunks = eligible_chunks(
            kb.query_df("SELECT * FROM code_chunks WHERE repo_id = ? ORDER BY rel_path, start_line, kind", [repo_id]),
            cfg,
        )
        chunks = chunks.copy()
        if not chunks.empty:
            chunks["_text_sha1"] = chunks.apply(lambda r: text_sha1(chunk_payload(r, max_chars)), axis=1)
        source_fingerprint = chunks_source_fingerprint(chunks)
        started = now_iso()
        if chunks.empty:
            replace_repo_embedding_rows(kb, repo_id, model_name, pd.DataFrame())
            record_embedding_status(kb, repo_id=repo_id, model_name=model_name, source_fingerprint=source_fingerprint, status="ok", chunk_count=0, vector_count=0, vector_dim=0, started_at=started, index_path=index_path, meta_path=meta_path, details="no eligible chunks")
            return {"repo_id": repo_id, "vectors": 0, "chunks": 0, "skipped": False}
        if rebuild:
            for path in (index_path, meta_path):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
        index, existing_meta = load_existing_repo_index(faiss, index_path, meta_path, chunks, model_name) if not rebuild else (None, pd.DataFrame())
        if not existing_meta.empty and len(existing_meta) == len(chunks):
            replace_repo_embedding_rows(kb, repo_id, model_name, existing_meta)
            vector_dim = int(existing_meta.iloc[0]["vector_dim"]) if not existing_meta.empty else 0
            record_embedding_status(kb, repo_id=repo_id, model_name=model_name, source_fingerprint=source_fingerprint, status="ok", chunk_count=len(chunks), vector_count=len(existing_meta), vector_dim=vector_dim, started_at=started, index_path=index_path, meta_path=meta_path, details="cached")
            return {"repo_id": repo_id, "vectors": len(existing_meta), "chunks": len(chunks), "skipped": True}

        existing_keys = set(zip(existing_meta["chunk_id"], existing_meta["text_sha1"])) if not existing_meta.empty else set()
        pending = chunks[~chunks.apply(lambda r: (r["chunk_id"], r["_text_sha1"]) in existing_keys, axis=1)].reset_index(drop=True)
        rows: List[Dict[str, Any]] = existing_meta.to_dict("records") if not existing_meta.empty else []
        vector_dim = int(existing_meta.iloc[0]["vector_dim"]) if not existing_meta.empty else 0
        replace_repo_embedding_rows(kb, repo_id, model_name, existing_meta)
        record_embedding_status(kb, repo_id=repo_id, model_name=model_name, source_fingerprint=source_fingerprint, status="running", chunk_count=len(chunks), vector_count=len(rows), vector_dim=vector_dim, started_at=started, index_path=index_path, meta_path=meta_path, details=f"pending={len(pending)}")
        model = load_embedder(emb)
        batches_done = 0
        for start in range(0, len(pending), batch_size):
            batch = pending.iloc[start:start + batch_size]
            texts = [chunk_payload(row, max_chars) for _, row in batch.iterrows()]
            vectors = model.encode(texts, batch_size=len(texts), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
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
                replace_repo_embedding_rows(kb, repo_id, model_name, meta)
                record_embedding_status(kb, repo_id=repo_id, model_name=model_name, source_fingerprint=source_fingerprint, status="running", chunk_count=len(chunks), vector_count=len(meta), vector_dim=vector_dim, started_at=started, index_path=index_path, meta_path=meta_path, details=f"pending={len(chunks) - len(meta)}")
        meta = pd.DataFrame(rows)
        if index is not None:
            safe_write_faiss(faiss, index, index_path)
            safe_write_parquet(meta, meta_path)
        replace_repo_embedding_rows(kb, repo_id, model_name, meta)
        record_embedding_status(kb, repo_id=repo_id, model_name=model_name, source_fingerprint=source_fingerprint, status="ok", chunk_count=len(chunks), vector_count=len(meta), vector_dim=vector_dim, started_at=started, index_path=index_path, meta_path=meta_path)
        return {"repo_id": repo_id, "vectors": len(meta), "chunks": len(chunks), "skipped": False}
    except Exception as e:
        try:
            record_embedding_status(kb, repo_id=repo_id, model_name=embedding_model_id(emb), source_fingerprint="", status="error", chunk_count=0, vector_count=0, vector_dim=0, started_at=now_iso(), index_path=repo_vector_paths(cfg, repo_id)[0], meta_path=repo_vector_paths(cfg, repo_id)[1], details=str(e))
        finally:
            raise
    finally:
        kb.close()
