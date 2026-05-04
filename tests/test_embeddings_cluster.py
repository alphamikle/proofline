from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from proofline.extractors import embeddings


class FakeEmbedder:
    def __init__(self, node_id: int):
        self.node_id = node_id
        self.calls = []

    def encode(
        self,
        texts,
        batch_size=None,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        self.calls.append({"texts": list(texts), "batch_size": batch_size})
        return np.asarray([[float(self.node_id), float(text[1:])] for text in texts], dtype="float32")


class EmbeddingClusterTests(unittest.TestCase):
    def test_cluster_preserves_input_order_while_distributing_texts(self) -> None:
        created = []
        fakes = []

        def fake_load_embedder(node_cfg):
            created.append(dict(node_cfg))
            fake = FakeEmbedder(len(created))
            fakes.append(fake)
            return fake

        cfg = {
            "model_name": "same-model",
            "servers": [
                {"name": "one", "provider": "openai_compatible", "base_url": "http://one.local/v1", "api_key_env": "", "batch_size": 2, "max_workers": 1},
                {"name": "two", "provider": "openai_compatible", "base_url": "http://two.local/v1", "api_key_env": "", "batch_size": 1, "max_workers": 2},
            ],
        }
        with patch.object(embeddings, "load_embedder", side_effect=fake_load_embedder):
            provider = embeddings.ClusterEmbeddingProvider(cfg)
            vectors = provider.encode([f"t{i}" for i in range(5)])

        self.assertEqual([row[1] for row in vectors.tolist()], [0, 1, 2, 3, 4])
        self.assertEqual([row[0] for row in vectors.tolist()], [1, 1, 2, 2, 1])
        self.assertEqual(created[0]["model_name"], "same-model")
        self.assertEqual(created[0]["provider"], "openai_compatible")
        self.assertEqual(created[1]["max_workers"], 2)
        self.assertNotIn("servers", created[0])
        self.assertEqual([call["batch_size"] for call in fakes[0].calls], [2, 2])
        self.assertEqual([call["batch_size"] for call in fakes[1].calls], [1, 1])

    def test_single_text_calls_rotate_across_servers(self) -> None:
        created = []

        def fake_load_embedder(node_cfg):
            created.append(dict(node_cfg))
            return FakeEmbedder(len(created))

        cfg = {
            "model_name": "same-model",
            "servers": [
                {"name": "one", "provider": "openai_compatible", "base_url": "http://one.local/v1", "api_key_env": "", "batch_size": 8},
                {"name": "two", "provider": "openai_compatible", "base_url": "http://two.local/v1", "api_key_env": "", "batch_size": 8},
            ],
        }
        with patch.object(embeddings, "load_embedder", side_effect=fake_load_embedder):
            provider = embeddings.ClusterEmbeddingProvider(cfg)
            first = provider.encode(["t0"])
            second = provider.encode(["t1"])

        self.assertEqual(first.tolist(), [[1.0, 0.0]])
        self.assertEqual(second.tolist(), [[2.0, 1.0]])

    def test_cluster_batch_size_uses_combined_server_capacity(self) -> None:
        cfg = {
            "batch_size": 4,
            "servers": [
                {"name": "one", "provider": "openai_compatible", "base_url": "http://one.local/v1", "batch_size": 8, "max_workers": 2},
                {"name": "two", "provider": "cli", "command": "embed", "batch_size": 3, "max_workers": 1},
            ],
        }

        self.assertEqual(embeddings.embedding_batch_size(cfg), 19)

    def test_cluster_model_id_does_not_depend_on_server_urls(self) -> None:
        model_id = embeddings.embedding_model_id({
            "model_name": "same-model",
            "servers": [
                {"name": "one", "provider": "openai_compatible", "base_url": "http://one.local/v1"},
                {"name": "two", "provider": "openai_compatible", "base_url": "http://two.local/v1"},
            ],
        })

        self.assertEqual(model_id, "cluster:same-model")


if __name__ == "__main__":
    unittest.main()
