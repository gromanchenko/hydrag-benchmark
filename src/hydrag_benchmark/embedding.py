"""Dense embedding interface and implementations.

Supports:
- TransformersEmbedder: GPU-accelerated via HuggingFace transformers
  (requires [gpu] extra: torch + transformers)
- HashEmbedder: Deterministic hash-based embedder for testing (no deps)
"""

from __future__ import annotations

import hashlib
import logging
import math
import struct
from dataclasses import dataclass

logger = logging.getLogger("hydrag_benchmark.embedding")


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding model."""

    model_name: str = "Alibaba-NLP/gte-Qwen2-7B-instruct"
    dimension: int = 3584
    batch_size: int = 32
    device: str = "cuda"
    dtype: str = "float16"


class TransformersEmbedder:
    """Dense embedder using HuggingFace transformers with last_token_pool.

    Implements the gte-Qwen2-7B-instruct pooling strategy:
    use the hidden state of the last non-padding token (last_token_pool),
    NOT mean pooling.

    Requires: pip install hydrag-benchmark[gpu]
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self._config = config or EmbeddingConfig()
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "GPU embedding requires torch and transformers. "
                "Install with: pip install hydrag-benchmark[gpu]"
            ) from exc

        logger.info("Loading embedding model: %s", self._config.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._config.model_name, trust_remote_code=True,
        )
        dtype = getattr(torch, self._config.dtype, torch.float16)
        self._model = AutoModel.from_pretrained(
            self._config.model_name, torch_dtype=dtype, trust_remote_code=True,
        ).to(self._config.device).eval()

    @property
    def dimension(self) -> int:
        return self._config.dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using last_token_pool strategy."""
        self._load()
        import torch

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._config.batch_size):
            batch = texts[i : i + self._config.batch_size]
            encoded = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=8192, return_tensors="pt",
            ).to(self._config.device)

            with torch.no_grad():
                outputs = self._model(**encoded)

            # last_token_pool: use the hidden state at the last non-padding position
            embs = self._last_token_pool(
                outputs.last_hidden_state, encoded["attention_mask"],
            )
            # L2 normalize
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            all_embeddings.extend(embs.cpu().tolist())

        return all_embeddings

    @staticmethod
    def _last_token_pool(
        last_hidden_states: "torch.Tensor",
        attention_mask: "torch.Tensor",
    ) -> "torch.Tensor":
        """Pool using the last non-padding token's hidden state."""
        import torch

        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

    def unload(self) -> None:
        """Free GPU memory."""
        self._model = None
        self._tokenizer = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class HashEmbedder:
    """Deterministic hash-based embedder for testing. No GPU required.

    Produces stable, reproducible embeddings based on text content hash.
    Useful for CI/CD and local development without GPU hardware.
    """

    def __init__(self, dimension: int = 3584) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]

    def _hash_embed(self, text: str) -> list[float]:
        """Generate a deterministic unit vector from text hash."""
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Expand hash to fill dimension by repeated hashing
        raw: list[float] = []
        seed = h
        while len(raw) < self._dimension:
            seed = hashlib.sha256(seed).digest()
            # Unpack 8 floats from 32 bytes (using 4-byte chunks)
            for j in range(0, 32, 4):
                val = struct.unpack("<I", seed[j : j + 4])[0]
                raw.append((val / 0xFFFFFFFF) * 2.0 - 1.0)

        raw = raw[: self._dimension]
        # L2 normalize
        norm = math.sqrt(sum(x * x for x in raw))
        if norm > 0:
            raw = [x / norm for x in raw]
        return raw
