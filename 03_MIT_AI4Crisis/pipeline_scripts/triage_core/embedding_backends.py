"""Embedding backend abstraction for pluggable embedding providers.

Provides a unified interface for OpenAI API embeddings and local HuggingFace
models (e.g. Qwen3-Embedding-0.6B), allowing easy provider switching via config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from .cache import SQLiteCache
from .utils import hash_text, progress_iter, safe_json_dumps

if TYPE_CHECKING:
    from openai import OpenAI


class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends."""

    name: str

    @abstractmethod
    def embed(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Args:
            texts: Input texts to embed.
            batch_size: Number of texts per batch.
            show_progress: Whether to display a progress bar.

        Returns:
            Array of shape (n_texts, embedding_dim).
        """

    @staticmethod
    def get_backend(
        name: str,
        model: str,
        cache: Optional[SQLiteCache] = None,
        **kwargs,
    ) -> "EmbeddingBackend":
        """Factory method to get backend by provider name.

        Args:
            name: Provider name ("openai", "huggingface", "sentence-transformers").
            model: Model identifier (e.g. "text-embedding-3-large" or
                   "Qwen/Qwen3-Embedding-0.6B").
            cache: Optional SQLiteCache for caching embeddings.
            **kwargs: Provider-specific parameters.

        Returns:
            An EmbeddingBackend instance.
        """
        backends = {
            "openai": OpenAIEmbeddingBackend,
            "huggingface": HuggingFaceEmbeddingBackend,
            "hf": HuggingFaceEmbeddingBackend,
            "sentence-transformers": SentenceTransformerEmbeddingBackend,
            "st": SentenceTransformerEmbeddingBackend,
        }
        backend_cls = backends.get(name.lower())
        if not backend_cls:
            available = sorted(set(backends.keys()))
            raise ValueError(f"Unknown embedding backend: {name}. Available: {available}")
        return backend_cls(model=model, cache=cache, **kwargs)


class OpenAIEmbeddingBackend(EmbeddingBackend):
    """OpenAI API embedding backend (existing behavior)."""

    name = "openai"

    def __init__(
        self,
        model: str,
        cache: Optional[SQLiteCache] = None,
        client: Optional["OpenAI"] = None,
        dry_run: bool = False,
        dry_run_dim: int = 1536,
        **kwargs,
    ):
        self.model = model
        self.cache = cache
        self.client = client
        self.dry_run = dry_run
        self.dry_run_dim = dry_run_dim
        self._precomputed: Dict[str, List[float]] = {}

    def _cache_key(self, text: str) -> str:
        payload = {"model": self.model, "text": text}
        return f"embed:{hash_text(safe_json_dumps(payload))}"

    def register_precomputed(self, texts: List[str], embeddings: np.ndarray) -> None:
        """Register precomputed embeddings for cache lookup."""
        for text, vec in zip(texts, embeddings):
            key = self._cache_key(text)
            self._precomputed[key] = np.asarray(vec, dtype=float).tolist()

    def embed(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        vectors: List[np.ndarray] = []
        total_batches = (len(texts) + batch_size - 1) // batch_size if texts else 0

        for start in progress_iter(
            range(0, len(texts), batch_size),
            total=total_batches,
            desc="OpenAI embeddings",
            enabled=show_progress,
        ):
            batch = texts[start : start + batch_size]
            batch_vectors: List[Optional[np.ndarray]] = []
            missing_texts: List[str] = []
            missing_keys: List[str] = []
            missing_pos: List[int] = []

            for idx, text in enumerate(batch):
                key = self._cache_key(text)
                precomputed = self._precomputed.get(key)
                if precomputed is not None:
                    batch_vectors.append(np.array(precomputed, dtype=float))
                    continue
                cached = self.cache.get(key) if self.cache else None
                if cached is None:
                    missing_texts.append(text)
                    missing_keys.append(key)
                    missing_pos.append(idx)
                    batch_vectors.append(None)
                else:
                    batch_vectors.append(np.array(cached, dtype=float))

            if missing_texts:
                if self.dry_run:
                    for pos, text, key in zip(missing_pos, missing_texts, missing_keys):
                        seed = int(hash_text(text)[:8], 16)
                        rng = np.random.default_rng(seed)
                        vec = rng.normal(size=self.dry_run_dim).astype(float)
                        batch_vectors[pos] = vec
                        if self.cache:
                            self.cache.set(key, vec.tolist())
                else:
                    if self.client is None:
                        raise RuntimeError(
                            "OpenAI client not available and dry_run is False; "
                            "cannot compute embeddings."
                        )
                    resp = self.client.embeddings.create(
                        model=self.model, input=missing_texts
                    )
                    for pos, key, embed in zip(missing_pos, missing_keys, resp.data):
                        vec = np.array(embed.embedding, dtype=float)
                        batch_vectors[pos] = vec
                        if self.cache:
                            self.cache.set(key, vec.tolist())

            batch_arr = np.vstack([v for v in batch_vectors if v is not None])
            vectors.append(batch_arr)

        return np.vstack(vectors)


class HuggingFaceEmbeddingBackend(EmbeddingBackend):
    """HuggingFace Transformers embedding backend for local models.

    Supports models like Qwen/Qwen3-Embedding-0.6B via the transformers library.
    Uses last-token pooling (appropriate for decoder-based embedding models).
    """

    name = "huggingface"

    def __init__(
        self,
        model: str,
        cache: Optional[SQLiteCache] = None,
        device: Optional[str] = None,
        max_length: int = 8192,
        instruction: Optional[str] = None,
        **kwargs,
    ):
        self.model_name = model
        self.cache = cache
        self.max_length = max_length
        self.instruction = instruction
        self._model = None
        self._tokenizer = None

        if device is None:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

    def _load_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left", trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True
        ).to(self._device)
        self._model.eval()

    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask):
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

    def _cache_key(self, text: str) -> str:
        payload = {"model": self.model_name, "text": text}
        return f"embed_hf:{hash_text(safe_json_dumps(payload))}"

    def _prepare_text(self, text: str) -> str:
        if self.instruction:
            return f"Instruct: {self.instruction}\nQuery: {text}"
        return text

    def embed(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        self._load_model()

        vectors: List[np.ndarray] = []
        total_batches = (len(texts) + batch_size - 1) // batch_size if texts else 0

        for start in progress_iter(
            range(0, len(texts), batch_size),
            total=total_batches,
            desc=f"HF embeddings ({self.model_name})",
            enabled=show_progress,
        ):
            batch = texts[start : start + batch_size]
            batch_vectors: List[Optional[np.ndarray]] = []
            missing_texts: List[str] = []
            missing_pos: List[int] = []

            for idx, text in enumerate(batch):
                key = self._cache_key(text)
                cached = self.cache.get(key) if self.cache else None
                if cached is not None:
                    batch_vectors.append(np.array(cached, dtype=float))
                else:
                    missing_texts.append(text)
                    missing_pos.append(idx)
                    batch_vectors.append(None)

            if missing_texts:
                prepared = [self._prepare_text(t) for t in missing_texts]
                inputs = self._tokenizer(
                    prepared,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self._device)

                with torch.no_grad():
                    outputs = self._model(**inputs)

                emb = self._last_token_pool(
                    outputs.last_hidden_state, inputs["attention_mask"]
                )
                emb = F.normalize(emb, p=2, dim=1).cpu().numpy()

                for i, (pos, text) in enumerate(zip(missing_pos, missing_texts)):
                    vec = emb[i]
                    batch_vectors[pos] = vec
                    if self.cache:
                        key = self._cache_key(text)
                        self.cache.set(key, vec.tolist())

            batch_arr = np.vstack([v for v in batch_vectors if v is not None])
            vectors.append(batch_arr)

        return np.vstack(vectors)


class SentenceTransformerEmbeddingBackend(EmbeddingBackend):
    """Sentence-Transformers embedding backend.

    Simpler interface for models that work well with sentence-transformers
    (e.g. Qwen/Qwen3-Embedding-0.6B).
    """

    name = "sentence-transformers"

    def __init__(
        self,
        model: str,
        cache: Optional[SQLiteCache] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        self.model_name = model
        self.cache = cache
        self._model = None
        self._device = device

    def _load_model(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        kwargs = {}
        if self._device:
            kwargs["device"] = self._device
        self._model = SentenceTransformer(self.model_name, **kwargs)

    def _cache_key(self, text: str) -> str:
        payload = {"model": self.model_name, "text": text}
        return f"embed_st:{hash_text(safe_json_dumps(payload))}"

    def embed(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        self._load_model()

        # Check cache first
        all_cached = True
        cached_vectors: List[Optional[np.ndarray]] = []
        missing_indices: List[int] = []

        for idx, text in enumerate(texts):
            key = self._cache_key(text)
            cached = self.cache.get(key) if self.cache else None
            if cached is not None:
                cached_vectors.append(np.array(cached, dtype=float))
            else:
                cached_vectors.append(None)
                missing_indices.append(idx)
                all_cached = False

        if all_cached:
            return np.vstack(cached_vectors)

        # Compute missing embeddings
        missing_texts = [texts[i] for i in missing_indices]
        emb = self._model.encode(
            missing_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )

        for i, idx in enumerate(missing_indices):
            vec = emb[i]
            cached_vectors[idx] = vec
            if self.cache:
                key = self._cache_key(texts[idx])
                self.cache.set(key, vec.tolist())

        return np.vstack(cached_vectors)
