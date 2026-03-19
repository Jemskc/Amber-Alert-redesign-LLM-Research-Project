from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI

from .cache import SQLiteCache
from .utils import hash_text, progress_iter, safe_json_dumps


class OpenAIClient:
    def __init__(
        self,
        api_key: Optional[str],
        cache_path: str,
        max_retries: int = 4,
        dry_run: bool = False,
        show_progress: bool = False,
        embedding_provider: str = "openai",
        llm_provider: str = "openai",
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if (self.api_key and not dry_run) else None
        self.cache = SQLiteCache(cache_path)
        self.max_retries = max_retries
        self.dry_run = dry_run
        self._supports_response_format: Optional[bool] = None
        self._supports_temperature: Optional[bool] = None
        self.show_progress = show_progress
        self._precomputed_embeddings: Dict[str, List[float]] = {}
        self._embedding_provider = embedding_provider
        self._llm_provider = llm_provider
        self._embedding_backend = None
        self._llm_backend = None

    def _require_client(self, purpose: str) -> OpenAI:
        if self.client is None:
            raise RuntimeError(
                f"OPENAI_API_KEY not set and dry_run is False; cannot call OpenAI for {purpose}."
            )
        return self.client

    def _cache_key(self, prefix: str, payload: Dict[str, Any]) -> str:
        return f"{prefix}:{hash_text(safe_json_dumps(payload))}"

    def _mock_probs(self, n: int, seed: int) -> List[float]:
        rng = np.random.default_rng(seed)
        raw = rng.random(n)
        raw = raw / raw.sum()
        return raw.tolist()

    def _prompt_with_schema(self, prompt: str, schema: Dict[str, Any]) -> str:
        schema_text = safe_json_dumps(schema)
        return (
            f"{prompt}\n\n"
            "JSON schema:\n"
            f"{schema_text}\n\n"
            "Return JSON only."
        )

    @staticmethod
    def _is_temperature_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "temperature" in msg and "not supported" in msg

    def _responses_create_basic(self, model: str, prompt: str, temperature: float, response_format: Optional[Dict[str, Any]] = None):
        client = self._require_client("responses")
        kwargs: Dict[str, Any] = {"model": model, "input": prompt}
        if response_format is not None:
            kwargs["response_format"] = response_format
        if self._supports_temperature is not False:
            kwargs["temperature"] = temperature
        try:
            return client.responses.create(**kwargs)
        except Exception as exc:
            if self._supports_temperature is not False and self._is_temperature_error(exc):
                self._supports_temperature = False
                kwargs.pop("temperature", None)
                return client.responses.create(**kwargs)
            raise

    def _responses_create(self, model: str, prompt: str, schema: Dict[str, Any], temperature: float):
        if self._supports_response_format is False:
            return self._responses_create_basic(
                model=model,
                prompt=self._prompt_with_schema(prompt, schema),
                temperature=temperature,
            )

        try:
            resp = self._responses_create_basic(
                model=model,
                prompt=prompt,
                temperature=temperature,
                response_format={"type": "json_schema", "json_schema": schema},
            )
            self._supports_response_format = True
            return resp
        except TypeError as exc:
            if "response_format" not in str(exc):
                raise
            self._supports_response_format = False
            return self._responses_create_basic(
                model=model,
                prompt=self._prompt_with_schema(prompt, schema),
                temperature=temperature,
            )

    @staticmethod
    def _response_text(resp) -> str:
        text = getattr(resp, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text
        output = getattr(resp, "output", None)
        if output:
            try:
                return output[0].content[0].text
            except Exception:
                pass
        if isinstance(text, str):
            return text
        return ""

    def register_embeddings(self, model: str, texts: List[str], embeddings: np.ndarray) -> None:
        if len(texts) != len(embeddings):
            raise ValueError("Embeddings length does not match texts length")
        if self._embedding_provider == "openai" and self._embedding_backend is not None:
            from .embedding_backends import OpenAIEmbeddingBackend
            if isinstance(self._embedding_backend, OpenAIEmbeddingBackend):
                self._embedding_backend.register_precomputed(texts, embeddings)
        for text, vec in zip(texts, embeddings):
            payload = {"model": model, "text": text}
            key = self._cache_key("embed", payload)
            self._precomputed_embeddings[key] = np.asarray(vec, dtype=float).tolist()

    def responses_json(self, model: str, schema: Dict[str, Any], prompt: str, temperature: float = 0) -> Dict[str, Any]:
        if self._llm_provider != "openai":
            backend = self._get_llm_backend(model)
            return backend.generate_json(prompt, schema, temperature=temperature)

        payload = {"model": model, "schema": schema, "prompt": prompt, "temperature": temperature}
        key = self._cache_key("responses", payload)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        if self.dry_run:
            seed = int(hash_text(prompt)[:8], 16)
            mock = {"mock": True, "seed": seed}
            self.cache.set(key, mock)
            return mock

        self._require_client("responses")
        for attempt in range(self.max_retries):
            try:
                resp = self._responses_create(model, prompt, schema, temperature)
                data = self._response_text(resp)
                result = json_loads_strict(data)
                if isinstance(result, list):
                    if len(result) == 1 and isinstance(result[0], dict):
                        result = result[0]
                    elif result and all(isinstance(item, dict) for item in result):
                        result = result[0]
                    else:
                        raise ValueError("Response JSON is a list, expected object")
                self.cache.set(key, result)
                return result
            except Exception:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt + 0.5)

        raise RuntimeError("OpenAI call failed after retries")
    def responses_json_with_repair(self, model: str, schema: Dict[str, Any], prompt: str, temperature: float = 0) -> Dict[str, Any]:
        if self._llm_provider != "openai":
            backend = self._get_llm_backend(model)
            return backend.generate_json_with_repair(prompt, schema, temperature=temperature)

        try:
            return self.responses_json(model, schema, prompt, temperature=temperature)
        except Exception:
            repair_prompt = (
                "Fix the following response to strictly match the JSON schema. "
                "Return JSON only.\n\n"
                f"Original prompt:\n{prompt}\n"
            )
            return self.responses_json(model, schema, repair_prompt, temperature=temperature)

    def _get_embedding_backend(self, model: str):
        """Lazily create and return the embedding backend."""
        if self._embedding_backend is not None:
            return self._embedding_backend
        if self._embedding_provider == "openai":
            from .embedding_backends import OpenAIEmbeddingBackend

            backend = OpenAIEmbeddingBackend(
                model=model,
                cache=self.cache,
                client=self.client,
                dry_run=self.dry_run,
            )
            self._embedding_backend = backend
            return backend
        from .embedding_backends import EmbeddingBackend

        backend = EmbeddingBackend.get_backend(
            name=self._embedding_provider,
            model=model,
            cache=self.cache,
        )
        self._embedding_backend = backend
        return backend

    def _get_llm_backend(self, model: str):
        """Lazily create and return the LLM backend."""
        if self._llm_backend is not None:
            return self._llm_backend
        if self._llm_provider == "openai":
            return None  # Use built-in OpenAI methods
        from .llm_backends import LLMBackend

        backend = LLMBackend.get_backend(
            name=self._llm_provider,
            model=model,
            cache=self.cache,
        )
        self._llm_backend = backend
        return backend

    def embeddings(self, model: str, texts: List[str], batch_size: int = 64) -> np.ndarray:
        if self._embedding_provider != "openai":
            backend = self._get_embedding_backend(model)
            return backend.embed(texts, batch_size=batch_size, show_progress=self.show_progress)

        vectors: List[np.ndarray] = []
        total_batches = (len(texts) + batch_size - 1) // batch_size if texts else 0
        for start in progress_iter(
            range(0, len(texts), batch_size),
            total=total_batches,
            desc="OpenAI embeddings",
            enabled=self.show_progress,
        ):
            batch = texts[start : start + batch_size]
            batch_vectors: List[np.ndarray | None] = []
            missing_texts: List[str] = []
            missing_keys: List[str] = []
            missing_pos: List[int] = []
            for idx, text in enumerate(batch):
                payload = {"model": model, "text": text}
                key = self._cache_key("embed", payload)
                precomputed = self._precomputed_embeddings.get(key)
                if precomputed is not None:
                    batch_vectors.append(np.array(precomputed, dtype=float))
                    continue
                cached = self.cache.get(key)
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
                        vec = rng.normal(size=1536).astype(float)
                        batch_vectors[pos] = vec
                        self.cache.set(key, vec.tolist())
                else:
                    client = self._require_client("embeddings")
                    resp = client.embeddings.create(model=model, input=missing_texts)
                    for pos, key, embed in zip(missing_pos, missing_keys, resp.data):
                        vec = np.array(embed.embedding, dtype=float)
                        batch_vectors[pos] = vec
                        self.cache.set(key, vec.tolist())

            batch_arr = np.vstack([vec for vec in batch_vectors if vec is not None])
            vectors.append(batch_arr)

        return np.vstack(vectors)


def json_loads_strict(text: str) -> Dict[str, Any]:
    import json
    import re

    if text is None:
        raise ValueError("Empty response text")

    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Empty response text")

    if cleaned.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.IGNORECASE | re.DOTALL)
        if match:
            cleaned = match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for marker in ("{", "["):
            idx = cleaned.find(marker)
            if idx == -1:
                continue
            try:
                obj, _ = decoder.raw_decode(cleaned[idx:])
                return obj
            except json.JSONDecodeError:
                continue
        raise
