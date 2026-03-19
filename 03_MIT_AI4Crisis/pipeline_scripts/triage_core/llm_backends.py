"""LLM backend abstraction for pluggable LLM providers.

Provides a unified interface for OpenAI API and local HuggingFace models
(e.g. Qwen/Qwen3-8B), allowing easy provider switching via config.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .cache import SQLiteCache
from .utils import hash_text, safe_json_dumps


def _json_loads_strict(text: str) -> Dict[str, Any]:
    """Parse JSON from LLM output, handling markdown fences and partial output."""
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
                if isinstance(obj, list):
                    if len(obj) == 1 and isinstance(obj[0], dict):
                        return obj[0]
                    if obj and all(isinstance(item, dict) for item in obj):
                        return obj[0]
                return obj
            except json.JSONDecodeError:
                continue
        raise


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    name: str

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0,
    ) -> Dict[str, Any]:
        """Generate a JSON response conforming to the given schema.

        Args:
            prompt: The input prompt.
            schema: JSON schema the output should conform to.
            temperature: Sampling temperature.

        Returns:
            Parsed JSON dict.
        """

    def generate_json_with_repair(
        self,
        prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0,
    ) -> Dict[str, Any]:
        """Generate JSON with automatic repair on malformed output.

        Tries once normally, then re-prompts with a repair instruction.
        """
        try:
            return self.generate_json(prompt, schema, temperature=temperature)
        except Exception:
            repair_prompt = (
                "Fix the following response to strictly match the JSON schema. "
                "Return JSON only.\n\n"
                f"Original prompt:\n{prompt}\n"
            )
            return self.generate_json(repair_prompt, schema, temperature=temperature)

    @staticmethod
    def get_backend(
        name: str,
        model: str,
        cache: Optional[SQLiteCache] = None,
        **kwargs,
    ) -> "LLMBackend":
        """Factory method to get backend by provider name.

        Args:
            name: Provider name ("openai", "huggingface").
            model: Model identifier.
            cache: Optional SQLiteCache for caching responses.
            **kwargs: Provider-specific parameters.

        Returns:
            An LLMBackend instance.
        """
        backends = {
            "huggingface": HuggingFaceLLMBackend,
            "hf": HuggingFaceLLMBackend,
        }
        backend_cls = backends.get(name.lower())
        if not backend_cls:
            available = sorted(set(backends.keys()))
            raise ValueError(f"Unknown LLM backend: {name}. Available: {available}")
        return backend_cls(model=model, cache=cache, **kwargs)


class HuggingFaceLLMBackend(LLMBackend):
    """HuggingFace Transformers LLM backend for local inference.

    Supports models like Qwen/Qwen3-8B via the transformers library.
    JSON output is enforced via prompt engineering and parsed from the response.
    """

    name = "huggingface"

    def __init__(
        self,
        model: str,
        cache: Optional[SQLiteCache] = None,
        device: Optional[str] = None,
        max_new_tokens: int = 4096,
        max_retries: int = 3,
        **kwargs,
    ):
        self.model_name = model
        self.cache = cache
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self._pipeline = None

        if device is None:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

    def _load_pipeline(self):
        if self._pipeline is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if not torch.cuda.is_available():
            self._model = self._model.to(self._device)
        self._model.eval()
        self._pipeline = True  # Flag that model is loaded

    def _cache_key(self, prompt: str, schema: Dict[str, Any]) -> str:
        payload = {"model": self.model_name, "prompt": prompt, "schema": schema}
        return f"llm_hf:{hash_text(safe_json_dumps(payload))}"

    def _build_prompt_with_schema(self, prompt: str, schema: Dict[str, Any]) -> str:
        schema_text = safe_json_dumps(schema)
        return (
            f"{prompt}\n\n"
            "JSON schema:\n"
            f"{schema_text}\n\n"
            "Return JSON only. Do not include any explanation or markdown formatting."
        )

    def generate_json(
        self,
        prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0,
    ) -> Dict[str, Any]:
        key = self._cache_key(prompt, schema)
        if self.cache:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        self._load_pipeline()
        import torch

        full_prompt = self._build_prompt_with_schema(prompt, schema)

        messages = [
            {"role": "system", "content": "You are a helpful assistant that outputs valid JSON only."},
            {"role": "user", "content": full_prompt},
        ]

        text_input = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text_input, return_tensors="pt").to(self._model.device)

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9

        for attempt in range(self.max_retries):
            try:
                with torch.no_grad():
                    output_ids = self._model.generate(**inputs, **gen_kwargs)

                # Decode only the generated tokens (exclude the input)
                new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
                response_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

                result = _json_loads_strict(response_text)
                if isinstance(result, list):
                    if len(result) == 1 and isinstance(result[0], dict):
                        result = result[0]
                    elif result and all(isinstance(item, dict) for item in result):
                        result = result[0]
                    else:
                        raise ValueError("Response JSON is a list, expected object")

                if self.cache:
                    self.cache.set(key, result)
                return result
            except Exception:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(0.5)

        raise RuntimeError("HuggingFace LLM call failed after retries")
