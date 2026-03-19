from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..category_definitions import format_category_definitions
from ..llm_schemas import category_schema, urgency_schema
from ..openai_client import OpenAIClient
from ..utils import compute_entropy, hash_text, progress_iter
from .base import BasePipeline
from .llm_joint import DEFAULT_URGENCY_RUBRIC


class LLMSequentialPipeline(BasePipeline):
    name = "llm_sequential"

    def __init__(
        self,
        client: OpenAIClient,
        model: str,
        categories: List[str],
        temperature: float = 0,
        include_vector: bool = True,
        urgency_rubric: str = DEFAULT_URGENCY_RUBRIC,
        show_progress: bool = False,
        checkpoint_path: Optional[str] = None,
        progress_log_path: Optional[str] = None,
        resume: bool = True,
    ):
        self.client = client
        self.model = model
        self.categories = categories
        self.temperature = temperature
        self.include_vector = include_vector
        self.urgency_rubric = urgency_rubric
        self.show_progress = show_progress
        self.checkpoint_path = checkpoint_path
        self.progress_log_path = progress_log_path
        self.resume = resume

    def _prompt_category(self, text: str) -> str:
        cat_list = ", ".join(self.categories)
        taxonomy = format_category_definitions()
        return (
            "You are a crisis triage assistant. Return JSON only following the schema. "
            "Provide a brief, clear rationale (1-2 sentences) for the category that explains the decision. "
            "Place the rationale before probability fields in the JSON.\n\n"
            f"Top-level categories: {cat_list}.\n"
            "Definitions:\n"
            f"{taxonomy}\n\n"
            "Instructions:\n"
            "- Messages may belong to multiple categories. Assign probabilities across all categories; sum to 1.\n"
            "- If uncertain, spread probability mass; do not invent details.\n\n"
            f"Message: {text}"
        )

    def _prompt_urgency(self, text: str, cat_probs: Dict[str, float]) -> str:
        top = sorted(cat_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        top_str = ", ".join([f"{k}:{v:.2f}" for k, v in top])
        taxonomy = format_category_definitions()
        return (
            "You are a crisis triage assistant. Return JSON only following the schema. "
            "Provide a brief, clear rationale (1-2 sentences) for the urgency that explains the decision. "
            "Place the rationale before probability fields in the JSON.\n\n"
            f"Top-level categories: {', '.join(self.categories)}.\n"
            "Definitions:\n"
            f"{taxonomy}\n\n"
            "Instructions:\n"
            "- Use the urgency rubric to produce a distribution over 0–5; sum to 1.\n"
            "- Provide urgency_by_category with a 0–5 score for each top-level category (0 = not urgent/NA).\n"
            "- Also provide urgency_vector_8 with dim_1..dim_8 matching the category order above.\n"
            "- Consider the top categories below, but prioritize the message content if they conflict.\n"
            "- If uncertain, spread probability mass; do not invent details.\n\n"
            f"Urgency rubric:\n{self.urgency_rubric}\n\n"
            f"Top categories (probabilities): {top_str}\n\n"
            f"Message: {text}"
        )

    def _mock_category(self, text: str) -> Dict[str, Any]:
        seed = int(hash_text(text)[:8], 16)
        rng = np.random.default_rng(seed)
        cat_probs = rng.random(len(self.categories))
        cat_probs = cat_probs / cat_probs.sum()
        return {
            "category_rationale": "No model call; this is a mock response.",
            "category_top": self.categories[int(np.argmax(cat_probs))],
            "category_probs": {c: float(p) for c, p in zip(self.categories, cat_probs)},
            "entropy_category": float(compute_entropy(cat_probs)),
            "self_report_uncertainty": float(rng.random()),
        }

    def _mock_urgency(self, text: str) -> Dict[str, Any]:
        seed = int(hash_text(text)[8:16], 16)
        rng = np.random.default_rng(seed)
        urg_probs = rng.random(6)
        urg_probs = urg_probs / urg_probs.sum()
        urg_by_cat = {cat: int(rng.integers(0, 6)) for cat in self.categories}
        return {
            "urgency_rationale": "No model call; this is a mock response.",
            "urgency_by_category": urg_by_cat,
            "urgency_probs_overall": [float(p) for p in urg_probs],
            "urgency_expected_overall": float(np.dot(np.arange(6), urg_probs)),
            "urgency_probs_high": float(urg_probs[4] + urg_probs[5]),
            "entropy_urgency": float(compute_entropy(urg_probs)),
            "urgency_vector_8": {f"dim_{i+1}": urg_by_cat.get(cat, 0) for i, cat in enumerate(self.categories[:8])},
            "self_report_uncertainty": float(rng.random()),
        }

    def predict(self, df) -> Dict[str, Any]:
        results = []
        checkpoint_file = None
        progress_file = None
        start_idx = 0
        if self.checkpoint_path:
            path = Path(self.checkpoint_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if self.resume and path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            results.append(json.loads(line))
                        except json.JSONDecodeError:
                            break
                start_idx = len(results)
            checkpoint_file = open(path, "a", encoding="utf-8")
        if self.progress_log_path:
            progress_path = Path(self.progress_log_path)
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_file = open(progress_path, "a", encoding="utf-8")

        cat_schema = category_schema(self.categories)
        urg_schema = urgency_schema(self.categories, include_vector=self.include_vector)
        texts = df["text"].tolist()
        msg_ids = df["msg_id"].tolist() if "msg_id" in df.columns else [str(i) for i in range(len(texts))]
        try:
            for idx, text in enumerate(progress_iter(texts, total=len(texts), desc="LLM sequential", enabled=self.show_progress)):
                if idx < start_idx:
                    continue
                cat_prompt = self._prompt_category(text)
                cat_output = self.client.responses_json_with_repair(self.model, cat_schema, cat_prompt, temperature=self.temperature)
                if cat_output.get("mock"):
                    cat_output = self._mock_category(text)

                cat_probs = cat_output.get("category_probs", {})
                if cat_probs:
                    total = sum(cat_probs.values()) or 1.0
                    cat_probs = {k: float(v) / total for k, v in cat_probs.items()}
                else:
                    cat_probs = {c: 1 / len(self.categories) for c in self.categories}

                urg_prompt = self._prompt_urgency(text, cat_probs)
                urg_output = self.client.responses_json_with_repair(self.model, urg_schema, urg_prompt, temperature=self.temperature)
                if urg_output.get("mock"):
                    urg_output = self._mock_urgency(text)

                urg_by_cat = urg_output.get("urgency_by_category", {})
                if not isinstance(urg_by_cat, dict):
                    urg_by_cat = {}
                urg_by_cat = {
                    cat: int(np.clip(urg_by_cat.get(cat, 0), 0, 5)) if isinstance(urg_by_cat.get(cat, 0), (int, float)) else 0
                    for cat in self.categories
                }
                if self.include_vector and not isinstance(urg_output.get("urgency_vector_8"), dict):
                    urg_output["urgency_vector_8"] = {f"dim_{i+1}": urg_by_cat.get(cat, 0) for i, cat in enumerate(self.categories[:8])}

                urg_probs = np.asarray(urg_output.get("urgency_probs_overall", [1 / 6] * 6), dtype=float)
                urg_probs = urg_probs / urg_probs.sum()

                record = {
                    "category_rationale": cat_output.get("category_rationale", ""),
                    "category_top": max(cat_probs, key=cat_probs.get),
                    "category_probs": cat_probs,
                    "category_confidence": float(max(cat_probs.values())),
                    "entropy_category": float(compute_entropy(cat_probs.values())),
                    "urgency_rationale": urg_output.get("urgency_rationale", ""),
                    "urgency_by_category": urg_by_cat,
                    "urgency_probs_overall": [float(p) for p in urg_probs],
                    "urgency_expected_overall": float(np.dot(np.arange(6), urg_probs)),
                    "urgency_probs_high": float(urg_probs[4] + urg_probs[5]),
                    "entropy_urgency": float(compute_entropy(urg_probs)),
                    "urgency_vector_8": urg_output.get("urgency_vector_8", {}),
                    "urgency_confidence": float(max(urg_probs)),
                    "self_report_uncertainty": float(urg_output.get("self_report_uncertainty", 0.5)),
                }
                results.append(record)
                if checkpoint_file is not None:
                    checkpoint_file.write(json.dumps(record, ensure_ascii=True) + "\n")
                    checkpoint_file.flush()
                if progress_file is not None:
                    payload = {"msg_id": msg_ids[idx], "text": text, "prediction": record}
                    progress_file.write(json.dumps(payload, ensure_ascii=True) + "\n")
                    progress_file.flush()
        finally:
            if checkpoint_file is not None:
                checkpoint_file.close()
            if progress_file is not None:
                progress_file.close()
        return {"records": results}

    def save(self, path):
        payload = {
            "name": self.name,
            "model": self.model,
            "categories": self.categories,
            "temperature": self.temperature,
            "include_vector": self.include_vector,
            "urgency_rubric": self.urgency_rubric,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    @classmethod
    def load(cls, path, client: OpenAIClient) -> "LLMSequentialPipeline":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(
            client=client,
            model=payload["model"],
            categories=payload["categories"],
            temperature=payload["temperature"],
            include_vector=payload.get("include_vector", True),
            urgency_rubric=payload.get("urgency_rubric", DEFAULT_URGENCY_RUBRIC),
        )
