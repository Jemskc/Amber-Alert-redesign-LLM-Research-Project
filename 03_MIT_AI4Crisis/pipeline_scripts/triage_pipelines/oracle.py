from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..llm_schemas import urgency_schema
from ..openai_client import OpenAIClient
from ..utils import compute_entropy, progress_iter
from .base import BasePipeline
from .llm_joint import DEFAULT_URGENCY_RUBRIC


class OracleCategoryPipeline(BasePipeline):
    name = "oracle_category"

    def __init__(
        self,
        client: OpenAIClient,
        mode: str,
        categories: List[str],
        model: Optional[str] = None,
        temperature: float = 0,
        urgency_rubric: str = DEFAULT_URGENCY_RUBRIC,
        embedding_model: Optional[str] = None,
        lgbm_model_path: Optional[str] = None,
        batch_size: int = 64,
        show_progress: bool = False,
    ):
        self.client = client
        self.mode = mode
        self.categories = categories
        self.model = model
        self.temperature = temperature
        self.urgency_rubric = urgency_rubric
        self.embedding_model = embedding_model
        self.lgbm_model_path = lgbm_model_path
        self.batch_size = batch_size
        self.ml_model = None
        self.show_progress = show_progress

    def fit(self, train_df, val_df=None) -> "OracleCategoryPipeline":
        if self.mode == "ml" and self.lgbm_model_path:
            from .ml_sequential import MLSequentialPipeline

            self.ml_model = MLSequentialPipeline.load(self.lgbm_model_path, client=self.client)
        return self

    def _prompt(self, text: str, category: str) -> str:
        return (
            "You are a crisis triage assistant. Return JSON only following the schema. "
            "Provide a brief, clear rationale (1-2 sentences) for urgency that explains the decision, before probability fields.\n\n"
            f"Urgency rubric:\n{self.urgency_rubric}\n\n"
            "Provide urgency_by_category with a 0–5 score for each top-level category (0 = not urgent/NA).\n"
            "Also provide urgency_vector_8 with dim_1..dim_8 matching the category order used elsewhere.\n\n"
            f"True category: {category}\n\n"
            f"Message: {text}"
        )

    def predict(self, df) -> Dict[str, Any]:
        results = []
        if self.mode == "ml":
            if self.ml_model is None:
                raise ValueError("ML model path required for oracle ML mode")
            texts = df["text"].tolist()
            emb = self.client.embeddings(self.embedding_model, texts, batch_size=self.batch_size)
            cat_onehot = np.zeros((len(df), len(self.categories)), dtype=float)
            cat_idx = [self.categories.index(c) if c in self.categories else 0 for c in df["category_gold"].astype(str)]
            cat_onehot[np.arange(len(df)), cat_idx] = 1.0
            feats = np.hstack([emb, cat_onehot])
            urg_probs = self.ml_model.urgency_model.predict_proba(feats)
            for idx in range(len(texts)):
                probs = urg_probs[idx]
                probs = probs / probs.sum()
                results.append(
                    {
                        "category_top": df.iloc[idx]["category_gold"],
                        "urgency_probs_overall": [float(p) for p in probs],
                        "urgency_expected_overall": float(np.dot(np.arange(6), probs)),
                        "urgency_probs_high": float(probs[4] + probs[5]),
                        "entropy_urgency": float(compute_entropy(probs)),
                    }
                )
            return {"records": results}

        schema = urgency_schema(self.categories, include_vector=True)
        rows = df.to_dict("records")
        for row in progress_iter(rows, total=len(rows), desc="Oracle LLM", enabled=self.show_progress):
            prompt = self._prompt(row["text"], row.get("category_gold", "unknown"))
            output = self.client.responses_json_with_repair(self.model, schema, prompt, temperature=self.temperature)
            if output.get("mock"):
                urg_probs = np.ones(6) / 6
                output = {
                    "urgency_rationale": "No model call; this is a mock response.",
                    "urgency_by_category": {cat: 0 for cat in self.categories},
                    "urgency_probs_overall": [float(p) for p in urg_probs],
                    "urgency_expected_overall": float(np.dot(np.arange(6), urg_probs)),
                    "urgency_probs_high": float(urg_probs[4] + urg_probs[5]),
                    "entropy_urgency": float(compute_entropy(urg_probs)),
                    "urgency_vector_8": {f"dim_{i+1}": 0 for i in range(8)},
                    "self_report_uncertainty": 0.5,
                }
            results.append(output)
        return {"records": results}

    def save(self, path: str | Path) -> None:
        payload = {
            "name": self.name,
            "mode": self.mode,
            "categories": self.categories,
            "model": self.model,
            "temperature": self.temperature,
            "embedding_model": self.embedding_model,
            "lgbm_model_path": self.lgbm_model_path,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    @classmethod
    def load(cls, path: str | Path, client: OpenAIClient) -> "OracleCategoryPipeline":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(
            client=client,
            mode=payload["mode"],
            categories=payload["categories"],
            model=payload.get("model"),
            temperature=payload.get("temperature", 0),
            embedding_model=payload.get("embedding_model"),
            lgbm_model_path=payload.get("lgbm_model_path"),
        )
