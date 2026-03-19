from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..category_definitions import format_category_definitions
from ..llm_schemas import joint_schema
from ..openai_client import OpenAIClient
from ..utils import compute_entropy, hash_text, progress_iter
from .base import BasePipeline


DEFAULT_URGENCY_RUBRIC = """
Urgency scale (0-5):
0 = Not applicable / coordination / request for info (no stated need)
1 = Low urgency (services available, minor needs, general info)
2 = Moderate urgency (needs attention; supplies requested without immediate harm)
3 = High urgency (clear unmet basic needs like food/water/medicine/shelter)
4 = Very high urgency (severe shortage, widespread risk, vulnerable groups at risk)
5 = Extreme urgency (imminent threat to life: deaths, people dying, trapped, fire, active violence)
""".strip()


PROMPT_TEMPLATE_NAME = "llm_joint_v3.txt"
_PROMPT_TEMPLATE_CACHE: Dict[str, str] = {}


class LLMJointPipeline(BasePipeline):
    name = "llm_joint"

    def __init__(
        self,
        client: OpenAIClient,
        model: str,
        categories: List[str],
        temperature: float = 0,
        include_vector: bool = True,
        urgency_rubric: str = DEFAULT_URGENCY_RUBRIC,
        prompt_template_name: str = PROMPT_TEMPLATE_NAME,
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
        self.prompt_template_name = prompt_template_name
        self.show_progress = show_progress
        self.checkpoint_path = checkpoint_path
        self.progress_log_path = progress_log_path
        self.resume = resume

    def _prompt(self, text: str) -> str:
        cat_list = ", ".join(self.categories)
        taxonomy = format_category_definitions()
        template = self._load_prompt_template(self.prompt_template_name)
        return template.format(
            cat_list=cat_list,
            taxonomy=taxonomy,
            urgency_rubric=self.urgency_rubric,
            message=text,
        )

    @staticmethod
    def _load_prompt_template(name: str) -> str:
        cached = _PROMPT_TEMPLATE_CACHE.get(name)
        if cached:
            return cached
        base_dir = Path(__file__).resolve().parents[1] / "prompts"
        prompt_path = base_dir / name
        if not prompt_path.exists():
            prompt_path = Path(__file__).resolve().parent / "prompts" / name
        template = prompt_path.read_text(encoding="utf-8")
        _PROMPT_TEMPLATE_CACHE[name] = template
        return template

    def _mock_output(self, text: str) -> Dict[str, Any]:
        seed = int(hash_text(text)[:8], 16)
        rng = np.random.default_rng(seed)
        # Independent probabilities [0, 1] for each category (not a distribution)
        cat_probs = rng.random(len(self.categories))
        urg_probs = rng.random(6)
        urg_probs = urg_probs / urg_probs.sum()
        urg_expected = float(np.dot(np.arange(6), urg_probs))
        urg_by_cat = {cat: int(rng.integers(0, 6)) for cat in self.categories}
        # Generate per-category urgency distributions
        urg_probs_by_cat = {}
        for cat in self.categories:
            cat_urg_probs = rng.random(6)
            cat_urg_probs = cat_urg_probs / cat_urg_probs.sum()
            urg_probs_by_cat[cat] = [float(p) for p in cat_urg_probs]
        # Compute entropy on normalized distribution for uncertainty measure
        cat_probs_norm = cat_probs / (cat_probs.sum() or 1.0)
        return {
            "category_rationale": "No model call; this is a mock response.",
            "category_top": self.categories[int(np.argmax(cat_probs))],
            "category_probs": {c: float(p) for c, p in zip(self.categories, cat_probs)},
            "entropy_category": float(compute_entropy(cat_probs_norm)),
            "urgency_rationale": "No model call; this is a mock response.",
            "urgency_by_category": urg_by_cat,
            "urgency_probs_by_category": urg_probs_by_cat,
            "urgency_probs_overall": [float(p) for p in urg_probs],
            "urgency_expected_overall": urg_expected,
            "urgency_probs_high": float(urg_probs[4] + urg_probs[5]),
            "entropy_urgency": float(compute_entropy(urg_probs)),
            "urgency_vector_8": {f"dim_{i+1}": urg_by_cat.get(cat, 0) for i, cat in enumerate(self.categories[:8])},
            "self_report_uncertainty": float(rng.random()),
        }

    def _normalize(self, output: Dict[str, Any]) -> Dict[str, Any]:
        cat_probs = output.get("category_probs", {})
        if not isinstance(cat_probs, dict) or not cat_probs:
            cat_probs = {c: 0.5 for c in self.categories}
        # Clip each category probability to [0, 1] (independent probabilities, not a distribution)
        output["category_probs"] = {k: float(np.clip(v, 0.0, 1.0)) for k, v in cat_probs.items()}
        output["category_top"] = max(output["category_probs"], key=output["category_probs"].get)
        # Entropy computed on normalized distribution for uncertainty measure
        probs_for_entropy = np.array(list(output["category_probs"].values()))
        probs_for_entropy = probs_for_entropy / (probs_for_entropy.sum() or 1.0)
        output["entropy_category"] = float(compute_entropy(probs_for_entropy))

        urg_by_cat = output.get("urgency_by_category", {})
        if not isinstance(urg_by_cat, dict):
            urg_by_cat = {}
        urg_by_cat = {
            cat: int(np.clip(urg_by_cat.get(cat, 0), 0, 5)) if isinstance(urg_by_cat.get(cat, 0), (int, float)) else 0
            for cat in self.categories
        }
        output["urgency_by_category"] = urg_by_cat

        # Normalize per-category urgency distributions
        urg_probs_by_cat = output.get("urgency_probs_by_category", {})
        if not isinstance(urg_probs_by_cat, dict):
            urg_probs_by_cat = {}
        normalized_urg_probs_by_cat = {}
        for cat in self.categories:
            cat_probs = urg_probs_by_cat.get(cat, [1 / 6] * 6)
            if not isinstance(cat_probs, (list, tuple)) or len(cat_probs) != 6:
                cat_probs = [1 / 6] * 6
            cat_probs = np.asarray(cat_probs, dtype=float)
            cat_probs = np.clip(cat_probs, 0, 1)
            total = cat_probs.sum()
            if total > 0:
                cat_probs = cat_probs / total
            else:
                cat_probs = np.full(6, 1 / 6, dtype=float)
            normalized_urg_probs_by_cat[cat] = [float(p) for p in cat_probs]
        output["urgency_probs_by_category"] = normalized_urg_probs_by_cat

        if self.include_vector:
            vec = output.get("urgency_vector_8", {})
            if not isinstance(vec, dict) or not vec:
                output["urgency_vector_8"] = {f"dim_{i+1}": urg_by_cat.get(cat, 0) for i, cat in enumerate(self.categories[:8])}

        urg_probs = output.get("urgency_probs_overall", [1 / 6] * 6)
        urg_probs = np.asarray(urg_probs, dtype=float)
        if urg_probs.size == 0:
            urg_probs = np.full(6, 1 / 6, dtype=float)
        urg_probs = urg_probs / urg_probs.sum()
        output["urgency_probs_overall"] = [float(p) for p in urg_probs]
        output["urgency_expected_overall"] = float(np.dot(np.arange(6), urg_probs))
        output["urgency_probs_high"] = float(urg_probs[4] + urg_probs[5])
        output["entropy_urgency"] = float(compute_entropy(urg_probs))
        return output

    def _fallback_output(self, reason: str) -> Dict[str, Any]:
        cat_probs = {c: 0.5 for c in self.categories}  # Uncertain: 0.5 for each independent category
        urg_probs = np.full(6, 1 / 6, dtype=float)
        urg_by_cat = {cat: 0 for cat in self.categories}
        # Uniform per-category urgency distributions (uncertain)
        urg_probs_by_cat = {cat: [1 / 6] * 6 for cat in self.categories}
        # Compute entropy on normalized distribution for uncertainty measure
        probs_for_entropy = np.array(list(cat_probs.values()))
        probs_for_entropy = probs_for_entropy / (probs_for_entropy.sum() or 1.0)
        output = {
            "category_rationale": f"NA: {reason}",
            "category_top": max(cat_probs, key=cat_probs.get),
            "category_probs": {k: float(v) for k, v in cat_probs.items()},
            "entropy_category": float(compute_entropy(probs_for_entropy)),
            "urgency_rationale": f"NA: {reason}",
            "urgency_by_category": urg_by_cat,
            "urgency_probs_by_category": urg_probs_by_cat,
            "urgency_probs_overall": [float(p) for p in urg_probs],
            "urgency_expected_overall": float(np.dot(np.arange(6), urg_probs)),
            "urgency_probs_high": float(urg_probs[4] + urg_probs[5]),
            "entropy_urgency": float(compute_entropy(urg_probs)),
            "self_report_uncertainty": 1.0,
        }
        if self.include_vector:
            output["urgency_vector_8"] = {f"dim_{i+1}": urg_by_cat.get(cat, 0) for i, cat in enumerate(self.categories[:8])}
        return output

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

        schema = joint_schema(self.categories, include_vector=self.include_vector)
        texts = df["text"].tolist()
        msg_ids = df["msg_id"].tolist() if "msg_id" in df.columns else [str(i) for i in range(len(texts))]
        try:
            for idx, text in enumerate(progress_iter(texts, total=len(texts), desc="LLM joint", enabled=self.show_progress)):
                if idx < start_idx:
                    continue
                prompt = self._prompt(text)
                max_attempts = 5
                for attempt in range(1, max_attempts + 1):
                    try:
                        output = self.client.responses_json_with_repair(
                            self.model, schema, prompt, temperature=self.temperature
                        )
                        if output.get("mock"):
                            output = self._mock_output(text)
                        output = self._normalize(output)
                        results.append(output)
                        if checkpoint_file is not None:
                            checkpoint_file.write(json.dumps(output, ensure_ascii=True) + "\n")
                            checkpoint_file.flush()
                        if progress_file is not None:
                            payload = {"msg_id": msg_ids[idx], "text": text, "prediction": output}
                            progress_file.write(json.dumps(payload, ensure_ascii=True) + "\n")
                            progress_file.flush()
                        break
                    except Exception as exc:
                        print(
                            f"[LLM joint] error row={idx} attempt={attempt}/{max_attempts}: {exc}",
                            flush=True,
                        )
                        if attempt >= max_attempts:
                            fallback = self._fallback_output(str(exc))
                            results.append(fallback)
                            if checkpoint_file is not None:
                                checkpoint_file.write(json.dumps(fallback, ensure_ascii=True) + "\n")
                                checkpoint_file.flush()
                            if progress_file is not None:
                                payload = {"msg_id": msg_ids[idx], "text": text, "prediction": fallback}
                                progress_file.write(json.dumps(payload, ensure_ascii=True) + "\n")
                                progress_file.flush()
                            break
                        time.sleep(1.0)
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
            "prompt_template_name": self.prompt_template_name,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    @classmethod
    def load(cls, path, client: OpenAIClient) -> "LLMJointPipeline":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(
            client=client,
            model=payload["model"],
            categories=payload["categories"],
            temperature=payload["temperature"],
            include_vector=payload.get("include_vector", True),
            urgency_rubric=payload.get("urgency_rubric", DEFAULT_URGENCY_RUBRIC),
            prompt_template_name=payload.get("prompt_template_name", PROMPT_TEMPLATE_NAME),
        )
