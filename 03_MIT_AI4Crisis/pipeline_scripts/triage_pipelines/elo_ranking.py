from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

from ..llm_schemas import judge_schema
from ..openai_client import OpenAIClient
from ..utils import hash_text, progress_iter
from .base import BasePipeline


class EloRankingPipeline(BasePipeline):
    name = "elo_ranking"

    def __init__(
        self,
        client: OpenAIClient,
        judge_model: str,
        embedding_model: str,
        urgency_columns: Optional[List[str]] = None,
        anchor_per_level: int = 10,
        max_comparisons: int = 8,
        strategy: str = "binary_search",
        rating_system: str = "elo",
        batch_size: int = 64,
        temperature: float = 0,
        show_progress: bool = False,
    ):
        self.client = client
        self.judge_model = judge_model
        self.embedding_model = embedding_model
        self.urgency_columns = urgency_columns
        self.anchor_per_level = anchor_per_level
        self.max_comparisons = max_comparisons
        self.strategy = strategy
        self.rating_system = rating_system
        self.batch_size = batch_size
        self.temperature = temperature
        self.show_progress = show_progress

        self.anchors: List[Dict[str, Any]] = []
        self.calibrator: Optional[IsotonicRegression] = None

    def _overall_urgency(self, df) -> np.ndarray:
        cols = [c for c in (self.urgency_columns or []) if c in df.columns]
        if not cols:
            return np.zeros(len(df), dtype=int)
        values = df[cols].fillna(0).astype(int).clip(0, 5).values
        return values.max(axis=1).astype(int)

    def _farthest_first(self, emb: np.ndarray, n: int) -> List[int]:
        if emb.shape[0] <= n:
            return list(range(emb.shape[0]))
        selected = [0]
        while len(selected) < n:
            dists = np.min(
                [np.linalg.norm(emb - emb[i], axis=1) for i in selected],
                axis=0,
            )
            idx = int(np.argmax(dists))
            if idx in selected:
                idx = random.randint(0, emb.shape[0] - 1)
            selected.append(idx)
        return selected

    def fit(self, train_df, val_df=None) -> "EloRankingPipeline":
        texts = train_df["text"].tolist()
        emb = self.client.embeddings(self.embedding_model, texts, batch_size=self.batch_size)
        urg = self._overall_urgency(train_df)

        anchors = []
        for level in range(6):
            idx = np.where(urg == level)[0]
            if len(idx) == 0:
                continue
            sub_emb = emb[idx]
            chosen = self._farthest_first(sub_emb, self.anchor_per_level)
            for i in chosen:
                anchor_idx = idx[i]
                anchors.append(
                    {
                        "msg_id": train_df.iloc[anchor_idx]["msg_id"],
                        "text": texts[anchor_idx],
                        "urgency": int(urg[anchor_idx]),
                    }
                )
        self.anchors = anchors

        if val_df is not None:
            preds = self.predict(val_df)
            y = self._overall_urgency(val_df)
            ratings = np.array(preds["ratings"], dtype=float)
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(ratings, y)

        return self

    def _judge(self, text_a: str, text_b: str) -> Dict[str, Any]:
        schema = judge_schema()
        prompt = (
            "You are an expert triage judge. Compare two messages and choose which is more urgent overall. "
            "Return JSON only.\n\n"
            f"A: {text_a}\n\n"
            f"B: {text_b}"
        )
        output = self.client.responses_json_with_repair(self.judge_model, schema, prompt, temperature=self.temperature)
        if output.get("mock"):
            seed = int(hash_text(text_a + text_b)[:8], 16)
            rng = random.Random(seed)
            winner = rng.choice(["A", "B", "TIE"])
            output = {"winner": winner, "confidence": 0.5}
        return output

    def _elo_update(self, rating: float, anchor_rating: float, outcome: float, k: float = 32.0) -> float:
        expected = 1.0 / (1.0 + 10 ** ((anchor_rating - rating) / 400.0))
        return rating + k * (outcome - expected)

    def _binary_search_rating(self, text: str) -> float:
        low, high = 0, 5
        rating = 1200.0
        for _ in range(self.max_comparisons):
            mid = (low + high) // 2
            anchors = [a for a in self.anchors if a["urgency"] == mid]
            if not anchors:
                break
            anchor = random.choice(anchors)
            result = self._judge(text, anchor["text"])
            if result["winner"] == "A":
                low = mid
                rating = self._elo_update(rating, 1200.0 + mid * 100, 1.0)
            elif result["winner"] == "B":
                high = mid
                rating = self._elo_update(rating, 1200.0 + mid * 100, 0.0)
            else:
                low = mid
                high = mid
                rating = self._elo_update(rating, 1200.0 + mid * 100, 0.5)
                break
            if high - low <= 1:
                break
        coarse = float((low + high) / 2.0)
        if self.rating_system == "elo":
            return float((rating - 1200.0) / 100.0)
        return coarse

    def _trueskill_rating(self, text: str) -> float:
        try:
            import trueskill
        except Exception:
            return self._binary_search_rating(text)
        rating = trueskill.Rating(mu=25.0, sigma=8.333)
        for _ in range(self.max_comparisons):
            anchor = random.choice(self.anchors)
            anchor_rating = trueskill.Rating(mu=anchor["urgency"] * 5.0 + 25.0, sigma=8.333)
            result = self._judge(text, anchor["text"])
            if result["winner"] == "A":
                rating, _ = trueskill.rate_1vs1(rating, anchor_rating)
            elif result["winner"] == "B":
                _, rating = trueskill.rate_1vs1(anchor_rating, rating)
            else:
                rating, _ = trueskill.rate_1vs1(rating, anchor_rating, drawn=True)
        return float(rating.mu / 5.0)

    def predict(self, df) -> Dict[str, Any]:
        ratings = []
        texts = df["text"].tolist()
        for text in progress_iter(texts, total=len(texts), desc="Elo ranking", enabled=self.show_progress):
            if self.rating_system == "trueskill":
                rating = self._trueskill_rating(text)
            else:
                rating = self._binary_search_rating(text)
            ratings.append(rating)
        if self.calibrator is not None:
            calibrated = self.calibrator.predict(ratings)
        else:
            calibrated = ratings
        urgency_expected = [float(r) for r in calibrated]
        urgency_probs_high = [float(1.0 if r >= 4 else 0.0) for r in calibrated]

        return {
            "ratings": ratings,
            "urgency_expected_overall": urgency_expected,
            "urgency_probs_high": urgency_probs_high,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": self.name,
            "judge_model": self.judge_model,
            "embedding_model": self.embedding_model,
            "urgency_columns": self.urgency_columns,
            "anchor_per_level": self.anchor_per_level,
            "max_comparisons": self.max_comparisons,
            "strategy": self.strategy,
            "rating_system": self.rating_system,
        }
        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=True, indent=2)
        with open(path / "anchors.json", "w", encoding="utf-8") as f:
            json.dump(self.anchors, f, ensure_ascii=True, indent=2)
        if self.calibrator is not None:
            joblib.dump(self.calibrator, path / "calibrator.joblib")

    @classmethod
    def load(cls, path: str | Path, client: OpenAIClient) -> "EloRankingPipeline":
        path = Path(path)
        with open(path / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        obj = cls(
            client=client,
            judge_model=meta["judge_model"],
            embedding_model=meta["embedding_model"],
            urgency_columns=meta.get("urgency_columns"),
            anchor_per_level=meta.get("anchor_per_level", 10),
            max_comparisons=meta.get("max_comparisons", 8),
            strategy=meta.get("strategy", "binary_search"),
            rating_system=meta.get("rating_system", "elo"),
        )
        if (path / "anchors.json").exists():
            with open(path / "anchors.json", "r", encoding="utf-8") as f:
                obj.anchors = json.load(f)
        if (path / "calibrator.joblib").exists():
            import joblib

            obj.calibrator = joblib.load(path / "calibrator.joblib")
        return obj
