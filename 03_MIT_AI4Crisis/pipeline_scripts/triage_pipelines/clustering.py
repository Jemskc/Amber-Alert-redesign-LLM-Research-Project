from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..openai_client import OpenAIClient
from ..utils import ensure_dir
from .base import BasePipeline


class ClusteringPipeline(BasePipeline):
    name = "clustering"

    def __init__(
        self,
        client: OpenAIClient,
        embedding_model: str,
        umap_dim: int = 5,
        min_cluster_size: int = 15,
        batch_size: int = 64,
    ):
        self.client = client
        self.embedding_model = embedding_model
        self.umap_dim = umap_dim
        self.min_cluster_size = min_cluster_size
        self.batch_size = batch_size

        self.umap_model = None
        self.cluster_model = None

    def fit(self, df, val_df=None) -> "ClusteringPipeline":
        return self

    def predict(self, df) -> Dict[str, Any]:
        try:
            import umap
            import hdbscan
        except Exception:
            return {"skipped": True, "reason": "umap-learn or hdbscan missing"}

        texts = df["text"].tolist()
        emb = self.client.embeddings(self.embedding_model, texts, batch_size=self.batch_size)

        self.umap_model = umap.UMAP(n_components=self.umap_dim, random_state=13)
        reduced = self.umap_model.fit_transform(emb)

        self.cluster_model = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size)
        labels = self.cluster_model.fit_predict(reduced)

        summaries = self._summarize_clusters(texts, labels)

        return {
            "cluster_ids": labels.tolist(),
            "reduced": reduced.tolist(),
            "summaries": summaries,
        }

    def _summarize_clusters(self, texts: List[str], labels: np.ndarray) -> List[Dict[str, Any]]:
        summaries = []
        vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        tfidf = vectorizer.fit_transform(texts)
        vocab = np.array(vectorizer.get_feature_names_out())

        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                continue
            idx = np.where(labels == cluster_id)[0]
            if len(idx) == 0:
                continue
            cluster_tfidf = tfidf[idx].mean(axis=0)
            top_idx = np.argsort(np.asarray(cluster_tfidf).ravel())[::-1][:10]
            top_terms = vocab[top_idx].tolist()
            examples = [texts[i] for i in idx[:3]]
            summaries.append(
                {
                    "cluster_id": int(cluster_id),
                    "size": int(len(idx)),
                    "top_terms": ", ".join(top_terms),
                    "examples": examples,
                }
            )
        return summaries

    def save(self, path: str | Path) -> None:
        path = Path(path)
        ensure_dir(path)
        meta = {
            "name": self.name,
            "embedding_model": self.embedding_model,
            "umap_dim": self.umap_dim,
            "min_cluster_size": self.min_cluster_size,
        }
        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=True, indent=2)

    @classmethod
    def load(cls, path: str | Path, client: OpenAIClient) -> "ClusteringPipeline":
        path = Path(path)
        with open(path / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        return cls(
            client=client,
            embedding_model=meta["embedding_model"],
            umap_dim=meta.get("umap_dim", 5),
            min_cluster_size=meta.get("min_cluster_size", 15),
        )
