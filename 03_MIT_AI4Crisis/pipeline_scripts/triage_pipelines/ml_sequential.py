from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from ..data import CATEGORY_ID_MAP
from ..model_backends import ModelBackend
from ..openai_client import OpenAIClient
from ..utils import compute_entropy
from .base import BasePipeline


class MLSequentialPipeline(BasePipeline):
    name = "ml_sequential"

    def __init__(
        self,
        client: OpenAIClient,
        embedding_model: str,
        categories: List[str],
        urgency_columns: Optional[List[str]] = None,
        option: str = "feature_concat",
        lgbm_params: Optional[Dict[str, Any]] = None,
        lgbm_log_period: int = 0,
        batch_size: int = 64,
        overall_strategy: str = "max",
        category_multilabel: Optional[bool] = None,
        category_threshold: float = 0.5,
        # Multi-backend support
        model_backend: str = "lightgbm",
        model_params: Optional[Dict[str, Any]] = None,
    ):
        self.client = client
        self.embedding_model = embedding_model
        self.categories = categories
        self.urgency_columns = urgency_columns
        self.option = option
        self.lgbm_params = lgbm_params or {}
        self.batch_size = batch_size
        self.overall_strategy = overall_strategy
        self.category_multilabel = category_multilabel
        self.category_threshold = category_threshold
        self.lgbm_log_period = lgbm_log_period

        # Initialize model backend (merge legacy lgbm_params with model_params)
        self.model_backend_name = model_backend
        _params = {**(lgbm_params or {}), **(model_params or {})}
        self.backend = ModelBackend.get_backend(model_backend, _params)

        self.category_model: Optional[Any] = None
        self.category_multi_models: Dict[str, Any] = {}
        self.urgency_model: Optional[Any] = None
        self.urgency_per_category: Dict[str, Any] = {}
        self.urgency_dim_models: Dict[str, Any] = {}

    @staticmethod
    def _parse_category_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(v) for v in value if str(v).strip()]
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed if str(v).strip()]
            except Exception:
                return [raw]
            return []
        if value is None:
            return []
        return [str(value)]

    def _compute_overall(self, df) -> Optional[np.ndarray]:
        cols = self.urgency_columns or []
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return None
        values = df[cols].fillna(0).astype(int).clip(0, 5).values
        if self.overall_strategy == "mean":
            return np.round(values.mean(axis=1)).astype(int)
        return values.max(axis=1).astype(int)

    def _augment_features(self, emb: np.ndarray, cat_probs: np.ndarray) -> np.ndarray:
        return np.hstack([emb, cat_probs])

    def _expand_cat_probs(self, probs: np.ndarray) -> np.ndarray:
        if self.category_model is None:
            return probs
        full = np.zeros((probs.shape[0], len(self.categories)), dtype=float)
        for idx, label in enumerate(self.category_model.classes_):
            if label in self.categories:
                full[:, self.categories.index(label)] = probs[:, idx]
        return full

    def _lgbm_callbacks(self, label: str):
        period = int(self.lgbm_log_period or 0)
        if period <= 0:
            return None
        logger = logging.getLogger("triage")

        def _callback(env):
            iteration = env.iteration + 1
            total = env.end_iteration
            if iteration == 1:
                logger.info("LightGBM %s: %d iterations", label, total)
            if iteration % period != 0 and iteration != total:
                return
            metrics = []
            for item in env.evaluation_result_list or []:
                if len(item) >= 3:
                    data_name, metric_name, value = item[:3]
                    metrics.append(f"{data_name}:{metric_name}={value:.5f}")
            if metrics:
                logger.info("LightGBM %s iter %d/%d: %s", label, iteration, total, ", ".join(metrics))
            else:
                logger.info("LightGBM %s iter %d/%d", label, iteration, total)

        _callback.order = 10
        return [_callback]

    def _fit_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        label: str,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        logger = logging.getLogger("triage")
        logger.info("Training %s %s on %d rows.", self.backend.name, label, X.shape[0])
        X = np.asarray(X)

        # Only use callbacks and eval_set if backend supports them
        callbacks = None
        eval_set = None
        if self.backend.supports_callbacks:
            callbacks = self._lgbm_callbacks(label)
        if self.backend.supports_eval_set and callbacks:
            eval_set = [(X, y)]

        self.backend.fit(
            model, X, y,
            sample_weight=sample_weight,
            eval_set=eval_set,
            callbacks=callbacks,
        )

        # Clean up feature names to avoid serialization issues
        if hasattr(model, "feature_names_in_"):
            try:
                delattr(model, "feature_names_in_")
            except Exception:
                model.feature_names_in_ = None

    def fit(self, train_df, val_df=None) -> "MLSequentialPipeline":
        train_texts = train_df["text"].tolist()
        train_emb = self.client.embeddings(self.embedding_model, train_texts, batch_size=self.batch_size)

        use_multilabel = self.category_multilabel
        if use_multilabel is None and "category_list" in train_df.columns:
            lists = [self._parse_category_list(v) for v in train_df["category_list"].tolist()]
            use_multilabel = any(len(items) > 0 for items in lists)

        if use_multilabel and "category_list" in train_df.columns:
            cat_lists = [self._parse_category_list(v) for v in train_df["category_list"].tolist()]
            self.category_multi_models = {}
            for cat in self.categories:
                y = np.array([1 if cat in items else 0 for items in cat_lists], dtype=int)
                if y.sum() == 0 or y.sum() == len(y):
                    continue
                model = self.backend.create_classifier("binary")
                self._fit_model(model, train_emb, y, f"category:{cat}")
                self.category_multi_models[cat] = model
            self.category_multilabel = bool(self.category_multi_models)
        else:
            self.category_multilabel = False
            if "category_gold" in train_df.columns and train_df["category_gold"].notna().any():
                y_cat = train_df["category_gold"].astype(str)
                self.category_model = self.backend.create_classifier(
                    "multiclass", num_class=len(self.categories)
                )
                self._fit_model(self.category_model, train_emb, y_cat, "category")
            else:
                return self

        y_overall = self._compute_overall(train_df)
        if y_overall is None:
            return self

        if self.option == "per_category":
            for cat in self.categories:
                if "category_list" in train_df.columns and self.category_multilabel:
                    cat_lists = [self._parse_category_list(v) for v in train_df["category_list"].tolist()]
                    idx = np.array([cat in items for items in cat_lists], dtype=bool)
                else:
                    idx = train_df["category_gold"].astype(str) == cat
                if idx.sum() < 20:
                    continue
                model = self.backend.create_classifier("multiclass", num_class=6)
                self._fit_model(model, train_emb[idx], y_overall[idx], f"urgency:{cat}")
                self.urgency_per_category[cat] = model
        else:
            if "category_list" in train_df.columns and self.category_multilabel:
                cat_lists = [self._parse_category_list(v) for v in train_df["category_list"].tolist()]
                cat_onehot = np.zeros((len(train_df), len(self.categories)), dtype=float)
                for row_idx, items in enumerate(cat_lists):
                    for item in items:
                        if item in self.categories:
                            cat_onehot[row_idx, self.categories.index(item)] = 1.0
            else:
                cat_onehot = np.zeros((len(train_df), len(self.categories)), dtype=float)
                cat_idx = [self.categories.index(c) if c in self.categories else 0 for c in train_df["category_gold"].astype(str)]
                cat_onehot[np.arange(len(train_df)), cat_idx] = 1.0
            feats = self._augment_features(train_emb, cat_onehot)
            self.urgency_model = self.backend.create_classifier("multiclass", num_class=6)
            self._fit_model(self.urgency_model, feats, y_overall, "urgency_overall")

        # --- Per-dimension urgency models ---
        logger = logging.getLogger("triage")
        if self.urgency_columns:
            # Build category one-hot from gold labels for augmenting per-dim features
            if "category_list" in train_df.columns and self.category_multilabel:
                cat_lists = [self._parse_category_list(v) for v in train_df["category_list"].tolist()]
                cat_onehot_dim = np.zeros((len(train_df), len(self.categories)), dtype=float)
                for row_idx, items in enumerate(cat_lists):
                    for item in items:
                        if item in self.categories:
                            cat_onehot_dim[row_idx, self.categories.index(item)] = 1.0
            else:
                cat_onehot_dim = np.zeros((len(train_df), len(self.categories)), dtype=float)
                cat_idx = [self.categories.index(c) if c in self.categories else 0 for c in train_df["category_gold"].astype(str)]
                cat_onehot_dim[np.arange(len(train_df)), cat_idx] = 1.0
            dim_feats = self._augment_features(train_emb, cat_onehot_dim)

            min_filtered = 10
            for col in self.urgency_columns:
                if col not in train_df.columns:
                    continue
                y_dim = train_df[col].fillna(0).astype(int).clip(0, 5).values
                pos_mask = y_dim > 0
                n_pos = int(pos_mask.sum())
                if n_pos >= min_filtered:
                    filtered_feats = dim_feats[pos_mask]
                    filtered_y = y_dim[pos_mask]
                    n_classes = len(np.unique(filtered_y))
                    logger.info("urgency_dim:%s filtered to %d/%d positive samples (%d classes).",
                                col, n_pos, len(y_dim), n_classes)
                    model = self.backend.create_classifier("multiclass", num_class=n_classes)
                    self._fit_model(model, filtered_feats, filtered_y, f"urgency_dim:{col}")
                else:
                    logger.warning("urgency_dim:%s only %d positive samples (< %d), training on all %d.",
                                   col, n_pos, min_filtered, len(y_dim))
                    model = self.backend.create_classifier("multiclass", num_class=6)
                    self._fit_model(model, dim_feats, y_dim, f"urgency_dim:{col}")
                self.urgency_dim_models[col] = model
            logger.info("Trained %d per-dimension urgency models.", len(self.urgency_dim_models))

        return self

    def predict(self, df) -> Dict[str, Any]:
        texts = df["text"].tolist()
        emb = self.client.embeddings(self.embedding_model, texts, batch_size=self.batch_size)
        results = []

        # Build mapping from category name to urgency dimension index.
        cat_to_dim_idx: Dict[str, int] = {}
        for di, uc in enumerate(self.urgency_columns or []):
            m = re.search(r"\d+", uc)
            if m:
                cat_name = CATEGORY_ID_MAP.get(m.group())
                if cat_name:
                    cat_to_dim_idx[cat_name] = di

        cat_probs = None
        multi_label = bool(self.category_multi_models)
        if multi_label:
            probs = np.zeros((len(texts), len(self.categories)), dtype=float)
            for idx, cat in enumerate(self.categories):
                model = self.category_multi_models.get(cat)
                if model is None:
                    continue
                probs[:, idx] = model.predict_proba(emb)[:, 1]
            cat_probs = probs
        elif self.category_model is not None:
            cat_probs = self.category_model.predict_proba(emb)
            cat_probs = self._expand_cat_probs(cat_probs)

        for idx in range(len(texts)):
            record: Dict[str, Any] = {}
            if cat_probs is not None:
                probs = cat_probs[idx]
                total = probs.sum()
                norm = probs / total if total > 0 else np.full_like(probs, 1 / len(probs)) if len(probs) else probs
                if multi_label:
                    record["category_probs"] = {c: float(p) for c, p in zip(self.categories, probs)}
                    top_idx = int(np.argmax(probs)) if len(probs) else 0
                    record["category_top"] = self.categories[top_idx] if self.categories else None
                    record["entropy_category"] = float(compute_entropy(norm))
                    record["category_confidence"] = float(probs[top_idx]) if len(probs) else 0.0
                    pred_list = [c for c, p in zip(self.categories, probs) if p >= self.category_threshold]
                    if not pred_list and self.categories:
                        pred_list = [self.categories[top_idx]]
                    record["category_pred_list"] = pred_list
                else:
                    record["category_probs"] = {c: float(p) for c, p in zip(self.category_model.classes_, norm)}
                    record["category_top"] = self.category_model.classes_[int(np.argmax(norm))]
                    record["entropy_category"] = float(compute_entropy(norm))
                    record["category_confidence"] = float(norm.max())

            urg_probs = np.full(6, 1 / 6, dtype=float)
            if self.option == "per_category" and record.get("category_top") in self.urgency_per_category:
                model = self.urgency_per_category[record["category_top"]]
                urg_probs = model.predict_proba(emb[idx : idx + 1])[0]
            elif self.urgency_model is not None and cat_probs is not None:
                if multi_label:
                    cat_feats = (cat_probs[idx : idx + 1] >= self.category_threshold).astype(float)
                    if cat_feats.sum() == 0 and len(self.categories):
                        top_idx = int(np.argmax(cat_probs[idx]))
                        cat_feats[0, top_idx] = 1.0
                else:
                    cat_feats = cat_probs[idx : idx + 1]
                feats = self._augment_features(emb[idx : idx + 1], cat_feats)
                urg_probs = self.urgency_model.predict_proba(feats)[0]

            urg_probs = np.asarray(urg_probs, dtype=float)
            urg_probs = urg_probs / urg_probs.sum()
            record["urgency_probs_overall"] = [float(p) for p in urg_probs]
            record["urgency_expected_overall"] = float(np.dot(np.arange(6), urg_probs))
            record["urgency_probs_high"] = float(urg_probs[4] + urg_probs[5])
            record["entropy_urgency"] = float(compute_entropy(urg_probs))

            # --- Per-dimension urgency predictions ---
            if self.urgency_dim_models:
                record["urgency_vector_8"] = {}
                record["urgency_probs_by_category"] = {}
                record["urgency_by_category"] = {}
                # Build set of urgency dimension indices for predicted categories
                pred_dim_indices: set = set()
                if record.get("category_pred_list"):
                    for cat_name in record["category_pred_list"]:
                        dim_for_cat = cat_to_dim_idx.get(cat_name)
                        if dim_for_cat is not None:
                            pred_dim_indices.add(dim_for_cat)
                # Build augmented features for this sample
                if multi_label:
                    sample_cat_feats = (cat_probs[idx : idx + 1] >= self.category_threshold).astype(float)
                    if sample_cat_feats.sum() == 0 and len(self.categories):
                        top_idx = int(np.argmax(cat_probs[idx]))
                        sample_cat_feats[0, top_idx] = 1.0
                elif cat_probs is not None:
                    sample_cat_feats = cat_probs[idx : idx + 1]
                else:
                    sample_cat_feats = np.zeros((1, len(self.categories)), dtype=float)
                sample_feats = self._augment_features(emb[idx : idx + 1], sample_cat_feats)

                for dim_idx, col in enumerate(self.urgency_columns):
                    model = self.urgency_dim_models.get(col)
                    if model is None:
                        continue
                    # If this category was not predicted, urgency is 0
                    if pred_dim_indices and dim_idx not in pred_dim_indices:
                        dim_probs = np.zeros(6, dtype=float)
                        dim_probs[0] = 1.0
                        pred_class = 0
                    else:
                        raw_probs = self.backend.predict_proba(model, sample_feats)[0]
                        dim_probs = np.zeros(6, dtype=float)
                        classes = getattr(model, "classes_", None)
                        if classes is not None and len(classes) == len(raw_probs):
                            for ci, cls in enumerate(classes):
                                if 0 <= int(cls) <= 5:
                                    dim_probs[int(cls)] = raw_probs[ci]
                        elif len(raw_probs) == 6:
                            dim_probs = np.asarray(raw_probs, dtype=float)
                        else:
                            dim_probs[: len(raw_probs)] = raw_probs
                        dim_probs = dim_probs / max(dim_probs.sum(), 1e-12)
                        pred_class = int(np.argmax(dim_probs))
                    record["urgency_vector_8"][col] = pred_class
                    record["urgency_by_category"][col] = pred_class
                    record["urgency_probs_by_category"][col] = [float(p) for p in dim_probs]

            results.append(record)

        return {"records": results}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": self.name,
            "embedding_model": self.embedding_model,
            "categories": self.categories,
            "urgency_columns": self.urgency_columns,
            "option": self.option,
            "lgbm_params": self.lgbm_params,
            "overall_strategy": self.overall_strategy,
            "category_multilabel": self.category_multilabel,
            "category_threshold": self.category_threshold,
            # Model backend
            "model_backend": self.model_backend_name,
        }
        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=True, indent=2)

        if self.category_model is not None:
            joblib.dump(self.category_model, path / "category_model.joblib")
        if self.category_multi_models:
            joblib.dump(self.category_multi_models, path / "category_multi_models.joblib")
        if self.urgency_model is not None:
            joblib.dump(self.urgency_model, path / "urgency_model.joblib")
        if self.urgency_per_category:
            joblib.dump(self.urgency_per_category, path / "urgency_per_category.joblib")
        if self.urgency_dim_models:
            joblib.dump(self.urgency_dim_models, path / "urgency_dim_models.joblib")

    @classmethod
    def load(cls, path: str | Path, client: OpenAIClient) -> "MLSequentialPipeline":
        path = Path(path)
        with open(path / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        obj = cls(
            client=client,
            embedding_model=meta["embedding_model"],
            categories=meta["categories"],
            urgency_columns=meta.get("urgency_columns"),
            option=meta.get("option", "feature_concat"),
            lgbm_params=meta.get("lgbm_params"),
            overall_strategy=meta.get("overall_strategy", "max"),
            category_multilabel=meta.get("category_multilabel"),
            category_threshold=meta.get("category_threshold", 0.5),
            # Model backend
            model_backend=meta.get("model_backend", "lightgbm"),
        )
        if (path / "category_model.joblib").exists():
            obj.category_model = joblib.load(path / "category_model.joblib")
        if (path / "category_multi_models.joblib").exists():
            obj.category_multi_models = joblib.load(path / "category_multi_models.joblib")
        if (path / "urgency_model.joblib").exists():
            obj.urgency_model = joblib.load(path / "urgency_model.joblib")
        if (path / "urgency_per_category.joblib").exists():
            obj.urgency_per_category = joblib.load(path / "urgency_per_category.joblib")
        if (path / "urgency_dim_models.joblib").exists():
            obj.urgency_dim_models = joblib.load(path / "urgency_dim_models.joblib")
        return obj
