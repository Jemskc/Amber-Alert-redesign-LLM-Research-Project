from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from ..calibration import BinaryCalibrator
from ..data import CATEGORY_ID_MAP
from ..model_backends import ModelBackend
from ..openai_client import OpenAIClient
from ..utils import compute_entropy
from .base import BasePipeline


class MLJointPipeline(BasePipeline):
    name = "ml_joint"

    def __init__(
        self,
        client: OpenAIClient,
        embedding_model: str,
        categories: List[str],
        urgency_columns: Optional[List[str]] = None,
        ordinal: bool = False,
        calibration_method: str = "temperature",
        overall_strategy: str = "max",
        lgbm_params: Optional[Dict[str, Any]] = None,
        lgbm_log_period: int = 0,
        batch_size: int = 64,
        category_multilabel: Optional[bool] = None,
        category_threshold: float = 0.5,
        # New improvement options
        use_class_weights: bool = False,
        use_feature_scaling: bool = False,
        use_category_weights: bool = False,
        # Multi-backend support
        model_backend: str = "lightgbm",
        model_params: Optional[Dict[str, Any]] = None,
    ):
        self.client = client
        self.embedding_model = embedding_model
        self.categories = categories
        self.urgency_columns = urgency_columns
        self.ordinal = ordinal
        self.calibration_method = calibration_method
        self.overall_strategy = overall_strategy
        self.lgbm_params = lgbm_params or {}
        self.batch_size = batch_size
        self.category_multilabel = category_multilabel
        self.category_threshold = category_threshold
        self.lgbm_log_period = lgbm_log_period
        # New improvement options
        self.use_class_weights = use_class_weights
        self.use_feature_scaling = use_feature_scaling
        self.use_category_weights = use_category_weights

        # Initialize model backend (merge legacy lgbm_params with model_params)
        self.model_backend_name = model_backend
        _params = {**(lgbm_params or {}), **(model_params or {})}
        self.backend = ModelBackend.get_backend(model_backend, _params)

        self.category_model: Optional[Any] = None
        self.category_multi_models: Dict[str, Any] = {}
        self.urgency_model: Optional[Any] = None
        self.urgency_binary_models: List[Any] = []
        self.urgency_dim_models: Dict[str, Any] = {}
        self.p_high_calibrator: Optional[BinaryCalibrator] = None
        self.cat_conf_calibrator: Optional[BinaryCalibrator] = None
        self.scaler: Optional[StandardScaler] = None
        self.urgency_class_weights: Optional[Dict[int, float]] = None

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

    def _fit_urgency_dim_models(self, train_emb: np.ndarray, train_df) -> None:
        """Train per-dimension urgency models. Subclasses can override."""
        if not self.urgency_columns:
            return
        for col in self.urgency_columns:
            if col not in train_df.columns:
                continue
            y_dim = train_df[col].fillna(0).astype(int).clip(0, 5).values
            # Compute sample weights for per-dimension urgency
            dim_sample_weight = None
            if self.use_class_weights:
                counts = np.bincount(y_dim, minlength=6).astype(float)
                counts = np.maximum(counts, 1)
                weights = len(y_dim) / (6 * counts)
                dim_sample_weight = np.array([weights[int(y)] for y in y_dim])
            model = self.backend.create_classifier("multiclass", num_class=6)
            self._fit_model(model, train_emb, y_dim, f"urgency_dim:{col}", sample_weight=dim_sample_weight)
            self.urgency_dim_models[col] = model

    def fit(self, train_df, val_df=None) -> "MLJointPipeline":
        logger = logging.getLogger("triage")
        train_texts = train_df["text"].tolist()
        train_emb = self.client.embeddings(self.embedding_model, train_texts, batch_size=self.batch_size)

        # Feature scaling (optional)
        if self.use_feature_scaling:
            self.scaler = StandardScaler()
            train_emb = self.scaler.fit_transform(train_emb)
            logger.info("Applied feature scaling to embeddings.")

        # Compute urgency class weights (optional)
        y_overall_for_weights = self._compute_overall(train_df)
        if self.use_class_weights and y_overall_for_weights is not None:
            counts = np.bincount(y_overall_for_weights, minlength=6).astype(float)
            counts = np.maximum(counts, 1)  # Avoid division by zero
            weights = len(y_overall_for_weights) / (6 * counts)
            self.urgency_class_weights = {i: w for i, w in enumerate(weights)}
            logger.info("Computed urgency class weights: %s", self.urgency_class_weights)

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
                # Compute sample weights for imbalanced categories (optional)
                cat_sample_weight = None
                if self.use_category_weights:
                    n_pos = y.sum()
                    n_neg = len(y) - n_pos
                    weight_pos = len(y) / (2 * max(n_pos, 1))
                    weight_neg = len(y) / (2 * max(n_neg, 1))
                    cat_sample_weight = np.where(y == 1, weight_pos, weight_neg)
                model = self.backend.create_classifier("binary")
                self._fit_model(model, train_emb, y, f"category:{cat}", sample_weight=cat_sample_weight)
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

        y_overall = self._compute_overall(train_df)
        if y_overall is not None:
            # Compute sample weights from class weights (optional)
            urgency_sample_weight = None
            if self.use_class_weights and self.urgency_class_weights:
                urgency_sample_weight = np.array([self.urgency_class_weights[int(y)] for y in y_overall])

            if self.ordinal:
                self.urgency_binary_models = []
                for k in range(1, 6):
                    # For ordinal binary models, compute separate weights
                    y_binary = (y_overall >= k).astype(int)
                    ordinal_weight = None
                    if self.use_class_weights:
                        n_pos = y_binary.sum()
                        n_neg = len(y_binary) - n_pos
                        weight_pos = len(y_binary) / (2 * max(n_pos, 1))
                        weight_neg = len(y_binary) / (2 * max(n_neg, 1))
                        ordinal_weight = np.where(y_binary == 1, weight_pos, weight_neg)
                    model = self.backend.create_classifier("binary")
                    self._fit_model(model, train_emb, y_binary, f"urgency>= {k}", sample_weight=ordinal_weight)
                    self.urgency_binary_models.append(model)
            else:
                self.urgency_model = self.backend.create_classifier(
                    "multiclass", num_class=6
                )
                self._fit_model(self.urgency_model, train_emb, y_overall, "urgency_overall", sample_weight=urgency_sample_weight)

        self._fit_urgency_dim_models(train_emb, train_df)

        if val_df is not None:
            val_texts = val_df["text"].tolist()
            val_emb = self.client.embeddings(self.embedding_model, val_texts, batch_size=self.batch_size)
            # Apply feature scaling to validation embeddings
            if self.use_feature_scaling and self.scaler is not None:
                val_emb = self.scaler.transform(val_emb)
            if self.urgency_model or self.urgency_binary_models:
                probs = self._predict_urgency_probs(val_emb)
                p_high = probs[:, 4] + probs[:, 5]
                y_val = self._compute_overall(val_df)
                if y_val is not None:
                    self.p_high_calibrator = BinaryCalibrator(self.calibration_method).fit(p_high, (y_val >= 4).astype(int))

            if self.category_model is not None:
                cat_probs = self.category_model.predict_proba(val_emb)
                cat_max = cat_probs.max(axis=1)
                y_cat = val_df["category_gold"].astype(str)
                y_bin = (y_cat.isin(self.categories)).astype(int)
                self.cat_conf_calibrator = BinaryCalibrator(self.calibration_method).fit(cat_max, y_bin)

        return self

    def _predict_urgency_probs(self, emb: np.ndarray) -> np.ndarray:
        if self.ordinal and self.urgency_binary_models:
            ge_probs = [model.predict_proba(emb)[:, 1] for model in self.urgency_binary_models]
            ge_probs = np.vstack(ge_probs)
            probs = []
            for i in range(emb.shape[0]):
                p_ge = ge_probs[:, i]
                p_ge = np.concatenate([[1.0], p_ge, [0.0]])
                p = [max(p_ge[k] - p_ge[k + 1], 0.0) for k in range(6)]
                total = sum(p) or 1.0
                probs.append([v / total for v in p])
            return np.asarray(probs)
        if self.urgency_model is None:
            return np.full((emb.shape[0], 6), 1 / 6, dtype=float)
        probs = self.urgency_model.predict_proba(emb)
        return np.asarray(probs)

    def predict(self, df) -> Dict[str, Any]:
        texts = df["text"].tolist()
        emb = self.client.embeddings(self.embedding_model, texts, batch_size=self.batch_size)
        # Apply feature scaling if enabled
        if self.use_feature_scaling and self.scaler is not None:
            emb = self.scaler.transform(emb)
        results = []

        # Build mapping from category name to urgency dimension index once.
        # CATEGORY_ID_MAP: "1" -> "Emergency"; urgency_columns: "Urg 1", ...
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

        urg_probs = self._predict_urgency_probs(emb)

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
            urg = urg_probs[idx]
            urg = urg / urg.sum()
            record["urgency_probs_overall"] = [float(p) for p in urg]
            record["urgency_expected_overall"] = float(np.dot(np.arange(6), urg))
            record["urgency_probs_high"] = float(urg[4] + urg[5])
            record["entropy_urgency"] = float(compute_entropy(urg))
            if self.p_high_calibrator is not None:
                record["urgency_probs_high_calibrated"] = float(self.p_high_calibrator.predict([record["urgency_probs_high"]])[0])

            if self.urgency_dim_models:
                record["urgency_vector_8"] = {}
                record["urgency_probs_by_category"] = {}
                record["urgency_by_category"] = {}
                # Build set of urgency dimension indices for predicted categories.
                pred_dim_indices: set = set()
                if record.get("category_pred_list"):
                    for cat_name in record["category_pred_list"]:
                        dim_for_cat = cat_to_dim_idx.get(cat_name)
                        if dim_for_cat is not None:
                            pred_dim_indices.add(dim_for_cat)
                for dim_idx, col in enumerate(self.urgency_columns):
                    model = self.urgency_dim_models.get(col)
                    if model is None:
                        continue
                    # If this category was not predicted, urgency is 0
                    if pred_dim_indices and dim_idx not in pred_dim_indices:
                        dim_probs = np.zeros(6, dtype=float)
                        dim_probs[0] = 1.0  # All probability on class 0
                        pred_class = 0
                    else:
                        raw_probs = self.backend.predict_proba(model, emb[idx : idx + 1])[0]
                        # Map probabilities to full 6-class array (0-5).
                        # RF/LightGBM may return fewer columns when not all
                        # classes are present in training data; XGBoost backend
                        # already handles remapping internally.
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
            "ordinal": self.ordinal,
            "calibration_method": self.calibration_method,
            "overall_strategy": self.overall_strategy,
            "lgbm_params": self.lgbm_params,
            "category_multilabel": self.category_multilabel,
            "category_threshold": self.category_threshold,
            # New options
            "use_class_weights": self.use_class_weights,
            "use_feature_scaling": self.use_feature_scaling,
            "use_category_weights": self.use_category_weights,
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
        if self.urgency_binary_models:
            joblib.dump(self.urgency_binary_models, path / "urgency_binary_models.joblib")
        if self.urgency_dim_models:
            joblib.dump(self.urgency_dim_models, path / "urgency_dim_models.joblib")
        if self.p_high_calibrator is not None:
            joblib.dump(self.p_high_calibrator, path / "p_high_calibrator.joblib")
        if self.cat_conf_calibrator is not None:
            joblib.dump(self.cat_conf_calibrator, path / "cat_conf_calibrator.joblib")
        if self.scaler is not None:
            joblib.dump(self.scaler, path / "scaler.joblib")

    @classmethod
    def load(cls, path: str | Path, client: OpenAIClient) -> "MLJointPipeline":
        path = Path(path)
        with open(path / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        obj = cls(
            client=client,
            embedding_model=meta["embedding_model"],
            categories=meta["categories"],
            urgency_columns=meta.get("urgency_columns"),
            ordinal=meta.get("ordinal", False),
            calibration_method=meta.get("calibration_method", "temperature"),
            overall_strategy=meta.get("overall_strategy", "max"),
            lgbm_params=meta.get("lgbm_params"),
            category_multilabel=meta.get("category_multilabel"),
            category_threshold=meta.get("category_threshold", 0.5),
            # New options
            use_class_weights=meta.get("use_class_weights", False),
            use_feature_scaling=meta.get("use_feature_scaling", False),
            use_category_weights=meta.get("use_category_weights", False),
            # Model backend
            model_backend=meta.get("model_backend", "lightgbm"),
        )
        if (path / "category_model.joblib").exists():
            obj.category_model = joblib.load(path / "category_model.joblib")
        if (path / "category_multi_models.joblib").exists():
            obj.category_multi_models = joblib.load(path / "category_multi_models.joblib")
        if (path / "urgency_model.joblib").exists():
            obj.urgency_model = joblib.load(path / "urgency_model.joblib")
        if (path / "urgency_binary_models.joblib").exists():
            obj.urgency_binary_models = joblib.load(path / "urgency_binary_models.joblib")
        if (path / "urgency_dim_models.joblib").exists():
            obj.urgency_dim_models = joblib.load(path / "urgency_dim_models.joblib")
        if (path / "p_high_calibrator.joblib").exists():
            obj.p_high_calibrator = joblib.load(path / "p_high_calibrator.joblib")
        if (path / "cat_conf_calibrator.joblib").exists():
            obj.cat_conf_calibrator = joblib.load(path / "cat_conf_calibrator.joblib")
        if (path / "scaler.joblib").exists():
            obj.scaler = joblib.load(path / "scaler.joblib")
        return obj


class MLJointFilteredPipeline(MLJointPipeline):
    """ML Joint pipeline that trains per-dim urgency models only on positive samples.

    For each urgency dimension, training data is filtered to samples where
    urgency > 0, so models learn to distinguish levels 1-5 instead of being
    overwhelmed by zeros. Category masking at prediction time still handles
    the "is this category relevant?" question.
    """

    name = "ml_joint_filtered"

    def __init__(self, *args, min_filtered_samples: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_filtered_samples = min_filtered_samples

    def _fit_urgency_dim_models(self, train_emb: np.ndarray, train_df) -> None:
        """Train per-dimension urgency models on positive-only samples."""
        logger = logging.getLogger("triage")
        if not self.urgency_columns:
            return
        for col in self.urgency_columns:
            if col not in train_df.columns:
                continue
            y_dim = train_df[col].fillna(0).astype(int).clip(0, 5).values
            pos_mask = y_dim > 0
            n_positive = int(pos_mask.sum())
            n_total = len(y_dim)

            if n_positive < self.min_filtered_samples:
                # Fall back to all-sample training
                logger.warning(
                    "urgency_dim_filtered:%s only %d positive samples (< %d); "
                    "falling back to all-sample training.",
                    col, n_positive, self.min_filtered_samples,
                )
                y_filtered = y_dim
                emb_filtered = train_emb
            else:
                logger.info(
                    "urgency_dim_filtered:%s filtered to %d/%d positive samples.",
                    col, n_positive, n_total,
                )
                y_filtered = y_dim[pos_mask]
                emb_filtered = train_emb[pos_mask]

            unique_classes = np.unique(y_filtered)
            n_classes = len(unique_classes)

            # Compute sample weights on the filtered subset
            dim_sample_weight = None
            if self.use_class_weights:
                counts = np.bincount(y_filtered, minlength=6).astype(float)
                counts = np.maximum(counts, 1)
                weights = len(y_filtered) / (n_classes * counts)
                dim_sample_weight = np.array([weights[int(y)] for y in y_filtered])

            model = self.backend.create_classifier("multiclass", num_class=n_classes)
            self._fit_model(
                model, emb_filtered, y_filtered,
                f"urgency_dim_filtered:{col}",
                sample_weight=dim_sample_weight,
            )
            self.urgency_dim_models[col] = model

    def save(self, path: str | Path) -> None:
        super().save(path)
        path = Path(path)
        with open(path / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["min_filtered_samples"] = self.min_filtered_samples
        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=True, indent=2)

    @classmethod
    def load(cls, path: str | Path, client: "OpenAIClient") -> "MLJointFilteredPipeline":
        path = Path(path)
        with open(path / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        obj = cls(
            client=client,
            embedding_model=meta["embedding_model"],
            categories=meta["categories"],
            urgency_columns=meta.get("urgency_columns"),
            ordinal=meta.get("ordinal", False),
            calibration_method=meta.get("calibration_method", "temperature"),
            overall_strategy=meta.get("overall_strategy", "max"),
            lgbm_params=meta.get("lgbm_params"),
            category_multilabel=meta.get("category_multilabel"),
            category_threshold=meta.get("category_threshold", 0.5),
            use_class_weights=meta.get("use_class_weights", False),
            use_feature_scaling=meta.get("use_feature_scaling", False),
            use_category_weights=meta.get("use_category_weights", False),
            model_backend=meta.get("model_backend", "lightgbm"),
            min_filtered_samples=meta.get("min_filtered_samples", 10),
        )
        if (path / "category_model.joblib").exists():
            obj.category_model = joblib.load(path / "category_model.joblib")
        if (path / "category_multi_models.joblib").exists():
            obj.category_multi_models = joblib.load(path / "category_multi_models.joblib")
        if (path / "urgency_model.joblib").exists():
            obj.urgency_model = joblib.load(path / "urgency_model.joblib")
        if (path / "urgency_binary_models.joblib").exists():
            obj.urgency_binary_models = joblib.load(path / "urgency_binary_models.joblib")
        if (path / "urgency_dim_models.joblib").exists():
            obj.urgency_dim_models = joblib.load(path / "urgency_dim_models.joblib")
        if (path / "p_high_calibrator.joblib").exists():
            obj.p_high_calibrator = joblib.load(path / "p_high_calibrator.joblib")
        if (path / "cat_conf_calibrator.joblib").exists():
            obj.cat_conf_calibrator = joblib.load(path / "cat_conf_calibrator.joblib")
        if (path / "scaler.joblib").exists():
            obj.scaler = joblib.load(path / "scaler.joblib")
        return obj
