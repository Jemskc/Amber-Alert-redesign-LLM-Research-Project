"""Model backend abstraction for pluggable ML algorithms.

Provides a unified interface for LightGBM, Random Forest, and XGBoost,
allowing easy model switching via configuration.
"""

from __future__ import annotations

import os
import platform
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
from sklearn.ensemble import RandomForestClassifier

_IS_MACOS = platform.system() == "Darwin"

# Prevent OpenMP fork-safety segfaults on macOS
if _IS_MACOS:
    os.environ.setdefault("OMP_NUM_THREADS", "1")

if TYPE_CHECKING:
    from lightgbm import LGBMClassifier
    import xgboost as xgb


class ModelBackend(ABC):
    """Abstract base class for model backends."""

    name: str

    @abstractmethod
    def create_classifier(
        self, objective: str, num_class: Optional[int] = None, **kwargs
    ) -> Any:
        """Create a classifier for the given objective.

        Args:
            objective: "multiclass" or "binary"
            num_class: Number of classes for multiclass
            **kwargs: Additional parameters to override defaults

        Returns:
            A classifier instance
        """
        pass

    @abstractmethod
    def fit(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[List] = None,
        callbacks: Optional[List] = None,
    ) -> Any:
        """Fit the model with backend-specific handling.

        Args:
            model: The classifier to fit
            X: Training features
            y: Training labels
            sample_weight: Optional sample weights
            eval_set: Optional evaluation set (backend-specific support)
            callbacks: Optional callbacks (backend-specific support)

        Returns:
            The fitted model
        """
        pass

    @abstractmethod
    def predict_proba(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Get probability predictions.

        Args:
            model: The fitted classifier
            X: Features to predict

        Returns:
            Probability predictions
        """
        pass

    @property
    def supports_callbacks(self) -> bool:
        """Whether this backend supports training callbacks."""
        return False

    @property
    def supports_eval_set(self) -> bool:
        """Whether this backend supports evaluation sets during training."""
        return False

    @staticmethod
    def get_backend(name: str, params: Optional[Dict[str, Any]] = None) -> "ModelBackend":
        """Factory method to get backend by name.

        Args:
            name: Backend name (lightgbm, lgbm, randomforest, rf, xgboost, xgb)
            params: Backend-specific parameters

        Returns:
            A ModelBackend instance

        Raises:
            ValueError: If backend name is unknown
        """
        backends = {
            "lightgbm": LightGBMBackend,
            "lgbm": LightGBMBackend,
            "randomforest": RandomForestBackend,
            "rf": RandomForestBackend,
            "xgboost": XGBoostBackend,
            "xgb": XGBoostBackend,
        }
        backend_cls = backends.get(name.lower())
        if not backend_cls:
            available = sorted(set(backends.keys()))
            raise ValueError(f"Unknown backend: {name}. Available: {available}")
        return backend_cls(params or {})


class LightGBMBackend(ModelBackend):
    """LightGBM backend (existing behavior)."""

    name = "lightgbm"

    def __init__(self, params: Dict[str, Any]):
        # Force n_jobs=1 on macOS to prevent OpenMP fork-safety segfaults
        default_n_jobs = 1 if _IS_MACOS else -1
        self.base_params = {
            "n_estimators": params.get("n_estimators", 200),
            "learning_rate": params.get("learning_rate", 0.05),
            "max_depth": params.get("max_depth", -1),
            "num_leaves": params.get("num_leaves", 63),
            "verbosity": params.get("verbosity", -1),
            "n_jobs": params.get("n_jobs", default_n_jobs),
        }
        # Include any extra params passed through
        for key, value in params.items():
            if key not in self.base_params:
                self.base_params[key] = value

    def create_classifier(
        self, objective: str, num_class: Optional[int] = None, **kwargs
    ) -> "LGBMClassifier":
        from lightgbm import LGBMClassifier

        params = {**self.base_params, **kwargs}
        # Remove objective and num_class if present, we'll set them explicitly
        params.pop("objective", None)
        params.pop("num_class", None)

        if objective == "multiclass" and num_class:
            params["objective"] = "multiclass"
            params["num_class"] = num_class
        else:
            params["objective"] = "binary"
        return LGBMClassifier(**params)

    def fit(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[List] = None,
        callbacks: Optional[List] = None,
    ) -> Any:
        fit_params: Dict[str, Any] = {}
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
        if callbacks is not None:
            fit_params["callbacks"] = callbacks
        return model.fit(X, y, **fit_params)

    def predict_proba(self, model: Any, X: np.ndarray) -> np.ndarray:
        return model.predict_proba(X)

    @property
    def supports_callbacks(self) -> bool:
        return True

    @property
    def supports_eval_set(self) -> bool:
        return True


class RandomForestBackend(ModelBackend):
    """Random Forest backend."""

    name = "randomforest"

    def __init__(self, params: Dict[str, Any]):
        self.base_params = {
            "n_estimators": params.get("n_estimators", 200),
            "max_depth": params.get("max_depth", None),  # None = unlimited
            "min_samples_split": params.get("min_samples_split", 2),
            "min_samples_leaf": params.get("min_samples_leaf", 1),
            "max_features": params.get("max_features", "sqrt"),
            "n_jobs": params.get("n_jobs", -1),
            "random_state": params.get("random_state", 42),
            "class_weight": params.get("class_weight", "balanced"),
        }
        # Include any extra RF-compatible params
        rf_params = {
            "bootstrap", "oob_score", "warm_start", "ccp_alpha",
            "max_samples", "criterion", "min_weight_fraction_leaf",
            "max_leaf_nodes", "min_impurity_decrease",
        }
        for key, value in params.items():
            if key in rf_params and key not in self.base_params:
                self.base_params[key] = value

    def create_classifier(
        self, objective: str, num_class: Optional[int] = None, **kwargs
    ) -> RandomForestClassifier:
        params = {**self.base_params, **kwargs}
        # RF handles multiclass automatically, no special handling needed
        # Remove any LGBM-specific params that may have been passed
        for lgbm_param in ["objective", "num_class", "num_leaves", "learning_rate", "verbosity"]:
            params.pop(lgbm_param, None)
        return RandomForestClassifier(**params)

    def fit(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[List] = None,
        callbacks: Optional[List] = None,
    ) -> Any:
        # RF doesn't support eval_set or callbacks, just ignore them
        fit_params: Dict[str, Any] = {}
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight
        return model.fit(X, y, **fit_params)

    def predict_proba(self, model: Any, X: np.ndarray) -> np.ndarray:
        return model.predict_proba(X)


class XGBoostBackend(ModelBackend):
    """XGBoost backend."""

    name = "xgboost"

    def __init__(self, params: Dict[str, Any]):
        self.base_params = {
            "n_estimators": params.get("n_estimators", 200),
            "learning_rate": params.get("learning_rate", 0.05),
            "max_depth": params.get("max_depth", 6),
            "subsample": params.get("subsample", 0.8),
            "colsample_bytree": params.get("colsample_bytree", 0.8),
            "n_jobs": params.get("n_jobs", -1),
            "random_state": params.get("random_state", 42),
            "verbosity": params.get("verbosity", 0),
        }
        # Include any extra XGBoost-compatible params
        xgb_params = {
            "gamma", "min_child_weight", "reg_alpha", "reg_lambda",
            "scale_pos_weight", "base_score", "booster", "tree_method",
            "grow_policy", "max_leaves", "max_bin", "importance_type",
        }
        for key, value in params.items():
            if key in xgb_params and key not in self.base_params:
                self.base_params[key] = value

    def create_classifier(
        self, objective: str, num_class: Optional[int] = None, **kwargs
    ) -> "xgb.XGBClassifier":
        import xgboost as xgb

        params = {**self.base_params, **kwargs}
        # Remove any LGBM-specific params
        for lgbm_param in ["num_leaves", "num_class"]:
            params.pop(lgbm_param, None)
        # Remove objective, we'll set it explicitly
        params.pop("objective", None)

        if objective == "multiclass" and num_class:
            params["objective"] = "multi:softprob"
            params["num_class"] = num_class
        else:
            params["objective"] = "binary:logistic"
        return xgb.XGBClassifier(**params)

    def fit(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[List] = None,
        callbacks: Optional[List] = None,
    ) -> Any:
        fit_params: Dict[str, Any] = {}
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
            fit_params["verbose"] = False

        # XGBoost requires contiguous integer labels (0..n-1).
        # Some urgency dimensions have gaps (e.g. [0,1,2,3,5]).
        # Remap labels and store mapping on the model for predict_proba.
        unique_labels = np.unique(y)
        needs_remap = not np.array_equal(unique_labels, np.arange(len(unique_labels)))

        if needs_remap:
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            y_mapped = np.array([label_to_idx[v] for v in y])
            model._xgb_label_map = unique_labels  # store for reverse mapping
            if eval_set is not None:
                fit_params["eval_set"] = [
                    (eX, np.array([label_to_idx[v] for v in ey]))
                    for eX, ey in eval_set
                ]
            return model.fit(X, y_mapped, **fit_params)
        else:
            model._xgb_label_map = None
            return model.fit(X, y, **fit_params)

    def predict_proba(self, model: Any, X: np.ndarray) -> np.ndarray:
        probs = model.predict_proba(X)
        label_map = getattr(model, "_xgb_label_map", None)
        if label_map is not None:
            # Expand probabilities back to original label space.
            # E.g. if labels were [0,1,2,3,5], XGBoost outputs 5 columns
            # but we need 6 columns with column 4 = 0.
            max_label = int(label_map.max()) + 1
            if probs.shape[1] < max_label:
                full_probs = np.zeros((probs.shape[0], max_label), dtype=probs.dtype)
                for idx, original_label in enumerate(label_map):
                    full_probs[:, int(original_label)] = probs[:, idx]
                return full_probs
        return probs

    @property
    def supports_eval_set(self) -> bool:
        return True
