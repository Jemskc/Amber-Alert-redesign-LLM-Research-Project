from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

from .utils import ensure_dir, load_yaml, now_timestamp, save_yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "run_name": "experiment",
    "seed": 13,
    "output_root": "/mnt/data/triage_outputs",
    "data": {
        "urgency_xlsx": "/mnt/data/Ushahidi-Urgency-Dataset.xlsx",
        "category_xlsx": "/mnt/data/Ushahidi-Categories-Dataset.xlsx",
        "urgency_sheet": None,
        "category_sheet": None,
        "text_column": None,
        "urgency_columns": None,
        "category_columns": None,
        "group_column": None,
        "aggregate_threads": False,
        "rn_columns": None,
        "responder_columns": None,
        "crowd_columns": None,
        "merge": {
            "fuzzy": True,
            "fuzzy_threshold": 0.92,
        },
    },
    "splits": {
        "test_size": 0.2,
        "val_size": 0.1,
        "group_column": "group_id",
        "stratify_column": "category_gold",
        "cv_folds": 0,
        "cv_group_column": "group_id",
    },
    "openai": {
        "model": "gpt-4.1-mini",
        "judge_model": "gpt-4.1-mini",
        "embedding_model": "text-embedding-3-large",
        "temperature": 0,
        "max_retries": 4,
        "cache_path": "artifacts/openai_cache.sqlite",
        "batch_size": 64,
        "dry_run": False,
        "embedding_provider": "openai",
        "llm_provider": "openai",
        "save_embeddings": False,
        "embeddings_output_dir": "artifacts/embeddings",
        "progress": True,
    },
    "ml": {
        "lgbm_params": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": -1,
            "num_leaves": 63,
            "objective": "multiclass",
        },
        "lgbm_log_period": 0,
        "category_multilabel": None,
        "category_threshold": 0.5,
        "ordinal": False,
        "calibration": {
            "method": "temperature",
        },
        "grid_search": {
            "enabled": False,
            "metric": "category_macro_f1",
            "max_trials": 0,
            "apply_to_cv": False,
            "param_grid": {
                "num_leaves": [31, 63],
                "min_data_in_leaf": [20, 50],
                "feature_fraction": [0.6, 0.8],
                "bagging_fraction": [0.8, 1.0],
                "bagging_freq": [0, 1],
                "lambda_l2": [0.0, 1.0],
            },
        },
    },
    "elo": {
        "anchor_per_level": 10,
        "strategy": "binary_search",
        "max_comparisons": 8,
        "rating_system": "elo",
    },
    "policy": {
        "t_high": 0.6,
        "t_high2": 0.8,
        "t_low": 0.8,
        "t_cat": 0.55,
        "max_disagreement": 1.0,
    },
    "cluster": {
        "umap_dim": 5,
        "min_cluster_size": 15,
        "llm_labeling": False,
    },
    "report": {
        "ndcg_k": 50,
        "recall_k": 50,
    },
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    cfg = load_yaml(path)
    return deep_merge(DEFAULT_CONFIG, cfg)


def create_run_dir(cfg: Dict[str, Any]) -> Path:
    run_name = cfg.get("run_name", "experiment")
    out_root = cfg.get("output_root", "/mnt/data/triage_outputs")
    run_dir = ensure_dir(Path(out_root) / f"{now_timestamp()}_{run_name}")
    ensure_dir(run_dir / "figures")
    ensure_dir(run_dir / "artifacts")
    return run_dir


def save_config(cfg: Dict[str, Any], run_dir: Path) -> None:
    save_yaml(cfg, run_dir / "config_used.yaml")


def resolve_path(path_value: str | Path, base_dir: Path) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())
