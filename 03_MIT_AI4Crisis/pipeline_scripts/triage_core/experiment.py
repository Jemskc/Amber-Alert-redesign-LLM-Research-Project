from __future__ import annotations

import json
import itertools
import os
import random
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import create_run_dir, load_config, resolve_path, save_config
from .category_definitions import CATEGORY_DEFINITIONS, format_category_definitions
from .data import aggregate_to_threads, load_categories_dataset, load_urgency_dataset, merge_datasets
from .eval import (
    cost_sensitive_error,
    default_cost_matrix,
    ece_mce,
    evaluate_category,
    evaluate_category_auc,
    evaluate_ranking,
    evaluate_urgency,
    evaluate_urgency_per_category,
    plot_confusion_matrix,
    plot_reliability,
    reliability_curve,
    save_metrics,
)
from .openai_client import OpenAIClient
from .pipelines import (
    ClusteringPipeline,
    EloRankingPipeline,
    LLMJointPipeline,
    LLMSequentialPipeline,
    MLJointPipeline,
    MLSequentialPipeline,
    OracleCategoryPipeline,
)
from .policy import apply_policy
from .report import build_leaderboard, plot_coverage_risk, plot_ndcg_comparison
from .split import train_val_test_split
from .utils import configure_logger, ensure_dir, normalize_text, progress_iter, set_seed


def compute_overall_urgency(df: pd.DataFrame, urgency_columns: List[str], strategy: str = "max") -> np.ndarray:
    cols = [c for c in urgency_columns if c in df.columns]
    if not cols:
        return np.array([])
    coerced = df[cols].apply(pd.to_numeric, errors="coerce")
    valid_cols = [c for c in cols if coerced[c].notna().any()]
    if not valid_cols:
        return np.array([])
    values = coerced[valid_cols].fillna(0).clip(0, 5).astype(int).values
    if strategy == "mean":
        return np.round(values.mean(axis=1)).astype(int)
    return values.max(axis=1).astype(int)


# Canonical mapping from urgency column index to category name
# (matches insertion order of CATEGORY_DEFINITIONS)
URGENCY_COL_TO_CATEGORY = list(CATEGORY_DEFINITIONS.keys())


def _pred_overall_urgency(pred: Dict[str, Any]) -> int:
    """Compute predicted overall urgency as max of per-category urgencies.

    This matches the ground truth computation (max across Urg 1-8).
    Falls back to argmax of urgency_probs_overall if urgency_by_category
    is missing or empty.
    """
    urg_by_cat = pred.get("urgency_by_category", {})
    if urg_by_cat:
        return int(np.clip(max(urg_by_cat.values()), 0, 5))
    probs = pred.get("urgency_probs_overall", [1 / 6] * 6)
    return int(np.argmax(probs))


def _urgency_columns_from_df(df: pd.DataFrame) -> List[str]:
    exclude = {
        "msg_id",
        "text",
        "group_id",
        "category_hint",
        "category_hint_probs",
        "category_list",
        "category_multi",
        "has_urgency",
    }
    return [c for c in df.columns if c not in exclude]


def prepare_data(cfg: Dict[str, Any], logger) -> tuple[pd.DataFrame, List[str]]:
    urgency_df = load_urgency_dataset(
        cfg["data"]["urgency_xlsx"],
        sheet_name=cfg["data"].get("urgency_sheet"),
        text_column=cfg["data"].get("text_column"),
        urgency_columns=cfg["data"].get("urgency_columns"),
        group_column=cfg["data"].get("group_column"),
    )
    urgency_cols = _urgency_columns_from_df(urgency_df)
    if not urgency_cols and cfg["openai"].get("dry_run", False):
        rng = np.random.default_rng(cfg.get("seed", 13))
        urgency_cols = [f"urgency_dim_{i+1}" for i in range(8)]
        probs = np.array([0.7, 0.1, 0.08, 0.06, 0.04, 0.02], dtype=float)
        probs = probs / probs.sum()
        for col in urgency_cols:
            urgency_df[col] = rng.choice(np.arange(6), size=len(urgency_df), p=probs)
        urgency_df["has_urgency"] = True
        logger.warning("No urgency columns detected; created synthetic urgency columns for dry run.")
    category_df = load_categories_dataset(
        cfg["data"]["category_xlsx"],
        sheet_name=cfg["data"].get("category_sheet"),
        text_column=cfg["data"].get("text_column"),
        category_columns=cfg["data"].get("category_columns"),
        rn_columns=cfg["data"].get("rn_columns"),
        responder_columns=cfg["data"].get("responder_columns"),
        crowd_columns=cfg["data"].get("crowd_columns"),
        strategy=cfg["data"].get("category_strategy", "majority"),
        weights=cfg["data"].get("category_weights"),
    )
    merged, stats = merge_datasets(
        urgency_df,
        category_df,
        fuzzy=cfg["data"].get("merge", {}).get("fuzzy", True),
        fuzzy_threshold=cfg["data"].get("merge", {}).get("fuzzy_threshold", 0.92),
    )
    logger.info("Merge stats: %s", stats)

    if cfg["data"].get("aggregate_threads", False):
        pre_count = len(merged)
        merged = aggregate_to_threads(merged, "group_id", urgency_cols)
        logger.info(
            "Thread aggregation: %d posts -> %d threads",
            pre_count,
            len(merged),
        )

    return merged, urgency_cols


def _pipeline_enabled(cfg: Dict[str, Any], name: str, default: bool = True) -> bool:
    return bool(cfg.get("pipelines", {}).get(name, default))


def _grid_param_combinations(
    param_grid: Dict[str, Any],
    max_trials: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    values = []
    for key in keys:
        val = param_grid[key]
        if isinstance(val, (list, tuple)):
            values.append(list(val))
        else:
            values.append([val])
    combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    if max_trials and len(combos) > max_trials:
        rng = random.Random(seed)
        combos = rng.sample(combos, max_trials)
    return combos


def _score_from_metrics(metrics: Dict[str, Any], metric: str) -> Tuple[float, bool]:
    metric_map = {
        "category_macro_f1": ("category", "macro_f1", True),
        "category_accuracy": ("category", "accuracy", True),
        "category_jaccard": ("category", "jaccard", True),
        "urgency_mae": ("urgency", "mae", False),
        "urgency_qwk": ("urgency", "qwk", True),
        "urgency_high_f1": ("urgency", "high_f1", True),
        "urgency_high_auc": ("urgency", "high_auc", True),
        "urgency_cost_sensitive": ("urgency", "cost_sensitive", False),
        "urgency_weighted_mae": ("urgency", "weighted_mae", False),
        "urgency_boundary_crossing_rate": ("urgency", "boundary_crossing_rate", False),
    }
    if metric not in metric_map:
        raise ValueError(f"Unsupported grid_search metric: {metric}")
    section, key, maximize = metric_map[metric]
    score = metrics.get(section, {}).get(key)
    if score is None:
        raise ValueError(f"Metric {metric} missing from evaluation results")
    return float(score), maximize


def run_lgbm_grid_search(
    pipeline_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: Dict[str, Any],
    categories: List[str],
    urgency_columns: List[str],
    client: OpenAIClient,
    logger,
    run_dir: Path,
) -> Dict[str, Any]:
    grid_cfg = cfg.get("ml", {}).get("grid_search", {})
    if not grid_cfg.get("enabled"):
        return dict(cfg.get("ml", {}).get("lgbm_params", {}))

    metric = grid_cfg.get("metric", "category_macro_f1")
    max_trials = grid_cfg.get("max_trials")
    seed = cfg.get("seed", 13)
    param_grid = grid_cfg.get("param_grid", {})
    combos = _grid_param_combinations(param_grid, max_trials, seed)
    logger.info("Grid search %s: %d trials (metric=%s)", pipeline_name, len(combos), metric)

    best_score = None
    best_params: Dict[str, Any] = {}
    results: List[Dict[str, Any]] = []

    show_progress = grid_cfg.get("progress", cfg.get("openai", {}).get("progress", True))
    for params in progress_iter(combos, total=len(combos), desc=f"{pipeline_name} grid", enabled=show_progress):
        lgbm_params = dict(cfg.get("ml", {}).get("lgbm_params", {}))
        lgbm_params.update(params)
        try:
            if pipeline_name == "ml_joint":
                pipeline = MLJointPipeline(
                    client=client,
                    embedding_model=cfg["openai"]["embedding_model"],
                    categories=categories,
                    urgency_columns=urgency_columns,
                    ordinal=cfg["ml"].get("ordinal", False),
                    calibration_method=cfg["ml"].get("calibration", {}).get("method", "temperature"),
                    lgbm_params=lgbm_params,
                    lgbm_log_period=cfg["ml"].get("lgbm_log_period", 0),
                    category_multilabel=cfg["ml"].get("category_multilabel"),
                    category_threshold=cfg["ml"].get("category_threshold", 0.5),
                )
            else:
                pipeline = MLSequentialPipeline(
                    client=client,
                    embedding_model=cfg["openai"]["embedding_model"],
                    categories=categories,
                    urgency_columns=urgency_columns,
                    option=cfg["ml"].get("sequential_option", "feature_concat"),
                    lgbm_params=lgbm_params,
                    lgbm_log_period=cfg["ml"].get("lgbm_log_period", 0),
                    category_multilabel=cfg["ml"].get("category_multilabel"),
                    category_threshold=cfg["ml"].get("category_threshold", 0.5),
                )
            pipeline.fit(train_df, val_df)
            preds = run_pipeline_predictions(pipeline, val_df)
            metrics = evaluate_predictions_simple(val_df, preds, categories, urgency_columns)
            score, maximize = _score_from_metrics(metrics, metric)
        except Exception as exc:
            logger.warning("Grid search %s params %s failed: %s", pipeline_name, params, exc)
            continue

        results.append({"params": params, "score": score, "metrics": metrics})
        if best_score is None:
            best_score = score
            best_params = params
        else:
            if (maximize and score > best_score) or (not maximize and score < best_score):
                best_score = score
                best_params = params

    out_path = run_dir / f"grid_search_{pipeline_name}.json"
    payload = {
        "pipeline": pipeline_name,
        "metric": metric,
        "best_score": best_score,
        "best_params": best_params,
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    if best_score is None:
        logger.warning("Grid search %s had no successful trials; using base params.", pipeline_name)
        best_params = {}
    logger.info("Grid search %s best score=%s params=%s", pipeline_name, best_score, best_params)
    logger.info("Saved grid search results: %s", out_path)

    merged_params = dict(cfg.get("ml", {}).get("lgbm_params", {}))
    merged_params.update(best_params)
    return merged_params


def run_pipeline_predictions(pipeline, df: pd.DataFrame) -> List[Dict[str, Any]]:
    preds = pipeline.predict(df)
    return preds.get("records", preds)


def evaluate_predictions_simple(
    df: pd.DataFrame,
    preds: List[Dict[str, Any]],
    categories: List[str],
    urgency_columns: List[str],
    overall_strategy: str = "max",
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    n = min(len(df), len(preds))
    df = df.iloc[:n].copy()
    preds = preds[:n]

    if df["has_category"].any():
        mask = df["has_category"].to_numpy()
        if "category_list" in df.columns:
            y_true = [parse_category_list(v) for v in df.loc[mask, "category_list"].tolist()]
        else:
            y_true = df.loc[mask, "category_gold"].astype(str).tolist()
        if "category_list" in df.columns and any("category_pred_list" in preds[i] for i in range(n) if mask[i]):
            y_pred = [preds[i].get("category_pred_list") or [] for i in range(n) if mask[i]]
        else:
            y_pred = [preds[i].get("category_top") for i in range(n) if mask[i]]
        y_pred_probs = [preds[i].get("category_probs", {}) for i in range(n) if mask[i]]
        metrics["category"] = evaluate_category(y_true, y_pred, y_pred_probs=y_pred_probs, labels=categories)

    if df["has_urgency"].any():
        mask = df["has_urgency"].to_numpy()
        y_true = compute_overall_urgency(df.loc[mask], urgency_columns, strategy=overall_strategy)
        if len(y_true) > 0:
            y_pred = [_pred_overall_urgency(preds[i]) for i in range(n) if mask[i]]
            p_high = [preds[i].get("urgency_probs_high", 0.0) for i in range(n) if mask[i]]
            metrics["urgency"] = evaluate_urgency(y_true.tolist(), y_pred, p_high)
            metrics["urgency"]["cost_sensitive"] = cost_sensitive_error(y_true.tolist(), y_pred, default_cost_matrix())

    return metrics


def summarize_cv_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for metrics in metrics_list:
        for section, values in metrics.items():
            summary.setdefault(section, {})
            for key, val in values.items():
                summary[section].setdefault(key, []).append(val)
    for section, values in summary.items():
        for key, vals in list(values.items()):
            arr = np.asarray(vals, dtype=float)
            summary[section][f"{key}_mean"] = float(arr.mean())
            summary[section][f"{key}_std"] = float(arr.std())
            summary[section].pop(key, None)
    return summary


def run_ml_cross_val(
    df: pd.DataFrame,
    pipeline_name: str,
    cfg: Dict[str, Any],
    categories: List[str],
    urgency_columns: List[str],
    client: OpenAIClient,
    lgbm_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

    folds = cfg["splits"].get("cv_folds", 0)
    if folds <= 1:
        return {}

    group_col = cfg["splits"].get("cv_group_column", cfg["splits"].get("group_column"))
    if group_col and group_col in df.columns:
        splitter = GroupKFold(n_splits=folds)
        split_iter = splitter.split(df, groups=df[group_col])
    elif cfg["splits"].get("stratify_column") in df.columns:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg.get("seed", 13))
        split_iter = splitter.split(df, df[cfg["splits"]["stratify_column"]])
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=cfg.get("seed", 13))
        split_iter = splitter.split(df)

    metrics_list: List[Dict[str, Any]] = []
    for train_idx, val_idx in split_iter:
        train_fold = df.iloc[train_idx].copy()
        val_fold = df.iloc[val_idx].copy()

        if pipeline_name == "ml_joint":
            pipeline = MLJointPipeline(
                client=client,
                embedding_model=cfg["openai"]["embedding_model"],
                categories=categories,
                urgency_columns=urgency_columns,
                ordinal=cfg["ml"].get("ordinal", False),
                calibration_method=cfg["ml"].get("calibration", {}).get("method", "temperature"),
                lgbm_params=lgbm_params or cfg["ml"].get("lgbm_params", {}),
                lgbm_log_period=cfg["ml"].get("lgbm_log_period", 0),
                category_multilabel=cfg["ml"].get("category_multilabel"),
                category_threshold=cfg["ml"].get("category_threshold", 0.5),
            )
        else:
            pipeline = MLSequentialPipeline(
                client=client,
                embedding_model=cfg["openai"]["embedding_model"],
                categories=categories,
                urgency_columns=urgency_columns,
                option=cfg["ml"].get("sequential_option", "feature_concat"),
                lgbm_params=lgbm_params or cfg["ml"].get("lgbm_params", {}),
                lgbm_log_period=cfg["ml"].get("lgbm_log_period", 0),
                category_multilabel=cfg["ml"].get("category_multilabel"),
                category_threshold=cfg["ml"].get("category_threshold", 0.5),
            )

        pipeline.fit(train_fold, val_fold)
        preds = run_pipeline_predictions(pipeline, val_fold)
        metrics_list.append(evaluate_predictions_simple(val_fold, preds, categories, urgency_columns))

    return summarize_cv_metrics(metrics_list)


def evaluate_predictions(
    df: pd.DataFrame,
    preds: List[Dict[str, Any]],
    categories: List[str],
    urgency_columns: List[str],
    output_dir: Path,
    prefix: str,
    overall_strategy: str = "max",
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    n = min(len(df), len(preds))
    df = df.iloc[:n].copy()
    preds = preds[:n]

    if df["has_category"].any():
        mask = df["has_category"].to_numpy()
        list_col = None
        if "category_list" in df.columns:
            list_col = "category_list"
        elif "category_list_cat" in df.columns:
            list_col = "category_list_cat"
        elif "category_list_urg" in df.columns:
            list_col = "category_list_urg"

        if list_col:
            y_true_multi = [parse_category_list(v) for v in df.loc[mask, list_col].tolist()]
            if any("category_pred_list" in preds[i] for i in range(n) if mask[i]):
                y_pred = [preds[i].get("category_pred_list") or [] for i in range(n) if mask[i]]
            else:
                y_pred = []
                for i in range(n):
                    if not mask[i]:
                        continue
                    top = preds[i].get("category_top")
                    y_pred.append([top] if top else [])
            y_pred_probs = [preds[i].get("category_probs", {}) for i in range(n) if mask[i]]
            metrics["category"] = evaluate_category(y_true_multi, y_pred, y_pred_probs=y_pred_probs, labels=categories)
            # Per-category AUC
            auc_metrics = evaluate_category_auc(y_true_multi, y_pred_probs, categories)
            if auc_metrics:
                metrics["category_auc"] = auc_metrics
            if y_true_multi and all(len(items) == 1 for items in y_true_multi):
                y_true = [items[0] if items else None for items in y_true_multi]
                y_pred_single = []
                for pred in y_pred:
                    if isinstance(pred, (list, tuple, set)) and pred:
                        y_pred_single.append(list(pred)[0])
                    else:
                        y_pred_single.append(None)
                plot_confusion_matrix(y_true, y_pred_single, categories, output_dir / f"figures/confusion_{prefix}.png")
        else:
            y_true = df.loc[mask, "category_gold"].astype(str).tolist()
            y_pred = [preds[i].get("category_top") for i in range(n) if mask[i]]
            y_pred_probs = [preds[i].get("category_probs", {}) for i in range(n) if mask[i]]
            metrics["category"] = evaluate_category(y_true, y_pred, y_pred_probs=y_pred_probs, labels=categories)
            plot_confusion_matrix(y_true, y_pred, categories, output_dir / f"figures/confusion_{prefix}.png")

    if df["has_urgency"].any():
        mask = df["has_urgency"].to_numpy()
        y_true = compute_overall_urgency(df.loc[mask], urgency_columns, strategy=overall_strategy)
        if len(y_true) > 0:
            y_pred = [_pred_overall_urgency(preds[i]) for i in range(n) if mask[i]]
            p_high = [preds[i].get("urgency_probs_high", 0.0) for i in range(n) if mask[i]]
            metrics["urgency"] = evaluate_urgency(y_true.tolist(), y_pred, p_high)
            metrics["urgency"]["cost_sensitive"] = cost_sensitive_error(y_true.tolist(), y_pred, default_cost_matrix())
            mean_pred, frac_pos = reliability_curve((y_true >= 4).astype(int), p_high)
            plot_reliability(mean_pred, frac_pos, output_dir / f"figures/calibration_{prefix}.png")
            ece, mce = ece_mce((y_true >= 4).astype(int), p_high)
            metrics["calibration"] = {"ece": ece, "mce": mce}

            # Per-category urgency metrics
            preds_for_urgency = [preds[i] for i in range(n) if mask[i]]
            urgency_per_cat = evaluate_urgency_per_category(
                df.loc[mask], preds_for_urgency, urgency_columns,
                category_names=URGENCY_COL_TO_CATEGORY,
            )
            if urgency_per_cat:
                metrics["urgency_per_category"] = urgency_per_cat

    save_metrics(metrics, output_dir / f"metrics_{prefix}.json")
    return metrics


def save_predictions(df: pd.DataFrame, preds: List[Dict[str, Any]], output_dir: Path, prefix: str) -> None:
    pred_df = pd.DataFrame(preds)
    out = pd.concat([df.reset_index(drop=True), pred_df], axis=1)
    out.to_csv(output_dir / f"predictions_{prefix}.csv", index=False)


def save_embeddings(df: pd.DataFrame, embeddings: np.ndarray, path: Path) -> None:
    cols = [f"e{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=cols)
    out = pd.concat([df[["msg_id", "text"]].reset_index(drop=True), emb_df], axis=1)
    out.to_csv(path, index=False)


def parse_category_list(value: Any) -> List[str]:
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


def _normalize_msg_id(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    mask = s.str.match(r"^\\d+\\.0$")
    if mask.any():
        s = s.where(~mask, s.str.replace(r"\\.0$", "", regex=True))
    return s


def load_embeddings_partial(path: Path, df: pd.DataFrame, logger) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not path.exists():
        return None, None
    try:
        try:
            emb_df = pd.read_csv(path, dtype={"msg_id": "string"})
        except ValueError:
            emb_df = pd.read_csv(path)
    except Exception as exc:
        logger.warning("Failed to load embeddings from %s: %s", path, exc)
        return None, None

    emb_cols = [col for col in emb_df.columns if col.startswith("e")]
    if not emb_cols:
        logger.warning("Embeddings file %s has no embedding columns; skipping reuse.", path)
        return None, None

    total = len(df)
    if total == 0:
        return np.array([], dtype=bool), np.empty((0, len(emb_cols)), dtype=float)

    if "msg_id" in emb_df.columns and "msg_id" in df.columns:
        emb_df["msg_id"] = _normalize_msg_id(emb_df["msg_id"])
        df_ids = _normalize_msg_id(df["msg_id"])
        emb_df = emb_df.set_index("msg_id")
        if emb_df.index.duplicated().any():
            dup_count = int(emb_df.index.duplicated().sum())
            logger.warning(
                "Embeddings file %s has %d duplicate msg_id rows; keeping first occurrence.",
                path,
                dup_count,
            )
            emb_df = emb_df[~emb_df.index.duplicated(keep="first")]
        matched_mask = df_ids.isin(emb_df.index).to_numpy()
        matched_count = int(matched_mask.sum())
        missing_count = total - matched_count
        coverage = 100.0 * matched_count / total if total else 0.0
        logger.info(
            "Embeddings reuse %s: matched=%d missing=%d coverage=%.1f%%",
            path.name,
            matched_count,
            missing_count,
            coverage,
        )
        if missing_count:
            sample_missing = df_ids[~matched_mask].head(3).tolist()
            logger.info("Embeddings reuse %s: missing_sample=%s", path.name, sample_missing)
        if matched_count == 0:
            sample_df = df_ids.head(3).tolist()
            sample_emb = emb_df.index.astype(str).to_series().head(3).tolist()
            logger.warning(
                "Embeddings reuse %s: no msg_id matches. df dtype=%s emb dtype=%s df_sample=%s emb_sample=%s",
                path.name,
                df["msg_id"].dtype,
                emb_df.index.dtype,
                sample_df,
                sample_emb,
            )
        if matched_count == 0:
            return None, None
        matched_ids = df_ids[matched_mask]
        matched_emb = emb_df.loc[matched_ids, emb_cols].to_numpy(dtype=float)
        return matched_mask, matched_emb

    if len(emb_df) != total:
        logger.warning("Embeddings file %s row count mismatch; recomputing.", path)
        return None, None

    matched_mask = np.ones(total, dtype=bool)
    matched_emb = emb_df[emb_cols].to_numpy(dtype=float)
    logger.info("Embeddings reuse %s: matched=%d missing=0 coverage=100.0%%", path.name, total)
    return matched_mask, matched_emb


def load_embeddings_union(paths: List[Path], df: pd.DataFrame, logger) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    frames = []
    emb_cols: Optional[List[str]] = None
    for path in paths:
        if not path.exists():
            continue
        try:
            try:
                emb_df = pd.read_csv(path, dtype={"msg_id": "string"})
            except ValueError:
                emb_df = pd.read_csv(path)
        except Exception as exc:
            logger.warning("Failed to load embeddings from %s: %s", path, exc)
            continue
        cols = [col for col in emb_df.columns if col.startswith("e")]
        if not cols:
            logger.warning("Embeddings file %s has no embedding columns; skipping.", path)
            continue
        if emb_cols is None:
            emb_cols = cols
        elif len(cols) != len(emb_cols):
            logger.warning("Embeddings file %s dim mismatch; expected %d got %d.", path, len(emb_cols), len(cols))
            continue
        cols_to_keep = ["msg_id"]
        if "text" in emb_df.columns:
            cols_to_keep.append("text")
        if "msg_id" in emb_df.columns:
            emb_df["msg_id"] = _normalize_msg_id(emb_df["msg_id"])
            emb_df = emb_df.drop_duplicates("msg_id", keep="first")
            frames.append(emb_df[cols_to_keep + emb_cols])

    if not frames or emb_cols is None:
        return None, None

    merged = pd.concat(frames, ignore_index=True)
    if merged["msg_id"].duplicated().any():
        merged = merged.drop_duplicates("msg_id", keep="first")
    merged = merged.set_index("msg_id")

    total = len(df)
    if total == 0:
        return np.array([], dtype=bool), np.empty((0, len(emb_cols)), dtype=float)

    if "msg_id" in df.columns:
        df_ids = _normalize_msg_id(df["msg_id"])
        matched_mask = df_ids.isin(merged.index).to_numpy()
        matched_count = int(matched_mask.sum())
        missing_count = total - matched_count
        coverage = 100.0 * matched_count / total if total else 0.0
        logger.info(
            "Embeddings reuse union: matched=%d missing=%d coverage=%.1f%%",
            matched_count,
            missing_count,
            coverage,
        )
        if missing_count:
            sample_missing = df_ids[~matched_mask].head(3).tolist()
            logger.info("Embeddings reuse union: missing_sample=%s", sample_missing)
        if matched_count == 0 and "text" in df.columns and "text" in merged.columns:
            df_text = df["text"].astype(str).map(normalize_text)
            merged_text = merged["text"].astype(str).map(normalize_text)
            merged_text_index = merged_text.reset_index().drop_duplicates("text").set_index("text")
            matched_mask = df_text.isin(merged_text_index.index).to_numpy()
            matched_count = int(matched_mask.sum())
            missing_count = total - matched_count
            coverage = 100.0 * matched_count / total if total else 0.0
            logger.info(
                "Embeddings reuse union (text fallback): matched=%d missing=%d coverage=%.1f%%",
                matched_count,
                missing_count,
                coverage,
            )
            if matched_count == 0:
                return None, None
            matched_texts = df_text[matched_mask]
            matched_emb = merged_text_index.loc[matched_texts, emb_cols].to_numpy(dtype=float)
            return matched_mask, matched_emb
        if matched_count == 0:
            return None, None
        matched_ids = df_ids[matched_mask]
        matched_emb = merged.loc[matched_ids, emb_cols].to_numpy(dtype=float)
        return matched_mask, matched_emb

    return None, None


def find_previous_embeddings_dir(
    output_root: Path,
    run_name: str,
    rel_emb_dir: Path,
    current_run_dir: Path,
    logger,
) -> Optional[Path]:
    if not output_root.exists():
        return None
    required_files = {"embeddings_train.csv", "embeddings_val.csv", "embeddings_test.csv"}
    current_run_dir = current_run_dir.resolve()
    candidates = sorted(
        [p for p in output_root.glob(f"*_{run_name}") if p.is_dir() and p.resolve() != current_run_dir],
        reverse=True,
    )
    for cand in candidates:
        emb_dir = (cand / rel_emb_dir).resolve()
        if not emb_dir.exists():
            continue
        if not any((emb_dir / name).exists() for name in required_files):
            logger.info("Skipping prior embeddings dir (no files): %s", emb_dir)
            continue
        logger.info("Found prior embeddings directory: %s", emb_dir)
        return emb_dir

    generic = sorted(output_root.glob(f"*/{rel_emb_dir.as_posix()}"), reverse=True)
    for emb_dir in generic:
        emb_dir = emb_dir.resolve()
        if current_run_dir in emb_dir.parents:
            continue
        if emb_dir.exists():
            if not any((emb_dir / name).exists() for name in required_files):
                logger.info("Skipping prior embeddings dir (no files): %s", emb_dir)
                continue
            logger.info("Found prior embeddings directory (fallback): %s", emb_dir)
            return emb_dir
    return None


def _checkpoint_has_fields(path: Path, required: List[str]) -> bool:
    if not path.exists():
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                return all(key in data for key in required)
    except Exception:
        return False
    return False


def copy_previous_checkpoint(
    output_root: Path,
    run_name: str,
    filename: str,
    run_dir: Path,
    required_fields: Optional[List[str]],
    logger,
) -> None:
    dest = run_dir / "artifacts" / filename
    if dest.exists():
        return
    candidates = sorted([p for p in output_root.glob(f"*_{run_name}") if p.is_dir() and p.resolve() != run_dir.resolve()], reverse=True)
    for cand in candidates:
        src = cand / "artifacts" / filename
        if src.exists() and (not required_fields or _checkpoint_has_fields(src, required_fields)):
            ensure_dir(dest.parent)
            shutil.copy2(src, dest)
            logger.info("Reused checkpoint from %s", src)
            return
        if src.exists():
            logger.info("Skipping checkpoint without required fields: %s", src)
    generic = sorted(output_root.glob(f"*/artifacts/{filename}"), reverse=True)
    for src in generic:
        src = src.resolve()
        if run_dir.resolve() in src.parents:
            continue
        if src.exists() and (not required_fields or _checkpoint_has_fields(src, required_fields)):
            ensure_dir(dest.parent)
            shutil.copy2(src, dest)
            logger.info("Reused checkpoint from %s", src)
            return
        if src.exists():
            logger.info("Skipping checkpoint without required fields: %s", src)


def run_all(config_path: str) -> Path:
    cfg = load_config(config_path)
    run_dir = create_run_dir(cfg)
    save_config(cfg, run_dir)
    logger = configure_logger(str(run_dir / "run.log"))
    set_seed(cfg.get("seed", 13))

    df, urgency_columns = prepare_data(cfg, logger)

    train_df, val_df, test_df = train_val_test_split(
        df,
        test_size=cfg["splits"]["test_size"],
        val_size=cfg["splits"]["val_size"],
        group_column=cfg["splits"].get("group_column"),
        stratify_column=cfg["splits"].get("stratify_column"),
        seed=cfg.get("seed", 13),
    )

    if "category_list" in df.columns:
        category_set = set()
        for raw in df["category_list"].dropna().tolist():
            category_set.update(parse_category_list(raw))
        categories = sorted(category_set)
    else:
        categories = sorted(df["category_gold"].dropna().unique().tolist())
    if not categories:
        logger.warning("No category labels detected; using placeholder category.")
        categories = ["unspecified"]
    logger.info("Category taxonomy:\\n%s", format_category_definitions())
    if not urgency_columns:
        urgency_columns = cfg["data"].get("urgency_columns") or []

    show_progress = cfg["openai"].get("progress", True)
    api_key = cfg["openai"].get("api_key") or os.environ.get("OPENAI_API_KEY")
    dry_run = cfg["openai"].get("dry_run", False)
    if not api_key and not dry_run:
        dry_run = True
        logger.warning("OPENAI_API_KEY not set; running offline and requiring cached embeddings.")

    cache_path = resolve_path(cfg["openai"]["cache_path"], base_dir=Path.cwd())
    client = OpenAIClient(
        api_key=api_key,
        cache_path=cache_path,
        max_retries=cfg["openai"]["max_retries"],
        dry_run=dry_run,
        show_progress=show_progress,
        embedding_provider=cfg["openai"].get("embedding_provider", "openai"),
        llm_provider=cfg["openai"].get("llm_provider", "openai"),
    )

    if cfg["openai"].get("save_embeddings", False):
        rel_emb_dir = Path(cfg["openai"].get("embeddings_output_dir", "artifacts/embeddings"))
        emb_dir = rel_emb_dir
        if not emb_dir.is_absolute():
            emb_dir = run_dir / emb_dir
        ensure_dir(emb_dir)
        train_path = emb_dir / "embeddings_train.csv"
        val_path = emb_dir / "embeddings_val.csv"
        test_path = emb_dir / "embeddings_test.csv"
        batch_size = cfg["openai"].get("batch_size", 64)
        embedding_model = cfg["openai"]["embedding_model"]
        reuse_dir = None
        if not rel_emb_dir.is_absolute():
            reuse_dir = find_previous_embeddings_dir(Path(cfg["output_root"]), cfg["run_name"], rel_emb_dir, run_dir, logger)

        def build_embeddings(path: Path, df: pd.DataFrame) -> np.ndarray:
            candidate_paths = [path]
            if reuse_dir is not None:
                candidate_paths.extend(
                    [
                        reuse_dir / "embeddings_train.csv",
                        reuse_dir / "embeddings_val.csv",
                        reuse_dir / "embeddings_test.csv",
                    ]
                )
            matched_mask, matched_emb = load_embeddings_union(candidate_paths, df, logger)

            if matched_mask is None:
                if dry_run:
                    raise RuntimeError(
                        f"Missing cached embeddings for {path.name} and OPENAI_API_KEY is not set."
                    )
                logger.info("Embeddings reuse %s: no prior matches; computing all %d rows.", path.name, len(df))
                emb = client.embeddings(embedding_model, df["text"].tolist(), batch_size=batch_size)
                save_embeddings(df, emb, path)
                return emb

            if matched_mask.all():
                if matched_emb.shape[0] != len(df):
                    logger.warning(
                        "Embeddings reuse %s: matched_emb rows %d != df rows %d; recomputing all.",
                        path.name,
                        matched_emb.shape[0],
                        len(df),
                    )
                    emb = client.embeddings(embedding_model, df["text"].tolist(), batch_size=batch_size)
                    save_embeddings(df, emb, path)
                    return emb
                emb = matched_emb
                save_embeddings(df, emb, path)
                return emb

            missing_texts = df.loc[~matched_mask, "text"].tolist()
            logger.info("Embeddings reuse %s: computing %d missing rows.", path.name, len(missing_texts))
            if dry_run and missing_texts:
                raise RuntimeError(
                    f"Missing {len(missing_texts)} embeddings for {path.name} and OPENAI_API_KEY is not set."
                )
            missing_emb = client.embeddings(embedding_model, missing_texts, batch_size=batch_size)
            dim = matched_emb.shape[1] if matched_emb.size else missing_emb.shape[1]
            emb = np.zeros((len(df), dim), dtype=float)
            emb[matched_mask] = matched_emb
            emb[~matched_mask] = missing_emb
            save_embeddings(df, emb, path)
            return emb

        train_emb = build_embeddings(train_path, train_df)
        client.register_embeddings(embedding_model, train_df["text"].tolist(), train_emb)
        val_emb = build_embeddings(val_path, val_df)
        client.register_embeddings(embedding_model, val_df["text"].tolist(), val_emb)
        test_emb = build_embeddings(test_path, test_df)
        client.register_embeddings(embedding_model, test_df["text"].tolist(), test_emb)

    metrics_all: Dict[str, Dict[str, Any]] = {}
    preds_cache: Dict[str, List[Dict[str, Any]]] = {}

    run_llm_joint = _pipeline_enabled(cfg, "llm_joint", True)
    run_llm_sequential = _pipeline_enabled(cfg, "llm_sequential", True)
    run_oracle_llm = _pipeline_enabled(cfg, "oracle_llm", True)
    run_ml_joint = _pipeline_enabled(cfg, "ml_joint", True)
    run_ml_sequential = _pipeline_enabled(cfg, "ml_sequential", True)
    run_ml_cv = _pipeline_enabled(cfg, "ml_cv", True)
    run_clustering = _pipeline_enabled(cfg, "clustering", True)
    run_elo = _pipeline_enabled(cfg, "elo_ranking", True)
    run_policy = _pipeline_enabled(cfg, "policy", True)

    if run_llm_joint:
        copy_previous_checkpoint(
            Path(cfg["output_root"]),
            cfg["run_name"],
            "llm_joint.jsonl",
            run_dir,
            required_fields=["urgency_by_category"],
            logger=logger,
        )
    if run_llm_sequential:
        copy_previous_checkpoint(
            Path(cfg["output_root"]),
            cfg["run_name"],
            "llm_sequential.jsonl",
            run_dir,
            required_fields=["urgency_by_category"],
            logger=logger,
        )

    if run_llm_joint:
        llm_joint = LLMJointPipeline(
            client=client,
            model=cfg["openai"]["model"],
            categories=categories,
            temperature=cfg["openai"]["temperature"],
            show_progress=show_progress,
            checkpoint_path=str(run_dir / "artifacts" / "llm_joint.jsonl"),
            progress_log_path=str(run_dir / "artifacts" / "llm_joint_progress.jsonl"),
        )
        preds = run_pipeline_predictions(llm_joint, test_df)
        save_predictions(test_df, preds, run_dir, "llm_joint")
        preds_cache["llm_joint"] = preds
        metrics_all["llm_joint"] = evaluate_predictions(test_df, preds, categories, urgency_columns, run_dir, "llm_joint")
    else:
        logger.info("Skipping LLM joint (pipelines.llm_joint=false).")

    if run_llm_sequential:
        llm_seq = LLMSequentialPipeline(
            client=client,
            model=cfg["openai"]["model"],
            categories=categories,
            temperature=cfg["openai"]["temperature"],
            show_progress=show_progress,
            checkpoint_path=str(run_dir / "artifacts" / "llm_sequential.jsonl"),
            progress_log_path=str(run_dir / "artifacts" / "llm_sequential_progress.jsonl"),
        )
        preds = run_pipeline_predictions(llm_seq, test_df)
        save_predictions(test_df, preds, run_dir, "llm_sequential")
        preds_cache["llm_sequential"] = preds
        metrics_all["llm_sequential"] = evaluate_predictions(test_df, preds, categories, urgency_columns, run_dir, "llm_sequential")
    else:
        logger.info("Skipping LLM sequential (pipelines.llm_sequential=false).")

    lgbm_params_joint = dict(cfg.get("ml", {}).get("lgbm_params", {}))
    lgbm_params_seq = dict(cfg.get("ml", {}).get("lgbm_params", {}))
    grid_cfg = cfg.get("ml", {}).get("grid_search", {})
    apply_to_cv = bool(grid_cfg.get("apply_to_cv", False))
    if grid_cfg.get("enabled"):
        if run_ml_joint:
            lgbm_params_joint = run_lgbm_grid_search(
                "ml_joint",
                train_df,
                val_df,
                cfg,
                categories,
                urgency_columns,
                client,
                logger,
                run_dir,
            )
        if run_ml_sequential:
            lgbm_params_seq = run_lgbm_grid_search(
                "ml_sequential",
                train_df,
                val_df,
                cfg,
                categories,
                urgency_columns,
                client,
                logger,
                run_dir,
            )

    # ML cross-validation (optional)
    cv_folds = cfg["splits"].get("cv_folds", 0)
    if run_ml_cv and cv_folds and cv_folds > 1:
        combined = pd.concat([train_df, val_df], ignore_index=True)
        metrics_all["ml_joint_cv"] = run_ml_cross_val(
            combined,
            "ml_joint",
            cfg,
            categories,
            urgency_columns,
            client,
            lgbm_params=lgbm_params_joint if apply_to_cv else None,
        )
        save_metrics(metrics_all["ml_joint_cv"], run_dir / "metrics_ml_joint_cv.json")
        metrics_all["ml_sequential_cv"] = run_ml_cross_val(
            combined,
            "ml_sequential",
            cfg,
            categories,
            urgency_columns,
            client,
            lgbm_params=lgbm_params_seq if apply_to_cv else None,
        )
        save_metrics(metrics_all["ml_sequential_cv"], run_dir / "metrics_ml_sequential_cv.json")
    elif not run_ml_cv:
        logger.info("Skipping ML CV (pipelines.ml_cv=false).")

    # ML joint
    if run_ml_joint:
        ml_joint = MLJointPipeline(
            client=client,
            embedding_model=cfg["openai"]["embedding_model"],
            categories=categories,
            urgency_columns=urgency_columns,
            ordinal=cfg["ml"].get("ordinal", False),
            calibration_method=cfg["ml"].get("calibration", {}).get("method", "temperature"),
            lgbm_params=lgbm_params_joint,
            lgbm_log_period=cfg["ml"].get("lgbm_log_period", 0),
            category_multilabel=cfg["ml"].get("category_multilabel"),
            category_threshold=cfg["ml"].get("category_threshold", 0.5),
        )
        ml_joint.fit(train_df, val_df)
        preds = run_pipeline_predictions(ml_joint, test_df)
        save_predictions(test_df, preds, run_dir, "ml_joint")
        preds_cache["ml_joint"] = preds
        metrics_all["ml_joint"] = evaluate_predictions(test_df, preds, categories, urgency_columns, run_dir, "ml_joint")
    else:
        logger.info("Skipping ML joint (pipelines.ml_joint=false).")

    # ML sequential
    if run_ml_sequential:
        ml_seq = MLSequentialPipeline(
            client=client,
            embedding_model=cfg["openai"]["embedding_model"],
            categories=categories,
            urgency_columns=urgency_columns,
            option=cfg["ml"].get("sequential_option", "feature_concat"),
            lgbm_params=lgbm_params_seq,
            lgbm_log_period=cfg["ml"].get("lgbm_log_period", 0),
            category_multilabel=cfg["ml"].get("category_multilabel"),
            category_threshold=cfg["ml"].get("category_threshold", 0.5),
        )
        ml_seq.fit(train_df, val_df)
        preds = run_pipeline_predictions(ml_seq, test_df)
        save_predictions(test_df, preds, run_dir, "ml_sequential")
        preds_cache["ml_sequential"] = preds
        metrics_all["ml_sequential"] = evaluate_predictions(test_df, preds, categories, urgency_columns, run_dir, "ml_sequential")
    else:
        logger.info("Skipping ML sequential (pipelines.ml_sequential=false).")

    # Oracle category upper bound
    if run_oracle_llm:
        oracle = OracleCategoryPipeline(
            client=client,
            mode="llm",
            categories=categories,
            model=cfg["openai"]["model"],
            show_progress=show_progress,
        )
        preds = run_pipeline_predictions(oracle, test_df)
        save_predictions(test_df, preds, run_dir, "oracle_llm")
        preds_cache["oracle_llm"] = preds
        metrics_all["oracle_llm"] = evaluate_predictions(test_df, preds, categories, urgency_columns, run_dir, "oracle_llm")
    else:
        logger.info("Skipping oracle LLM (pipelines.oracle_llm=false).")

    # Clustering
    if run_clustering:
        cluster_scope = cfg.get("cluster", {}).get("scope", "test")
        cluster_df = df if cluster_scope == "all" else test_df
        cluster = ClusteringPipeline(
            client=client,
            embedding_model=cfg["openai"]["embedding_model"],
            umap_dim=cfg["cluster"].get("umap_dim", 5),
            min_cluster_size=cfg["cluster"].get("min_cluster_size", 15),
        )
        cluster_out = cluster.predict(cluster_df)
        if not cluster_out.get("skipped"):
            assignments = pd.DataFrame({"msg_id": cluster_df["msg_id"], "cluster_id": cluster_out["cluster_ids"]})
            assignments.to_csv(run_dir / "cluster_assignments.csv", index=False)
            pd.DataFrame(cluster_out["summaries"]).to_csv(run_dir / "cluster_summary.csv", index=False)
            reduced = np.array(cluster_out["reduced"])
            if reduced.shape[1] >= 2:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(reduced[:, 0], reduced[:, 1], c=cluster_out["cluster_ids"], s=8, cmap="tab20")
                ax.set_title("UMAP Clusters")
                fig.tight_layout()
                fig.savefig(run_dir / "figures/umap_clusters.png")
                plt.close(fig)
    else:
        logger.info("Skipping clustering (pipelines.clustering=false).")

    # Elo ranking
    if run_elo:
        elo = EloRankingPipeline(
            client=client,
            judge_model=cfg["openai"]["judge_model"],
            embedding_model=cfg["openai"]["embedding_model"],
            urgency_columns=urgency_columns,
            anchor_per_level=cfg["elo"].get("anchor_per_level", 10),
            max_comparisons=cfg["elo"].get("max_comparisons", 8),
            strategy=cfg["elo"].get("strategy", "binary_search"),
            show_progress=show_progress,
        )
        elo.fit(train_df, val_df)
        elo_preds = elo.predict(test_df)
        ratings = elo_preds["urgency_expected_overall"]
        y_true = compute_overall_urgency(test_df, urgency_columns)
        metrics_all["elo_ranking"] = {"ranking": evaluate_ranking(y_true.tolist(), ratings, k=cfg["report"]["ndcg_k"])}
    else:
        logger.info("Skipping Elo ranking (pipelines.elo_ranking=false).")

    # Policy
    if run_policy:
        policy_records = {
            name: preds_cache[name]
            for name in ["llm_joint", "ml_joint", "ml_sequential"]
            if name in preds_cache
        }
        if "ml_joint" in policy_records and len(policy_records) >= 2:
            decisions = apply_policy(policy_records, cfg["policy"])
            pd.DataFrame(decisions).to_csv(run_dir / "policy_decisions.csv", index=False)
            points = []
            y_true = compute_overall_urgency(test_df, urgency_columns)
            base_preds = policy_records["ml_joint"]
            for t in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                thresholds = dict(cfg["policy"])
                thresholds["t_high"] = t
                thresholds["t_high2"] = max(t, cfg["policy"]["t_high2"])
                decisions_t = apply_policy(policy_records, thresholds)
                auto_idx = [i for i, d in enumerate(decisions_t) if d["decision"] != "DEFER"]
                if auto_idx:
                    pred_labels = [
                        int(np.argmax(base_preds[i].get("urgency_probs_overall", [1 / 6] * 6)))
                        for i in auto_idx
                    ]
                    risk = float(np.mean(np.abs(y_true[auto_idx] - np.array(pred_labels)))) if len(auto_idx) else 0.0
                else:
                    risk = 0.0
                points.append({"coverage": len(auto_idx) / len(test_df), "risk": risk})
            plot_coverage_risk(points, run_dir / "figures/coverage_risk.png")
        else:
            logger.info(
                "Skipping policy: need ml_joint and at least one other pipeline. available=%s",
                list(policy_records.keys()),
            )
    else:
        logger.info("Skipping policy (pipelines.policy=false).")

    # Leaderboard + figures
    build_leaderboard(metrics_all, run_dir / "leaderboard.csv")
    plot_ndcg_comparison(metrics_all, run_dir / "figures/ndcg_comparison.png")

    return run_dir
