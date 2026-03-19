from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.metrics import cohen_kappa_score


def evaluate_category_auc(
    y_true: List[List[str]],
    y_pred_probs: List[Dict[str, float]],
    labels: List[str],
) -> Dict[str, Any]:
    """
    Compute per-category AUC scores for multi-label classification.

    Args:
        y_true: List of true category sets per sample
        y_pred_probs: List of predicted probability dicts per sample
        labels: List of all category labels

    Returns:
        Dict with per-category AUC and macro/micro averages
    """
    if not labels or not y_true or not y_pred_probs:
        return {}

    n_samples = len(y_true)
    n_labels = len(labels)

    # Build binary matrices
    y_true_bin = np.zeros((n_samples, n_labels), dtype=int)
    y_pred_scores = np.zeros((n_samples, n_labels), dtype=float)

    for i, (true_set, probs) in enumerate(zip(y_true, y_pred_probs)):
        for j, label in enumerate(labels):
            y_true_bin[i, j] = 1 if label in true_set else 0
            y_pred_scores[i, j] = float(probs.get(label, 0.0))

    # Per-category AUC
    per_category_auc = {}
    valid_aucs = []
    for j, label in enumerate(labels):
        y_true_col = y_true_bin[:, j]
        y_pred_col = y_pred_scores[:, j]
        # Need both classes present for AUC
        if len(np.unique(y_true_col)) > 1:
            auc = roc_auc_score(y_true_col, y_pred_col)
            per_category_auc[label] = float(auc)
            valid_aucs.append(auc)
        else:
            per_category_auc[label] = None  # Can't compute AUC

    result = {"per_category": per_category_auc}

    # Macro AUC (average of per-category AUCs)
    if valid_aucs:
        result["macro_auc"] = float(np.mean(valid_aucs))

    # Micro AUC (flatten all predictions)
    y_true_flat = y_true_bin.ravel()
    y_pred_flat = y_pred_scores.ravel()
    if len(np.unique(y_true_flat)) > 1:
        result["micro_auc"] = float(roc_auc_score(y_true_flat, y_pred_flat))

    return result


def evaluate_urgency_per_category(
    df,
    preds: List[Dict[str, Any]],
    urgency_columns: List[str],
    category_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute per-category urgency metrics (MAE, AUC for high urgency).

    Args:
        df: DataFrame with ground truth urgency columns
        preds: List of prediction dicts with urgency_by_category or urgency_vector_8
        urgency_columns: List of urgency column names (Urg 1, Urg 2, ...)
        category_names: List of category names corresponding to urgency columns
            (e.g. ["Emergency", "Vital Lines", ...]). When provided, used for
            direct name-based lookup in urgency_by_category predictions.

    Returns:
        Dict with per-category urgency metrics
    """
    results = {}
    n = min(len(df), len(preds))

    for i, col in enumerate(urgency_columns):
        if col not in df.columns:
            continue

        y_true = df[col].iloc[:n].fillna(0).astype(int).values
        y_pred = []
        y_pred_probs_high = []

        # Determine the category name for this urgency column
        cat_name = category_names[i] if category_names and i < len(category_names) else None

        for j in range(n):
            pred = preds[j]
            # Try urgency_by_category first
            urg_by_cat = pred.get("urgency_by_category", {})
            urg_probs_by_cat = pred.get("urgency_probs_by_category", {})

            pred_val = 0
            prob_high = 0.0

            # Try category name match first (most reliable), then column name, then dim_N
            if cat_name and cat_name in urg_by_cat:
                pred_val = int(urg_by_cat[cat_name])
            elif col in urg_by_cat:
                pred_val = int(urg_by_cat[col])
            elif f"dim_{i+1}" in urg_by_cat:
                pred_val = int(urg_by_cat[f"dim_{i+1}"])

            # Get probability of high urgency from per-category distributions
            if cat_name and cat_name in urg_probs_by_cat:
                probs = urg_probs_by_cat[cat_name]
                if len(probs) == 6:
                    prob_high = probs[4] + probs[5]
            elif col in urg_probs_by_cat:
                probs = urg_probs_by_cat[col]
                if len(probs) == 6:
                    prob_high = probs[4] + probs[5]
            elif f"dim_{i+1}" in urg_probs_by_cat:
                probs = urg_probs_by_cat[f"dim_{i+1}"]
                if len(probs) == 6:
                    prob_high = probs[4] + probs[5]

            y_pred.append(pred_val)
            y_pred_probs_high.append(prob_high)

        y_pred = np.array(y_pred)
        y_pred_probs_high = np.array(y_pred_probs_high)

        # Compute metrics
        cat_metrics = {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "n_samples": int((y_true > 0).sum()),  # Non-zero samples
        }

        # AUC for high urgency (>=4) if we have both classes
        y_high = (y_true >= 4).astype(int)
        if len(np.unique(y_high)) > 1 and y_pred_probs_high.sum() > 0:
            cat_metrics["high_auc"] = float(roc_auc_score(y_high, y_pred_probs_high))

        results[col] = cat_metrics

    return results


def evaluate_category(
    y_true: List[str] | List[List[str]],
    y_pred: List[str] | List[List[str]],
    y_pred_probs: Optional[List[Dict[str, float]]] = None,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if y_true and isinstance(y_true[0], (list, tuple, set)):
        label_set = set(labels or [])
        label_set.update({label for item in y_true for label in item})
        for pred in y_pred:
            if isinstance(pred, (list, tuple, set)):
                label_set.update(pred)
            elif pred:
                label_set.add(pred)
        if y_pred_probs:
            for probs in y_pred_probs:
                if isinstance(probs, dict):
                    label_set.update(probs.keys())
        labels = sorted(label_set)

        y_true_bin = [[1 if label in item else 0 for label in labels] for item in y_true]
        pred_is_list = any(isinstance(pred, (list, tuple, set)) for pred in y_pred)
        if pred_is_list:
            y_pred_bin = [[1 if label in (pred or []) else 0 for label in labels] for pred in y_pred]
            exact = float(np.mean([set(pred or []) == set(true) for pred, true in zip(y_pred, y_true)]))
            jaccard = []
            for pred, true in zip(y_pred, y_true):
                p = set(pred or [])
                t = set(true)
                denom = len(p | t)
                jaccard.append(len(p & t) / denom if denom else 0.0)
            metrics = {
                "accuracy": exact,
                "macro_f1": float(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)),
                "micro_f1": float(f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)),
                "jaccard": float(np.mean(jaccard)) if jaccard else 0.0,
            }
        else:
            y_pred_bin = [[1 if label == pred else 0 for label in labels] for pred in y_pred]
            hit_rate = float(np.mean([pred in item for pred, item in zip(y_pred, y_true)]))
            metrics = {
                "accuracy": hit_rate,
                "macro_f1": float(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)),
                "micro_f1": float(f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)),
            }
        return metrics
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    return metrics


def evaluate_urgency(y_true: List[int], y_pred: List[int], p_high: List[float]) -> Dict[str, Any]:
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
    }
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_high = (y_true_arr >= 4).astype(int)
    p_high = np.asarray(p_high)
    precision, recall, f1, _ = precision_recall_fscore_support(y_high, p_high >= 0.5, average="binary", zero_division=0)
    metrics.update(
        {
            "high_precision": float(precision),
            "high_recall": float(recall),
            "high_f1": float(f1),
        }
    )
    if len(np.unique(y_high)) > 1:
        metrics["high_auc"] = float(roc_auc_score(y_high, p_high))

    # Add boundary crossing metrics (community vs individual urgency)
    boundary_metrics = boundary_crossing_error(y_true, y_pred)
    metrics.update(boundary_metrics)

    return metrics


def boundary_crossing_error(y_true: List[int], y_pred: List[int], boundary: float = 3.5) -> Dict[str, float]:
    """
    Compute metrics that penalize crossing the community (>=4) vs individual (<4) boundary.

    In crisis triage, urgency levels 4-5 indicate community-level emergencies while 1-3
    indicate individual-level needs. Misclassifying across this boundary is more costly
    than errors within the same category.

    Args:
        y_true: Ground truth urgency scores (0-5)
        y_pred: Predicted urgency scores (0-5)
        boundary: The threshold separating individual (<boundary) from community (>=boundary)

    Returns:
        Dict with:
        - boundary_crossing_rate: Fraction of predictions that cross the boundary
        - weighted_mae: MAE with 2x penalty for boundary crossings
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_community = y_true >= boundary
    pred_community = y_pred >= boundary

    # Count boundary crossings
    crossings = true_community != pred_community
    crossing_rate = float(crossings.mean()) if len(crossings) > 0 else 0.0

    # Weighted MAE: double penalty for boundary crossings
    mae_base = np.abs(y_true - y_pred).astype(float)
    mae_weighted = mae_base.copy()
    mae_weighted[crossings] *= 2.0

    return {
        "boundary_crossing_rate": crossing_rate,
        "weighted_mae": float(mae_weighted.mean()) if len(mae_weighted) > 0 else 0.0,
    }


def cost_sensitive_error(y_true: List[int], y_pred: List[int], cost_matrix: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cost = 0.0
    for t, p in zip(y_true, y_pred):
        cost += cost_matrix[t, p]
    return float(cost / len(y_true))


def default_cost_matrix(size: int = 6, under_penalty: float = 2.0, over_penalty: float = 1.0) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=float)
    for i in range(size):
        for j in range(size):
            if i == j:
                matrix[i, j] = 0.0
            elif j < i:
                matrix[i, j] = (i - j) * under_penalty
            else:
                matrix[i, j] = (j - i) * over_penalty
    return matrix


def evaluate_ranking(y_true: List[int], scores: List[float], k: int = 50) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    order = np.argsort(scores)[::-1]
    gains = y_true[order]
    ndcg = ndcg_at_k(gains, k)
    recall_k = float((gains[:k] >= 4).sum() / max((y_true >= 4).sum(), 1))
    spearman = spearmanr(y_true, scores).correlation
    kendall = kendalltau(y_true, scores).correlation
    return {
        "ndcg@k": float(ndcg),
        "recall@k": float(recall_k),
        "spearman": float(spearman if spearman is not None else 0.0),
        "kendall": float(kendall if kendall is not None else 0.0),
    }


def ndcg_at_k(gains: np.ndarray, k: int) -> float:
    gains = gains[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float((gains * discounts).sum())
    ideal = np.sort(gains)[::-1]
    idcg = float((ideal * discounts).sum())
    return dcg / idcg if idcg > 0 else 0.0


def reliability_curve(y_true: List[int], p_high: List[float], n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    p_high = np.asarray(p_high)
    frac_pos, mean_pred = calibration_curve(y_true, p_high, n_bins=n_bins, strategy="uniform")
    return mean_pred, frac_pos


def ece_mce(y_true: List[int], p_high: List[float], n_bins: int = 10) -> Tuple[float, float]:
    y_true = np.asarray(y_true)
    p_high = np.asarray(p_high)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        mask = (p_high >= bins[i]) & (p_high < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = p_high[mask].mean()
        gap = abs(acc - conf)
        ece += gap * (mask.sum() / len(y_true))
        mce = max(mce, gap)
    return float(ece), float(mce)


def plot_confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str], path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_reliability(mean_pred: np.ndarray, frac_pos: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(mean_pred, frac_pos, marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Mean predicted")
    ax.set_ylabel("Fraction positive")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_metrics(metrics: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=True, indent=2)
