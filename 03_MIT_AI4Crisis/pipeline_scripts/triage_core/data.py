from __future__ import annotations

import json
from dataclasses import dataclass
import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import hash_text, normalize_text


TEXT_CANDIDATES = ["message", "text", "description", "tweet", "content"]
CATEGORY_TEXT_CANDIDATES = ["category", "categories"]
ID_CANDIDATES = ["message_id", "msg_id", "id", "uid", "#", "number"]
GROUP_CANDIDATES = ["event_id", "incident_id", "crisis_id", "event", "incident", "group"]
CATEGORY_ID_MAP = {
    "1": "Emergency",
    "2": "Vital Lines",
    "3": "Public Health",
    "4": "Security Threats",
    "5": "Infrastructure Damage",
    "6": "Natural Hazards",
    "7": "Services Available",
    "8": "Other",
}


@dataclass
class MatchStats:
    direct_id: int = 0
    direct_text: int = 0
    fuzzy: int = 0
    total: int = 0


def detect_text_column(columns: List[str]) -> str:
    for cand in TEXT_CANDIDATES:
        for col in columns:
            if cand in col.lower():
                return col
    return columns[0]


def detect_id_column(columns: List[str]) -> Optional[str]:
    for cand in ID_CANDIDATES:
        for col in columns:
            if cand == col.lower():
                return col
    return None


def detect_group_column(columns: List[str]) -> Optional[str]:
    for cand in GROUP_CANDIDATES:
        for col in columns:
            if cand == col.lower():
                return col
    return None


def detect_category_text_column(columns: List[str]) -> Optional[str]:
    for cand in CATEGORY_TEXT_CANDIDATES:
        for col in columns:
            if cand == col.lower():
                return col
    return None


def normalize_category_label(raw: str) -> str:
    raw = raw.strip()
    match = re.match(r"^\s*([1-8])", raw)
    if match:
        return CATEGORY_ID_MAP.get(match.group(1), raw)
    if "|" in raw:
        raw = raw.split("|", 1)[1].strip()
    raw = re.sub(r"^[0-9]+[a-z]?\.?\s*", "", raw, flags=re.IGNORECASE)
    for key, value in CATEGORY_ID_MAP.items():
        if value.lower() in raw.lower():
            return value
    return raw


def detect_urgency_columns(df: pd.DataFrame, min_valid_ratio: float = 0.8) -> List[str]:
    named = [col for col in df.columns if re.match(r"^\s*urg\s*\d+\s*$", str(col), flags=re.IGNORECASE)]
    if len(named) >= 8:
        return named[:8]
    numeric_cols = []
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        valid = series.dropna()
        if len(valid) == 0:
            continue
        in_range = valid[(valid >= 0) & (valid <= 5)]
        if len(in_range) / len(valid) >= min_valid_ratio:
            numeric_cols.append(col)
    if len(numeric_cols) >= 8:
        return numeric_cols[:8]
    return numeric_cols


def _dedupe_by_msg_id(df: pd.DataFrame, cols_max: List[str], logger: Optional[Any] = None) -> pd.DataFrame:
    if "msg_id" not in df.columns:
        return df
    dup_count = int(df["msg_id"].duplicated().sum())
    if dup_count == 0:
        return df
    if logger:
        logger.warning("Found %d duplicate msg_id rows; collapsing by max for %s.", dup_count, cols_max)
    agg: Dict[str, Any] = {}
    for col in df.columns:
        if col in cols_max:
            agg[col] = "max"
        else:
            agg[col] = "first"
    return df.groupby("msg_id", as_index=False).agg(agg)


def load_urgency_dataset(
    path: str,
    sheet_name: Optional[str] = None,
    text_column: Optional[str] = None,
    urgency_columns: Optional[List[str]] = None,
    group_column: Optional[str] = None,
    category_text_column: Optional[str] = None,
) -> pd.DataFrame:
    if sheet_name is None:
        try:
            xls = pd.ExcelFile(path)
            if "Final Merged" in xls.sheet_names:
                sheet_name = "Final Merged"
            elif "Urgency" in xls.sheet_names:
                sheet_name = "Urgency"
            else:
                sheet_name = 0
        except Exception:
            sheet_name = 0
    df = pd.read_excel(path, sheet_name=sheet_name)
    if isinstance(df, dict):
        df = next(iter(df.values()))
    if text_column is None and "INCIDENT TITLE" in df.columns and "DESCRIPTION" in df.columns:
        df["__combined_text__"] = df["INCIDENT TITLE"].astype(str) + " " + df["DESCRIPTION"].astype(str)
        text_col = "__combined_text__"
    else:
        text_col = text_column or detect_text_column(df.columns.tolist())
    id_col = detect_id_column(df.columns.tolist())
    group_col = group_column or detect_group_column(df.columns.tolist())
    urgency_cols = urgency_columns or detect_urgency_columns(df)
    category_col = category_text_column or detect_category_text_column(df.columns.tolist())

    df = df.copy()
    df["text"] = df[text_col].astype(str)
    if id_col and id_col in df.columns:
        df["msg_id"] = df[id_col].astype(str)
    else:
        df["msg_id"] = [hash_text(normalize_text(t)) for t in df["text"].tolist()]

    if group_col and group_col in df.columns:
        df["group_id"] = df[group_col].astype(str)
    else:
        df["group_id"] = df["msg_id"].astype(str)

    for col in urgency_cols:
        df[col] = df[col].fillna(0)
        df[col] = df[col].astype(int)
        df[col] = df[col].clip(0, 5)

    if category_col and category_col in df.columns:
        categories_raw = df[category_col].fillna("").astype(str)
        category_lists = []
        for text in categories_raw:
            items = [c.strip() for c in text.split(",") if c.strip()]
            normalized = []
            for item in items:
                label = normalize_category_label(item)
                if label and label not in normalized:
                    normalized.append(label)
            category_lists.append(normalized)
        category_primary = [lst[0] if lst else None for lst in category_lists]
        category_probs = []
        for lst in category_lists:
            if not lst:
                category_probs.append({})
                continue
            prob = 1.0 / len(lst)
            category_probs.append({c: prob for c in lst})
        df["category_hint"] = category_primary
        df["category_hint_probs"] = [json.dumps(p, ensure_ascii=True) for p in category_probs]
        df["category_list"] = [json.dumps(lst, ensure_ascii=True) for lst in category_lists]
        df["category_multi"] = [len(lst) > 1 for lst in category_lists]

    df["has_urgency"] = bool(urgency_cols)
    # Note: No deduplication by msg_id - every message is processed individually.
    # Messages with the same group_id will be kept together in train/val/test splits.
    keep_cols = ["msg_id", "text", "group_id"] + urgency_cols + ["has_urgency"]
    if "category_hint" in df.columns:
        keep_cols += ["category_hint", "category_hint_probs", "category_list", "category_multi"]
    return df[keep_cols]


def _extract_labelers(columns: List[str], labeler_hint: Optional[List[str]]) -> List[str]:
    if labeler_hint:
        return [c for c in labeler_hint if c in columns]
    return []


def _infer_labeler_columns(columns: List[str], keyword: str) -> List[str]:
    matches = []
    for col in columns:
        if keyword in col.lower():
            matches.append(col)
    return matches


def _majority_vote(labels: List[Any]) -> Tuple[Optional[str], Dict[str, float]]:
    labels = [l for l in labels if isinstance(l, str) and l.strip()]
    if not labels:
        return None, {}
    counts: Dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    total = sum(counts.values())
    probs = {k: v / total for k, v in counts.items()}
    best = max(counts, key=counts.get)
    return best, probs


def _weighted_vote(labels: List[Any], weights: List[float]) -> Tuple[Optional[str], Dict[str, float]]:
    scores: Dict[str, float] = {}
    for label, weight in zip(labels, weights):
        if not isinstance(label, str) or not label.strip():
            continue
        scores[label] = scores.get(label, 0.0) + weight
    if not scores:
        return None, {}
    total = sum(scores.values())
    probs = {k: v / total for k, v in scores.items()}
    best = max(scores, key=scores.get)
    return best, probs


def load_categories_dataset(
    path: str,
    sheet_name: Optional[str] = None,
    text_column: Optional[str] = None,
    category_columns: Optional[List[str]] = None,
    rn_columns: Optional[List[str]] = None,
    responder_columns: Optional[List[str]] = None,
    crowd_columns: Optional[List[str]] = None,
    strategy: str = "majority",
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    if sheet_name is None:
        try:
            xls = pd.ExcelFile(path)
            if "Final Merged" in xls.sheet_names:
                sheet_name = "Final Merged"
            else:
                sheet_name = 0
        except Exception:
            sheet_name = 0
    df = pd.read_excel(path, sheet_name=sheet_name)
    if isinstance(df, dict):
        df = next(iter(df.values()))
    text_col = text_column or detect_text_column(df.columns.tolist())
    id_col = detect_id_column(df.columns.tolist())

    df = df.copy()
    df["text"] = df[text_col].astype(str) if text_col in df.columns else ""
    if id_col and id_col in df.columns:
        df["msg_id"] = df[id_col].astype(str)
    else:
        df["msg_id"] = [hash_text(normalize_text(t)) for t in df["text"].tolist()]

    obj_cols: List[str] = []
    if category_columns is None:
        rn_cols = _extract_labelers(df.columns.tolist(), rn_columns) or _infer_labeler_columns(df.columns.tolist(), "rn")
        responder_cols = _extract_labelers(df.columns.tolist(), responder_columns) or _infer_labeler_columns(
            df.columns.tolist(), "responder"
        )
        crowd_cols = _extract_labelers(df.columns.tolist(), crowd_columns) or _infer_labeler_columns(
            df.columns.tolist(), "crowd"
        )
        if strategy == "rn_only" and rn_cols:
            category_columns = rn_cols
        else:
            category_columns = rn_cols + responder_cols + crowd_cols

        if not category_columns:
            keywords = ["category", "label", "class", "topic", "type"]
            for col in df.columns.tolist():
                col_l = col.lower()
                if col in {text_col, id_col}:
                    continue
                if any(k in col_l for k in keywords):
                    category_columns.append(col)

        if not category_columns:
            obj_cols = [
                col
                for col in df.columns.tolist()
                if col not in {text_col, id_col} and df[col].dtype == object
            ]
            category_columns = [col for col in obj_cols if df[col].nunique(dropna=True) <= 50]
    else:
        rn_cols = rn_columns or []
        responder_cols = responder_columns or []
        crowd_cols = crowd_columns or []

    category_names = category_columns
    mapped_names = []
    for col in category_columns:
        match = re.match(r"^c([1-8])_rn$", col.strip().lower())
        if match:
            mapped_names.append(CATEGORY_ID_MAP.get(match.group(1), col))
        else:
            mapped_names.append(col)
    if mapped_names and len(mapped_names) == len(category_columns):
        category_names = mapped_names

    numeric_cols = []
    for col in category_columns:
        series = pd.to_numeric(df[col], errors="coerce")
        valid_ratio = series.notna().mean()
        if valid_ratio >= 0.9:
            df[col] = series.fillna(0)
            numeric_cols.append(col)
    # Note: No deduplication by msg_id - every message is processed individually.

    gold_labels = []
    gold_probs = []
    gold_lists = []
    for _, row in df.iterrows():
        labels = [row[col] for col in category_columns if col in df.columns]
        if labels and all(isinstance(v, (int, float, np.number)) for v in labels):
            values = np.array([float(v) for v in labels], dtype=float)
            total = float(values.sum())
            if total > 0:
                active = [name for name, val in zip(category_names, values) if val > 0]
                label = active[0] if active else None
                probs = {col: float(val / total) for col, val in zip(category_names, values) if val > 0}
            else:
                label, probs = None, {}
        else:
            if strategy == "rn_only" and rn_cols:
                labels = [row[col] for col in rn_cols if col in df.columns]
                label, probs = _majority_vote(labels)
            elif strategy == "weighted" and (rn_cols or responder_cols or crowd_cols):
                weights_map = weights or {"rn": 2.0, "responder": 1.5, "crowd": 1.0}
                combined = []
                weight_list = []
                for col in rn_cols:
                    combined.append(row.get(col, None))
                    weight_list.append(weights_map.get("rn", 2.0))
                for col in responder_cols:
                    combined.append(row.get(col, None))
                    weight_list.append(weights_map.get("responder", 1.5))
                for col in crowd_cols:
                    combined.append(row.get(col, None))
                    weight_list.append(weights_map.get("crowd", 1.0))
                label, probs = _weighted_vote(combined, weight_list)
            else:
                label, probs = _majority_vote(labels)
        gold_labels.append(label)
        gold_probs.append(json.dumps(probs, ensure_ascii=True))
        if labels and all(isinstance(v, (int, float, np.number)) for v in labels):
            active = [name for name, val in zip(category_names, values) if val > 0]
            gold_lists.append(json.dumps(active, ensure_ascii=True))
        elif label:
            gold_lists.append(json.dumps([label], ensure_ascii=True))
        else:
            gold_lists.append(json.dumps([], ensure_ascii=True))

    df["category_gold"] = gold_labels
    df["category_prob_gold"] = gold_probs
    df["category_list"] = gold_lists
    df["category_multi"] = df["category_list"].apply(lambda x: len(json.loads(x)) > 1 if isinstance(x, str) else False)
    df["has_category"] = df["category_gold"].notna()
    keep_cols = ["msg_id", "text", "category_gold", "category_prob_gold", "category_list", "category_multi", "has_category"]
    return df[keep_cols]


def _fuzzy_match(needle: str, haystack: List[str], threshold: float) -> Optional[str]:
    best = None
    best_score = 0.0
    for candidate in haystack:
        score = SequenceMatcher(None, needle, candidate).ratio()
        if score > best_score:
            best_score = score
            best = candidate
    if best_score >= threshold:
        return best
    return None


def aggregate_to_threads(
    df: pd.DataFrame,
    group_column: str,
    urgency_columns: List[str],
) -> pd.DataFrame:
    """Aggregate post-level data to thread-level by grouping on group_column.

    For each thread (group):
    - text: concatenate all posts with newline separator
    - urgency: take max per dimension across posts
    - categories: union of all category labels; probability = max per category
    - post_count: number of posts in thread

    Args:
        df: Post-level DataFrame with group_column, text, urgency, and category columns.
        group_column: Column name to group by (e.g. "group_id").
        urgency_columns: List of urgency dimension column names.

    Returns:
        Thread-level DataFrame with one row per group.
    """
    if group_column not in df.columns:
        return df

    # Deduplicate rows within each group (the outer merge in merge_datasets
    # can create N*N rows for a thread with N posts).  We keep one row per
    # unique text within each group so post_count reflects actual posts.
    dedup_cols = [group_column, "text"]
    df = df.drop_duplicates(subset=dedup_cols, keep="first")

    agg: Dict[str, Any] = {}
    agg["text"] = lambda x: "\n---\n".join(str(v) for v in x if pd.notna(v))
    agg["msg_id"] = "first"

    for col in urgency_columns:
        if col in df.columns:
            agg[col] = "max"

    if "has_urgency" in df.columns:
        agg["has_urgency"] = "any"
    if "has_category" in df.columns:
        agg["has_category"] = "any"

    if "category_gold" in df.columns:
        agg["category_gold"] = "first"
    if "category_prob_gold" in df.columns:
        def _merge_probs(series):
            merged: Dict[str, float] = {}
            for raw in series.dropna():
                try:
                    probs = json.loads(raw) if isinstance(raw, str) else raw
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(probs, dict):
                    for k, v in probs.items():
                        merged[k] = max(merged.get(k, 0.0), float(v))
            return json.dumps(merged, ensure_ascii=True)
        agg["category_prob_gold"] = _merge_probs

    if "category_list" in df.columns:
        def _merge_cat_lists(series):
            cats: List[str] = []
            for raw in series.dropna():
                try:
                    lst = json.loads(raw) if isinstance(raw, str) else raw
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(lst, list):
                    for item in lst:
                        if item and item not in cats:
                            cats.append(item)
            return json.dumps(cats, ensure_ascii=True)
        agg["category_list"] = _merge_cat_lists

    if "category_multi" in df.columns:
        agg["category_multi"] = "any"

    grouped = df.groupby(group_column, as_index=False).agg(agg)
    grouped["post_count"] = df.groupby(group_column)["text"].nunique().clip(lower=1).values
    grouped["group_id"] = grouped[group_column].astype(str)

    return grouped


def merge_datasets(
    urgency_df: pd.DataFrame,
    category_df: pd.DataFrame,
    fuzzy: bool = True,
    fuzzy_threshold: float = 0.92,
) -> Tuple[pd.DataFrame, MatchStats]:
    stats = MatchStats(total=max(len(urgency_df), len(category_df)))

    merged = urgency_df.merge(category_df, on="msg_id", how="outer", suffixes=("_urg", "_cat"))
    if "text_urg" in merged.columns or "text_cat" in merged.columns:
        merged["text"] = merged["text_urg"].combine_first(merged["text_cat"])
    merged["has_urgency"] = merged.get("has_urgency", False).fillna(False).infer_objects(copy=False).astype(bool)
    merged["has_category"] = merged.get("has_category", False).fillna(False).infer_objects(copy=False).astype(bool)
    if "group_id" in merged.columns:
        merged["group_id"] = merged["group_id"].fillna(merged["msg_id"].astype(str))

    if "category_gold" in merged.columns:
        stats.direct_id = merged["category_gold"].notna().sum()

    # Fill missing category by text match
    cat_lookup = {}
    for t, row in zip(category_df["text"], category_df.to_dict("records")):
        norm = normalize_text(t)
        if not norm:
            continue
        cat_lookup[norm] = row
    if "category_hint" in merged.columns:
        hint_mask = merged["category_gold"].isna() & merged["category_hint"].notna()
        merged.loc[hint_mask, "category_gold"] = merged.loc[hint_mask, "category_hint"]
        merged.loc[hint_mask, "category_prob_gold"] = merged.loc[hint_mask, "category_hint_probs"]
        merged.loc[hint_mask, "has_category"] = True
        merged = merged.drop(columns=[c for c in ["category_hint", "category_hint_probs"] if c in merged.columns])

    missing_cat = merged[merged["category_gold"].isna() & merged["text"].notna()].copy()
    for idx, text in zip(missing_cat.index, missing_cat["text"].tolist()):
        norm = normalize_text(text)
        if not norm:
            continue
        if norm in cat_lookup:
            match = cat_lookup[norm]
            stats.direct_text += 1
        elif fuzzy:
            key = _fuzzy_match(norm, list(cat_lookup.keys()), threshold=fuzzy_threshold)
            match = cat_lookup.get(key) if key else None
            if match:
                stats.fuzzy += 1
        else:
            match = None
        if match is not None:
            merged.loc[idx, "category_gold"] = match.get("category_gold")
            merged.loc[idx, "category_prob_gold"] = match.get("category_prob_gold")
            merged.loc[idx, "has_category"] = True

    merged["has_category"] = merged["category_gold"].notna()
    merged["has_urgency"] = merged["has_urgency"].astype(bool)
    # Normalize category list columns after merge so pipelines can detect multilabel data.
    if "category_list" not in merged.columns:
        if "category_list_cat" in merged.columns:
            merged["category_list"] = merged["category_list_cat"]
        elif "category_list_urg" in merged.columns:
            merged["category_list"] = merged["category_list_urg"]
    if "category_multi" not in merged.columns:
        if "category_multi_cat" in merged.columns:
            merged["category_multi"] = merged["category_multi_cat"]
        elif "category_multi_urg" in merged.columns:
            merged["category_multi"] = merged["category_multi_urg"]
    merged = merged.drop(columns=[c for c in ["text_urg", "text_cat"] if c in merged.columns])
    return merged, stats
