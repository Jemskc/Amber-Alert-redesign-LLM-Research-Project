from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from .config import load_config, resolve_path
from .data import aggregate_to_threads, load_categories_dataset, load_urgency_dataset, merge_datasets
from .openai_client import OpenAIClient
from .split import train_val_test_split
from .utils import configure_logger, set_seed


def load_data_and_split(config_path: str) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, OpenAIClient]:
    cfg = load_config(config_path)
    logger = configure_logger()
    set_seed(cfg.get("seed", 13))

    urgency_df = load_urgency_dataset(
        cfg["data"]["urgency_xlsx"],
        sheet_name=cfg["data"].get("urgency_sheet"),
        text_column=cfg["data"].get("text_column"),
        urgency_columns=cfg["data"].get("urgency_columns"),
        group_column=cfg["data"].get("group_column"),
    )
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
        urgency_cols = cfg["data"].get("urgency_columns") or [
            c for c in merged.columns if c.lower().startswith("urg")
        ]
        pre_count = len(merged)
        merged = aggregate_to_threads(merged, "group_id", urgency_cols)
        logger.info(
            "Thread aggregation: %d posts -> %d threads",
            pre_count,
            len(merged),
        )

    train_df, val_df, test_df = train_val_test_split(
        merged,
        test_size=cfg["splits"]["test_size"],
        val_size=cfg["splits"]["val_size"],
        group_column=cfg["splits"].get("group_column"),
        stratify_column=cfg["splits"].get("stratify_column"),
        seed=cfg.get("seed", 13),
    )

    show_progress = cfg["openai"].get("progress", True)
    cache_path = resolve_path(cfg["openai"]["cache_path"], base_dir=Path.cwd())
    client = OpenAIClient(
        api_key=None,
        cache_path=cache_path,
        max_retries=cfg["openai"]["max_retries"],
        dry_run=cfg["openai"].get("dry_run", False),
        show_progress=show_progress,
        embedding_provider=cfg["openai"].get("embedding_provider", "openai"),
        llm_provider=cfg["openai"].get("llm_provider", "openai"),
    )

    return cfg, train_df, val_df, test_df, client
