from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit


def train_val_test_split(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    group_column: str | None = None,
    stratify_column: str | None = None,
    seed: int = 13,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if group_column and group_column in df.columns:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(df, groups=df[group_column]))
    else:
        stratify = df[stratify_column] if stratify_column and stratify_column in df.columns else None
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(df, stratify))

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    if group_column and group_column in df.columns:
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        tr_idx, val_idx = next(splitter.split(train_df, groups=train_df[group_column]))
    else:
        stratify = train_df[stratify_column] if stratify_column and stratify_column in train_df.columns else None
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        tr_idx, val_idx = next(splitter.split(train_df, stratify))

    return train_df.iloc[tr_idx].copy(), train_df.iloc[val_idx].copy(), test_df
