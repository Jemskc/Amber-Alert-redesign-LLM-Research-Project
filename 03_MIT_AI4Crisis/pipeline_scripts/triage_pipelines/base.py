from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


class BasePipeline:
    name: str = "base"

    def fit(self, train_df, val_df=None) -> "BasePipeline":
        return self

    def predict(self, df) -> Dict[str, Any]:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> "BasePipeline":
        raise NotImplementedError
