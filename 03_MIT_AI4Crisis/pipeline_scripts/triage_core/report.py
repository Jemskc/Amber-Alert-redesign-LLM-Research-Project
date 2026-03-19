from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def build_leaderboard(metrics: Dict[str, Dict[str, Any]], path: Path) -> None:
    rows = []
    for name, m in metrics.items():
        rows.append(
            {
                "pipeline": name,
                "category_macro_f1": m.get("category", {}).get("macro_f1"),
                "urgency_mae": m.get("urgency", {}).get("mae"),
                "urgency_qwk": m.get("urgency", {}).get("qwk"),
                "high_recall": m.get("urgency", {}).get("high_recall"),
                "calibration_ece": m.get("calibration", {}).get("ece"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def plot_coverage_risk(points: List[Dict[str, float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([p["coverage"] for p in points], [p["risk"] for p in points], marker="o")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_ndcg_comparison(metrics: Dict[str, Dict[str, Any]], path: Path) -> None:
    names = []
    values = []
    for name, m in metrics.items():
        if "ranking" in m:
            names.append(name)
            values.append(m["ranking"].get("ndcg@k", 0.0))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, values)
    ax.set_ylabel("NDCG@K")
    ax.set_xticklabels(names, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
