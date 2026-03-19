from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def _extract_p_high(record: Dict[str, Any]) -> float:
    if "urgency_probs_high_calibrated" in record:
        return float(record["urgency_probs_high_calibrated"])
    if "urgency_probs_high" in record:
        return float(record["urgency_probs_high"])
    probs = record.get("urgency_probs_overall")
    if probs:
        return float(probs[4] + probs[5])
    return 0.0


def _extract_expected(record: Dict[str, Any]) -> float:
    if "urgency_expected_overall" in record:
        return float(record["urgency_expected_overall"])
    probs = record.get("urgency_probs_overall")
    if probs:
        return float(sum(i * p for i, p in enumerate(probs)))
    return 0.0


def apply_policy(records: Dict[str, List[Dict[str, Any]]], thresholds: Dict[str, float]) -> List[Dict[str, Any]]:
    decisions = []
    model_names = list(records.keys())
    num_items = len(next(iter(records.values()))) if records else 0

    for idx in range(num_items):
        model_outputs = [records[name][idx] for name in model_names]
        p_highs = np.array([_extract_p_high(r) for r in model_outputs])
        expected = np.array([_extract_expected(r) for r in model_outputs])
        max_p_high = float(p_highs.max()) if p_highs.size else 0.0
        median_expected = float(np.median(expected)) if expected.size else 0.0
        disagreement = float(expected.max() - expected.min()) if expected.size else 0.0

        if max_p_high > thresholds["t_high"] and median_expected <= 2:
            decision = "DEFER"
        elif np.all(expected <= 2) and np.all(p_highs <= 1 - thresholds["t_low"]) and disagreement <= thresholds["max_disagreement"]:
            decision = "AUTO"
        elif max_p_high > thresholds["t_high2"] and disagreement <= thresholds["max_disagreement"]:
            decision = "AUTO_HIGH_AUDIT"
        else:
            decision = "DEFER"

        decisions.append(
            {
                "decision": decision,
                "max_p_high": max_p_high,
                "median_expected": median_expected,
                "disagreement": disagreement,
            }
        )

    return decisions
