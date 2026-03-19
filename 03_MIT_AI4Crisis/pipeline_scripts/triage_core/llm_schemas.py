from __future__ import annotations

from typing import Dict, List, Optional


def category_schema(categories: List[str]) -> Dict:
    return {
        "name": "category_output",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "category_rationale": {"type": "string"},
                "category_top": {"type": "string"},
                "category_probs": {
                    "type": "object",
                    "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "entropy_category": {"type": "number"},
                "self_report_uncertainty": {"type": "number"},
            },
            "required": [
                "category_rationale",
                "category_top",
                "category_probs",
                "entropy_category",
                "self_report_uncertainty",
            ],
        },
    }


def _urgency_by_category_schema(categories: List[str]) -> Dict:
    """Point estimates (0-5) per category."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {cat: {"type": "number", "minimum": 0, "maximum": 5} for cat in categories},
        "required": categories,
    }


def _urgency_probs_by_category_schema(categories: List[str]) -> Dict:
    """Full probability distribution [P(0), P(1), ..., P(5)] per category."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            cat: {
                "type": "array",
                "items": {"type": "number", "minimum": 0, "maximum": 1},
                "minItems": 6,
                "maxItems": 6,
            }
            for cat in categories
        },
        "required": categories,
    }


def urgency_schema(categories: Optional[List[str]] = None, include_vector: bool = True) -> Dict:
    props = {
        "urgency_rationale": {"type": "string"},
        "urgency_probs_overall": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 6,
            "maxItems": 6,
        },
        "urgency_expected_overall": {"type": "number"},
        "urgency_probs_high": {"type": "number"},
        "entropy_urgency": {"type": "number"},
        "self_report_uncertainty": {"type": "number"},
    }
    required = [
        "urgency_rationale",
        "urgency_probs_overall",
        "urgency_expected_overall",
        "urgency_probs_high",
        "entropy_urgency",
        "self_report_uncertainty",
    ]
    if categories:
        props["urgency_by_category"] = _urgency_by_category_schema(categories)
        props["urgency_probs_by_category"] = _urgency_probs_by_category_schema(categories)
        required.append("urgency_by_category")
        required.append("urgency_probs_by_category")
    if include_vector:
        props["urgency_vector_8"] = {
            "type": "object",
            "additionalProperties": {"type": "integer"},
        }
        required.append("urgency_vector_8")

    return {
        "name": "urgency_output",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": props,
            "required": required,
        },
    }


def joint_schema(categories: List[str], include_vector: bool = True) -> Dict:
    return {
        "name": "joint_output",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "category_rationale": {"type": "string"},
                "category_top": {"type": "string"},
                "category_probs": {
                    "type": "object",
                    "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "entropy_category": {"type": "number"},
                "urgency_rationale": {"type": "string"},
                "urgency_by_category": _urgency_by_category_schema(categories),
                "urgency_probs_by_category": _urgency_probs_by_category_schema(categories),
                "urgency_probs_overall": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 6,
                    "maxItems": 6,
                },
                "urgency_expected_overall": {"type": "number"},
                "urgency_probs_high": {"type": "number"},
                "entropy_urgency": {"type": "number"},
                "urgency_vector_8": {
                    "type": "object",
                    "additionalProperties": {"type": "integer"},
                },
                "self_report_uncertainty": {"type": "number"},
            },
            "required": [
                "category_rationale",
                "category_top",
                "category_probs",
                "entropy_category",
                "urgency_rationale",
                "urgency_by_category",
                "urgency_probs_by_category",
                "urgency_probs_overall",
                "urgency_expected_overall",
                "urgency_probs_high",
                "entropy_urgency",
                "self_report_uncertainty",
            ]
            + (["urgency_vector_8"] if include_vector else []),
        },
    }


def judge_schema() -> Dict:
    return {
        "name": "judge_output",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "winner": {"type": "string", "enum": ["A", "B", "TIE"]},
                "confidence": {"type": "number"},
                "rationale": {"type": "string"},
            },
            "required": ["winner", "confidence"],
        },
    }
