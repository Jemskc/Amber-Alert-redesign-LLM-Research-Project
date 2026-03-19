Date: 2026-02-24
Prepared by: Codex (for Jems KC)

Subject: Error and code-change note for local LLM fallback issues

Summary
- Observed fallback error pattern in DeepSeek run output:
  - `'>=' not supported between instances of 'dict' and 'float'`
- Count observed in run `outputs/triage_outputs/20260223_215030_llm_joint`:
  - Test fallback rows: 15
  - Train fallback rows: 22
  - Total fallback rows: 37

Root cause for this specific error
- In `llm_joint.py`, `category_probs` values were clipped directly with `np.clip(...)`.
- Some local LLM responses (DeepSeek/Qwen via HF backend) returned dict-shaped values instead of scalar floats.
- This triggered a type comparison failure during clipping.

Jems KC-tagged code changes found and documented

1) `triage_system/src/triage/pipelines/llm_joint.py`
- `Changed on 2026-02-23 by Jems KC` (existing)
  - Per-category urgency probability vector hardening:
    - Keeps original code commented.
    - Uses `_coerce_prob_vector(..., size=6)` to handle malformed length/type/NaN.
  - Overall urgency probability vector hardening:
    - Keeps original code commented.
    - Uses `_coerce_prob_vector(..., size=6)` to prevent shape mismatch and NaN propagation.
- `Changed on 2026-02-24 by Jems KC` (new)
  - Category probability hardening for dict-shaped local LLM outputs:
    - Keeps original one-line clip code commented.
    - Replaces with robust scalar coercion per expected category.
    - Adds helper: `_coerce_probability_scalar(raw, default=0.5)`.
    - Handles nested dict keys (`probability`, `prob`, `score`, `value`) and non-numeric/NaN/inf safely.

2) `triage_system/src/triage/eval.py`
- `Changed on 2026-02-23 by Jems KC`
  - `p_high` sanitation before `roc_auc_score(...)`:
    - Original code kept commented.
    - Converts to float array, applies `nan_to_num`, clamps to `[0,1]`.
  - Prevents evaluation crash from NaN/inf/non-numeric local outputs.

3) `triage_system/src/triage/experiment.py`
- `Changed on 2026-02-23 by Jems KC`
  - `urgency_probs_high` sanitation while preparing urgency metrics:
    - Original list-comprehension kept commented.
    - Adds robust float parsing + finite check + clipping.
  - Prevents malformed local LLM values from breaking metric computation.

Related config notes (Jems-tagged, not explicitly "Jems KC")

4) `configs/llm_joint.yaml`
- `Changed on: 2026-02-23 by Jems`
  - Model switch comments recorded:
    - DeepSeek set
    - GLM trial
    - Switched back to DeepSeek

Reference note
- Prior context note exists at:
  - `NOTE_for_professor_local_llm_issues_2026-02-23.md`
