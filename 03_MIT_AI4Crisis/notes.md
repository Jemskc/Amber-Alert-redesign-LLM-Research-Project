# Project Notes & Remarks

## ML Pipeline Differences

### Why ML Joint and ML Sequential produce different category metrics

Even though both pipelines train one binary classifier per category in multi-label mode on the same data, their category results differ because:

1. **Feature scaling**: ML Joint applies `StandardScaler` to embeddings when `use_feature_scaling: true`. ML Sequential does not support this parameter at all — it always trains on raw embeddings.

2. **Category sample weights**: ML Joint applies inverse-frequency sample weights to category training when `use_category_weights: true`. ML Sequential trains its category classifiers without any sample weighting.

Both are enabled in `full_study.yaml` but only affect ML Joint. To get a fair urgency-only comparison between the two pipelines, either disable these in the config or add the same support to ML Sequential.

### Pipeline architecture summary

| Pipeline | Category | Urgency | Key idea |
|---|---|---|---|
| **ML Joint** | Multi-label binary classifiers per category | Overall model + per-dimension models (one per Urg column). Category masking at prediction: if category not predicted, urgency forced to 0. | Everything trained independently on embeddings |
| **ML Joint Filtered** | Same as ML Joint (inherited) | Same as ML Joint, except per-dimension models trained only on samples where urgency > 0. Falls back to all-sample if < 10 positive samples. | Avoids zero-class domination in per-dim urgency |
| **ML Sequential** | Same binary classifiers, but no scaling/weighting | Overall model + per-dimension models using augmented features (embeddings + category one-hot). Category masking at prediction. Also supports per-category mode and feature_concat mode for overall urgency. | Urgency conditioned on predicted category |

### Config flags that control which pipelines run

In `pipelines:` section of YAML configs:
- `ml_joint: true` — runs MLJointPipeline
- `ml_joint_filtered: true` — runs MLJointFilteredPipeline (new)
- `ml_sequential: true` — runs MLSequentialPipeline

These are checked in `triage_train_ml.py`. ML Joint and ML Sequential always run (no config gate currently). ML Joint Filtered is gated by `pipelines.ml_joint_filtered`.

## Per-Category AUC showing N/A (single class)

AUC requires both positive and negative examples in the test set. Categories like "Collapsed structure", "Food Shortage", "Food distribution point", "Water shortage" showed N/A because they had zero positive samples in the test split.

**Root cause found**: These were not real categories — they were sub-categories (e.g., "2a. Food Shortage") that leaked through due to a regex bug in `normalize_category_label()` in `data.py`. The bug was double backslashes in raw strings:

```python
# BUG: r"^\\s*" matches literal backslash+s, not whitespace
match = re.match(r"^\\s*([1-8])", raw)

# FIX: r"^\s*" matches whitespace
match = re.match(r"^\s*([1-8])", raw)
```

This caused sub-categories like "2a. Penurie d'aliments | Food Shortage" to not match the leading digit regex, falling through to produce "Food Shortage" as a standalone category instead of mapping to "Vital Lines" (category 2).

**Fix applied**: Corrected three regex patterns in `data.py` (lines 74, 79, 87). After the fix, all sub-categories map to their 8 parent categories:
- 2a Food Shortage → Vital Lines
- 2b Water shortage → Vital Lines
- 5a Collapsed structure → Infrastructure Damage
- 7a Food distribution point → Services Available

## Per-Category Urgency: Identical scores across pipelines (BUG)

Per-category urgency MAE values were identical between RF and MLP (and between ML Joint and ML Sequential). Two causes:

1. **MLP missing field (bug, now fixed)**: `mlp_joint.py` produced `urgency_vector_8` but not `urgency_by_category`. The evaluation function `evaluate_urgency_per_category` looks for `urgency_by_category` → finds empty dict → defaults `pred_val = 0` for all samples.

2. **Zero-class domination**: Even when `urgency_by_category` is present (ML Joint, RF), per-dimension models trained on all samples predict class 0 for nearly every sample because ~95% of training data has urgency=0 for any given dimension. So `pred_val ≈ 0` regardless.

Both effects converge to MAE = mean(|y_true|), which is a constant determined by the test data, not the model.

**Fix applied**: Added `urgency_by_category` field to MLP predictions (`mlp_joint.py:452`). The ML Joint Filtered pipeline should also help with zero-class domination by training per-dim models on positive samples only.

**ML Sequential** now trains per-dimension urgency models using augmented features (embeddings + category one-hot), matching its conditioning-on-category architecture. Category masking is applied at prediction time using CATEGORY_ID_MAP, same as ML Joint.

## Category Masking Index Bug (FIXED)

The per-dimension urgency prediction in ML Joint uses "category masking" — if a category wasn't predicted for a sample, that dimension's urgency is forced to 0. The masking used `self.categories.index(cat_name)` to find the dimension index, but `self.categories` is alphabetically sorted while `self.urgency_columns` follows the numeric CATEGORY_ID_MAP order:

```
categories[0] = "Emergency"            → masked dim_idx 0 (Urg 1) ✓
categories[1] = "Infrastructure Damage" → masked dim_idx 1 (Urg 2) ✗ should be Urg 5
categories[7] = "Vital Lines"          → masked dim_idx 7 (Urg 8) ✗ should be Urg 2
```

Only 2 of 8 mappings were correct. This caused most dimensions to be wrongly masked to 0.

**Fix applied**: Build an explicit mapping from category names to urgency dimension indices using CATEGORY_ID_MAP + extracting the number from "Urg N" column names. Now "Emergency" → Urg 1 (dim 0), "Vital Lines" → Urg 2 (dim 1), etc.

Combined with zero-class domination, this meant ALL per-dim urgency predictions were 0 for every sample, producing identical MAE = mean(|y_true|) across all pipelines.

**Impact measurement** (RF run before fix):
- ML Joint regular: 4/2992 non-zero per-dim predictions (0.1%)
- ML Joint Filtered: 410/2992 non-zero (13.7%)

## MLP Per-Dimension Loss Masking (FIXED)

The MLP per-dimension urgency heads (`urgency_heads` in `TriageMLPModel`) were trained using `cross_entropy` on ALL samples, including those with urgency=0. Despite using inverse-frequency `urgency_weights`, the zero class still dominated (~95% of samples per dimension), causing the heads to predict 0 for nearly everything.

**Symptom**: Urg 2 MAE = 2.1203 (identical to the all-zeros baseline), confirming the heads learned nothing useful.

**Fix applied**: Added loss masking in both `_train_epoch` and `_eval_epoch` (`mlp_joint.py`). The per-dimension loss is now computed only on samples where `targets > 0` for that dimension:
```python
mask = targets > 0
if mask.any():
    per_cat_losses.append(F.cross_entropy(logits[mask], targets[mask], weight=self.urgency_weights))
```

This is the neural network equivalent of the positive-sample filtering used in ML Joint Filtered and ML Sequential. The overall urgency head (`urg_logits`) is left unchanged since it needs all samples.

## Open Questions

- Should ML Sequential be updated to support `use_feature_scaling` and `use_category_weights` for fair comparison?
- ~~Should ML Sequential produce per-dimension urgency predictions like ML Joint does?~~ Done — ML Sequential now trains per-dim models with augmented features.
- With only 8 categories after the regex fix, will the N/A AUC issue resolve (since the rare sub-categories are now merged into their parents)?
- What is the minimum number of samples per category needed for reliable AUC computation?
- **MLP shared backbone tradeoff**: The per-dim urgency loss masking (positive samples only) changes the gradient signal flowing through the shared backbone (512→256→128), which also feeds the category head. This caused a Jaccard drop (~0.72 vs ~0.74-0.75 from ML models). Option A: detach gradients so per-dim heads don't affect the backbone (isolates category performance). Option B: accept the tradeoff. Need to decide.
