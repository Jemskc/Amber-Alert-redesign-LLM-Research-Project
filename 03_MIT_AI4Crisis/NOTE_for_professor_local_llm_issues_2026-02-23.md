Date: 2026-02-23
Prepared by: Jems KC

Subject: Local LLM instability observed with DeepSeek and GLM in triage LLM Joint pipeline

Summary
- While testing local HuggingFace backends in `triage_run_llm_joint.py`, we observed repeated output-format failures for:
  - `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
  - `zai-org/glm-edge-4b-chat`
- Baseline `Qwen/Qwen2.5-7B-Instruct` completed cleanly in the same pipeline.

Observed failures
- Shape mismatch errors in urgency vector calculations:
  - `shapes (6,) and (7,) not aligned`
  - also seen with `(6,) and (1|3|5|12)`
- Type/format errors from model JSON:
  - `float() argument must be a string or a real number, not 'dict'`
  - `'numpy.float64' object is not iterable`
- Evaluation-time crash:
  - `ValueError: Input contains NaN`
  - Crash site: `roc_auc_score(y_high, p_high)` in `triage_system/src/triage/eval.py`

Impact
- Runs do not always crash during prediction because fallback outputs are used after retries.
- However, malformed outputs can propagate into evaluation and terminate the run.
- This reduces reliability and trustworthiness of DeepSeek/GLM results compared with Qwen.

Current status
- Code hardening was added on 2026-02-23 to sanitize malformed probabilities and prevent NaN-related evaluation crashes.
- Change annotations were added in code with date, author (`Jems KC`), and reason.

Recommendation
- For stable experiments, use Qwen as primary local baseline.
- Keep DeepSeek/GLM as exploratory models until output schema compliance improves.
