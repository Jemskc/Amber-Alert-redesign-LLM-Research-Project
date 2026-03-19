# AGENTS.md

Orientation guide for AI agents working in this repository.

---

## What This Project Is

**AI4Crisis** is a research codebase for automating the triage of disaster "Request-for-Help" (RFH) messages submitted to crisis mapping platforms like Ushahidi. The dataset comes from the 2010 Haiti earthquake. The system predicts two things for each **thread** (incident): which of 8 crisis categories apply (multi-label) and how urgent the message is on a 0-5 scale (per category).

There is no ground truth. The labels were assigned post-event by registered nurses (RNs), so the evaluation framework is built around **plausibility** (inter-annotator agreement), not accuracy.

---

## Research Context

This repo supports an active research program across multiple publications. Understanding the lineage matters because the codebase implements and extends ideas from each paper.

**Published:**

- **MISQ 2025** — "Design Principles for Information Categorization Quality in Crowdsourced Crisis Mapping Platforms" (Valecha, Oh, Rao). Established the sensemaking-based design principles for how crowd volunteers categorize RFH messages. This is the conceptual foundation. Published in MIS Quarterly Vol. 49(2), June 2025. PDF in `documentation/`.

- **BizAI 2024** — "Expert Knowledge or Contextual Examples: Effective Prompting Strategies for Utilizing Large Language Models for Triaging Emergency Request-for-Help Messages" (Rodriguez, Vishwamitra, Bina, Valecha, Rao). Showed that expert-knowledge prompting outperforms few-shot examples for LLM-based urgency triage. PDF in `documentation/`.

**Under Review:**

- **JAIS** — "Design Principles for Crisis Mapping Platforms: Exploring Information in Crisis Messages using Large Language Models" (Patel, Valecha, Vishwamitra, Rao). Investigates when LLMs should augment vs. substitute human categorizers. Under 3rd round review. PDF in `documentation/`.

**In Progress:**

- **MISQ Urgency Triage Draft** — "Automating Urgency Triage in Crisis Mapping: A Multi-Stage Human-AI Collaboration Framework." This is the paper the codebase most directly supports right now. It compares tree-based, neural, ensemble, and LLM architectures across joint and sequential configurations for urgency triage. The working draft is at `documentation/MISQ_Urgency_Triage_Draft.md`. Target journal: MIS Quarterly.

---

## Key Concepts

**Thread-level processing:** The dataset has 2,267 posts across 1,494 threads (incidents). Multiple posts can belong to the same thread (up to 17). When `aggregate_threads: true`, posts are concatenated per thread and predictions are made at thread level. The `#` column in the Excel data is the thread identifier.

**8 Crisis Categories:** Emergency, Vital Lines, Public Health, Security Threats, Infrastructure Damage, Natural Hazards, Services Available, Other. Defined in `triage_system/src/triage/category_definitions.py`.

**Urgency Scale (0-5):** 0 = not applicable, 1-3 = individual-level needs, 4-5 = community-level emergencies. The boundary between 3 and 4 is operationally critical — crossing it in either direction changes resource allocation decisions. The codebase tracks "boundary crossing" as a first-class metric.

**Plausibility, not accuracy:** Since only the victim knows the true situation, we measure how well predictions align with RN consensus, not against ground truth. Primary metric is Jaccard Index (set overlap of predicted vs. RN-assigned categories).

**Joint vs. Sequential:** Joint pipelines predict category and urgency in a single pass. Sequential pipelines predict category first, then condition urgency on the predicted category. The draft paper's central finding is that the better approach depends on the model type — tree-based models benefit from sequential; neural models do well with joint.

---

## Repo Structure

```
ai4crisis/
├── AGENTS.md                          # You are here
├── README.md                          # Project overview and quick start
├── requirements.txt                   # Python dependencies
├── Final_Merged_All_Columns.xlsx      # Primary merged dataset
│
├── configs/                           # YAML experiment configurations
│   ├── full_study.yaml                # Main config — thread-level, all improvements
│   ├── ml_baseline.yaml               # ML with no improvements (baseline)
│   ├── ml_all_improvements.yaml       # ML with all optimizations
│   ├── ml_joint.yaml                  # Minimal ML joint config
│   ├── ml_randomforest.yaml           # Random Forest backend
│   ├── ml_xgboost.yaml               # XGBoost backend
│   ├── llm_joint.yaml                 # Minimal LLM joint config
│   ├── cluster.yaml                   # Unsupervised clustering
│   └── elo.yaml                       # Elo-based ranking
│
├── data/raw/                          # Original Ushahidi datasets
│   ├── Ushahidi-Categories-Dataset.xlsx
│   └── Ushahidi-Urgency-Dataset.xlsx
│
├── triage_system/                     # Main codebase
│   ├── README.md                      # Detailed pipeline docs
│   ├── src/triage/                    # Core library (~4,000 lines)
│   │   ├── experiment.py              # Experiment orchestration
│   │   ├── data.py                    # Data loading, merging, thread aggregation
│   │   ├── eval.py                    # All evaluation metrics
│   │   ├── config.py                  # Config loading with defaults
│   │   ├── cli_utils.py              # Shared data loading for scripts
│   │   ├── llm_schemas.py            # Pydantic schemas for LLM output
│   │   ├── model_backends.py          # LightGBM / XGBoost / RF abstraction
│   │   ├── embedding_backends.py      # OpenAI / HuggingFace / ST embedding abstraction
│   │   ├── llm_backends.py           # OpenAI / HuggingFace LLM abstraction
│   │   ├── openai_client.py           # OpenAI API wrapper with caching
│   │   ├── calibration.py             # Temperature scaling
│   │   ├── policy.py                  # Automation vs. augmentation decisions
│   │   ├── split.py                   # Train/val/test splitting
│   │   └── pipelines/                 # All pipeline implementations
│   │       ├── llm_joint.py           # Single LLM call: category + urgency
│   │       ├── llm_sequential.py      # Two LLM calls: category then urgency
│   │       ├── ml_joint.py            # Tree-based: parallel predictions
│   │       ├── ml_sequential.py       # Tree-based: category → urgency
│   │       ├── mlp_joint.py           # Neural network, multi-task
│   │       ├── clustering.py          # UMAP + HDBSCAN
│   │       ├── elo_ranking.py         # Pairwise LLM ranking
│   │       └── oracle.py             # Ground-truth upper bound
│   ├── scripts/                       # Entry points
│   │   ├── triage_run_all.py          # Run all enabled pipelines
│   │   ├── triage_train_ml.py         # Train ML (joint + sequential)
│   │   ├── triage_run_ml_cv.py        # K-fold cross-validation
│   │   ├── triage_run_llm_joint.py    # LLM joint pipeline
│   │   ├── triage_run_llm.py          # Both LLM pipelines
│   │   ├── triage_train_mlp.py        # Neural network pipeline
│   │   ├── triage_cluster.py          # Unsupervised clustering
│   │   ├── triage_run_elo.py          # Elo ranking
│   │   ├── ensemble_predictions.py    # Ensemble multiple models
│   │   ├── triage_eval.py             # Evaluate saved predictions
│   │   └── triage_prepare_data.py     # Data preparation
│   └── tests/                         # Unit tests (pytest)
│
├── documentation/                     # Research papers and presentations
│   ├── MISQ_Urgency_Triage_Draft.md   # Active working draft (MISQ target)
│   ├── MISQ Crowd Reliability - final manuscript.pdf
│   ├── JAIS_LLM_Categorization.pdf
│   ├── BizAI-final_Manuscript_LLM in Urgency.pdf
│   └── *.pptx, *.pdf                  # Presentations
│
├── web/                               # React/Vite frontend for cluster visualization
├── outputs/                           # Timestamped experiment outputs
└── artifacts/                         # Cached embeddings (SQLite)
```

---

## Running Things

All scripts expect `PYTHONPATH=triage_system/src` and an `OPENAI_API_KEY` environment variable.

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"

# ML pipelines (LightGBM default, thread-level)
PYTHONPATH=triage_system/src python triage_system/scripts/triage_train_ml.py \
  --config configs/full_study.yaml

# ML with XGBoost
PYTHONPATH=triage_system/src python triage_system/scripts/triage_train_ml.py \
  --config configs/ml_xgboost.yaml

# ML with Random Forest
PYTHONPATH=triage_system/src python triage_system/scripts/triage_train_ml.py \
  --config configs/ml_randomforest.yaml

# MLP (neural network)
PYTHONPATH=triage_system/src python triage_system/scripts/triage_train_mlp.py \
  --config configs/full_study.yaml

# LLM joint pipeline (makes API calls — use --max-samples to limit cost)
PYTHONPATH=triage_system/src python triage_system/scripts/triage_run_llm_joint.py \
  --config configs/llm_joint.yaml --skip-train --max-samples 10

# Cross-validation
PYTHONPATH=triage_system/src python triage_system/scripts/triage_run_ml_cv.py \
  --config configs/full_study.yaml

# Tests
PYTHONPATH=triage_system/src pytest triage_system/tests/
```

Set `dry_run: true` in config YAML to test without making OpenAI API calls.

---

## Architecture: Pluggable Backends

The codebase uses an abstract backend pattern for ML models, embeddings, and LLMs, enabling easy provider switching via config.

### ML Model Backends (`model_backends.py`)
ABC with `create_classifier()`, `fit()`, `predict_proba()`. Implementations:
- **LightGBMBackend** — default, gradient boosting
- **XGBoostBackend** — with label remapping for non-contiguous urgency classes
- **RandomForestBackend** — scikit-learn ensemble

Config: `ml.model_backend: lightgbm | xgboost | randomforest`

### Embedding Backends (`embedding_backends.py`)
ABC with `embed()` returning `(n_texts, dim)` array. Implementations:
- **OpenAIEmbeddingBackend** — `text-embedding-3-large` (1536 dims), with caching and dry_run
- **HuggingFaceEmbeddingBackend** — local inference with last-token pooling (e.g., `Qwen/Qwen3-Embedding-0.6B`, up to 1024 dims)
- **SentenceTransformerEmbeddingBackend** — sentence-transformers library

Config: `openai.embedding_provider: openai | huggingface | sentence-transformers`

### LLM Backends (`llm_backends.py`)
ABC with `generate_json()` and `generate_json_with_repair()`. Implementations:
- **OpenAI** — built into `openai_client.py` (uses Responses API with JSON schema enforcement)
- **HuggingFaceLLMBackend** — local inference with chat templates and prompt-based JSON extraction (e.g., `Qwen/Qwen3-8B`)

Config: `openai.llm_provider: openai | huggingface`

**Important**: Embeddings and ML models are coupled — models trained on 1536-dim OpenAI embeddings won't work with 1024-dim Qwen embeddings. Always retrain together.

---

## Important Conventions

- **Thread-level processing:** Data is aggregated from posts to threads before classification. Controlled by `data.aggregate_threads: true` and `data.group_column: "#"` in config. The `#` column maps to `group_id` internally.
- **Data leakage prevention:** Always use `group_column: group_id` in split configs to keep related messages in the same split.
- **Embeddings are cached** in `artifacts/openai_cache.sqlite`. They persist across runs and are expensive to regenerate. When thread aggregation is active, old per-post embeddings from CSV files are skipped — only the SQLite cache is used (which keys by text content).
- **Outputs are timestamped** under `outputs/` (e.g., `20260130_041035_full_study/`). Each run produces predictions, metrics, and figures.
- **Configs drive everything.** Pipeline toggles, model selection, hyperparameters, data paths — it's all in the YAML files under `configs/`.
- **Evaluation uses hard Jaccard only.** Soft Jaccard was removed. Categories with probability > 0.5 are treated as predicted (binary).

---

## Current State of Work

The primary active effort is the MISQ urgency triage paper (`documentation/MISQ_Urgency_Triage_Draft.md`). Key empirical results so far:

- Tree-based models (LightGBM) benefit from sequential conditioning (20-35% improvement)
- Neural models (MLP) achieve competitive performance with joint architectures
- Ensembling is most robust: MAE 0.96, boundary-crossing rate 11% (vs. 15-19% for single models)
- All models exhibit a precision-recall tradeoff for high-urgency (4-5) detection: ~80% precision but only ~14% recall

### Recent Changes (Feb 2026)
- **Thread-level aggregation**: Data is now processed at the thread level (1,494 threads from 2,267 posts) instead of post level
- **Soft Jaccard removed**: Only hard Jaccard (binary at 0.5 threshold) is used for evaluation
- **Embedding backend abstraction**: Pluggable embedding providers (OpenAI, HuggingFace, sentence-transformers) for testing Qwen 0.6B embeddings
- **LLM backend abstraction**: Pluggable LLM providers (OpenAI, HuggingFace) for testing Qwen 8B and other open-source models
- **Legacy code removed**: Old top-level `src/` and `scripts/` directories deleted; all code lives in `triage_system/`

The codebase is actively evolving to support additional experiments for the paper.
