# AI4Crisis - Urgency Triage System

Multi-model urgency scoring and triage pipeline for crisis "Request-for-Help" (RFH) messages.

## Research Context

This codebase supports research on **information categorization and urgency triage in crisis mapping platforms**. Crisis mapping platforms like Ushahidi enable disaster victims to submit brief "Requests for Help" (RFH) messages via SMS or mobile devices. These messages are then processed by crowd volunteers who categorize and prioritize them for first responders.

The research addresses two fundamental challenges in digital humanitarian response:

1. **Quality of Categorization**: RFH messages are often terse, incomplete, and ambiguous because victims submit them under high anxiety and time pressure. This makes accurate categorization difficult.

2. **Automation of Categorization**: Manual categorization by crowd volunteers is time-consuming (volunteers spend >30% of their time on categorization) and produces inconsistent results, with prior studies showing ~50% disagreement rates between independent reviewers.

### The Core Problem

In crisis situations, there is no ground truth—the actual applicable categories are only known to the victims submitting the RFHs. This creates a unique challenge where categorization quality must be measured through **plausibility** (degree of agreement among multiple evaluators) rather than accuracy against a known standard.

## Related Publications

This codebase supports and implements methods from three research papers:

### 1. MISQ 2025 - Design Principles for Crowd-Based Categorization Quality
**"Design Principles for Information Categorization Quality in Crowdsourced Crisis Mapping Platforms"**
*Valecha, Oh, & Rao — MIS Quarterly, Vol. 49(2), pp. 777-804, June 2025*

This paper proposes design principles to improve how crowd volunteers categorize RFH messages using **sensemaking theory**. Key contributions include:

- **Design Principle 1 (Extracting Contextual Cues)**: Platforms should extract social cues (who: victims, scale) and situational cues (when: urgency/time, what: needs/quantity, where: place/direction) from RFH messages
- **Design Principle 2 (Aggregating Interactive Posts)**: Platforms should aggregate crisis mapping posts (geolocation) and information gap-filling posts (clarifying incomplete information)
- A template interface that displays these cues was validated through multiple experiments showing significant improvement in categorization agreement

**Data**: 2010 Haiti earthquake RFH messages from the Ushahidi platform

### 2. JAIS - Human-AI Collaboration for Crisis Categorization
**"Design Principles for Crisis Mapping Platforms: Exploring Information in Crisis Messages using Large Language Models"**
*Patel, Valecha, Vishwamitra, & Rao — Under 3rd Round Review at JAIS*


This paper investigates how LLMs can be integrated into crisis mapping platforms to **augment or substitute** human categorizers (both expert registered nurses and crowd volunteers). Key findings:

- **Study 1 (Haiti)**: Compared LLM categorization against RN-led expert team
- **Study 2 (Chile)**: Compared LLM categorization against original crowd volunteers
- Uses 5 different LLMs with majority voting to improve categorization reliability
- Derives design principles for when LLMs should **augment** humans (require supervision) vs. **substitute** them (work independently)

**Design Principles for LLM Integration**:
- DP 1.1/2.1 (LLM Integration): Crisis mapping systems should integrate LLMs to augment/substitute experts and crowd volunteers
- DP 1.2/2.2 (LLM Consensus): Use majority voting across multiple LLMs for improved reliability
- DP 1.3-1.5/2.3-2.4 (Human-AI Collaboration): Context-dependent augmentation vs. substitution based on message themes (community vs. individual needs, generic vs. specific information)

**Data**: 2010 Haiti and 2010 Chile earthquake RFH messages

### 3. BizAI 2024 - LLM Prompting Strategies for Urgency Triage
**"Expert Knowledge or Contextual Examples: Effective Prompting Strategies for Utilizing Large Language Models for Triaging Emergency Request-for-Help Messages"**
*Rodriguez, Vishwamitra, Bina, Valecha, & Rao — BizAI Conference*

This paper examines how different prompting strategies affect LLM performance in **urgency triage** (assigning urgency scores 0-5). Key findings:

- Compares three prompting approaches: (1) generic prompting, (2) expert knowledge via RN-developed urgency scale, (3) contextual examples
- **Expert knowledge prompting significantly outperforms contextual examples** for urgency assessment
- LLMs prompted with expert knowledge show higher alignment with RN urgency assessments
- Highlights the importance of domain expertise in designing LLM-enabled crisis response systems

**Data**: 2010 Haiti earthquake RFH messages with RN-assigned urgency scores

## Task Definition

For each crisis **thread** (one or more related RFH messages from the same incident), predict:
1. **Category**: Which of 8 crisis categories the thread belongs to (multi-label)
2. **Urgency**: A 0-5 score per category indicating urgency level

## Quick Start

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"

# Run ML pipelines (LightGBM, thread-level)
PYTHONPATH=triage_system/src python triage_system/scripts/triage_train_ml.py \
  --config configs/full_study.yaml

# Run MLP pipeline (neural network, thread-level)
PYTHONPATH=triage_system/src python triage_system/scripts/triage_train_mlp.py \
  --config configs/full_study.yaml

# Run LLM pipeline (slower, makes API calls)
PYTHONPATH=triage_system/src python triage_system/scripts/triage_run_llm_joint.py \
  --config configs/llm_joint.yaml --skip-train --max-samples 10
```

## After You Have an OpenAI Key

If you already have a key, the minimal next steps are:

```bash
# Make the key available to the scripts
export OPENAI_API_KEY="your-key"

# (Optional) Verify it is set
echo $OPENAI_API_KEY

# Then run a pipeline (examples below)
```

## Get Started (Which Scripts to Run)

### ML Pipelines (tree-based models)

Each ML config runs three pipelines: **ML Joint**, **ML Joint Filtered**, and **ML Sequential**.

```bash
# LightGBM (default) — all three ML pipelines
PYTHONPATH=triage_system/src python triage_system/scripts/triage_train_ml.py \
  --config configs/full_study.yaml

# XGBoost backend — all three ML pipelines
PYTHONPATH=triage_system/src python triage_system/scripts/triage_train_ml.py \
  --config configs/ml_xgboost.yaml

# Random Forest backend — all three ML pipelines
PYTHONPATH=triage_system/src python triage_system/scripts/triage_train_ml.py \
  --config configs/ml_randomforest.yaml

# Ablation: baseline (no class weights, no scaling)
PYTHONPATH=triage_system/src python triage_system/scripts/triage_train_ml.py \
  --config configs/ml_baseline.yaml

# Ablation: class weights only
PYTHONPATH=triage_system/src python triage_system/scripts/triage_train_ml.py \
  --config configs/ml_class_weights_only.yaml

# Ablation: all improvements (class weights + feature scaling + category weights)
PYTHONPATH=triage_system/src python triage_system/scripts/triage_train_ml.py \
  --config configs/ml_all_improvements.yaml
```

### MLP Pipeline (neural network)

```bash
# MLP joint pipeline (shared backbone → category + urgency heads)
PYTHONPATH=triage_system/src python triage_system/scripts/triage_train_mlp.py \
  --config configs/full_study.yaml
```

### LLM Pipeline

```bash
# LLM joint pipeline (single API call per sample, slower)
PYTHONPATH=triage_system/src python triage_system/scripts/triage_run_llm_joint.py \
  --config configs/llm_joint.yaml --skip-train --max-samples 10
```

### Cross-Validation

```bash
# ML cross-validation (hyperparameter tuning)
PYTHONPATH=triage_system/src python triage_system/scripts/triage_run_ml_cv.py \
  --config configs/full_study.yaml
```

### What Each Config Produces

| Config | Backend | Pipelines | Key Settings |
|--------|---------|-----------|-------------|
| `full_study.yaml` | LightGBM | Joint + Filtered + Sequential | All improvements on, thread aggregation |
| `ml_xgboost.yaml` | XGBoost | Joint + Filtered + Sequential | Thread aggregation |
| `ml_randomforest.yaml` | Random Forest | Joint + Filtered + Sequential | Thread aggregation |
| `ml_baseline.yaml` | LightGBM | Joint + Filtered + Sequential | No class weights, no scaling |
| `ml_class_weights_only.yaml` | LightGBM | Joint + Filtered + Sequential | Class weights on, no scaling |
| `ml_all_improvements.yaml` | LightGBM | Joint + Filtered + Sequential | All improvements on |

## Available Pipelines

| Pipeline | Type | Description |
|----------|------|-------------|
| **MLJointPipeline** | ML | Per-category binary classifiers + per-dim urgency on embeddings. Category masking at prediction. |
| **MLJointFilteredPipeline** | ML | Same as ML Joint, but per-dim urgency models trained only on positive samples (urgency > 0). Falls back to all-sample training if < 10 positive samples. |
| **MLSequentialPipeline** | ML | Category first, then urgency conditioned on category via feature concatenation. Per-dim urgency with positive-sample filtering. |
| **MLPJointPipeline** | Neural | Multi-task neural network (shared backbone → category + urgency heads). Per-dim loss masking on positive samples. Category masking at prediction. |
| **LLMJointPipeline** | LLM | Single API call for category + urgency with rationales |

## Thread-Level Processing

The dataset contains 2,267 individual posts across 1,494 threads (incidents). Multiple posts can belong to the same thread (up to 17 posts per incident). When `aggregate_threads: true` is set in config:

- All posts within a thread are concatenated into a single text
- Urgency scores are aggregated (max per dimension across posts)
- Category labels are unioned across posts
- One prediction is made per thread, not per post
- Embeddings are computed on the concatenated thread text

This is controlled by two config settings:
```yaml
data:
  group_column: "#"          # Column identifying threads
  aggregate_threads: true    # Enable thread-level aggregation
```

## Pluggable Backends

### ML Model Backends
The ML pipelines support multiple model backends via `ml.model_backend`:
- `lightgbm` (default) — LightGBM gradient boosting
- `xgboost` — XGBoost gradient boosting
- `randomforest` — Scikit-learn Random Forest

### Embedding Backends
Embeddings can be generated via different providers using `openai.embedding_provider`:
- `openai` (default) — OpenAI API (`text-embedding-3-large`, 1536 dims)
- `huggingface` — Local HuggingFace models (e.g., `Qwen/Qwen3-Embedding-0.6B`)
- `sentence-transformers` — Sentence-Transformers library

### LLM Backends
LLM calls can use different providers via `openai.llm_provider`:
- `openai` (default) — OpenAI API
- `huggingface` — Local HuggingFace models (e.g., `Qwen/Qwen3-8B`)

## Methodology (Joint Pipelines)

### ML Joint (LightGBM on Embeddings)
**How it works**
- Each thread is embedded once using the configured embedding model.
- **Category prediction**: trains one binary classifier per category (multi-label, independent probabilities).
- **Urgency prediction (overall)**: multiclass classifier over 0-5 (or ordinal mode if enabled).
- **Per-category urgency**: if `Urg 1..Urg 8` columns exist, trains one multiclass classifier per category.
- Optional calibration on a validation split for high-urgency probability and category confidence.
- Supports class weights, feature scaling, and category weights for handling class imbalance.

**Why it's "joint"**
- A single embedding pass feeds multiple predictors (category + urgency in parallel), instead of conditioning urgency on a prior category prediction.

### LLM Joint (Single-Pass LLM)
**How it works**
- A single LLM call predicts **independent category probabilities** (multi-label, 0-1 each).
- The same call returns **per-category urgency distributions** `[P(0),...,P(5)]`.
- Outputs are evaluated with plausibility metrics (Jaccard) and urgency metrics.

**Why it's "joint"**
- Category and urgency are produced in one structured response, so urgency isn't conditioned on a separate ML step.

## Key Features

- **Thread-level classification**: Posts are aggregated by incident before classification
- **Multi-label classification**: Independent probability (0-1) per category
- **Per-category urgency**: Full probability distribution `[P(0),...,P(5)]` for each category
- **Plausibility metrics**: Jaccard Index for category overlap (used because ground truth is absent)
- **Boundary metrics**: Penalizes crossing community (4-5) vs individual (1-3) urgency threshold
- **Pluggable backends**: Swap between LightGBM, XGBoost, RF, and between OpenAI/HuggingFace embeddings and LLMs

## Category Taxonomy

The 8 categories were established by Ushahidi during the Haiti earthquake deployment:

| # | Category | Description | Examples |
|---|----------|-------------|----------|
| 1 | **Emergency** | Medical emergencies, people trapped, fire | "People are trapped under rubble" |
| 2 | **Vital Lines** | Food, water, shelter, fuel, power shortages | "We need tents, food, water" |
| 3 | **Public Health** | Infectious disease, medical needs | "Epidemic of fever and diarrhea" |
| 4 | **Security Threats** | Looting, violence, riots | "Reports of looting in the area" |
| 5 | **Infrastructure Damage** | Collapsed structures, blocked roads | "Bridge destroyed in Pudahuel" |
| 6 | **Natural Hazards** | Deaths, missing persons, earthquake effects | "Aftershock felt in Port-au-Prince" |
| 7 | **Services Available** | Distribution points, hospitals, shelters | "Medical supplies available at..." |
| 8 | **Other** | IDP concentration, aid issues, search & rescue | "Search and rescue needed" |

## Urgency Scale

The urgency scale was developed with input from registered nurses (RNs) with emergency management expertise:

| Level | Meaning | Example |
|-------|---------|---------|
| 0 | Not applicable / No urgency | Information-only messages |
| 1-3 | Individual-level urgency | Single person or family needs |
| 4-5 | Community-level urgency (critical) | Large groups, immediate life-threatening situations |

**Note**: The boundary between individual (1-3) and community (4-5) urgency is particularly important for resource allocation decisions.

## Documentation

See **[triage_system/README.md](triage_system/README.md)** for detailed documentation including:
- All available scripts and how to run them
- Configuration options
- Output format and prediction columns
- Metrics explanation

For research context, see the **[documentation/](documentation/)** folder containing:
- MISQ 2025 paper
- JAIS paper
- BizAI 2024 paper
- Summary presentation

## Project Structure

```
ai4crisis/
├── configs/                    # YAML configuration files
│   ├── full_study.yaml        # Main config (thread-level, all improvements)
│   ├── ml_xgboost.yaml       # XGBoost backend
│   ├── ml_randomforest.yaml   # Random Forest backend
│   └── ...
├── triage_system/
│   ├── src/triage/            # Core library
│   │   ├── pipelines/         # LLM and ML pipelines
│   │   ├── data.py            # Data loading, merging, thread aggregation
│   │   ├── eval.py            # Evaluation metrics
│   │   ├── experiment.py      # Experiment orchestration
│   │   ├── model_backends.py  # LightGBM/XGBoost/RF abstraction
│   │   ├── embedding_backends.py  # OpenAI/HuggingFace/ST embedding abstraction
│   │   ├── llm_backends.py    # OpenAI/HuggingFace LLM abstraction
│   │   └── openai_client.py   # API wrapper with caching
│   ├── scripts/               # Runnable scripts
│   └── tests/                 # Unit tests
├── documentation/             # Research papers and presentations
├── outputs/                   # Run outputs (predictions, metrics, figures)
└── artifacts/                 # Cached embeddings (SQLite)
```

## Key Methodological Notes

- **Thread-level processing**: Posts are grouped by incident ID (`#` column) and classified as threads, not individual posts
- **Labels come from registered nurses** post-event (establishing plausibility baselines, not ground truth)
- **Use group splits by incident ID** to prevent data leakage across related messages
- **Embeddings are automatically cached** in SQLite and reused across runs
- **Set `dry_run: true`** in config to test without API calls
- **Jaccard Index** is used as the primary agreement metric (intersection over union of categories)
