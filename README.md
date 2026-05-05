# Amber Alert Redesign — LLM Research Project

A research project investigating how Large Language Models (LLMs) can improve the classification, analysis, and topic modeling of Amber Alert content, with the goal of enhancing alert effectiveness and information processing.

---

## Overview

Amber Alerts are time-critical public safety communications. This project applies modern NLP techniques — including multi-label LLM classification and BERTopic-based topic modeling — to understand patterns in alert content, evaluate model capabilities, and explore how AI can support more effective alert systems.

---

## Repository Structure

```
.
├── 01_Amber_Alert_Classification/
│   ├── scripts/
│   │   ├── a2.py                          # Classification pipeline
│   │   └── gpuA100_1.py                   # GPU-accelerated classification (A100)
│   └── analysis_results/
│       ├── classified_comments.csv
│       ├── ALL_MULTILABEL_COMMENTS.csv
│       ├── FINAL_STATISTICS_REPORT.csv
│       ├── MODEL_DETECTION_COUNTS.csv
│       ├── FINAL_CAPABILITY_MATRIX-2026-3-18.csv
│       ├── final_grounded_output.csv
│       └── New-task-sheet.csv
│
└── 02_Topic_Modeling/
    ├── scripts/
    │   ├── bertopic_all_dataset.py         # BERTopic on full dataset
    │   └── bertopic_none_set.py            # BERTopic on filtered subset
    ├── input_data/
    └── outputs/
        ├── none_set_full_138_topics/       # Full topic resolution (138 topics)
        └── none_set_reduced_24_topics/     # Reduced topic resolution (24 topics)
```

---

## Modules

### 01 — Amber Alert Classification

Uses LLMs to apply multi-label classification to Amber Alert text. Scripts are designed to run on GPU hardware (NVIDIA A100) for efficient inference at scale.

**Key outputs:**
| File | Description |
|------|-------------|
| `classified_comments.csv` | Per-comment classification labels |
| `ALL_MULTILABEL_COMMENTS.csv` | Full multi-label annotation results |
| `FINAL_STATISTICS_REPORT.csv` | Aggregate statistics across the dataset |
| `MODEL_DETECTION_COUNTS.csv` | Detection counts per model |
| `FINAL_CAPABILITY_MATRIX-2026-3-18.csv` | Cross-model capability assessment matrix |
| `final_grounded_output.csv` | Final grounded/verified model output |

---

### 02 — Topic Modeling

Applies [BERTopic](https://maartengr.github.io/BERTopic/) to discover latent themes across the Amber Alert dataset. Two resolution levels are provided: a fine-grained 138-topic model and a consolidated 24-topic model.

**Key outputs:**
| Directory | Description |
|-----------|-------------|
| `none_set_full_138_topics/` | Full-resolution topic model results |
| `none_set_reduced_24_topics/` | Reduced topic model for interpretability |

---

## Requirements

- Python 3.9+
- GPU recommended (NVIDIA A100 for classification scripts)
- Key packages: `transformers`, `bertopic`, `pandas`, `torch`

Install dependencies:
```bash
pip install transformers bertopic pandas torch
```

---

## Usage

**Run classification:**
```bash
python 01_Amber_Alert_Classification/scripts/a2.py
```

**Run topic modeling:**
```bash
python 02_Topic_Modeling/scripts/bertopic_all_dataset.py
```

---

## Tech Stack

- **LLMs** — Hugging Face Transformers for text classification
- **BERTopic** — Topic modeling with sentence embeddings
- **Python / Pandas** — Data processing and analysis
- **CUDA / A100** — GPU-accelerated inference

---

## Author

**Jemskc** — [GitHub](https://github.com/Jemskc)

---

*This project is part of ongoing research into AI-assisted public safety communication systems.*
