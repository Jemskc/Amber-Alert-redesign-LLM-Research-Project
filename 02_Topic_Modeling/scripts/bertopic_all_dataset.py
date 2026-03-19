"""
BERTopic - ALL DATASET
------------------------
Dataset : Tpoic-modeling-for-all-dataset.xlsx
          All 98,712 Amber Alert Reddit comments (all capability labels)
Model   : all-mpnet-base-v2  (best quality, runs on RTX 4060 GPU)

Strategy (Grootendorst recommended):
  Step 1 → Find ALL fine-grained topics (MIN_TOPIC_SIZE=100)
  Step 2 → Intelligently merge similar topics down to 30
  Step 3 → Save BOTH versions (full + reduced) for paper
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from datetime import datetime
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_FILE      = "Tpoic-modeling-for-all-dataset.xlsx"
DATASET_TAG     = "ALL_DATASET"
EMBED_MODEL     = "all-mpnet-base-v2"

MIN_TOPIC_SIZE  = 100   # slightly larger threshold for 98K dataset
N_GRAM_RANGE    = (1, 2)
TOP_N_WORDS     = 10
REDUCED_TOPICS  = 30    # target for paper-ready version (more topics — larger dataset)

# ─────────────────────────────────────────────
# OUTPUT FOLDER WITH DATETIME STAMP
# ─────────────────────────────────────────────
timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M")
output_dir = os.path.join("outputs", f"amber_alert_{DATASET_TAG}_{timestamp}")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "full_topics",     "visualizations"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "reduced_30_topics", "visualizations"), exist_ok=True)

print("=" * 60)
print("  BERTopic — AMBER ALERT ALL DATASET")
print(f"  Timestamp      : {timestamp}")
print(f"  Output folder  : {output_dir}")
print(f"  Strategy       : Full topics → Reduce to {REDUCED_TOPICS}")
print("=" * 60)

# ─────────────────────────────────────────────
# STEP 1 — LOAD & CLEAN DATA
# ─────────────────────────────────────────────
print("\n[1/7] Loading dataset...")
df = pd.read_excel(INPUT_FILE)
print(f"      Loaded {len(df):,} rows")

df["comment"] = df["comment"].astype(str).str.strip()
df = df[df["comment"].notna()]
df = df[df["comment"] != ""]
df = df[df["comment"] != "nan"]
df = df[df["comment"] != "[deleted]"]
df = df[df["comment"] != "[removed]"]
df = df.reset_index(drop=True)

comments = df["comment"].tolist()
print(f"      After cleaning: {len(comments):,} comments ready")

# ─────────────────────────────────────────────
# STEP 2 — LOAD EMBEDDING MODEL (GPU)
# ─────────────────────────────────────────────
print(f"\n[2/7] Loading embedding model: {EMBED_MODEL}")
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"      Using device : {device.upper()}")
if device == "cuda":
    print(f"      GPU          : {torch.cuda.get_device_name(0)}")

embedding_model = SentenceTransformer(EMBED_MODEL, device=device)
print("      Model loaded successfully")

# ─────────────────────────────────────────────
# STEP 3 — GENERATE EMBEDDINGS
# ─────────────────────────────────────────────
print(f"\n[3/7] Generating embeddings for {len(comments):,} comments...")
print("      (GPU accelerated — ~5-8 minutes for 98K comments)")
embeddings = embedding_model.encode(
    comments,
    show_progress_bar=True,
    batch_size=256,
    device=device
)
print(f"      Embeddings shape: {embeddings.shape}")

# ─────────────────────────────────────────────
# STEP 4 — BUILD & FIT BERTOPIC (FULL)
# ─────────────────────────────────────────────
print("\n[4/7] Building BERTopic model (full fine-grained)...")

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42
)

hdbscan_model = HDBSCAN(
    min_cluster_size=MIN_TOPIC_SIZE,
    min_samples=10,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True
)

vectorizer_model = CountVectorizer(
    ngram_range=N_GRAM_RANGE,
    stop_words="english",
    min_df=10
)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    top_n_words=TOP_N_WORDS,
    verbose=True
)

print("      Fitting BERTopic (UMAP + HDBSCAN + c-TF-IDF)...")
topics, probabilities = topic_model.fit_transform(comments, embeddings)
print("      Done!")

topic_info_full = topic_model.get_topic_info()
num_topics_full = len(topic_info_full[topic_info_full["Topic"] != -1])
num_outliers    = int(topic_info_full[topic_info_full["Topic"] == -1]["Count"].values[0]) \
                  if -1 in topic_info_full["Topic"].values else 0

print(f"\n      Full topics found  : {num_topics_full}")
print(f"      Outliers (Topic-1) : {num_outliers:,}")

# ─────────────────────────────────────────────
# HELPER — build outputs for any topic state
# ─────────────────────────────────────────────
def build_topic_name_map(model, topic_info):
    name_map = {}
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            name_map[tid] = "Outlier"
        else:
            words = model.get_topic(tid)
            if words:
                name = "_".join([w[0] for w in words[:4]])
                name_map[tid] = f"Topic{tid}_{name}"
            else:
                name_map[tid] = f"Topic{tid}"
    return name_map

def get_keywords_string(model, topic_id):
    if topic_id == -1:
        return "outlier — no keywords"
    words = model.get_topic(topic_id)
    return ", ".join([w[0] for w in words[:10]]) if words else ""

def save_all_outputs(model, topic_info, df_comments, topics_col,
                     probs_col, out_dir, file_prefix, label):
    """Save topic summary, comments, keywords, representatives and charts."""

    os.makedirs(os.path.join(out_dir, "visualizations"), exist_ok=True)
    name_map = build_topic_name_map(model, topic_info)

    num_real    = len(topic_info[topic_info["Topic"] != -1])
    num_outlier = int(topic_info[topic_info["Topic"] == -1]["Count"].values[0]) \
                  if -1 in topic_info["Topic"].values else 0

    # ── Topic Summary ─────────────────────────
    summary = topic_info.copy()
    summary.columns = [c.lower().replace(" ", "_") for c in summary.columns]
    summary["top_keywords"] = summary["topic"].apply(
        lambda t: get_keywords_string(model, t))
    summary["human_label"]  = ""

    f = os.path.join(out_dir, f"{file_prefix}_topic_summary.csv")
    summary.to_csv(f, index=False, encoding="utf-8-sig")
    print(f"      Saved: {f}")

    # ── Comments with Topic Labels ────────────
    df_out = df_comments.copy()
    df_out["topic_id"]          = topics_col
    df_out["topic_name"]        = [name_map.get(t, str(t)) for t in topics_col]
    df_out["topic_probability"] = [round(float(p), 4) for p in probs_col]
    df_out["top_keywords"]      = [get_keywords_string(model, t) for t in topics_col]
    df_out["human_label"]       = ""

    f = os.path.join(out_dir, f"{file_prefix}_comments_with_topics.csv")
    df_out.to_csv(f, index=False, encoding="utf-8-sig")
    print(f"      Saved: {f}")

    # ── Per-Topic Keywords with Scores ────────
    kw_rows = []
    for tid in topic_info["Topic"]:
        if tid == -1:
            continue
        for word, score in model.get_topic(tid):
            kw_rows.append({
                "topic_id"   : tid,
                "topic_name" : name_map.get(tid, ""),
                "keyword"    : word,
                "score"      : round(score, 6)
            })
    f = os.path.join(out_dir, f"{file_prefix}_topic_keywords.csv")
    pd.DataFrame(kw_rows).to_csv(f, index=False, encoding="utf-8-sig")
    print(f"      Saved: {f}")

    # ── Representative Comments ───────────────
    rep_rows = []
    for tid, docs in model.get_representative_docs().items():
        for doc in docs:
            rep_rows.append({
                "topic_id"              : tid,
                "topic_name"            : name_map.get(tid, ""),
                "representative_comment": doc
            })
    f = os.path.join(out_dir, f"{file_prefix}_representative_comments.csv")
    pd.DataFrame(rep_rows).to_csv(f, index=False, encoding="utf-8-sig")
    print(f"      Saved: {f}")

    # ── Visualizations ────────────────────────
    viz_dir = os.path.join(out_dir, "visualizations")
    charts = [
        ("visualize_barchart",  {"top_n_topics": min(num_real, 30), "n_words": 10}, "topic_barchart"),
        ("visualize_topics",    {},                                                   "intertopic_map"),
        ("visualize_hierarchy", {},                                                   "topic_hierarchy"),
        ("visualize_heatmap",   {},                                                   "topic_heatmap"),
    ]
    for method, kwargs, name in charts:
        try:
            fig = getattr(model, method)(**kwargs)
            fp  = os.path.join(viz_dir, f"{file_prefix}_{name}.html")
            fig.write_html(fp)
            print(f"      Saved: {fp}")
        except Exception as e:
            print(f"      {name} skipped: {e}")

    # ── Mini summary print ────────────────────
    print(f"\n      [{label}] Topics: {num_real} | Outliers: {num_outlier:,}")
    print(f"      {'ID':<6} {'Size':<8} {'Top Keywords'}")
    print("      " + "-" * 55)
    for _, row in summary[summary["topic"] != -1].head(10).iterrows():
        print(f"      {int(row['topic']):<6} {int(row['count']):<8} {str(row['top_keywords'])[:50]}")

    return df_out

# ─────────────────────────────────────────────
# STEP 5 — SAVE FULL VERSION (all topics)
# ─────────────────────────────────────────────
print("\n[5/7] Saving FULL version outputs...")
full_dir    = os.path.join(output_dir, "full_topics")
full_prefix = f"amber_alert_ALL_FULL_{timestamp}"

df_full = save_all_outputs(
    model        = topic_model,
    topic_info   = topic_info_full,
    df_comments  = df,
    topics_col   = topics,
    probs_col    = probabilities,
    out_dir      = full_dir,
    file_prefix  = full_prefix,
    label        = f"FULL — {num_topics_full} topics"
)

# ─────────────────────────────────────────────
# STEP 6 — REDUCE TO 30 TOPICS (paper version)
# ─────────────────────────────────────────────
print(f"\n[6/7] Reducing to {REDUCED_TOPICS} topics (intelligent semantic merge)...")
print("      Similar topics are merged — no comments are deleted")

topic_model.reduce_topics(
    comments,
    nr_topics=REDUCED_TOPICS
)
topics_reduced = topic_model.topics_
probs_reduced  = topic_model.probabilities_

topic_info_reduced = topic_model.get_topic_info()
num_topics_reduced = len(topic_info_reduced[topic_info_reduced["Topic"] != -1])
print(f"      Reduced to {num_topics_reduced} topics")

reduced_dir    = os.path.join(output_dir, "reduced_30_topics")
reduced_prefix = f"amber_alert_ALL_REDUCED30_{timestamp}"

df_reduced = save_all_outputs(
    model        = topic_model,
    topic_info   = topic_info_reduced,
    df_comments  = df,
    topics_col   = topics_reduced,
    probs_col    = probs_reduced,
    out_dir      = reduced_dir,
    file_prefix  = reduced_prefix,
    label        = f"REDUCED — {num_topics_reduced} topics"
)

# ─────────────────────────────────────────────
# STEP 7 — FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n[7/7] Done!")
print("\n" + "=" * 60)
print("  COMPLETE — AMBER ALERT ALL DATASET")
print("=" * 60)
print(f"  Total comments    : {len(comments):,}")
print(f"  Full topics       : {num_topics_full}  → {full_dir}")
print(f"  Reduced topics    : {num_topics_reduced} → {reduced_dir}")
print(f"  Outliers          : {num_outliers:,} comments")
print("=" * 60)
print("\n  OUTPUT STRUCTURE:")
print(f"  {output_dir}/")
print(f"  ├── full_topics/")
print(f"  │   ├── *_topic_summary.csv")
print(f"  │   ├── *_comments_with_topics.csv")
print(f"  │   ├── *_topic_keywords.csv")
print(f"  │   ├── *_representative_comments.csv")
print(f"  │   └── visualizations/*.html")
print(f"  └── reduced_30_topics/")
print(f"      ├── *_topic_summary.csv        ← USE FOR PAPER")
print(f"      ├── *_comments_with_topics.csv ← USE FOR PAPER")
print(f"      ├── *_topic_keywords.csv")
print(f"      ├── *_representative_comments.csv")
print(f"      └── visualizations/*.html")
print("=" * 60)
