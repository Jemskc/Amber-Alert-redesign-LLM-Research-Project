from __future__ import annotations

import json
import logging
import os
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "PyTorch is required for MLPJointPipeline. "
        "Install with `pip install torch`."
    ) from exc

from ..data import CATEGORY_ID_MAP
from ..openai_client import OpenAIClient
from ..utils import compute_entropy
from .base import BasePipeline


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    """Focal loss for multi-label classification to handle class imbalance."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    probs = torch.sigmoid(logits)
    pt = targets * probs + (1 - targets) * (1 - probs)
    focal_weight = (1 - pt) ** gamma
    # Alpha weighting: alpha for positive, (1-alpha) for negative
    alpha_weight = targets * alpha + (1 - targets) * (1 - alpha)
    loss = alpha_weight * focal_weight * bce
    return loss.mean()


def _parse_category_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if str(v).strip()]
        except Exception:
            return [raw]
        return []
    if value is None:
        return []
    return [str(value)]


def _compute_overall_urgency(df, urgency_columns: List[str], strategy: str = "max") -> np.ndarray:
    cols = [c for c in urgency_columns if c in df.columns]
    if not cols:
        return np.array([], dtype=int)
    values = df[cols].fillna(0).astype(int).clip(0, 5).values
    if strategy == "mean":
        return np.round(values.mean(axis=1)).astype(int)
    return values.max(axis=1).astype(int)


class MLPJointNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: List[int],
        dropout: float,
        num_categories: int,
        urgency_columns: List[str],
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        self.shared = nn.Sequential(*layers)

        self.category_head = nn.Linear(in_dim, num_categories) if num_categories > 0 else None
        self.urgency_head = nn.Linear(in_dim, 6) if urgency_columns else None
        self.urgency_heads = nn.ModuleDict({col: nn.Linear(in_dim, 6) for col in urgency_columns})

    def forward(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        h = self.shared(x)
        cat_logits = self.category_head(h) if self.category_head is not None else None
        urg_logits = self.urgency_head(h) if self.urgency_head is not None else None
        urg_by_cat = {col: head(h) for col, head in self.urgency_heads.items()}
        return cat_logits, urg_logits, urg_by_cat


@dataclass
class MLPConfig:
    hidden_sizes: List[int]
    dropout: float
    lr: float
    weight_decay: float
    batch_size: int
    epochs: int
    patience: int
    loss_weights: Dict[str, float]
    # Advanced training options
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    urgency_class_weights: Optional[List[float]] = None  # Weights for classes 0-5
    scheduler: str = "none"  # "none", "cosine", "plateau"
    scheduler_patience: int = 3  # For plateau scheduler


class MLPJointPipeline(BasePipeline):
    name = "mlp_joint"

    def __init__(
        self,
        client: OpenAIClient,
        embedding_model: str,
        categories: List[str],
        urgency_columns: Optional[List[str]] = None,
        overall_strategy: str = "max",
        category_threshold: float = 0.5,
        config: Optional[MLPConfig] = None,
        device: Optional[str] = None,
    ):
        self.client = client
        self.embedding_model = embedding_model
        self.categories = categories
        self.urgency_columns = urgency_columns or []
        self.overall_strategy = overall_strategy
        self.category_threshold = category_threshold
        self.config = config or MLPConfig(
            hidden_sizes=[512, 256, 128],
            dropout=0.2,
            lr=1e-3,
            weight_decay=1e-4,
            batch_size=64,
            epochs=50,
            patience=10,
            loss_weights={"category": 1.0, "urgency_overall": 2.0, "urgency_per_category": 1.5},
            use_focal_loss=True,
            focal_gamma=2.0,
            focal_alpha=0.25,
            urgency_class_weights=None,
            scheduler="cosine",
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[MLPJointNet] = None
        self.scaler: Optional[StandardScaler] = None
        self.urgency_weights: Optional[torch.Tensor] = None

    def _encode_categories(self, df) -> np.ndarray:
        if not self.categories:
            return np.zeros((len(df), 0), dtype=np.float32)
        if "category_list" in df.columns:
            lists = [_parse_category_list(v) for v in df["category_list"].tolist()]
        else:
            lists = [[str(v)] if v is not None and str(v).strip() else [] for v in df["category_gold"].tolist()]
        y = np.zeros((len(df), len(self.categories)), dtype=np.float32)
        for i, items in enumerate(lists):
            for j, cat in enumerate(self.categories):
                if cat in items:
                    y[i, j] = 1.0
        return y

    def _encode_urgency(self, df) -> Tuple[np.ndarray, np.ndarray]:
        y_overall = _compute_overall_urgency(df, self.urgency_columns, strategy=self.overall_strategy)
        if y_overall.size == 0:
            y_overall = np.zeros((len(df),), dtype=np.int64)
        y_overall = y_overall.astype(np.int64)

        y_by_cat = np.zeros((len(df), len(self.urgency_columns)), dtype=np.int64)
        for idx, col in enumerate(self.urgency_columns):
            if col not in df.columns:
                continue
            y_by_cat[:, idx] = df[col].fillna(0).astype(int).clip(0, 5).values
        return y_overall, y_by_cat

    def _build_dataloader(
        self,
        x: np.ndarray,
        y_cat: np.ndarray,
        y_urg_overall: np.ndarray,
        y_urg_by_cat: np.ndarray,
        shuffle: bool,
    ) -> DataLoader:
        tensors = [
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y_cat, dtype=torch.float32),
            torch.tensor(y_urg_overall, dtype=torch.long),
            torch.tensor(y_urg_by_cat, dtype=torch.long),
        ]
        dataset = TensorDataset(*tensors)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle, num_workers=0)

    def _train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        total_loss = 0.0
        w_cat = self.config.loss_weights.get("category", 1.0)
        w_urg = self.config.loss_weights.get("urgency_overall", 1.0)
        w_urg_cat = self.config.loss_weights.get("urgency_per_category", 1.0)

        for x, y_cat, y_urg_overall, y_urg_by_cat in loader:
            x = x.to(self.device)
            y_cat = y_cat.to(self.device)
            y_urg_overall = y_urg_overall.to(self.device)
            y_urg_by_cat = y_urg_by_cat.to(self.device)

            optimizer.zero_grad()
            cat_logits, urg_logits, urg_by_cat = self.model(x)
            loss = 0.0

            if cat_logits is not None and y_cat.shape[1] > 0:
                if self.config.use_focal_loss:
                    loss += w_cat * focal_loss(cat_logits, y_cat, self.config.focal_gamma, self.config.focal_alpha)
                else:
                    loss += w_cat * F.binary_cross_entropy_with_logits(cat_logits, y_cat)

            if urg_logits is not None:
                loss += w_urg * F.cross_entropy(urg_logits, y_urg_overall, weight=self.urgency_weights)

            if urg_by_cat:
                per_cat_losses = []
                for idx, col in enumerate(self.urgency_columns):
                    logits = urg_by_cat[col]
                    targets = y_urg_by_cat[:, idx]
                    # Only compute loss on samples where urgency > 0
                    mask = targets > 0
                    if mask.any():
                        per_cat_losses.append(F.cross_entropy(logits[mask], targets[mask], weight=self.urgency_weights))
                if per_cat_losses:
                    loss += w_urg_cat * torch.stack(per_cat_losses).mean()

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        return total_loss / max(len(loader.dataset), 1)

    def _eval_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        w_cat = self.config.loss_weights.get("category", 1.0)
        w_urg = self.config.loss_weights.get("urgency_overall", 1.0)
        w_urg_cat = self.config.loss_weights.get("urgency_per_category", 1.0)

        with torch.no_grad():
            for x, y_cat, y_urg_overall, y_urg_by_cat in loader:
                x = x.to(self.device)
                y_cat = y_cat.to(self.device)
                y_urg_overall = y_urg_overall.to(self.device)
                y_urg_by_cat = y_urg_by_cat.to(self.device)

                cat_logits, urg_logits, urg_by_cat = self.model(x)
                loss = 0.0

                if cat_logits is not None and y_cat.shape[1] > 0:
                    if self.config.use_focal_loss:
                        loss += w_cat * focal_loss(cat_logits, y_cat, self.config.focal_gamma, self.config.focal_alpha)
                    else:
                        loss += w_cat * F.binary_cross_entropy_with_logits(cat_logits, y_cat)
                if urg_logits is not None:
                    loss += w_urg * F.cross_entropy(urg_logits, y_urg_overall, weight=self.urgency_weights)
                if urg_by_cat:
                    per_cat_losses = []
                    for idx, col in enumerate(self.urgency_columns):
                        logits = urg_by_cat[col]
                        targets = y_urg_by_cat[:, idx]
                        mask = targets > 0
                        if mask.any():
                            per_cat_losses.append(F.cross_entropy(logits[mask], targets[mask], weight=self.urgency_weights))
                    if per_cat_losses:
                        loss += w_urg_cat * torch.stack(per_cat_losses).mean()

                total_loss += loss.item() * x.size(0)

        return total_loss / max(len(loader.dataset), 1)

    def fit(self, train_df, val_df=None) -> "MLPJointPipeline":
        debug = os.environ.get("TRIAGE_MLP_DEBUG") == "1"
        logger = logging.getLogger("triage")

        # Always limit threads on macOS to prevent segfaults from fork safety issues
        if platform.system() == "Darwin":
            try:
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
            except Exception:
                pass
        train_texts = train_df["text"].tolist()
        train_emb = self.client.embeddings(self.embedding_model, train_texts, batch_size=self.config.batch_size)
        if debug:
            logger.info("MLP debug: train_emb shape=%s dtype=%s", train_emb.shape, train_emb.dtype)
            try:
                logger.info("MLP debug: train_emb finite=%s", bool(np.isfinite(train_emb).all()))
            except Exception:
                logger.info("MLP debug: train_emb finite=unknown")

        self.scaler = StandardScaler()
        train_emb = self.scaler.fit_transform(train_emb)
        if debug:
            logger.info("MLP debug: scaled train_emb mean=%.6f std=%.6f", float(np.mean(train_emb)), float(np.std(train_emb)))

        y_cat = self._encode_categories(train_df)
        y_urg_overall, y_urg_by_cat = self._encode_urgency(train_df)

        # Compute urgency class weights from training distribution
        if self.config.urgency_class_weights is None:
            counts = np.bincount(y_urg_overall, minlength=6).astype(float)
            counts = np.maximum(counts, 1)  # Avoid division by zero
            weights = len(y_urg_overall) / (6 * counts)
            self.urgency_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        else:
            self.urgency_weights = torch.tensor(self.config.urgency_class_weights, dtype=torch.float32).to(self.device)

        if debug:
            logger.info("MLP debug: y_cat shape=%s y_urg_overall shape=%s y_urg_by_cat shape=%s", y_cat.shape, y_urg_overall.shape, y_urg_by_cat.shape)
            logger.info("MLP debug: urgency_weights=%s", self.urgency_weights.cpu().numpy())
        train_loader = self._build_dataloader(train_emb, y_cat, y_urg_overall, y_urg_by_cat, shuffle=True)

        input_dim = train_emb.shape[1]
        self.model = MLPJointNet(
            input_dim=input_dim,
            hidden_sizes=self.config.hidden_sizes,
            dropout=self.config.dropout,
            num_categories=len(self.categories),
            urgency_columns=self.urgency_columns,
        ).to(self.device)
        if debug:
            logger.info("MLP debug: model initialized on %s", self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler
        scheduler = None
        if self.config.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        elif self.config.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=self.config.scheduler_patience
            )

        if debug:
            logger.info("MLP debug: optimizer initialized, scheduler=%s", self.config.scheduler)

        best_val = float("inf")
        best_state = None
        patience_left = self.config.patience

        val_loader = None
        if val_df is not None and len(val_df) > 0:
            val_texts = val_df["text"].tolist()
            val_emb = self.client.embeddings(self.embedding_model, val_texts, batch_size=self.config.batch_size)
            val_emb = self.scaler.transform(val_emb)
            y_val_cat = self._encode_categories(val_df)
            y_val_urg, y_val_urg_by_cat = self._encode_urgency(val_df)
            val_loader = self._build_dataloader(val_emb, y_val_cat, y_val_urg, y_val_urg_by_cat, shuffle=False)
            if debug:
                logger.info("MLP debug: val_emb shape=%s", val_emb.shape)

        for _ in range(self.config.epochs):
            if debug:
                logger.info("MLP debug: starting epoch")
            self._train_epoch(train_loader, optimizer)

            # Step scheduler
            if scheduler is not None:
                if self.config.scheduler == "cosine":
                    scheduler.step()

            if val_loader is None:
                continue
            val_loss = self._eval_epoch(val_loader)

            # Step plateau scheduler with validation loss
            if scheduler is not None and self.config.scheduler == "plateau":
                scheduler.step(val_loss)

            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience_left = self.config.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, df) -> Dict[str, Any]:
        if self.model is None or self.scaler is None:
            raise RuntimeError("MLPJointPipeline must be fit before predict().")
        texts = df["text"].tolist()
        emb = self.client.embeddings(self.embedding_model, texts, batch_size=self.config.batch_size)
        emb = self.scaler.transform(emb)
        x = torch.tensor(emb, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            cat_logits, urg_logits, urg_by_cat = self.model(x)

        results: List[Dict[str, Any]] = []
        cat_probs = None
        if cat_logits is not None:
            cat_probs = torch.sigmoid(cat_logits).cpu().numpy()

        urg_probs = None
        if urg_logits is not None:
            urg_probs = F.softmax(urg_logits, dim=-1).cpu().numpy()

        urg_probs_by_cat: Dict[str, np.ndarray] = {}
        for col, logits in urg_by_cat.items():
            urg_probs_by_cat[col] = F.softmax(logits, dim=-1).cpu().numpy()

        # Build mapping from category name to urgency dimension index.
        cat_to_dim_idx: Dict[str, int] = {}
        for di, uc in enumerate(self.urgency_columns or []):
            m = re.search(r"\d+", uc)
            if m:
                cat_name = CATEGORY_ID_MAP.get(m.group())
                if cat_name:
                    cat_to_dim_idx[cat_name] = di

        for i in range(len(texts)):
            record: Dict[str, Any] = {}
            if cat_probs is not None and len(self.categories) > 0:
                probs = cat_probs[i]
                record["category_probs"] = {c: float(p) for c, p in zip(self.categories, probs)}
                top_idx = int(np.argmax(probs)) if len(probs) else 0
                record["category_top"] = self.categories[top_idx] if self.categories else None
                record["category_confidence"] = float(probs[top_idx]) if len(probs) else 0.0
                norm = probs / probs.sum() if probs.sum() > 0 else np.full_like(probs, 1 / len(probs))
                record["entropy_category"] = float(compute_entropy(norm))
                pred_list = [c for c, p in zip(self.categories, probs) if p >= self.category_threshold]
                if not pred_list and self.categories:
                    pred_list = [self.categories[top_idx]]
                record["category_pred_list"] = pred_list

            if urg_probs is not None:
                up = urg_probs[i]
                record["urgency_probs_overall"] = [float(p) for p in up]
                record["urgency_expected_overall"] = float(np.dot(np.arange(6), up))
                record["urgency_probs_high"] = float(up[4] + up[5])
                record["entropy_urgency"] = float(compute_entropy(up))

            if urg_probs_by_cat:
                record["urgency_probs_by_category"] = {}
                record["urgency_by_category"] = {}
                record["urgency_vector_8"] = {}
                # Build set of predicted category dimension indices
                pred_dim_indices: set = set()
                if record.get("category_pred_list"):
                    for cat_name in record["category_pred_list"]:
                        dim_for_cat = cat_to_dim_idx.get(cat_name)
                        if dim_for_cat is not None:
                            pred_dim_indices.add(dim_for_cat)
                for dim_idx, col in enumerate(self.urgency_columns):
                    if col not in urg_probs_by_cat:
                        continue
                    # Category masking: if category not predicted, force urgency to 0
                    if pred_dim_indices and dim_idx not in pred_dim_indices:
                        dim_probs = np.zeros(6, dtype=float)
                        dim_probs[0] = 1.0
                        pred_class = 0
                    else:
                        dim_probs = urg_probs_by_cat[col][i]
                        pred_class = int(np.argmax(dim_probs))
                    record["urgency_probs_by_category"][col] = [float(p) for p in dim_probs]
                    record["urgency_by_category"][col] = pred_class
                    record["urgency_vector_8"][col] = pred_class

            results.append(record)

        return {"records": results}

    def save(self, path: str | Path) -> None:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model must be trained before saving.")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        joblib.dump(self.scaler, path / "scaler.joblib")
        meta = {
            "categories": self.categories,
            "urgency_columns": self.urgency_columns,
            "overall_strategy": self.overall_strategy,
            "category_threshold": self.category_threshold,
            "config": self.config.__dict__,
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, client: OpenAIClient, embedding_model: str) -> "MLPJointPipeline":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        cfg = MLPConfig(**meta["config"])
        pipeline = cls(
            client=client,
            embedding_model=embedding_model,
            categories=meta["categories"],
            urgency_columns=meta["urgency_columns"],
            overall_strategy=meta.get("overall_strategy", "max"),
            category_threshold=meta.get("category_threshold", 0.5),
            config=cfg,
        )
        pipeline.scaler = joblib.load(path / "scaler.joblib")
        input_dim = pipeline.scaler.mean_.shape[0]
        pipeline.model = MLPJointNet(
            input_dim=input_dim,
            hidden_sizes=cfg.hidden_sizes,
            dropout=cfg.dropout,
            num_categories=len(pipeline.categories),
            urgency_columns=pipeline.urgency_columns,
        ).to(pipeline.device)
        pipeline.model.load_state_dict(torch.load(path / "model.pt", map_location=pipeline.device))
        return pipeline
