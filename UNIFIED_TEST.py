# Create a single runnable Python test script that merges 3 models and compares metrics side by side.
# It will also save two comparison graphs: (1) mAP per model, (2) C-F1 per model.
# The file will be saved to /mnt/data/UNIFIED_TEST.py

"""
UNIFIED_TEST.py
One script to evaluate THREE multi-label classifiers (e.g., 2x ResNet + 1x ViT) on the same EMOTIC-style test set,
and compare metrics side by side.

Metrics:
- AP per class (sklearn.average_precision_score)
- mAP (nanmean of class-wise AP)
- C-F1 (macro F1 across classes)
- O-F1 (example-based F1; samples with both true & pred empty count as F1=1)

Also saves two charts:
- map_per_model.png
- cf1_per_model.png

EDIT the CONFIG block below to set your data paths and model checkpoints/thresholds.
This script is Mac-friendly (uses MPS if available). It will also work on CPU/CUDA.
"""

import os
import math
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50

from PIL import Image
from sklearn.metrics import average_precision_score, f1_score

# ViT (optional; only needed if you use a ViT model below)
try:
    from transformers import ViTForImageClassification, ViTConfig
    HAS_VIT = True
except Exception:
    HAS_VIT = False

import matplotlib
matplotlib.use("Agg")  # safe for headless
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# =========================
# CONFIG — EDIT THIS BLOCK
# =========================

# =========================
# CONFIG — EDIT THIS BLOCK
# =========================

# Test CSV and image directory (.npy files)
ANNOT_PATH = "archive/annots_arrs/annot_arrs_test.csv"  # <-- put your test CSV path
IMG_DIR    = "archive/img_arrs"                         # <-- folder with .npy images

# If you know label columns, list them; otherwise keep None to auto-detect
LABEL_COLUMNS = None

# EMOTIC has 26 classes
NUM_CLASSES = 26

# Batch size for evaluation
BATCH_SIZE = 32

# Define your THREE models here (names, types, checkpoints, thresholds)
MODEL_CONFIGS = [
    {
        "name": "ResNet_BCE",
        "type": "resnet50",
        "checkpoint_path": "weighted_bce_resnet50_emotic.pth",        # <-- point to your .pth
        "thresholds_path": "bce_dynamic_thresholds.npy"                           # or e.g. "resnet_bce_thresholds.npy"
    },
    {
        "name": "ResNet_Focal",
        "type": "resnet50",
        "checkpoint_path": "focal_loss_resnet50_emotic.pth",      # <-- point to your .pth
        "thresholds_path": "focal_dynamic_thresholds.npy"  # <-- if you have per-class thresholds
    },
    {
        "name": "ViT_Emotic",
        "type": "vit",
        "checkpoint_path": "best_vit_emotic.pth",        # <-- point to your .pth
        "thresholds_path": "vit_dynamic_thresholds.npy"  # <-- if you have per-class thresholds
    },
]



# ===============
# Device helpers
# ===============
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =========================
# Dataset / Data utilities
# =========================
class EmoticNpyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, label_cols: List[str], image_size: int = 224):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.label_cols = label_cols
        self.size = image_size
        self.to_tensor = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        crop_name = row["Crop_name"]
        path = os.path.join(self.img_dir, crop_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

        arr = np.load(path)
        if arr.ndim == 2:  # grayscale -> RGB
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] != 3:
            raise ValueError(f"Unexpected image shape: {arr.shape} for {path}")

        pil = Image.fromarray(arr.astype(np.uint8))
        x = self.to_tensor(pil)  # [3, H, W] float32 in [0,1]
        y = torch.tensor(row[self.label_cols].values.astype(np.float32))  # multi-hot
        return x, y


def auto_detect_label_columns(df: pd.DataFrame) -> List[str]:
    excluded = {"Crop_name", "img_path", "image_path", "path"}
    candidates = []
    for col in df.columns:
        if col in excluded:
            continue
        # Consider numeric columns that look like binary labels
        if pd.api.types.is_numeric_dtype(df[col]):
            vals = df[col].dropna().unique()
            if len(vals) == 0:
                continue
            # Accept {0,1} or {0.0, 1.0}
            if set(np.unique(vals)).issubset({0, 1, 0.0, 1.0}):
                candidates.append(col)

    if not candidates:
        raise ValueError("Could not auto-detect label columns. Please set LABEL_COLUMNS explicitly.")
    return candidates


# ==============
# Model builders
# ==============
class ResNetMultiLabel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def build_vit(num_classes: int) -> nn.Module:
    if not HAS_VIT:
        raise RuntimeError("transformers is not installed. Install transformers to use ViT.")
    try:
        # Try to init from base weights; classifier will be replaced anyway
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        hidden = model.config.hidden_size
        model.classifier = nn.Linear(hidden, num_classes)
        model.config.num_labels = num_classes
        return model
    except Exception:
        # Offline / fallback
        cfg = ViTConfig(num_labels=num_classes, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072, image_size=224, patch_size=16)
        model = ViTForImageClassification(cfg)
        return model


def build_model(mtype: str, num_classes: int) -> nn.Module:
    if mtype.lower() == "resnet50":
        return ResNetMultiLabel(num_classes)
    elif mtype.lower() == "vit":
        return build_vit(num_classes)
    else:
        raise ValueError(f"Unknown model type: {mtype}")


def load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device):
    if ckpt_path is None or ckpt_path == "" or not os.path.exists(ckpt_path):
        logging.warning(f"Checkpoint not found: {ckpt_path}. Using randomly initialized weights.")
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logging.warning(f"Missing keys when loading {ckpt_path}: {missing}")
    if unexpected:
        logging.warning(f"Unexpected keys when loading {ckpt_path}: {unexpected}")


def load_thresholds(path: Optional[str], num_classes: int) -> np.ndarray:
    if path and os.path.exists(path):
        arr = np.load(path)
        if arr.shape[0] != num_classes:
            raise ValueError(f"Thresholds shape {arr.shape} != (NUM_CLASSES,)")
        return arr.astype(np.float32)
    return np.full((num_classes,), 0.5, dtype=np.float32)


# ==============
# Eval utilities
# ==============
@torch.no_grad()
def infer(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_labels = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logits = logits.detach().cpu().float().numpy()
        labels = yb.detach().cpu().float().numpy()
        all_logits.append(logits)
        all_labels.append(labels)
    return np.vstack(all_logits), np.vstack(all_labels)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def per_class_ap(y_true: np.ndarray, y_scores: np.ndarray) -> np.ndarray:
    """Returns AP per class; NaN for classes without both positive and negative examples."""
    num_classes = y_true.shape[1]
    aps = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        yt = y_true[:, c]
        ys = y_scores[:, c]
        try:
            ap = average_precision_score(yt, ys)
        except Exception:
            ap = np.nan
        aps[c] = ap
    return aps


def c_f1_macro(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> float:
    # Macro F1 across classes (label-based)
    return float(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0))


def o_f1_example(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> float:
    # Example-based F1; if both true & pred are all-zero for a sample, count F1=1 for that sample
    n = y_true_bin.shape[0]
    scores = []
    for i in range(n):
        t = y_true_bin[i]
        p = y_pred_bin[i]
        if t.sum() == 0 and p.sum() == 0:
            scores.append(1.0)
        else:
            scores.append(f1_score(t, p, average="binary", zero_division=1))
    return float(np.mean(scores))


def binarize(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (scores >= thresholds.reshape(1, -1)).astype(np.int32)


# =====
# MAIN
# =====
def main():
    device = get_device()
    logging.info(f"Using device: {device}")

    if not os.path.exists(ANNOT_PATH):
        raise FileNotFoundError(f"ANNOT_PATH not found: {ANNOT_PATH}")
    if not os.path.isdir(IMG_DIR):
        raise NotADirectoryError(f"IMG_DIR is not a directory: {IMG_DIR}")

    df = pd.read_csv(ANNOT_PATH)
    if LABEL_COLUMNS is None:
        label_cols = auto_detect_label_columns(df)
        logging.info(f"Auto-detected {len(label_cols)} label columns.")
    else:
        label_cols = LABEL_COLUMNS
    if len(label_cols) != NUM_CLASSES:
        logging.warning(f"NUM_CLASSES={NUM_CLASSES} but detected {len(label_cols)} label columns. "
                        f"Proceeding with detected columns.")

    # Dataset / Loader
    ds = EmoticNpyDataset(df, IMG_DIR, label_cols, image_size=224)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)

    # Storage
    summary_rows = []
    ap_tables = {}  # model_name -> AP vector

    # Evaluate each model
    for cfg in MODEL_CONFIGS:
        name = cfg["name"]
        mtype = cfg["type"]
        ckpt = cfg.get("checkpoint_path", None)
        th_path = cfg.get("thresholds_path", None)

        logging.info(f"--- Evaluating {name} ({mtype}) ---")
        model = build_model(mtype, num_classes=len(label_cols))
        load_checkpoint(model, ckpt, device)
        model.to(device)

        logits, y_true = infer(model, loader, device)
        probs = sigmoid(logits)

        thresholds = load_thresholds(th_path, num_classes=len(label_cols))
        y_pred = binarize(probs, thresholds)

        # Metrics
        ap_vec = per_class_ap(y_true, probs)               # per-class AP
        map_score = float(np.nanmean(ap_vec))              # mAP
        cf1 = c_f1_macro(y_true, y_pred)                   # C-F1
        of1 = o_f1_example(y_true, y_pred)                 # O-F1

        summary_rows.append({
            "Model": name,
            "Type": mtype,
            "mAP": round(map_score, 5),
            "C-F1": round(cf1, 5),
            "O-F1": round(of1, 5)
        })
        ap_tables[name] = ap_vec

    # Build summary table
    summary_df = pd.DataFrame(summary_rows).set_index("Model")
    print("\n===== SUMMARY (side by side) =====")
    print(summary_df)

    # Build per-class AP comparison
    ap_df = pd.DataFrame(ap_tables)  # rows: class idx, cols: model names
    ap_df.index = [f"class_{i}" for i in range(ap_df.shape[0])]
    print("\n===== Per-class AP (rows=classes) =====")
    print(ap_df)

    # Save CSVs
    summary_df.to_csv("summary_metrics.csv")
    ap_df.to_csv("per_class_ap.csv")
    logging.info("Saved CSVs: summary_metrics.csv, per_class_ap.csv")

    # =============
    # Plot charts
    # =============
    # 1) mAP per model
    plt.figure()
    plt.title("mAP per Model")
    plt.bar(summary_df.index.tolist(), summary_df["mAP"].tolist())
    plt.ylabel("mAP")
    plt.xlabel("Model")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("map_per_model.png", dpi=200)
    plt.close()

    # 2) C-F1 per model
    plt.figure()
    plt.title("C-F1 per Model")
    plt.bar(summary_df.index.tolist(), summary_df["C-F1"].tolist())
    plt.ylabel("C-F1")
    plt.xlabel("Model")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("cf1_per_model.png", dpi=200)
    plt.close()

    logging.info("Saved charts: map_per_model.png, cf1_per_model.png")

if __name__ == "__main__":
    main()

