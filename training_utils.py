"""
training_utils.py
=================
Utilities extracted from the training script for use in testing.
Place this file in the same directory as test_models.py
"""

import os
import numpy as np
import pandas as pd
import torch
from PIL import Image


def parse_annotations(csv_path):
    """Parse EMOTIC annotations from CSV file"""
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Select only numeric category columns (columns 8:34)
    category_columns = df.columns[8:34]
    
    # Debugging: Check if the selected columns are numeric
    print("Selected category columns:", category_columns)
    print("Column types:", df[category_columns].dtypes)
    
    # Ensure all data in category columns is numeric
    for col in category_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col} contains non-numeric data.")
    
    # Calculate class counts
    class_counts = df[category_columns].sum().to_numpy(dtype=np.float32)
    
    # Parse annotations
    annotations = []
    for _, row in df.iterrows():
        categories = [int(idx) for idx, val in enumerate(row[category_columns]) if val == 1]
        annotation = {"filename": row["Crop_name"], "categories": categories}
        annotations.append(annotation)
    
    return annotations, class_counts


class EMOTICDataset(torch.utils.data.Dataset):
    """Dataset class for EMOTIC emotion recognition"""
    
    def __init__(self,
                 annotations,
                 img_dir,
                 feature_extractor,
                 num_categories,
                 aug_dir=None,
                 mode="none",
                 strict_checks=True):
        self.annotations = annotations
        self.img_dir = img_dir
        self.feature_extractor = feature_extractor
        self.num_categories = num_categories
        self.aug_dir = aug_dir
        self.mode = mode
        self.strict_checks = bool(strict_checks)
        
        # Get device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Preflight checks for augmented mode
        if self.mode == "with_aug":
            if self.aug_dir is None:
                raise ValueError("AUG_MODE is 'with_aug' but aug_dir is None.")
            
            names = [str(e["filename"]) for e in self.annotations]
            aug_in_csv = {n for n in names if n.startswith("aug_")}
            
            try:
                aug_files_on_disk = set(os.listdir(self.aug_dir))
            except FileNotFoundError:
                raise FileNotFoundError(f"Augmented images directory not found: {self.aug_dir}")
            
            missing = sorted(aug_in_csv - aug_files_on_disk)
            if missing:
                msg = (f"[EMOTICDataset] {len(missing)} augmented files referenced in CSV "
                      f"are missing from {self.aug_dir}. Example: {missing[:3]}")
                if self.strict_checks:
                    raise FileNotFoundError(msg)
                else:
                    keep = set(names) & aug_files_on_disk
                    keep_mask = [(not fn.startswith("aug_")) or (fn in keep) for fn in names]
                    dropped = sum(not k for k in keep_mask)
                    print(msg + f" — Dropping {dropped} rows (strict_checks=False).")
                    self.annotations = [e for e, k in zip(self.annotations, keep_mask) if k]
                    names = [str(e["filename"]) for e in self.annotations]
            
            n_aug = sum(n.startswith("aug_") for n in names)
            n_all = len(names)
            print(f"[EMOTICDataset] with_aug mode: rows={n_all}, aug_rows={n_aug}, orig_rows={n_all - n_aug}")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        entry = self.annotations[idx]
        
        # Select the correct directory
        fname = str(entry['filename'])
        if fname.startswith("aug_"):
            base_dir = getattr(self, "aug_dir", None) or self.img_dir
        else:
            base_dir = self.img_dir
        
        img_path = os.path.join(base_dir, fname)
        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"File not found: {img_path} "
                f"(fname='{fname}', base_dir='{base_dir}')"
            )
        
        # Load the image array and ensure 3-channel RGB
        image = np.load(img_path)
        if image.ndim == 2:  # (H, W) grayscale
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[-1] == 1:  # (H, W, 1)
            image = np.repeat(image, 3, axis=-1)
        elif image.ndim == 3 and image.shape[-1] != 3:
            raise ValueError(f"Unexpected image shape {image.shape} for '{fname}'")
        
        # Preprocess with feature extractor
        inputs = self.feature_extractor(images=image, return_tensors="pt", antialias=True)
        inputs = {key: val.squeeze(0).to(self.device) for key, val in inputs.items()}
        
        # Multi-hot encoding for labels
        categories = torch.zeros(self.num_categories, dtype=torch.float32).to(self.device)
        for category in entry['categories']:
            if category < self.num_categories:
                categories[category] = 1.0
        
        inputs["labels"] = categories
        return inputs