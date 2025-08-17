import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm  # For progress bars
import logging
from pathlib import Path


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

SEED = 464
seed_everything(SEED)

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)

# -------------------------
# Device
# -------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# -------------------------
# Parse Annotations
# -------------------------
def parse_annotations(csv_path, num_categories=26):
    df = pd.read_csv(csv_path)
    annotations = []

    # Adjust category_columns based on your CSV
    category_columns = df.columns[9:39]

    for _, row in df.iterrows():
        categories = [int(idx) for idx, val in enumerate(row[category_columns]) if val == 1 and idx < num_categories]
        annotation = {
            'filename': row['Crop_name'],
            'categories': categories,
        }
        annotations.append(annotation)

    return annotations

def extend_with_augmentations(annotations, aug_dir):
    """
    Clone labels from originals for each aug file in aug_dir named like: aug_{k}_{base}.npy
    where {base}.npy exists in originals.
    """
    if not aug_dir or not os.path.isdir(aug_dir):
        logging.warning(f"No augmentation folder found at: {aug_dir}")
        return annotations

    # map base filename (without .npy) -> categories list from originals
    base_to_cats = {}
    for a in annotations:
        base = os.path.splitext(a['filename'])[0]
        base_to_cats[base] = a['categories']

    new_entries = []
    for fname in os.listdir(aug_dir):
        if not fname.endswith(".npy") or not fname.startswith("aug_"):
            continue
        # robustly extract base after the numeric counter
        parts = fname.split("_", 2)  # ['aug', '{k}', '{base}...']
        if len(parts) < 3:
            continue
        base = os.path.splitext(parts[2])[0]
        if base in base_to_cats:
            new_entries.append({"filename": fname, "categories": base_to_cats[base]})

    if not new_entries:
        logging.warning("No augmented items matched originals; check filenames like 'aug_{k}_{base}.npy'.")
        return annotations

    logging.info(f"Extending train annotations with {len(new_entries)} augmented items.")
    return annotations + new_entries


# -------------------------
# Dataset Definition
# -------------------------
class EMOTICDataset(Dataset):
    def __init__(self, annotations, img_dir, transform=None, num_categories=26, aug_dir=None):
        self.annotations = annotations
        self.img_dir = img_dir
        self.aug_dir = aug_dir
        self.transform = transform
        self.num_categories = num_categories

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        entry = self.annotations[idx]
        fname = entry['filename']
        if self.aug_dir and str(fname).startswith("aug_"):
            img_path = os.path.join(self.aug_dir, fname)
        else:
            img_path = os.path.join(self.img_dir, fname)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File not found: {img_path}")

        image = np.load(img_path)

        # Ensure image is RGB
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] != 3:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        image_tensor = self.transform(image)
        categories = torch.zeros(self.num_categories, dtype=torch.float32)
        for category in entry['categories']:
            if 0 <= category < self.num_categories:
                categories[category] = 1.0

        return image_tensor, categories


# -------------------------
# Model Definition
# -------------------------
class ResNet50EmotionModel(nn.Module):
    def __init__(self, num_classes=26, freeze=True):
        super(ResNet50EmotionModel, self).__init__()
        self.cnn = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Freeze earlier layers
        if freeze:
            for name, param in self.cnn.named_parameters():
                if "layer4" in name or "fc" in name:  # Fine-tune layer4 and fc
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        self.cnn.fc = nn.Identity()
        self.fc_categories = nn.Linear(2048, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        categories = self.fc_categories(features)
        return categories


# -------------------------
# Focal Loss Definition
# -------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Modulation
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# -------------------------
# Dynamic Threshold Calculation
# -------------------------
def calculate_dynamic_thresholds(model, val_loader, device):
    model.eval()
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for images, categories in tqdm(val_loader, desc="Calculating Thresholds", leave=True):
            images = images.to(device)
            categories = categories.to(device)

            outputs = model(images)
            all_targets.extend(categories.cpu().numpy())
            all_outputs.extend(torch.sigmoid(outputs).cpu().numpy())

    all_targets = np.vstack(all_targets)
    all_outputs = np.vstack(all_outputs)

    thresholds = []
    for i in range(all_targets.shape[1]):
        best_threshold = 0.5
        best_f1 = 0
        for threshold in np.arange(0.1, 0.9, 0.05):
            preds = (all_outputs[:, i] > threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(all_targets[:, i], preds, average="binary")
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        thresholds.append(best_threshold)

    return np.array(thresholds)


# -------------------------
# Training Function with Progress Bar
# -------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, epoch, num_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs}", leave=True)

    for batch_idx, (images, categories) in enumerate(progress_bar):
        images, categories = images.to(device), categories.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, categories)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"Batch Loss": loss.item()})

    avg_loss = total_loss / len(loader)
    logging.info(f"Epoch {epoch}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")
    return avg_loss


# -------------------------
# Main Function
# -------------------------
def main():
    # Paths
    train_annotations_path = "archive/annots_arrs/annot_arrs_train.csv"
    val_annotations_path = "archive/annots_arrs/annot_arrs_val.csv"
    img_dir = "archive/img_arrs"
    aug_dir = "archive/augmented_img_arrs"
    bce_model_path = "weighted_bce_resnet50_emotic.pth"
    focal_model_path = "focal_loss_resnet50_emotic.pth"
    bce_thresholds_path = "bce_dynamic_thresholds.npy"
    focal_thresholds_path = "focal_dynamic_thresholds.npy"

    # Hyperparameters
    batch_size = 16
    num_classes = 26
    lr = 1e-4
    # I make 5 because my computer is not suitable but u can adjust
    epochs = 10

    # Transformations
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Parse Annotations
    train_annotations = parse_annotations(train_annotations_path, num_categories=num_classes)
    train_annotations = extend_with_augmentations(train_annotations, aug_dir)
    val_annotations = parse_annotations(val_annotations_path, num_categories=num_classes)

    # Datasets and Loaders
    train_dataset = EMOTICDataset(train_annotations, img_dir=img_dir, aug_dir=aug_dir, transform=train_transform, num_categories=num_classes)
    val_dataset = EMOTICDataset(val_annotations, img_dir=img_dir, transform=val_transform, num_categories=num_classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Models
    bce_model = ResNet50EmotionModel(num_classes=num_classes).to(device)
    focal_model = ResNet50EmotionModel(num_classes=num_classes).to(device)

    # Optimizers
    bce_optimizer = torch.optim.Adam(bce_model.parameters(), lr=lr)
    focal_optimizer = torch.optim.Adam(focal_model.parameters(), lr=lr)

    # Class Weights
    class_counts = np.zeros(num_classes)
    for annotation in train_annotations:
        for category in annotation['categories']:
            if 0 <= category < num_classes:
                class_counts[category] += 1
    total_samples = len(train_annotations)
    class_weights = torch.tensor(total_samples / (num_classes * class_counts), dtype=torch.float32).to(device)

    # Loss Functions
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    focal_loss_fn = FocalLoss(gamma=2, alpha=class_weights)

    # Training Loop
    for epoch in range(1, epochs + 1):  # Start from 1 for better readability
        # Train BCE Model
        bce_loss = train_one_epoch(bce_model, train_loader, bce_optimizer, bce_loss_fn, epoch, epochs)

        # Train Focal Loss Model
        focal_loss = train_one_epoch(focal_model, train_loader, focal_optimizer, focal_loss_fn, epoch, epochs)

        logging.info(f"Epoch {epoch}/{epochs}: BCE Loss: {bce_loss:.4f}, Focal Loss: {focal_loss:.4f}")

    # Save Models
    torch.save(bce_model.state_dict(), bce_model_path)
    torch.save(focal_model.state_dict(), focal_model_path)

    # Calculate and Save Dynamic Thresholds for BCE Model
    bce_thresholds = calculate_dynamic_thresholds(bce_model, val_loader, device)
    np.save(bce_thresholds_path, bce_thresholds)

    # Calculate and Save Dynamic Thresholds for Focal Loss Model
    focal_thresholds = calculate_dynamic_thresholds(focal_model, val_loader, device)
    np.save(focal_thresholds_path, focal_thresholds)

    logging.info("Dynamic thresholds saved for both models.")


if __name__ == "__main__":
    main()
