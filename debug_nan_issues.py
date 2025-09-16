#!/usr/bin/env python
"""
debug_nan_issues.py
===================
Debug script to diagnose and fix NaN issues in model testing
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForImageClassification, AutoImageProcessor
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def check_model_weights(model_path):
    """Check if model weights contain NaN or Inf values"""
    print(f"\nChecking model weights: {model_path}")
    print("-" * 60)
    
    state_dict = torch.load(model_path, map_location='cpu')
    
    nan_layers = []
    inf_layers = []
    zero_layers = []
    
    for name, param in state_dict.items():
        if torch.isnan(param).any():
            nan_layers.append(name)
            nan_count = torch.isnan(param).sum().item()
            total_count = param.numel()
            print(f"  ⚠️  NaN found in {name}: {nan_count}/{total_count} values")
        
        if torch.isinf(param).any():
            inf_layers.append(name)
            inf_count = torch.isinf(param).sum().item()
            total_count = param.numel()
            print(f"  ⚠️  Inf found in {name}: {inf_count}/{total_count} values")
        
        # Check for layers that are all zeros (dead neurons)
        if (param == 0).all():
            zero_layers.append(name)
            print(f"  ⚠️  All zeros in {name}")
    
    if not nan_layers and not inf_layers and not zero_layers:
        print("  ✅ Model weights look healthy!")
    else:
        print(f"\n  Summary:")
        print(f"    - Layers with NaN: {len(nan_layers)}")
        print(f"    - Layers with Inf: {len(inf_layers)}")
        print(f"    - Dead layers (all zeros): {len(zero_layers)}")
    
    return nan_layers, inf_layers, zero_layers


def diagnose_model_outputs(run_dir, num_samples=10):
    """Run a quick diagnostic on model outputs"""
    print(f"\nDiagnosing model outputs for: {run_dir}")
    print("=" * 60)
    
    run_path = Path(run_dir)
    model_path = run_path / 'best_vit_emotic.pth'
    
    # Check model weights first
    nan_layers, inf_layers, zero_layers = check_model_weights(model_path)
    
    # Load model
    print("\nLoading model for inference test...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model = AutoModelForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        ignore_mismatched_sizes=True
    )
    
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.config.hidden_size, 26)
    )
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Test with random inputs
    print(f"\nTesting with {num_samples} random inputs...")
    print("-" * 60)
    
    feature_extractor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)
    
    issues_found = {
        'nan_outputs': 0,
        'inf_outputs': 0,
        'extreme_values': 0,
        'all_same': 0
    }
    
    for i in range(num_samples):
        # Create random image
        random_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        inputs = feature_extractor(images=random_image, return_tensors="pt")
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits)
            
            # Check for issues
            if torch.isnan(logits).any():
                issues_found['nan_outputs'] += 1
                print(f"  Sample {i+1}: ❌ NaN in logits")
            elif torch.isinf(logits).any():
                issues_found['inf_outputs'] += 1
                print(f"  Sample {i+1}: ❌ Inf in logits")
            elif logits.abs().max() > 100:
                issues_found['extreme_values'] += 1
                print(f"  Sample {i+1}: ⚠️  Extreme values (max: {logits.abs().max().item():.2f})")
            elif torch.allclose(logits[0], logits[0, 0]):
                issues_found['all_same'] += 1
                print(f"  Sample {i+1}: ⚠️  All outputs identical")
            else:
                print(f"  Sample {i+1}: ✅ OK (logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}])")
    
    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)
    
    if sum(issues_found.values()) == 0:
        print("✅ Model outputs appear normal!")
    else:
        print("❌ Issues detected:")
        for issue, count in issues_found.items():
            if count > 0:
                print(f"  - {issue.replace('_', ' ').title()}: {count}/{num_samples}")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if nan_layers or inf_layers:
        print("🔧 Model has corrupted weights. Recommendations:")
        print("  1. Check the training logs for gradient explosion")
        print("  2. Retrain with:")
        print("     - Lower learning rate (try 1e-5 or 5e-6)")
        print("     - Gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)")
        print("     - More stable optimizer (AdamW with weight decay)")
    
    elif issues_found['nan_outputs'] > 0 or issues_found['inf_outputs'] > 0:
        print("🔧 Model produces invalid outputs. Possible causes:")
        print("  1. Training instability in final epochs")
        print("  2. Check if loss went to NaN during training")
        print("  3. Try loading an earlier checkpoint if available")
    
    elif issues_found['extreme_values'] > 0:
        print("⚠️  Model produces extreme values. This might cause numerical instability.")
        print("  Consider:")
        print("  1. Adding logit clipping during inference")
        print("  2. Using mixed precision with care")
    
    elif zero_layers:
        print("⚠️  Some layers are dead (all zeros). The model might be undertrained.")
        print("  Consider:")
        print("  1. Training for more epochs")
        print("  2. Checking if learning rate was too low")
    
    else:
        print("✅ Model appears healthy! The test error might be data-related.")
        print("  Check:")
        print("  1. Test data for NaN values")
        print("  2. Test annotations for missing labels")


def test_with_nan_handling(run_dir):
    """Modified test function that handles NaN values gracefully"""
    from training_utils import parse_annotations, EMOTICDataset
    from test_models import TestConfig
    import pandas as pd
    from sklearn.metrics import average_precision_score
    
    print(f"\nRunning robust test with NaN handling for: {run_dir}")
    print("=" * 60)
    
    config = TestConfig()
    run_path = Path(run_dir)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load model
    model = AutoModelForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        ignore_mismatched_sizes=True
    )
    
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.config.hidden_size, 26)
    )
    
    model_path = run_path / 'best_vit_emotic.pth'
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Load test data
    test_annotations, _ = parse_annotations(config.get('data.test_annotations_path'))
    feature_extractor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)
    
    test_dataset = EMOTICDataset(
        test_annotations,
        config.get('data.img_dir'),
        feature_extractor,
        num_categories=26
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Run inference with NaN detection
    all_targets = []
    all_probabilities = []
    nan_batches = []
    
    print("\nRunning inference with NaN detection...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing")):
            images = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(images).logits
            
            # Check for NaN
            if torch.isnan(logits).any():
                nan_batches.append(batch_idx)
                # Replace NaN with zeros for now
                logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
            
            # Clip extreme values
            logits = torch.clamp(logits, min=-10, max=10)
            
            probabilities = torch.sigmoid(logits).cpu().numpy()
            
            all_targets.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities)
    
    all_targets = np.vstack(all_targets)
    all_probabilities = np.vstack(all_probabilities)
    
    print(f"\nNaN detected in {len(nan_batches)} out of {len(test_loader)} batches")
    if nan_batches:
        print(f"  Affected batches: {nan_batches[:10]}{'...' if len(nan_batches) > 10 else ''}")
    
    # Calculate metrics with NaN handling
    print("\nCalculating metrics with NaN handling...")
    
    # Replace any remaining NaN values
    nan_mask = np.isnan(all_probabilities)
    if nan_mask.any():
        print(f"  Found {nan_mask.sum()} NaN values in predictions, replacing with 0.5")
        all_probabilities = np.nan_to_num(all_probabilities, nan=0.5)
    
    # Calculate AP for categories with valid predictions
    emotion_names = [
        'Peace', 'Affection', 'Esteem', 'Anticipation', 'Engagement',
        'Confidence', 'Happiness', 'Pleasure', 'Excitement', 'Surprise',
        'Sympathy', 'Doubt/Confusion', 'Disconnection', 'Fatigue',
        'Embarrassment', 'Yearning', 'Disapproval', 'Aversion', 'Annoyance',
        'Anger', 'Sensitivity', 'Sadness', 'Disquietment', 'Fear', 'Pain',
        'Suffering'
    ]
    
    ap_scores = {}
    failed_categories = []
    
    for i, emotion in enumerate(emotion_names):
        if all_targets[:, i].sum() > 0:
            try:
                ap = average_precision_score(all_targets[:, i], all_probabilities[:, i])
                ap_scores[emotion] = ap
            except Exception as e:
                failed_categories.append(emotion)
                ap_scores[emotion] = 0.0
                print(f"  Failed to calculate AP for {emotion}: {e}")
        else:
            ap_scores[emotion] = float('nan')
    
    # Calculate mAP
    valid_scores = [score for score in ap_scores.values() if not np.isnan(score) and score > 0]
    map_score = np.mean(valid_scores) if valid_scores else 0.0
    
    print(f"\n" + "=" * 60)
    print("RESULTS (with NaN handling)")
    print("=" * 60)
    print(f"Mean Average Precision: {map_score:.4f}")
    print(f"Valid categories: {len(valid_scores)}/26")
    if failed_categories:
        print(f"Failed categories: {', '.join(failed_categories)}")
    
    # Save results
    results_dir = run_path / 'test_results_debug'
    results_dir.mkdir(exist_ok=True)
    
    results = {
        'map': map_score,
        'ap_scores': {k: float(v) if not np.isnan(v) else None for k, v in ap_scores.items()},
        'nan_batches': len(nan_batches),
        'total_batches': len(test_loader),
        'failed_categories': failed_categories
    }
    
    with open(results_dir / 'debug_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_dir / 'debug_results.json'}")
    
    return map_score, ap_scores


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python debug_nan_issues.py <run_directory>")
        print("Example: python debug_nan_issues.py runs/vit_emotic_20250819-192210")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    
    if not Path(run_dir).exists():
        print(f"Error: Directory '{run_dir}' not found!")
        sys.exit(1)
    
    # Run diagnosis
    diagnose_model_outputs(run_dir)
    
    # Ask if user wants to proceed with robust testing
    print("\n" + "=" * 60)
    response = input("Do you want to run the test with NaN handling? (y/n): ")
    
    if response.lower() == 'y':
        test_with_nan_handling(run_dir)