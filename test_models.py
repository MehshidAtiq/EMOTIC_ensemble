"""
Vision Transformer Model Testing Suite
======================================
Comprehensive testing framework for EMOTIC emotion recognition models.

Usage:
    python test_models.py --mode single --run-id vit_emotic_20250819-192210
    python test_models.py --mode all
    python test_models.py --mode recent --days 7
    python test_models.py --mode selective --pattern "aug"
    vit_emotic_20250819-192210
"""

import os
import json
import argparse
import glob
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Import necessary components from training script
from transformers import AutoModelForImageClassification, AutoImageProcessor


class TestConfig:
    """Configuration for model testing"""
    def __init__(self, config_path: Optional[str] = None):
        # Default configuration
        self.config = {
            'data': {
                'test_annotations_path': 'archive/annots_arrs/annot_arrs_test.csv',
                'img_dir': 'archive/img_arrs/',
                'batch_size': 32,
                'num_workers': 0
            },
            'evaluation': {
                'use_dynamic_thresholds': True,
                'fixed_threshold': 0.5,
                'calculate_optimal_thresholds': True
            },
            'metrics': {
                'calculate_ap': True,
                'calculate_map': True,
                'calculate_cf1': True,
                'calculate_of1': True,
                'save_predictions': True,
                'save_confusion_matrices': True
            },
            'output': {
                'generate_visualizations': True,
                'verbose': True,
                'save_per_sample_metrics': False
            }
        }
        
        # Load custom config if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                self._update_config(self.config, custom_config)
    
    def _update_config(self, base: dict, custom: dict):
        """Recursively update configuration"""
        for key, value in custom.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._update_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, key_path: str, default=None):
        """Get config value by dot notation (e.g., 'data.batch_size')"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value


class ModelTester:
    """Main testing class for Vision Transformer models"""
    
    def __init__(self, run_dir: str, config: TestConfig):
        self.run_dir = Path(run_dir)
        self.config = config
        self.device = self._get_device()
        self.manifest = self._load_manifest()
        self.model = None
        self.thresholds = None
        self.test_results_dir = self.run_dir / 'test_results'
        self.test_results_dir.mkdir(exist_ok=True)
        
        # Create timestamp for this test run
        self.test_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_test_dir = self.test_results_dir / f"test_{self.test_timestamp}"
        self.current_test_dir.mkdir(exist_ok=True)
        
    def _get_device(self):
        """Determine the best available device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _load_manifest(self) -> dict:
        """Load the manifest file for this model run"""
        manifest_path = self.run_dir / 'manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            return json.load(f)
    
    def load_model(self):
        """Load the trained model and thresholds"""
        print(f"Loading model from {self.run_dir}")
        
        # Initialize model architecture
        num_classes = 26
        self.model = AutoModelForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            ignore_mismatched_sizes=True
        )
        
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.model.config.hidden_size, num_classes)
        )
        
        # Load saved weights
        model_path = self.run_dir / 'best_vit_emotic.pth'
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Load thresholds
        if self.config.get('evaluation.use_dynamic_thresholds'):
            thresholds_path = self.run_dir / 'thresholds.json'
            if thresholds_path.exists():
                with open(thresholds_path, 'r') as f:
                    thresholds_data = json.load(f)
                    self.thresholds = np.array(thresholds_data['thresholds'])
            else:
                print("Warning: Dynamic thresholds not found, using fixed threshold")
                self.thresholds = np.full(26, self.config.get('evaluation.fixed_threshold', 0.5))
        else:
            self.thresholds = np.full(26, self.config.get('evaluation.fixed_threshold', 0.5))
    
    def load_test_data(self):
        """Load test dataset"""
        from training_utils import parse_annotations, EMOTICDataset  # Import from training script
        
        test_annotations_path = self.config.get('data.test_annotations_path')
        img_dir = self.config.get('data.img_dir')
        batch_size = self.config.get('data.batch_size', 32)
        
        # Parse annotations
        test_annotations, _ = parse_annotations(test_annotations_path)
        
        # Create feature extractor
        feature_extractor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)
        
        # Create dataset
        test_dataset = EMOTICDataset(
            test_annotations,
            img_dir,
            feature_extractor,
            num_categories=26
        )
        
        # Create dataloader
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.get('data.num_workers', 0)
        )
        
        return test_loader, test_annotations
    
    def run_inference(self, test_loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference on test data"""
        all_targets = []
        all_probabilities = []
        
        print("Running inference...")
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Processing batches"):
                images = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                logits = self.model(images).logits
                probabilities = torch.sigmoid(logits).cpu().numpy()
                
                all_targets.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities)
        
        all_targets = np.vstack(all_targets)
        all_probabilities = np.vstack(all_probabilities)
        
        # Apply thresholds for binary predictions
        all_predictions = (all_probabilities > self.thresholds).astype(int)
        
        return all_targets, all_probabilities, all_predictions
    
    def calculate_average_precision(self, targets: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
        """Calculate Average Precision for each category"""
        ap_scores = {}
        emotion_names = [
            'Peace', 'Affection', 'Esteem', 'Anticipation', 'Engagement',
            'Confidence', 'Happiness', 'Pleasure', 'Excitement', 'Surprise',
            'Sympathy', 'Doubt/Confusion', 'Disconnection', 'Fatigue',
            'Embarrassment', 'Yearning', 'Disapproval', 'Aversion', 'Annoyance',
            'Anger', 'Sensitivity', 'Sadness', 'Disquietment', 'Fear', 'Pain',
            'Suffering'
        ]
        
        for i, emotion in enumerate(emotion_names):
            if targets[:, i].sum() > 0:  # Only calculate if there are positive samples
                ap = average_precision_score(targets[:, i], probabilities[:, i])
                ap_scores[emotion] = ap
            else:
                ap_scores[emotion] = float('nan')
        
        return ap_scores
    
    def calculate_map(self, ap_scores: Dict[str, float]) -> float:
        """Calculate Mean Average Precision"""
        valid_scores = [score for score in ap_scores.values() if not np.isnan(score)]
        if valid_scores:
            return np.mean(valid_scores)
        return float('nan')
    
    def calculate_cf1(self, targets: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
        """Calculate Category-based F1 (Label-based F1)"""
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        # Macro average
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        # Weighted average
        weighted_f1 = np.average(f1, weights=support)
        
        return {
            'per_category_f1': f1.tolist(),
            'per_category_precision': precision.tolist(),
            'per_category_recall': recall.tolist(),
            'macro_f1': macro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'weighted_f1': weighted_f1,
            'support': support.tolist()
        }
    
    def calculate_of1(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate Overall F1 (Example-based F1)"""
        f1_scores = []
        
        for i in range(len(targets)):
            target = targets[i]
            pred = predictions[i]
            
            # Calculate F1 for this sample
            if target.sum() == 0 and pred.sum() == 0:
                f1_scores.append(1.0)
            elif target.sum() == 0 or pred.sum() == 0:
                f1_scores.append(0.0)
            else:
                tp = np.logical_and(target, pred).sum()
                fp = np.logical_and(np.logical_not(target), pred).sum()
                fn = np.logical_and(target, np.logical_not(pred)).sum()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0
                
                f1_scores.append(f1)
        
        return np.mean(f1_scores)
    
    def calculate_optimal_thresholds(self, targets: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        """Calculate optimal thresholds for each category based on F1 score"""
        optimal_thresholds = []
        
        for i in range(targets.shape[1]):
            best_threshold = 0.5
            best_f1 = 0
            
            for threshold in np.arange(0.1, 0.9, 0.05):
                preds = (probabilities[:, i] > threshold).astype(int)
                _, _, f1, _ = precision_recall_fscore_support(
                    targets[:, i], preds, average='binary', zero_division=0
                )
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            optimal_thresholds.append(best_threshold)
        
        return np.array(optimal_thresholds)
    
    def generate_visualizations(self, targets: np.ndarray, probabilities: np.ndarray, 
                              predictions: np.ndarray, metrics: dict):
        """Generate comprehensive visualizations"""
        viz_dir = self.current_test_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        emotion_names = [
            'Peace', 'Affection', 'Esteem', 'Anticipation', 'Engagement',
            'Confidence', 'Happiness', 'Pleasure', 'Excitement', 'Surprise',
            'Sympathy', 'Doubt/Confusion', 'Disconnection', 'Fatigue',
            'Embarrassment', 'Yearning', 'Disapproval', 'Aversion', 'Annoyance',
            'Anger', 'Sensitivity', 'Sadness', 'Disquietment', 'Fear', 'Pain',
            'Suffering'
        ]
        
        # 1. AP scores bar chart
        if 'average_precision' in metrics:
            plt.figure(figsize=(12, 6))
            ap_scores = metrics['average_precision']
            valid_emotions = [(e, s) for e, s in ap_scores.items() if not np.isnan(s)]
            emotions, scores = zip(*valid_emotions) if valid_emotions else ([], [])
            
            plt.bar(range(len(emotions)), scores)
            plt.xticks(range(len(emotions)), emotions, rotation=45, ha='right')
            plt.ylabel('Average Precision')
            plt.title(f'Average Precision per Category (mAP: {metrics.get("map", 0):.3f})')
            plt.tight_layout()
            plt.savefig(viz_dir / 'average_precision.png', dpi=100)
            plt.close()
        
        # 2. F1 scores comparison
        if 'cf1' in metrics:
            cf1_data = metrics['cf1']
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Per-category F1
            axes[0].bar(range(26), cf1_data['per_category_f1'])
            axes[0].set_xticks(range(26))
            axes[0].set_xticklabels(emotion_names, rotation=90, fontsize=8)
            axes[0].set_ylabel('F1 Score')
            axes[0].set_title('Per-Category F1 Scores')
            
            # Precision vs Recall
            axes[1].scatter(cf1_data['per_category_recall'], cf1_data['per_category_precision'])
            axes[1].set_xlabel('Recall')
            axes[1].set_ylabel('Precision')
            axes[1].set_title('Precision vs Recall per Category')
            axes[1].grid(True, alpha=0.3)
            
            # Summary metrics
            summary_metrics = {
                'Macro F1': cf1_data['macro_f1'],
                'Weighted F1': cf1_data['weighted_f1'],
                'Overall F1': metrics.get('of1', 0)
            }
            axes[2].bar(range(len(summary_metrics)), list(summary_metrics.values()))
            axes[2].set_xticks(range(len(summary_metrics)))
            axes[2].set_xticklabels(list(summary_metrics.keys()))
            axes[2].set_ylabel('Score')
            axes[2].set_title('Summary F1 Metrics')
            axes[2].set_ylim([0, 1])
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'f1_scores_comparison.png', dpi=100)
            plt.close()
        
        # 3. Confusion matrices for top 5 categories by support
        if self.config.get('metrics.save_confusion_matrices'):
            cf1_data = metrics['cf1']
            support = np.array(cf1_data['support'])
            top_indices = np.argsort(support)[-5:][::-1]
            
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            for idx, cat_idx in enumerate(top_indices):
                cm = confusion_matrix(targets[:, cat_idx], predictions[:, cat_idx])
                sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Blues')
                axes[idx].set_title(f'{emotion_names[cat_idx]}')
                axes[idx].set_xlabel('Predicted')
                axes[idx].set_ylabel('Actual')
            
            plt.suptitle('Confusion Matrices - Top 5 Categories by Support')
            plt.tight_layout()
            plt.savefig(viz_dir / 'confusion_matrices_top5.png', dpi=100)
            plt.close()
        
        # 4. Threshold analysis
        if self.config.get('evaluation.calculate_optimal_thresholds'):
            optimal_thresholds = self.calculate_optimal_thresholds(targets, probabilities)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(26)
            width = 0.35
            
            ax.bar(x - width/2, self.thresholds, width, label='Current Thresholds')
            ax.bar(x + width/2, optimal_thresholds, width, label='Optimal Thresholds')
            
            ax.set_xlabel('Emotion Category')
            ax.set_ylabel('Threshold Value')
            ax.set_title('Current vs Optimal Thresholds')
            ax.set_xticks(x)
            ax.set_xticklabels(emotion_names, rotation=45, ha='right', fontsize=8)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'threshold_comparison.png', dpi=100)
            plt.close()
        
        print(f"Visualizations saved to {viz_dir}")
    
    def save_results(self, metrics: dict, targets: np.ndarray, 
                    probabilities: np.ndarray, predictions: np.ndarray):
        """Save all test results"""
        
        # 1. Save metrics summary
        metrics_summary = {
            'test_timestamp': self.test_timestamp,
            'run_id': self.run_dir.name,
            'model_path': str(self.manifest.get('model_path', '')),
            'metrics': metrics,
            'config': self.config.config
        }
        
        with open(self.current_test_dir / 'metrics_summary.json', 'w') as f:
            json.dump(metrics_summary, f, indent=2, default=str)
        
        # 2. Save per-category metrics as CSV
        if 'cf1' in metrics and 'average_precision' in metrics:
            emotion_names = list(metrics['average_precision'].keys())
            cf1_data = metrics['cf1']
            
            category_df = pd.DataFrame({
                'Emotion': emotion_names,
                'AP': [metrics['average_precision'][e] for e in emotion_names],
                'F1': cf1_data['per_category_f1'],
                'Precision': cf1_data['per_category_precision'],
                'Recall': cf1_data['per_category_recall'],
                'Support': cf1_data['support'],
                'Threshold': self.thresholds.tolist()
            })
            
            category_df.to_csv(self.current_test_dir / 'per_category_metrics.csv', index=False)
        
        # 3. Save predictions if requested
        if self.config.get('metrics.save_predictions'):
            np.savez_compressed(
                self.current_test_dir / 'predictions.npz',
                targets=targets,
                probabilities=probabilities,
                predictions=predictions,
                thresholds=self.thresholds
            )
        
        # 4. Create a summary report
        self._create_summary_report(metrics)
        
        print(f"Results saved to {self.current_test_dir}")
    
    def _create_summary_report(self, metrics: dict):
        """Create a human-readable summary report"""
        report_lines = [
            "=" * 60,
            f"TEST REPORT - {self.run_dir.name}",
            f"Test Date: {self.test_timestamp}",
            "=" * 60,
            "",
            "OVERALL METRICS:",
            "-" * 30,
            f"Mean Average Precision (mAP): {metrics.get('map', 0):.4f}",
            f"Macro F1 Score (C-F1): {metrics['cf1']['macro_f1']:.4f}",
            f"Weighted F1 Score: {metrics['cf1']['weighted_f1']:.4f}",
            f"Example-based F1 (O-F1): {metrics.get('of1', 0):.4f}",
            "",
            "TOP 5 PERFORMING CATEGORIES (by AP):",
            "-" * 30
        ]
        
        # Add top performing categories
        ap_scores = metrics.get('average_precision', {})
        valid_ap = [(e, s) for e, s in ap_scores.items() if not np.isnan(s)]
        top_categories = sorted(valid_ap, key=lambda x: x[1], reverse=True)[:5]
        
        for i, (emotion, score) in enumerate(top_categories, 1):
            report_lines.append(f"{i}. {emotion}: {score:.4f}")
        
        report_lines.extend([
            "",
            "BOTTOM 5 PERFORMING CATEGORIES (by AP):",
            "-" * 30
        ])
        
        bottom_categories = sorted(valid_ap, key=lambda x: x[1])[:5]
        for i, (emotion, score) in enumerate(bottom_categories, 1):
            report_lines.append(f"{i}. {emotion}: {score:.4f}")
        
        # Save report
        with open(self.current_test_dir / 'test_report.txt', 'w') as f:
            f.write('\n'.join(report_lines))
    
    def run_test(self):
        """Main testing pipeline"""
        print("=" * 60)
        print(f"Testing model: {self.run_dir.name}")
        print("=" * 60)
        
        # Load model
        self.load_model()
        
        # Load test data
        test_loader, test_annotations = self.load_test_data()
        
        # Run inference
        targets, probabilities, predictions = self.run_inference(test_loader)
        
        # Calculate metrics
        metrics = {}
        
        if self.config.get('metrics.calculate_ap'):
            print("Calculating Average Precision...")
            metrics['average_precision'] = self.calculate_average_precision(targets, probabilities)
        
        if self.config.get('metrics.calculate_map'):
            print("Calculating Mean Average Precision...")
            metrics['map'] = self.calculate_map(metrics['average_precision'])
        
        if self.config.get('metrics.calculate_cf1'):
            print("Calculating Category-based F1...")
            metrics['cf1'] = self.calculate_cf1(targets, predictions)
        
        if self.config.get('metrics.calculate_of1'):
            print("Calculating Example-based F1...")
            metrics['of1'] = self.calculate_of1(targets, predictions)
        
        # Generate visualizations
        if self.config.get('output.generate_visualizations'):
            print("Generating visualizations...")
            self.generate_visualizations(targets, probabilities, predictions, metrics)
        
        # Save results
        self.save_results(metrics, targets, probabilities, predictions)
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"mAP: {metrics.get('map', 0):.4f}")
        print(f"Macro F1 (C-F1): {metrics['cf1']['macro_f1']:.4f}")
        print(f"Weighted F1: {metrics['cf1']['weighted_f1']:.4f}")
        print(f"Example-based F1 (O-F1): {metrics.get('of1', 0):.4f}")
        print("=" * 60)
        
        return metrics


class ModelTestRunner:
    """Orchestrates testing of multiple models"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.runs_base_dir = Path('runs')
        self.results_summary = []
    
    def find_models(self, mode: str, **kwargs) -> List[Path]:
        """Find models to test based on mode"""
        models = []
        
        if mode == 'single':
            run_id = kwargs.get('run_id')
            if not run_id:
                raise ValueError("run_id required for single mode")
            model_path = self.runs_base_dir / run_id
            if model_path.exists():
                models.append(model_path)
            else:
                raise FileNotFoundError(f"Model not found: {model_path}")
        
        elif mode == 'all':
            # Find all models with manifest.json
            for run_dir in self.runs_base_dir.iterdir():
                if run_dir.is_dir() and (run_dir / 'manifest.json').exists():
                    models.append(run_dir)
        
        elif mode == 'recent':
            days = kwargs.get('days', 7)
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for run_dir in self.runs_base_dir.iterdir():
                if run_dir.is_dir() and (run_dir / 'manifest.json').exists():
                    # Parse date from directory name
                    try:
                        date_str = run_dir.name.split('_')[-1].split('-')[0]
                        dir_date = datetime.strptime(date_str, '%Y%m%d')
                        if dir_date >= cutoff_date:
                            models.append(run_dir)
                    except:
                        continue
        
        elif mode == 'selective':
            pattern = kwargs.get('pattern', '')
            for run_dir in self.runs_base_dir.iterdir():
                if run_dir.is_dir() and pattern in run_dir.name and (run_dir / 'manifest.json').exists():
                    models.append(run_dir)
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return sorted(models)
    
    def test_models(self, mode: str, **kwargs):
        """Test models based on specified mode"""
        models = self.find_models(mode, **kwargs)
        
        if not models:
            print("No models found matching criteria")
            return
        
        print(f"Found {len(models)} model(s) to test")
        print("-" * 60)
        
        for model_dir in models:
            try:
                tester = ModelTester(str(model_dir), self.config)
                metrics = tester.run_test()
                
                self.results_summary.append({
                    'model': model_dir.name,
                    'test_timestamp': tester.test_timestamp,
                    'map': metrics.get('map', 0),
                    'macro_f1': metrics['cf1']['macro_f1'] if 'cf1' in metrics else 0,
                    'weighted_f1': metrics['cf1']['weighted_f1'] if 'cf1' in metrics else 0,
                    'of1': metrics.get('of1', 0),
                    'status': 'success'
                })
                
            except Exception as e:
                print(f"Error testing {model_dir.name}: {str(e)}")
                self.results_summary.append({
                    'model': model_dir.name,
                    'status': 'failed',
                    'error': str(e)
                })
            
            print("\n" + "=" * 60 + "\n")
        
        # Save overall summary
        self.save_overall_summary()
    
    def save_overall_summary(self):
        """Save summary of all tested models"""
        if not self.results_summary:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_dir = Path('test_summaries')
        summary_dir.mkdir(exist_ok=True)
        
        # Create DataFrame for easy viewing
        df = pd.DataFrame(self.results_summary)
        
        # Save as CSV
        csv_path = summary_dir / f'test_summary_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        
        # Save as JSON
        json_path = summary_dir / f'test_summary_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(self.results_summary, f, indent=2, default=str)
        
        # Print summary table
        print("\n" + "=" * 80)
        print("OVERALL TESTING SUMMARY")
        print("=" * 80)
        
        if 'map' in df.columns:
            successful_tests = df[df['status'] == 'success']
            if not successful_tests.empty:
                print(f"\nSuccessfully tested {len(successful_tests)} model(s):")
                print("-" * 80)
                print(successful_tests[['model', 'map', 'macro_f1', 'of1']].to_string(index=False))
                
                print("\n" + "-" * 80)
                print(f"Best model by mAP: {successful_tests.loc[successful_tests['map'].idxmax(), 'model']}")
                print(f"Best model by Macro F1: {successful_tests.loc[successful_tests['macro_f1'].idxmax(), 'model']}")
                print(f"Best model by O-F1: {successful_tests.loc[successful_tests['of1'].idxmax(), 'model']}")
        
        failed_tests = df[df['status'] == 'failed']
        if not failed_tests.empty:
            print(f"\nFailed to test {len(failed_tests)} model(s):")
            for _, row in failed_tests.iterrows():
                print(f"  - {row['model']}: {row.get('error', 'Unknown error')}")
        
        print(f"\nSummary saved to: {csv_path}")
        print("=" * 80)


def main():
    """Main entry point for the testing suite"""
    parser = argparse.ArgumentParser(description='Test Vision Transformer Models')
    
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'all', 'recent', 'selective'],
                       help='Testing mode')
    parser.add_argument('--run-id', type=str,
                       help='Specific run ID for single mode')
    parser.add_argument('--days', type=int, default=7,
                       help='Number of days for recent mode')
    parser.add_argument('--pattern', type=str,
                       help='Pattern to match for selective mode')
    parser.add_argument('--config', type=str,
                       help='Path to custom configuration file')
    parser.add_argument('--batch-size', type=int,
                       help='Override batch size')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--use-fixed-threshold', action='store_true',
                       help='Use fixed threshold instead of dynamic')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Fixed threshold value')
    
    args = parser.parse_args()
    
    # Load configuration
    config = TestConfig(args.config)
    
    # Override config with command line arguments
    if args.batch_size:
        config.config['data']['batch_size'] = args.batch_size
    if args.no_viz:
        config.config['output']['generate_visualizations'] = False
    if args.use_fixed_threshold:
        config.config['evaluation']['use_dynamic_thresholds'] = False
        config.config['evaluation']['fixed_threshold'] = args.threshold
    
    # Run tests
    runner = ModelTestRunner(config)
    
    if args.mode == 'single':
        runner.test_models('single', run_id=args.run_id)
    elif args.mode == 'all':
        runner.test_models('all')
    elif args.mode == 'recent':
        runner.test_models('recent', days=args.days)
    elif args.mode == 'selective':
        runner.test_models('selective', pattern=args.pattern)


if __name__ == "__main__":
    main()