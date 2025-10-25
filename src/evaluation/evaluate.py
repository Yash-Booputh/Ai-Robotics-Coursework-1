"""
Model Evaluation Module

Comprehensive evaluation of trained models (ResNet or YOLO) on test set
with detailed performance metrics and visualizations for academic reporting.
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import yaml
from datetime import datetime

from src.models.resnet_classifier import create_resnet_model
from src.models.yolo_classifier import YOLOClassifier
from src.data.dataset import DataLoaderFactory
from src.evaluation.metrics import (
    calculate_metrics,
    calculate_per_class_accuracy,
    get_confusion_matrix,
    get_classification_report,
    analyze_errors
)
from src.evaluation.visualization import (
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_class_distribution
)


class ModelEvaluator:

    def __init__(
            self,
            model_type: str,
            model_path: str,
            config_path: str = "config/config.yaml"
    ):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_type = model_type.lower()
        self.model_path = Path(model_path)

        # Create model-specific results directory
        base_results_path = Path(self.config['paths']['results'])
        self.results_path = base_results_path / self.model_type
        self.results_path.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print("\n" + "=" * 80)
        print("MODEL EVALUATION SYSTEM")
        print("=" * 80)
        print(f"Model Type:        {model_type.upper()}")
        print(f"Model Path:        {model_path}")
        print(f"Results Directory: {self.results_path}")
        print(f"Computation Device: {self.device}")
        print(f"Evaluation Date:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

    def load_model(self):
        """Load trained model and class names"""
        class_names_path = Path(self.config['paths']['models']) / 'class_names.json'

        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)

        self.num_classes = len(self.class_names)

        print(f"\nLoading model...")
        print(f"Number of classes: {self.num_classes}")
        print(f"Class names: {', '.join(self.class_names)}")

        if self.model_type in ['resnet34', 'resnet50']:
            self._load_resnet_model()
        elif self.model_type in ['yolo11m', 'yolo12m']:
            self._load_yolo_model()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _load_resnet_model(self):
        """Load ResNet model from checkpoint"""
        print(f"\nInitializing {self.model_type.upper()} architecture...")

        self.model = create_resnet_model(
            num_classes=self.num_classes,
            model_name=self.model_type,
            pretrained=False
        )

        print(f"Loading trained weights from: {self.model_path}")
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Model loaded successfully")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def _load_yolo_model(self):
        """Load YOLO model from checkpoint"""
        print(f"\nInitializing {self.model_type.upper()} architecture...")
        print(f"Loading trained weights from: {self.model_path}")

        self.model = YOLOClassifier(model_path=str(self.model_path))
        print(f"Model loaded successfully")

    def setup_data(self):
        """Setup data loaders for evaluation"""
        print("\nSetting up data loaders...")

        data_factory = DataLoaderFactory(config_path="config/config.yaml")
        self.dataloaders, self.dataset_sizes, _ = data_factory.create_dataloaders()

        print(f"Test set size: {self.dataset_sizes['test']} images")
        batch_size = self.config['training'].get('batch_size', 32)
        print(f"Batch size: {batch_size}")

    def evaluate_resnet(self):
        """Evaluate ResNet model on test set"""
        all_preds = []
        all_labels = []
        all_probs = []

        print("\n" + "-" * 80)
        print("Running inference on test set...")
        print("-" * 80)

        with torch.no_grad():
            for inputs, labels in tqdm(self.dataloaders['test'],
                                      desc='Processing batches',
                                      unit='batch'):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_preds), np.array(all_labels), np.array(all_probs)

    def evaluate_yolo(self):
        """Evaluate YOLO model on test set"""
        all_preds = []
        all_labels = []
        all_probs = []

        test_path = Path(self.config['data']['test_path'])

        print("\n" + "-" * 80)
        print("Running inference on test set...")
        print("-" * 80)

        total_images = 0
        for class_name in self.class_names:
            class_path = test_path / class_name
            images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
            total_images += len(images)

        print(f"Total test images: {total_images}")

        with tqdm(total=total_images, desc='Processing images', unit='img') as pbar:
            for class_idx, class_name in enumerate(self.class_names):
                class_path = test_path / class_name
                images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))

                for img_path in images:
                    results = self.model.predict(str(img_path), verbose=False)

                    probs = results[0].probs.data.cpu().numpy()
                    pred_idx = np.argmax(probs)

                    all_preds.append(pred_idx)
                    all_labels.append(class_idx)
                    all_probs.append(probs)

                    pbar.update(1)

        return np.array(all_preds), np.array(all_labels), np.array(all_probs)

    def evaluate(self):
        """Run complete evaluation pipeline"""
        print("\n" + "=" * 80)
        print(f"EVALUATION PIPELINE - {self.model_type.upper()}")
        print("=" * 80)

        # Load model
        self.load_model()

        # Run predictions
        if self.model_type in ['resnet34', 'resnet50']:
            self.setup_data()
            all_preds, all_labels, all_probs = self.evaluate_resnet()
        else:
            all_preds, all_labels, all_probs = self.evaluate_yolo()

        print("\nInference complete. Computing metrics...")

        # Calculate metrics
        metrics = calculate_metrics(all_labels, all_preds)
        class_accuracies = calculate_per_class_accuracy(all_labels, all_preds, self.num_classes)
        cm = get_confusion_matrix(all_labels, all_preds)
        report = get_classification_report(all_labels, all_preds, self.class_names)
        error_analysis = analyze_errors(cm, self.class_names, top_k=10)

        # Print results
        self.print_results(metrics, class_accuracies, cm, error_analysis)

        # Save results
        self.save_results(metrics, class_accuracies, cm, report, all_labels, all_preds, error_analysis)

        # Generate visualizations
        self.generate_visualizations(cm, metrics, class_accuracies)

        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
        print(f"\nAll results saved to: {self.results_path}")
        print("\nGenerated files:")
        print(f"  - test_results_{self.model_type}.json")
        print(f"  - classification_report_{self.model_type}.txt")
        print(f"  - confusion_matrix_{self.model_type}.png")
        print(f"  - confusion_matrix_normalized_{self.model_type}.png")
        print(f"  - per_class_performance_{self.model_type}.png")
        print(f"  - class_distribution_{self.model_type}.png")
        print("=" * 80 + "\n")

    def print_results(self, metrics: dict, class_accuracies: np.ndarray,
                     cm: np.ndarray, error_analysis: list):
        """Print detailed evaluation results"""
        print("\n" + "=" * 80)
        print("PERFORMANCE METRICS")
        print("=" * 80)

        print("\nOverall Performance:")
        print("-" * 80)
        print(f"Test Accuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
        print(f"Macro-averaged F1:      {metrics['macro_f1']:.4f} ({metrics['macro_f1'] * 100:.2f}%)")
        print(f"Weighted-averaged F1:   {metrics['weighted_f1']:.4f} ({metrics['weighted_f1'] * 100:.2f}%)")

        print("\n" + "-" * 80)
        print("Per-Class Performance:")
        print("-" * 80)
        print(f"{'Class':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 80)

        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:<15} "
                  f"{class_accuracies[i]:>10.4f}  "
                  f"{metrics['precision'][i]:>10.4f}  "
                  f"{metrics['recall'][i]:>10.4f}  "
                  f"{metrics['f1_score'][i]:>10.4f}  "
                  f"{int(metrics['support'][i]):>8}")

        print("-" * 80)
        print(f"{'MACRO AVERAGE':<15} "
              f"{np.mean(class_accuracies):>10.4f}  "
              f"{np.mean(metrics['precision']):>10.4f}  "
              f"{np.mean(metrics['recall']):>10.4f}  "
              f"{np.mean(metrics['f1_score']):>10.4f}  "
              f"{int(np.sum(metrics['support'])):>8}")

        # Confusion matrix statistics
        print("\n" + "-" * 80)
        print("Confusion Matrix Statistics:")
        print("-" * 80)

        total_predictions = np.sum(cm)
        correct_predictions = np.trace(cm)
        incorrect_predictions = total_predictions - correct_predictions

        print(f"Total predictions:      {int(total_predictions)}")
        print(f"Correct predictions:    {int(correct_predictions)} ({correct_predictions/total_predictions*100:.2f}%)")
        print(f"Incorrect predictions:  {int(incorrect_predictions)} ({incorrect_predictions/total_predictions*100:.2f}%)")

        # Error analysis
        if error_analysis:
            print("\n" + "-" * 80)
            print("Top 10 Most Common Misclassifications:")
            print("-" * 80)
            print(f"{'Count':<8} {'True Class':<15} {'Predicted As':<15}")
            print("-" * 80)

            for count, true_class, pred_class in error_analysis:
                print(f"{count:<8} {true_class:<15} {pred_class:<15}")

        # Best and worst performing classes
        best_class_idx = np.argmax(class_accuracies)
        worst_class_idx = np.argmin(class_accuracies)

        print("\n" + "-" * 80)
        print("Performance Highlights:")
        print("-" * 80)
        print(f"Best performing class:   {self.class_names[best_class_idx]} ({class_accuracies[best_class_idx]*100:.2f}%)")
        print(f"Worst performing class:  {self.class_names[worst_class_idx]} ({class_accuracies[worst_class_idx]*100:.2f}%)")
        print(f"Performance variance:    {np.std(class_accuracies):.4f}")

    def save_results(self, metrics: dict, class_accuracies: np.ndarray,
                     cm: np.ndarray, report: str, all_labels: np.ndarray,
                     all_preds: np.ndarray, error_analysis: list):
        """Save comprehensive evaluation results"""

        # Prepare detailed results dictionary
        results = {
            'model_information': {
                'model_type': self.model_type,
                'model_path': str(self.model_path),
                'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'device': str(self.device),
                'num_classes': self.num_classes,
                'class_names': self.class_names
            },
            'overall_metrics': {
                'test_accuracy': float(metrics['accuracy']),
                'macro_f1': float(metrics['macro_f1']),
                'weighted_f1': float(metrics['weighted_f1']),
                'macro_precision': float(np.mean(metrics['precision'])),
                'macro_recall': float(np.mean(metrics['recall']))
            },
            'per_class_metrics': {}
        }

        # Per-class detailed metrics
        for i, class_name in enumerate(self.class_names):
            results['per_class_metrics'][class_name] = {
                'accuracy': float(class_accuracies[i]),
                'precision': float(metrics['precision'][i]),
                'recall': float(metrics['recall'][i]),
                'f1_score': float(metrics['f1_score'][i]),
                'support': int(metrics['support'][i])
            }

        # Confusion matrix statistics
        results['confusion_matrix_stats'] = {
            'total_predictions': int(np.sum(cm)),
            'correct_predictions': int(np.trace(cm)),
            'incorrect_predictions': int(np.sum(cm) - np.trace(cm)),
            'accuracy_from_cm': float(np.trace(cm) / np.sum(cm))
        }

        # Error analysis
        results['error_analysis'] = [
            {
                'count': int(count),
                'true_class': true_class,
                'predicted_class': pred_class
            }
            for count, true_class, pred_class in error_analysis
        ]

        # Performance highlights
        best_class_idx = np.argmax(class_accuracies)
        worst_class_idx = np.argmin(class_accuracies)

        results['performance_highlights'] = {
            'best_class': {
                'name': self.class_names[best_class_idx],
                'accuracy': float(class_accuracies[best_class_idx])
            },
            'worst_class': {
                'name': self.class_names[worst_class_idx],
                'accuracy': float(class_accuracies[worst_class_idx])
            },
            'accuracy_std': float(np.std(class_accuracies)),
            'accuracy_range': float(class_accuracies[best_class_idx] - class_accuracies[worst_class_idx])
        }

        # Save JSON results
        results_file = self.results_path / f'test_results_{self.model_type}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved: {results_file}")

        # Save classification report
        report_file = self.results_path / f'classification_report_{self.model_type}.txt'
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"{self.model_type.upper()} - COMPREHENSIVE EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Model Type: {self.model_type.upper()}\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Classes: {self.num_classes}\n")
            f.write(f"Test Set Size: {len(all_labels)} images\n")
            f.write("\n" + "=" * 80 + "\n")
            f.write("CLASSIFICATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(report)
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("CONFUSION MATRIX ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total Predictions: {np.sum(cm)}\n")
            f.write(f"Correct: {np.trace(cm)} ({np.trace(cm)/np.sum(cm)*100:.2f}%)\n")
            f.write(f"Incorrect: {np.sum(cm) - np.trace(cm)} ({(np.sum(cm) - np.trace(cm))/np.sum(cm)*100:.2f}%)\n")

        print(f"Classification report saved: {report_file}")

    def generate_visualizations(self, cm: np.ndarray, metrics: dict, class_accuracies: np.ndarray):
        """Generate professional visualizations for report"""
        print("\nGenerating visualizations...")

        # Non-normalized confusion matrix
        plot_confusion_matrix(
            cm,
            self.class_names,
            save_path=str(self.results_path / f'confusion_matrix_{self.model_type}.png'),
            title=f'Confusion Matrix - {self.model_type.upper()}',
            normalize=False
        )
        print(f"  Confusion matrix saved")

        # Normalized confusion matrix
        plot_confusion_matrix(
            cm,
            self.class_names,
            save_path=str(self.results_path / f'confusion_matrix_normalized_{self.model_type}.png'),
            title=f'Normalized Confusion Matrix - {self.model_type.upper()}',
            normalize=True,
            cmap='RdYlGn'
        )
        print(f"  Normalized confusion matrix saved")

        # Per-class metrics
        plot_per_class_metrics(
            self.class_names,
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score'],
            save_path=str(self.results_path / f'per_class_performance_{self.model_type}.png'),
            title=f'Per-Class Performance Metrics - {self.model_type.upper()}'
        )
        print(f"  Per-class performance saved")

        # Class distribution
        plot_class_distribution(
            self.class_names,
            metrics['support'],
            save_path=str(self.results_path / f'class_distribution_{self.model_type}.png')
        )
        print(f"  Class distribution saved")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate trained model with comprehensive metrics')
    parser.add_argument('--model-type', type=str, required=True,
                        choices=['resnet34', 'resnet50', 'yolo11m', 'yolo12m'],
                        help='Model type to evaluate')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model weights')

    args = parser.parse_args()

    evaluator = ModelEvaluator(
        model_type=args.model_type,
        model_path=args.model_path
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()