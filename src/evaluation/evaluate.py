"""
Model Evaluation Module

Evaluates trained models (ResNet or YOLO) on test set
and generates comprehensive performance metrics.
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import yaml

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
        self.results_path = Path(self.config['paths']['results'])
        self.results_path.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(f"Evaluating {model_type} model")
        print(f"Model path: {model_path}")
        print(f"Device: {self.device}")

    def load_model(self):
        with open(self.config['paths']['models'] + '/class_names.json', 'r') as f:
            self.class_names = json.load(f)

        self.num_classes = len(self.class_names)

        if self.model_type in ['resnet34', 'resnet50']:
            self._load_resnet_model()
        elif self.model_type in ['yolo11m', 'yolo12m']:
            self._load_yolo_model()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _load_resnet_model(self):
        self.model = create_resnet_model(
            num_classes=self.num_classes,
            model_name=self.model_type,
            pretrained=False
        )
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"ResNet model loaded: {self.model_type}")

    def _load_yolo_model(self):
        self.model = YOLOClassifier(model_path=str(self.model_path))
        print(f"YOLO model loaded: {self.model_type}")

    def setup_data(self):
        data_factory = DataLoaderFactory(config_path="config/config.yaml")
        self.dataloaders, self.dataset_sizes, _ = data_factory.create_dataloaders()

        print(f"\nTest set: {self.dataset_sizes['test']} images")

    def evaluate_resnet(self):
        all_preds = []
        all_labels = []
        all_probs = []

        print("\nRunning predictions on test set...")
        with torch.no_grad():
            for inputs, labels in tqdm(self.dataloaders['test'], desc='Testing'):
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
        all_preds = []
        all_labels = []
        all_probs = []

        test_path = Path(self.config['data']['test_path'])

        print("\nRunning predictions on test set...")
        for class_idx, class_name in enumerate(self.class_names):
            class_path = test_path / class_name
            images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))

            for img_path in tqdm(images, desc=f'{class_name}'):
                results = self.model.predict(str(img_path), verbose=False)

                probs = results[0].probs.data.cpu().numpy()
                pred_idx = np.argmax(probs)

                all_preds.append(pred_idx)
                all_labels.append(class_idx)
                all_probs.append(probs)

        return np.array(all_preds), np.array(all_labels), np.array(all_probs)

    def evaluate(self):
        print("\n" + "=" * 70)
        print(f"EVALUATION - {self.model_type.upper()}")
        print("=" * 70)

        self.load_model()

        if self.model_type in ['resnet34', 'resnet50']:
            self.setup_data()
            all_preds, all_labels, all_probs = self.evaluate_resnet()
        else:
            all_preds, all_labels, all_probs = self.evaluate_yolo()

        metrics = calculate_metrics(all_labels, all_preds)
        class_accuracies = calculate_per_class_accuracy(all_labels, all_preds, self.num_classes)
        cm = get_confusion_matrix(all_labels, all_preds)
        report = get_classification_report(all_labels, all_preds, self.class_names)

        self.print_results(metrics, class_accuracies)
        self.save_results(metrics, class_accuracies, cm, report, all_labels, all_preds)
        self.generate_visualizations(cm, metrics, class_accuracies)

        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE")
        print("=" * 70)

    def print_results(self, metrics: dict, class_accuracies: np.ndarray):
        print(f"\nOverall Metrics:")
        print(f"  Test Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
        print(f"  Macro F1-Score:   {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1:      {metrics['weighted_f1']:.4f}")

        print(f"\nPer-Class Accuracy:")
        print(f"{'Class':<20} {'Accuracy':>10}")
        print("-" * 35)
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:<20} {class_accuracies[i]:>9.2%}")
        print("-" * 35)
        print(f"{'AVERAGE':<20} {np.mean(class_accuracies):>9.2%}")

    def save_results(self, metrics: dict, class_accuracies: np.ndarray,
                     cm: np.ndarray, report: str, all_labels: np.ndarray, all_preds: np.ndarray):

        results = {
            'model': self.model_type,
            'test_accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1'],
            'weighted_f1': metrics['weighted_f1'],
            'per_class_metrics': {}
        }

        for i, class_name in enumerate(self.class_names):
            results['per_class_metrics'][class_name] = {
                'accuracy': float(class_accuracies[i]),
                'precision': float(metrics['precision'][i]),
                'recall': float(metrics['recall'][i]),
                'f1_score': float(metrics['f1_score'][i]),
                'support': int(metrics['support'][i])
            }

        results_file = self.results_path / f'test_results_{self.model_type}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {results_file}")

        report_file = self.results_path / f'classification_report_{self.model_type}.txt'
        with open(report_file, 'w') as f:
            f.write(f"{self.model_type.upper()} - TEST SET RESULTS\n")
            f.write("=" * 70 + "\n\n")
            f.write(report)
        print(f"Report saved: {report_file}")

    def generate_visualizations(self, cm: np.ndarray, metrics: dict, class_accuracies: np.ndarray):
        plot_confusion_matrix(
            cm,
            self.class_names,
            save_path=str(self.results_path / f'confusion_matrix_{self.model_type}.png'),
            title=f'Confusion Matrix - {self.model_type.upper()}',
            normalize=False
        )
        print(f"Confusion matrix saved: confusion_matrix_{self.model_type}.png")

        plot_confusion_matrix(
            cm,
            self.class_names,
            save_path=str(self.results_path / f'confusion_matrix_normalized_{self.model_type}.png'),
            title=f'Confusion Matrix (Normalized) - {self.model_type.upper()}',
            normalize=True,
            cmap='RdYlGn'
        )
        print(f"Normalized confusion matrix saved: confusion_matrix_normalized_{self.model_type}.png")

        plot_per_class_metrics(
            self.class_names,
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score'],
            save_path=str(self.results_path / f'per_class_performance_{self.model_type}.png'),
            title=f'Per-Class Performance - {self.model_type.upper()}'
        )
        print(f"Per-class metrics saved: per_class_performance_{self.model_type}.png")

        plot_class_distribution(
            self.class_names,
            metrics['support'],
            save_path=str(self.results_path / f'class_distribution_{self.model_type}.png')
        )
        print(f"Class distribution saved: class_distribution_{self.model_type}.png")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate trained model')
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