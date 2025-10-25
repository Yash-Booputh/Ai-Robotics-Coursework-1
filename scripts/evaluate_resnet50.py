"""
Evaluate ResNet50 model on test set.

Usage:
    python scripts/evaluate_resnet50.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluate import ModelEvaluator


def main():
    evaluator = ModelEvaluator(
        model_type='resnet50',
        model_path='models/resnet50_best.pt',  # Changed from .pth to .pt
        config_path='config/config.yaml'
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()