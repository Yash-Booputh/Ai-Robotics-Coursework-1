"""
Evaluate ResNet34 model on test set.

Usage:
    python scripts/evaluate_resnet34.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluate import ModelEvaluator


def main():
    evaluator = ModelEvaluator(
        model_type='resnet34',
        model_path='models/resnet34_best.pth',
        config_path='config/config.yaml'
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()