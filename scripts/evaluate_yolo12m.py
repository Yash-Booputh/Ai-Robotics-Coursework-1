"""
Evaluate YOLO12m model on test set.

Usage:
    python scripts/evaluate_yolo12m.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluate import ModelEvaluator


def main():
    evaluator = ModelEvaluator(
        model_type='yolo12m',
        model_path='models/yolo12m_best.pt',
        config_path='config/config.yaml'
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()