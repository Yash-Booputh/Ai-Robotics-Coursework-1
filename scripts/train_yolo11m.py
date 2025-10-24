"""
Train YOLO11m classification model.

Usage:
    python scripts/train_yolo11m.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.train_yolo import YOLOTrainer


def main():
    trainer = YOLOTrainer(config_path="config/config.yaml")
    trainer.config['training']['yolo']['model_name'] = 'yolo11m-cls'
    trainer.model_name = 'yolo11m-cls'
    trainer.run()


if __name__ == "__main__":
    main()