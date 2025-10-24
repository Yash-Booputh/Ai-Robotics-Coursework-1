"""
Train ResNet50 model.

Usage:
    python scripts/train_resnet50.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.train_resnet import ResNetTrainer


def main():
    trainer = ResNetTrainer(config_path="config/config.yaml")
    trainer.config['training']['resnet']['model_name'] = 'resnet50'
    trainer.model_name = 'resnet50'
    trainer.run()


if __name__ == "__main__":
    main()