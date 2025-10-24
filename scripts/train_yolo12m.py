"""
Train YOLO12m classification model.

Usage:
    python scripts/train_yolo12m.py
"""

import sys
from pathlib import Path
import yaml
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.train_yolo import YOLOTrainer


def main():
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    config['training']['yolo']['model_name'] = 'yolo12m-cls'

    temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(config, temp_config, default_flow_style=False)
    temp_config.close()

    trainer = YOLOTrainer(config_path=temp_config.name)
    trainer.run()

    Path(temp_config.name).unlink()


if __name__ == "__main__":
    main()