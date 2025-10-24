"""
YOLO Training Module

Handles training, validation, and model saving
for YOLO classification models (YOLO11m/YOLO12m).
"""

import time
import json
import shutil
from pathlib import Path
import yaml

from src.models.yolo_classifier import create_yolo_model


class YOLOTrainer:

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_name = self.config['training']['yolo']['model_name']
        self.epochs = self.config['training']['yolo']['epochs']
        self.batch_size = self.config['training']['yolo']['batch_size']
        self.learning_rate = self.config['training']['yolo']['learning_rate']
        self.patience = self.config['training']['yolo']['patience']
        self.imgsz = self.config['training']['yolo']['image_size']

        self.data_path = Path(self.config['data']['train_path']).parent
        self.model_save_path = Path(self.config['paths']['models'])
        self.results_path = Path(self.config['paths']['results'])

        self.model_save_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)

        print(f"Training {self.model_name} for {self.epochs} epochs")

    def setup_model(self):
        self.model = create_yolo_model(model_name=self.model_name)
        print(f"Model loaded: {self.model_name}")

    def train(self) -> dict:
        print("\n" + "=" * 70)
        print(f"STARTING TRAINING - {self.model_name.upper()}")
        print("=" * 70)

        # Print debug info
        print(f"Current directory: {Path.cwd()}")
        print(f"Data path from config: {self.data_path}")
        print(f"Data path exists: {self.data_path.exists()}")

        start_time = time.time()

        # Use string path, not Path object
        data_str = str(self.data_path) if self.data_path.exists() else 'data'

        results = self.model.train(
            data_path=data_str,
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch_size,
            device='cuda' if self.config['inference']['device'] == 'cuda' else 'cpu',
            workers=4,
            patience=self.patience,
            save=True,
            plots=True,
            cache=False,
            optimizer='Adam',
            lr0=self.learning_rate,
            val=True,
            project='runs/classify',
            name=f'{self.model_name.replace("-cls", "")}_office'
        )

        training_time = time.time() - start_time

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print(f"Time: {training_time // 60:.0f}m {training_time % 60:.0f}s")
        print("=" * 70)

        return {'training_time': training_time, 'results': results}

    def validate(self):
        print("\n" + "=" * 70)
        print("VALIDATION")
        print("=" * 70)

        val_results = self.model.validate(data_path='data', split='val')

        print(f"\nValidation Results:")
        print(f"  Top-1 Accuracy: {val_results.top1 * 100:.2f}%")
        print(f"  Top-5 Accuracy: {val_results.top5 * 100:.2f}%")

        return val_results

    def test(self):
        print("\n" + "=" * 70)
        print("TESTING")
        print("=" * 70)

        test_results = self.model.validate(data_path='data', split='test')

        print(f"\nTest Results:")
        print(f"  Top-1 Accuracy: {test_results.top1 * 100:.2f}%")
        print(f"  Top-5 Accuracy: {test_results.top5 * 100:.2f}%")

        return test_results

    def save_model(self, training_time: float, test_results):
        model_short_name = self.model_name.replace('-cls', '')

        runs_base = Path('runs/classify')

        # Find all matching directories
        matching_dirs = sorted([d for d in runs_base.iterdir()
                              if d.is_dir() and d.name.startswith(f'{model_short_name}_office')])

        if not matching_dirs:
            print(f"\nError: No training run found in {runs_base}")
            return

        runs_path = matching_dirs[-1]
        print(f"\nUsing run directory: {runs_path}")

        best_model_path = runs_path / 'weights' / 'best.pt'
        last_model_path = runs_path / 'weights' / 'last.pt'

        print(f"Looking for weights at: {runs_path / 'weights'}")
        print(f"Best model exists: {best_model_path.exists()}")
        print(f"Last model exists: {last_model_path.exists()}")

        if best_model_path.exists():
            target_path = self.model_save_path / f'{model_short_name}_best.pt'
            shutil.copy2(best_model_path, target_path)
            print(f"Model saved: {target_path}")
        else:
            print(f"Warning: {best_model_path} not found")

        if last_model_path.exists():
            target_path = self.model_save_path / f'{model_short_name}_last.pt'
            shutil.copy2(last_model_path, target_path)
            print(f"Model saved: {target_path}")

        # Load or create class names
        class_names_file = self.model_save_path / 'class_names.json'
        if class_names_file.exists():
            with open(class_names_file, 'r') as f:
                class_names = json.load(f)
        else:
            class_names = self.config['data']['classes']
            with open(class_names_file, 'w') as f:
                json.dump(class_names, f, indent=2)
            print(f"Class names saved: {class_names_file}")

        results_dict = {
            'model': self.model_name,
            'test_accuracy': float(test_results.top1),
            'top5_accuracy': float(test_results.top5),
            'training_time_seconds': training_time,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'image_size': self.imgsz,
            'classes': class_names,
            'num_classes': len(class_names)
        }

        results_file = self.results_path / f'test_results_{model_short_name}.json'
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"Results saved: {results_file}")

        # Copy plots
        for filename, dest_name in [
            ('results.png', f'training_curves_{model_short_name}.png'),
            ('confusion_matrix_normalized.png', f'confusion_matrix_{model_short_name}.png')
        ]:
            src = runs_path / filename
            if src.exists():
                shutil.copy2(src, self.results_path / dest_name)
                print(f"Saved: {dest_name}")

    def run(self):
        self.setup_model()
        train_results = self.train()
        self.validate()
        test_results = self.test()
        self.save_model(train_results['training_time'], test_results)

        print("\n" + "=" * 70)
        print(f"{self.model_name.upper()} TRAINING COMPLETE")
        print("=" * 70)
        print(f"\nTest Accuracy: {test_results.top1 * 100:.2f}%")
        print(f"Top-5 Accuracy: {test_results.top5 * 100:.2f}%")
        print(f"Training Time: {train_results['training_time'] // 60:.0f}m")


def main():
    trainer = YOLOTrainer()
    trainer.run()


if __name__ == "__main__":
    main()