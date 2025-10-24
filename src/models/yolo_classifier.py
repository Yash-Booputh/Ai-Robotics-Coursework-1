"""
YOLO Classifier Module

Provides YOLO11m and YOLO12m classification models
using Ultralytics framework.
"""

from ultralytics import YOLO
from pathlib import Path
from typing import Optional, Dict, Any


class YOLOClassifier:

    def __init__(self, model_path: str = 'yolo11m-cls.pt'):
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.trained = False

    def train(
            self,
            data_path: str,
            epochs: int = 20,
            imgsz: int = 224,
            batch: int = 32,
            device: str = 'cuda',
            project: str = 'runs/classify',
            name: str = 'yolo_office',
            **kwargs
    ) -> Dict[str, Any]:
        results = self.model.train(
            data=data_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            **kwargs
        )

        self.trained = True
        return results

    def validate(self, data_path: str, split: str = 'val'):
        if not self.trained and not self._is_custom_model():
            raise RuntimeError("Model must be trained or loaded before validation")

        return self.model.val(data=data_path, split=split)

    def predict(self, source, **kwargs):
        return self.model.predict(source, **kwargs)

    def save(self, save_path: str):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(save_path)

    def load(self, model_path: str):
        self.model = YOLO(model_path)
        self.trained = True

    def _is_custom_model(self) -> bool:
        return 'best' in str(self.model_path) or 'last' in str(self.model_path)


def create_yolo_model(model_name: str = 'yolo11m-cls') -> YOLOClassifier:
    model_file = f"{model_name}.pt"
    return YOLOClassifier(model_path=model_file)