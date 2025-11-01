"""
Inference Module for Office Items Classification

"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List
import json
import yaml
from PIL import Image

from src.models.resnet_classifier import create_resnet_model
from src.models.yolo_classifier import YOLOClassifier
from src.data.dataset import get_inference_transform


class OfficeItemsPredictor:

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

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = self.config['inference']['confidence_threshold']

        self._load_class_names()
        self._load_model()

        if self.model_type in ['resnet34', 'resnet50']:
            self.transform = get_inference_transform(config_path)

        print(f"Predictor initialized: {model_type}")
        print(f"Device: {self.device}")
        print(f"Classes: {self.class_names}")

    def _load_class_names(self):
        class_names_path = Path(self.config['paths']['models']) / 'class_names.json'
        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
        self.num_classes = len(self.class_names)

    def _load_model(self):
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
        print(f"ResNet model loaded from: {self.model_path}")

    def _load_yolo_model(self):
        self.model = YOLOClassifier(model_path=str(self.model_path))
        print(f"YOLO model loaded from: {self.model_path}")

    def predict_image(self, image_path: Union[str, Path]) -> Tuple[str, float, np.ndarray]:
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if self.model_type in ['resnet34', 'resnet50']:
            return self._predict_resnet(image_path)
        else:
            return self._predict_yolo(image_path)

    def _predict_resnet(self, image_path: Path) -> Tuple[str, float, np.ndarray]:
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = self.class_names[predicted.item()]
        confidence_value = confidence.item()
        all_probs = probabilities.cpu().numpy()[0]

        return predicted_class, confidence_value, all_probs

    def _predict_yolo(self, image_path: Path) -> Tuple[str, float, np.ndarray]:
        results = self.model.predict(str(image_path), verbose=False)

        probs = results[0].probs.data.cpu().numpy()
        predicted_idx = np.argmax(probs)
        confidence_value = probs[predicted_idx]
        predicted_class = self.class_names[predicted_idx]

        return predicted_class, confidence_value, probs

    def predict_batch(self, image_paths: List[Union[str, Path]]) -> List[Tuple[str, float, np.ndarray]]:
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_image(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append((None, 0.0, None))

        return results

    def predict_from_camera(self, camera_id: int = 0, display: bool = True):
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")

        print(f"Camera opened. Press 'q' to quit, 's' to save frame.")

        frame_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame")
                break

            if frame_count % 5 == 0:
                temp_path = Path("temp_frame.jpg")
                cv2.imwrite(str(temp_path), frame)

                try:
                    predicted_class, confidence, probs = self.predict_image(temp_path)

                    self._draw_prediction(frame, predicted_class, confidence, probs)

                except Exception as e:
                    print(f"Prediction error: {e}")

                if temp_path.exists():
                    temp_path.unlink()

            if display:
                cv2.imshow('Office Items Classifier', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = f"captured_frame_{frame_count}.jpg"
                cv2.imwrite(save_path, frame)
                print(f"Frame saved: {save_path}")

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

    def _draw_prediction(self, frame: np.ndarray, predicted_class: str,
                         confidence: float, probs: np.ndarray):
        height, width = frame.shape[:2]

        color = (0, 255, 0) if confidence >= self.confidence_threshold else (0, 165, 255)

        text = f"{predicted_class}: {confidence * 100:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        cv2.rectangle(frame, (10, 10), (20 + text_width, 40 + text_height), (0, 0, 0), -1)
        cv2.putText(frame, text, (15, 35 + text_height // 2), font, font_scale, color, thickness)

        top_3_indices = np.argsort(probs)[-3:][::-1]
        y_offset = 80

        for idx in top_3_indices:
            class_text = f"{self.class_names[idx]}: {probs[idx] * 100:.1f}%"
            cv2.putText(frame, class_text, (15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 30

    def predict_from_video(self, video_path: str, output_path: str = None, display: bool = True):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        print(f"Processing video: {video_path}")

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_count % 5 == 0:
                temp_path = Path("temp_frame.jpg")
                cv2.imwrite(str(temp_path), frame)

                try:
                    predicted_class, confidence, probs = self.predict_image(temp_path)
                    self._draw_prediction(frame, predicted_class, confidence, probs)
                except Exception as e:
                    print(f"Prediction error on frame {frame_count}: {e}")

                if temp_path.exists():
                    temp_path.unlink()

            if writer:
                writer.write(frame)

            if display:
                cv2.imshow('Video Classification', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        print(f"Processed {frame_count} frames")
        if output_path:
            print(f"Output saved: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Office Items Classification Inference')
    parser.add_argument('--model-type', type=str, required=True,
                        choices=['resnet34', 'resnet50', 'yolo11m', 'yolo12m'],
                        help='Model type to use')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--source', type=str, required=True,
                        help='Input source: image path, camera (0), or video path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for video processing')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable display window')

    args = parser.parse_args()

    predictor = OfficeItemsPredictor(
        model_type=args.model_type,
        model_path=args.model_path
    )

    if args.source.isdigit():
        print("Starting camera prediction...")
        predictor.predict_from_camera(
            camera_id=int(args.source),
            display=not args.no_display
        )

    elif Path(args.source).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        print("Starting video prediction...")
        predictor.predict_from_video(
            video_path=args.source,
            output_path=args.output,
            display=not args.no_display
        )

    elif Path(args.source).exists():
        print("Predicting single image...")
        predicted_class, confidence, probs = predictor.predict_image(args.source)

        print("\n" + "=" * 50)
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence * 100:.2f}%")
        print("\nTop 3 Predictions:")
        top_3_indices = np.argsort(probs)[-3:][::-1]
        for idx in top_3_indices:
            print(f"  {predictor.class_names[idx]}: {probs[idx] * 100:.2f}%")
        print("=" * 50)

    else:
        print(f"Invalid source: {args.source}")


if __name__ == "__main__":
    main()