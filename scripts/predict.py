"""
Run inference on images, camera, or video.

Usage:
    # Single image
    python scripts/predict.py --model resnet34 --source path/to/image.jpg

    # Camera (0 = default camera)
    python scripts/predict.py --model yolo11m --source 0

    # Video file
    python scripts/predict.py --model resnet50 --source video.mp4 --output output.mp4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.predict import OfficeItemsPredictor
import argparse


def main():
    parser = argparse.ArgumentParser(description='Office Items Classification Inference')
    parser.add_argument('--model', type=str, required=True,
                        choices=['resnet34', 'resnet50', 'yolo11m', 'yolo12m'],
                        help='Model to use for prediction')
    parser.add_argument('--source', type=str, required=True,
                        help='Input source: image path, camera (0), or video path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for video processing')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable display window')

    args = parser.parse_args()

    model_paths = {
        'resnet34': 'models/resnet34_best.pth',
        'resnet50': 'models/resnet50_best.pth',
        'yolo11m': 'models/yolo11m_best.pt',
        'yolo12m': 'models/yolo12m_best.pt'
    }

    predictor = OfficeItemsPredictor(
        model_type=args.model,
        model_path=model_paths[args.model]
    )

    if args.source.isdigit():
        print("Starting camera prediction...")
        print("Press 'q' to quit, 's' to save frame")
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

        print("\n" + "=" * 70)
        print(f"Image: {args.source}")
        print(f"Predicted Class: {predicted_class.upper()}")
        print(f"Confidence: {confidence * 100:.2f}%")
        print("\nTop 3 Predictions:")
        print("-" * 70)

        import numpy as np
        top_3_indices = np.argsort(probs)[-3:][::-1]
        for rank, idx in enumerate(top_3_indices, 1):
            print(f"  {rank}. {predictor.class_names[idx]:<15} {probs[idx] * 100:>6.2f}%")
        print("=" * 70)

    else:
        print(f"Error: Invalid source '{args.source}'")
        print("Source must be:")
        print("  - Path to an image file")
        print("  - Camera ID (e.g., 0 for default camera)")
        print("  - Path to a video file")


if __name__ == "__main__":
    main()