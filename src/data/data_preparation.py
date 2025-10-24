"""
Data Preparation Module for Office Items Classification

This module handles:
- Extraction of dataset ZIP files
- Organization into train/val/test splits
- Dataset statistics and validation
"""

import os
import zipfile
import shutil
import random
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import yaml


class DataPreparation:

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.raw_path = Path(self.config['data']['raw_path'])
        self.train_path = Path(self.config['data']['train_path'])
        self.val_path = Path(self.config['data']['val_path'])
        self.test_path = Path(self.config['data']['test_path'])
        self.seed = self.config['data']['seed']
        self.split_ratio = self.config['data']['split_ratio']

        random.seed(self.seed)

    def find_zip_files(self) -> List[Path]:
        zip_files = list(self.raw_path.glob("*.zip"))

        if not zip_files:
            root_zips = list(Path(".").glob("*.zip"))
            if root_zips:
                print(f"Found {len(root_zips)} ZIP files in root directory")
                return root_zips

        return zip_files

    def extract_datasets(self) -> None:
        zip_files = self.find_zip_files()

        if not zip_files:
            raise FileNotFoundError("No ZIP files found in data/raw or root directory")

        print(f"Found {len(zip_files)} ZIP files")

        extract_path = self.raw_path / "extracted"

        if extract_path.exists():
            shutil.rmtree(extract_path)
        extract_path.mkdir(parents=True, exist_ok=True)

        print("Extracting datasets...")
        for zip_file in zip_files:
            class_name = zip_file.stem.replace('_dataset', '').replace('.zip', '')
            class_folder = extract_path / class_name
            class_folder.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(class_folder)

            print(f"  Extracted: {class_name}")

        print("Extraction complete")

    def analyze_structure(self) -> Dict[str, List[Path]]:
        extract_path = self.raw_path / "extracted"
        class_image_mapping = defaultdict(list)

        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

        for root, dirs, files in os.walk(extract_path):
            img_files = [f for f in files if Path(f).suffix.lower() in valid_extensions]

            if img_files:
                relative_path = Path(root).relative_to(extract_path)
                class_name = str(relative_path).split(os.sep)[0]

                for img in img_files:
                    class_image_mapping[class_name].append(Path(root) / img)

        return dict(class_image_mapping)

    def print_statistics(self, class_image_mapping: Dict[str, List[Path]]) -> None:
        print("\nDataset Statistics:")
        print("-" * 50)
        print(f"{'Class':<20} {'Images':>10}")
        print("-" * 50)

        total = 0
        for cls in sorted(class_image_mapping.keys()):
            count = len(class_image_mapping[cls])
            print(f"{cls:<20} {count:>10}")
            total += count

        print("-" * 50)
        print(f"{'TOTAL':<20} {total:>10}")
        print("-" * 50)

    def split_dataset(self, class_image_mapping: Dict[str, List[Path]]) -> Dict[str, Dict[str, int]]:
        for split_path in [self.train_path, self.val_path, self.test_path]:
            if split_path.exists():
                shutil.rmtree(split_path)
            split_path.mkdir(parents=True, exist_ok=True)

        for class_name in class_image_mapping.keys():
            (self.train_path / class_name).mkdir(exist_ok=True)
            (self.val_path / class_name).mkdir(exist_ok=True)
            (self.test_path / class_name).mkdir(exist_ok=True)

        split_stats = {}

        print("\nSplitting dataset...")
        for class_name, image_paths in sorted(class_image_mapping.items()):
            random.shuffle(image_paths)

            n_total = len(image_paths)
            n_train = int(n_total * self.split_ratio['train'])
            n_val = int(n_total * self.split_ratio['val'])

            train_imgs = image_paths[:n_train]
            val_imgs = image_paths[n_train:n_train + n_val]
            test_imgs = image_paths[n_train + n_val:]

            for i, img in enumerate(train_imgs):
                shutil.copy2(img, self.train_path / class_name / f"{i:04d}{img.suffix}")

            for i, img in enumerate(val_imgs):
                shutil.copy2(img, self.val_path / class_name / f"{i:04d}{img.suffix}")

            for i, img in enumerate(test_imgs):
                shutil.copy2(img, self.test_path / class_name / f"{i:04d}{img.suffix}")

            split_stats[class_name] = {
                'train': len(train_imgs),
                'val': len(val_imgs),
                'test': len(test_imgs)
            }

            print(f"  {class_name}: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

        return split_stats

    def save_split_statistics(self, split_stats: Dict[str, Dict[str, int]]) -> None:
        stats_path = Path(self.config['data']['train_path']).parent / "split_stats.json"

        with open(stats_path, 'w') as f:
            json.dump(split_stats, f, indent=2)

        print(f"\nSplit statistics saved to: {stats_path}")

    def print_split_summary(self, split_stats: Dict[str, Dict[str, int]]) -> None:
        print("\nSplit Summary:")
        print("-" * 65)
        print(f"{'Class':<20} {'Train':>12} {'Val':>12} {'Test':>12}")
        print("-" * 65)

        total_train = total_val = total_test = 0

        for cls in sorted(split_stats.keys()):
            train = split_stats[cls]['train']
            val = split_stats[cls]['val']
            test = split_stats[cls]['test']

            print(f"{cls:<20} {train:>12} {val:>12} {test:>12}")

            total_train += train
            total_val += val
            total_test += test

        print("-" * 65)
        print(f"{'TOTAL':<20} {total_train:>12} {total_val:>12} {total_test:>12}")
        print("-" * 65)

    def prepare(self) -> None:
        print("=" * 70)
        print("DATA PREPARATION - OFFICE ITEMS CLASSIFIER")
        print("=" * 70)

        self.extract_datasets()
        class_image_mapping = self.analyze_structure()
        self.print_statistics(class_image_mapping)
        split_stats = self.split_dataset(class_image_mapping)
        self.save_split_statistics(split_stats)
        self.print_split_summary(split_stats)

        print("\nData preparation complete!")
        print("=" * 70)


def main():
    try:
        prep = DataPreparation()
        prep.prepare()
    except Exception as e:
        print(f"Error during data preparation: {e}")
        raise


if __name__ == "__main__":
    main()