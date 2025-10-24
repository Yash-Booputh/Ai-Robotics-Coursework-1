"""
Dataset and DataLoader utilities

Handles data loading, augmentation, and transformations
for training and evaluation.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Dict, Tuple
import yaml


class DataLoaderFactory:

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.train_path = self.config['data']['train_path']
        self.val_path = self.config['data']['val_path']
        self.test_path = self.config['data']['test_path']
        self.batch_size = self.config['training']['resnet']['batch_size']

    def get_transforms(self) -> Dict[str, transforms.Compose]:
        aug_config = self.config['augmentation']

        train_transform = transforms.Compose([
            transforms.Resize(aug_config['resize']),
            transforms.RandomResizedCrop(
                aug_config['crop_size'],
                scale=tuple(aug_config['random_crop_scale'])
            ),
            transforms.RandomHorizontalFlip(p=aug_config['horizontal_flip']),
            transforms.RandomRotation(aug_config['rotation']),
            transforms.ColorJitter(
                brightness=aug_config['color_jitter']['brightness'],
                contrast=aug_config['color_jitter']['contrast'],
                saturation=aug_config['color_jitter']['saturation'],
                hue=aug_config['color_jitter']['hue']
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomGrayscale(p=aug_config['random_grayscale']),
            transforms.RandomPerspective(distortion_scale=0.2, p=aug_config['random_perspective']),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=aug_config['random_erasing'], scale=(0.02, 0.15))
        ])

        val_test_transform = transforms.Compose([
            transforms.Resize(aug_config['resize']),
            transforms.CenterCrop(aug_config['crop_size']),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        return {
            'train': train_transform,
            'val': val_test_transform,
            'test': val_test_transform
        }

    def create_dataloaders(self) -> Tuple[Dict[str, DataLoader], Dict[str, int], list]:
        data_transforms = self.get_transforms()

        image_datasets = {
            'train': datasets.ImageFolder(self.train_path, data_transforms['train']),
            'val': datasets.ImageFolder(self.val_path, data_transforms['val']),
            'test': datasets.ImageFolder(self.test_path, data_transforms['test'])
        }

        dataloaders = {
            'train': DataLoader(
                image_datasets['train'],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4
            ),
            'val': DataLoader(
                image_datasets['val'],
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4
            ),
            'test': DataLoader(
                image_datasets['test'],
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4
            )
        }

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
        class_names = image_datasets['train'].classes

        return dataloaders, dataset_sizes, class_names


def get_inference_transform(config_path: str = "config/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    aug_config = config['augmentation']

    return transforms.Compose([
        transforms.Resize(aug_config['resize']),
        transforms.CenterCrop(aug_config['crop_size']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])