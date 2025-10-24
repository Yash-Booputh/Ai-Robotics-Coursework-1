"""
ResNet Training Module

Handles training loop, validation, and model checkpointing
for ResNet models (ResNet34/ResNet50).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import copy
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm
import yaml

from src.models.resnet_classifier import create_resnet_model
from src.data.dataset import DataLoaderFactory


class ResNetTrainer:

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_name = self.config['training']['resnet']['model_name']
        self.epochs = self.config['training']['resnet']['epochs']
        self.learning_rate = self.config['training']['resnet']['learning_rate']

        self.model_save_path = Path(self.config['paths']['models'])
        self.model_save_path.mkdir(parents=True, exist_ok=True)

        print(f"Device: {self.device}")
        print(f"Training {self.model_name} for {self.epochs} epochs")

    def setup_data(self):
        data_factory = DataLoaderFactory(config_path="config/config.yaml")
        self.dataloaders, self.dataset_sizes, self.class_names = data_factory.create_dataloaders()
        self.num_classes = len(self.class_names)

        print(f"\nDataset loaded:")
        print(f"  Train: {self.dataset_sizes['train']} images")
        print(f"  Val:   {self.dataset_sizes['val']} images")
        print(f"  Test:  {self.dataset_sizes['test']} images")
        print(f"  Classes: {self.class_names}")

    def setup_model(self):
        self.model = create_resnet_model(
            num_classes=self.num_classes,
            model_name=self.model_name,
            pretrained=True
        )
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.get_trainable_parameters(),
            lr=self.learning_rate
        )

        scheduler_config = self.config['training']['resnet']['scheduler']
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=scheduler_config['step_size'],
            gamma=scheduler_config['gamma']
        )

        print(f"\nModel: {self.model_name} (pretrained on ImageNet)")
        print(f"Optimizer: Adam (lr={self.learning_rate})")
        print(f"Scheduler: StepLR")

    def train_epoch(self, phase: str) -> Tuple[float, float]:
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        running_corrects = 0

        pbar = tqdm(self.dataloaders[phase], desc=f'{phase}')

        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        if phase == 'train':
            self.scheduler.step()

        epoch_loss = running_loss / self.dataset_sizes[phase]
        epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

        return epoch_loss, epoch_acc.item()

    def train(self) -> Dict:
        print("\n" + "=" * 70)
        print(f"STARTING TRAINING - {self.model_name.upper()}")
        print("=" * 70)

        start_time = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

        for epoch in range(self.epochs):
            print(f'\nEpoch {epoch + 1}/{self.epochs}')
            print('-' * 40)

            for phase in ['train', 'val']:
                epoch_loss, epoch_acc = self.train_epoch(phase)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                history[f'{phase}_loss'].append(epoch_loss)
                history[f'{phase}_acc'].append(epoch_acc)

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    print(f'Best model saved! Val Acc: {best_acc:.4f}')

        training_time = time.time() - start_time

        print('\n' + '=' * 70)
        print('TRAINING COMPLETE')
        print(f'Time: {training_time // 60:.0f}m {training_time % 60:.0f}s')
        print(f'Best val accuracy: {best_acc:.4f} ({best_acc * 100:.2f}%)')
        print('=' * 70)

        self.model.load_state_dict(best_model_wts)

        return history

    def save_model(self, history: Dict):
        model_path = self.model_save_path / f'{self.model_name}_best.pth'
        torch.save(self.model.state_dict(), model_path)
        print(f"\nModel saved: {model_path}")

        class_names_path = self.model_save_path / 'class_names.json'
        with open(class_names_path, 'w') as f:
            json.dump(self.class_names, f, indent=2)
        print(f"Class names saved: {class_names_path}")

        history_path = self.model_save_path / f'training_history_{self.model_name}.pkl'
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)
        print(f"Training history saved: {history_path}")

    def run(self):
        self.setup_data()
        self.setup_model()
        history = self.train()
        self.save_model(history)

        print("\n" + "=" * 70)
        print(f"{self.model_name.upper()} TRAINING COMPLETE")
        print("=" * 70)
        print(f"\nBest val accuracy: {max(history['val_acc']) * 100:.2f}%")
        print(f"Final train accuracy: {history['train_acc'][-1] * 100:.2f}%")


def main():
    trainer = ResNetTrainer()
    trainer.run()


if __name__ == "__main__":
    main()