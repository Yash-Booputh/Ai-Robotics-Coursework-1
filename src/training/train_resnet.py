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

    def plot_training_curves(self):
        """Plot and save training curves"""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        epochs_range = range(1, len(self.train_acc_history) + 1)

        # Plot Accuracy
        ax1.plot(epochs_range, self.train_acc_history, 'o-', label='Training Accuracy',
                 color='#2E86AB', linewidth=2.5, markersize=6)
        ax1.plot(epochs_range, self.val_acc_history, 's-', label='Validation Accuracy',
                 color='#A23B72', linewidth=2.5, markersize=6)
        ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
        ax1.set_title('Model Accuracy', fontsize=15, fontweight='bold', pad=15)
        ax1.legend(loc='lower right', fontsize=11, framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim([0, 1.05])

        # Plot Loss
        ax2.plot(epochs_range, self.train_loss_history, 'o-', label='Training Loss',
                 color='#F18F01', linewidth=2.5, markersize=6)
        ax2.plot(epochs_range, self.val_loss_history, 's-', label='Validation Loss',
                 color='#C73E1D', linewidth=2.5, markersize=6)
        ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=13, fontweight='bold')
        ax2.set_title('Model Loss', fontsize=15, fontweight='bold', pad=15)
        ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle='--')

        plt.suptitle(f'{self.model_name.upper()} - Training History',
                     fontsize=17, fontweight='bold', y=1.02)
        plt.tight_layout()

        save_path = self.results_path / f'training_curves_{self.model_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Training curves saved: {save_path}")

        # Check for overfitting
        final_train_acc = self.train_acc_history[-1]
        final_val_acc = self.val_acc_history[-1]
        acc_gap = final_train_acc - final_val_acc

        if acc_gap > 0.05:
            print(f"\n⚠️  Potential overfitting detected:")
            print(f"   Training Accuracy: {final_train_acc:.4f}")
            print(f"   Validation Accuracy: {final_val_acc:.4f}")
            print(f"   Gap: {acc_gap:.4f}")
        else:
            print(f"\n✅ Model generalizes well (Train-Val gap: {acc_gap:.4f})")

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
        self.train()
        self.plot_training_curves()  # Add this line
        test_results = self.test()
        self.save_model(test_results)

        print("\n" + "=" * 70)
        print(f"{self.model_name.upper()} TRAINING COMPLETE")
        print("=" * 70)
        print(f"\nTest Accuracy: {test_results['accuracy']:.4f}")
        print(f"Best Validation Accuracy: {self.best_acc:.4f}")
        print(f"Training Time: {self.total_training_time // 60:.0f}m {self.total_training_time % 60:.0f}s")


def main():
    trainer = ResNetTrainer()
    trainer.run()


if __name__ == "__main__":
    main()