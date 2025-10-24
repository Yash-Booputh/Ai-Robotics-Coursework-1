"""
Visualization utilities for evaluation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Optional


def plot_confusion_matrix(
        cm: np.ndarray,
        class_names: List[str],
        save_path: str,
        title: str = "Confusion Matrix",
        normalize: bool = False,
        cmap: str = 'Blues'
):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage' if normalize else 'Count'},
        square=True
    )

    plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
    plt.ylabel('True Label', fontsize=13, fontweight='bold')
    plt.title(title, fontsize=15, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_class_metrics(
        class_names: List[str],
        precision: np.ndarray,
        recall: np.ndarray,
        f1: np.ndarray,
        save_path: str,
        title: str = "Per-Class Performance Metrics"
):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    x = np.arange(len(class_names))
    width = 0.6

    axes[0].bar(x, precision * 100, width, color='skyblue', edgecolor='navy', linewidth=1.5)
    axes[0].set_ylabel('Precision (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Precision by Class', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].axhline(y=np.mean(precision) * 100, color='red', linestyle='--',
                    linewidth=2, label=f'Macro Avg: {np.mean(precision) * 100:.2f}%')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(x, recall * 100, width, color='lightcoral', edgecolor='darkred', linewidth=1.5)
    axes[1].set_ylabel('Recall (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Recall by Class', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].axhline(y=np.mean(recall) * 100, color='red', linestyle='--',
                    linewidth=2, label=f'Macro Avg: {np.mean(recall) * 100:.2f}%')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    axes[2].bar(x, f1 * 100, width, color='lightgreen', edgecolor='darkgreen', linewidth=1.5)
    axes[2].set_ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
    axes[2].set_title('F1-Score by Class', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Class', fontsize=12, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(class_names, rotation=45, ha='right')
    axes[2].axhline(y=np.mean(f1) * 100, color='red', linestyle='--',
                    linewidth=2, label=f'Macro Avg: {np.mean(f1) * 100:.2f}%')
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_class_distribution(class_names: List[str], support: np.ndarray, save_path: str):
    plt.figure(figsize=(12, 6))

    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    bars = plt.bar(class_names, support, color=colors, edgecolor='black', linewidth=1.5)

    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
    plt.title('Test Set Class Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()