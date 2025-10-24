"""
Visualization utilities for evaluation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List


def plot_confusion_matrix(
        cm: np.ndarray,
        class_names: List[str],
        save_path: str,
        title: str = "Confusion Matrix",
        normalize: bool = False,
        cmap: str = 'Blues'
):
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        vmin, vmax = 0, 1
    else:
        cm_display = cm
        fmt = 'd'
        vmin, vmax = None, None

    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(cm_display, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.figure.colorbar(im, ax=ax, label='Percentage' if normalize else 'Count')

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted Label',
           ylabel='True Label',
           title=title)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm_display.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                text = f'{cm_display[i, j]:.1%}\n({cm[i, j]})'
            else:
                text = f'{cm[i, j]}'

            ax.text(j, i, text,
                   ha="center", va="center",
                   color="white" if cm_display[i, j] > thresh else "black",
                   fontsize=9 if normalize else 11,
                   fontweight='bold')

    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    fig.tight_layout()
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
    fig, axes = plt.subplots(3, 1, figsize=(15, 14))
    x = np.arange(len(class_names))
    width = 0.65

    colors_precision = plt.cm.Blues(np.linspace(0.4, 0.8, len(class_names)))
    colors_recall = plt.cm.Reds(np.linspace(0.4, 0.8, len(class_names)))
    colors_f1 = plt.cm.Greens(np.linspace(0.4, 0.8, len(class_names)))

    # Precision
    bars0 = axes[0].bar(x, precision * 100, width, color=colors_precision,
                        edgecolor='darkblue', linewidth=2)
    axes[0].set_ylabel('Precision (%)', fontsize=13, fontweight='bold')
    axes[0].set_title('Precision by Class', fontsize=14, fontweight='bold', pad=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right', fontsize=11)
    axes[0].axhline(y=np.mean(precision) * 100, color='red', linestyle='--',
                    linewidth=2.5, label=f'Macro Avg: {np.mean(precision)*100:.2f}%', alpha=0.8)
    axes[0].legend(fontsize=11, loc='lower right')
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].set_ylim([0, 105])

    # Add value labels on bars
    for bar in bars0:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    # Recall
    bars1 = axes[1].bar(x, recall * 100, width, color=colors_recall,
                        edgecolor='darkred', linewidth=2)
    axes[1].set_ylabel('Recall (%)', fontsize=13, fontweight='bold')
    axes[1].set_title('Recall by Class', fontsize=14, fontweight='bold', pad=10)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right', fontsize=11)
    axes[1].axhline(y=np.mean(recall) * 100, color='red', linestyle='--',
                    linewidth=2.5, label=f'Macro Avg: {np.mean(recall)*100:.2f}%', alpha=0.8)
    axes[1].legend(fontsize=11, loc='lower right')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1].set_ylim([0, 105])

    for bar in bars1:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    # F1-Score
    bars2 = axes[2].bar(x, f1 * 100, width, color=colors_f1,
                        edgecolor='darkgreen', linewidth=2)
    axes[2].set_ylabel('F1-Score (%)', fontsize=13, fontweight='bold')
    axes[2].set_title('F1-Score by Class', fontsize=14, fontweight='bold', pad=10)
    axes[2].set_xlabel('Class', fontsize=13, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(class_names, rotation=45, ha='right', fontsize=11)
    axes[2].axhline(y=np.mean(f1) * 100, color='red', linestyle='--',
                    linewidth=2.5, label=f'Macro Avg: {np.mean(f1)*100:.2f}%', alpha=0.8)
    axes[2].legend(fontsize=11, loc='lower right')
    axes[2].grid(axis='y', alpha=0.3, linestyle='--')
    axes[2].set_ylim([0, 105])

    for bar in bars2:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    plt.suptitle(title, fontsize=17, fontweight='bold', y=0.998)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_class_distribution(class_names: List[str], support: np.ndarray, save_path: str):
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
    bars = ax.bar(class_names, support, color=colors, edgecolor='black', linewidth=2)

    ax.set_xlabel('Class', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=13, fontweight='bold')
    ax.set_title('Test Set Class Distribution', fontsize=15, fontweight='bold', pad=15)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels and percentage
    total_samples = np.sum(support)
    for bar in bars:
        height = bar.get_height()
        percentage = (height / total_samples) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height + max(support)*0.01,
                f'{int(height)}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Add total count
    ax.text(0.98, 0.98, f'Total: {int(total_samples)} samples',
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()