"""
Compare all trained models and generate comparison visualizations.

Usage:
    python scripts/compare_models.py
"""

import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results():
    results_path = Path('results')
    models = ['resnet34', 'resnet50', 'yolo11m', 'yolo12m']

    all_results = {}

    for model in models:
        result_file = results_path / f'test_results_{model}.json'
        if result_file.exists():
            with open(result_file, 'r') as f:
                all_results[model] = json.load(f)

    return all_results


def create_comparison_table(results: dict) -> pd.DataFrame:
    comparison_data = {
        'Model': [],
        'Accuracy (%)': [],
        'Macro F1 (%)': [],
        'Weighted F1 (%)': [],
        'Model Size (MB)': []
    }

    model_sizes = {
        'resnet34': 84,
        'resnet50': 98,
        'yolo11m': 20,
        'yolo12m': 22
    }

    for model_name, result in results.items():
        comparison_data['Model'].append(model_name.upper())
        comparison_data['Accuracy (%)'].append(result['test_accuracy'] * 100)
        comparison_data['Macro F1 (%)'].append(result['macro_f1'] * 100)
        comparison_data['Weighted F1 (%)'].append(result['weighted_f1'] * 100)
        comparison_data['Model Size (MB)'].append(model_sizes[model_name])

    return pd.DataFrame(comparison_data)


def plot_comparison(df: pd.DataFrame, save_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    axes[0, 0].bar(df['Model'], df['Accuracy (%)'], color=colors,
                   edgecolor='black', linewidth=1.5)
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Test Accuracy Comparison', fontsize=13, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=15)
    for i, v in enumerate(df['Accuracy (%)']):
        axes[0, 0].text(i, v + 1, f'{v:.2f}%', ha='center', fontweight='bold', fontsize=10)

    axes[0, 1].bar(df['Model'], df['Macro F1 (%)'], color=colors,
                   edgecolor='black', linewidth=1.5)
    axes[0, 1].set_ylabel('Macro F1-Score (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Macro F1-Score Comparison', fontsize=13, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=15)
    for i, v in enumerate(df['Macro F1 (%)']):
        axes[0, 1].text(i, v + 1, f'{v:.2f}%', ha='center', fontweight='bold', fontsize=10)

    axes[1, 0].bar(df['Model'], df['Weighted F1 (%)'], color=colors,
                   edgecolor='black', linewidth=1.5)
    axes[1, 0].set_ylabel('Weighted F1 (%)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Weighted F1-Score Comparison', fontsize=13, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=15)
    for i, v in enumerate(df['Weighted F1 (%)']):
        axes[1, 0].text(i, v + 1, f'{v:.2f}%', ha='center', fontweight='bold', fontsize=10)

    axes[1, 1].bar(df['Model'], df['Model Size (MB)'], color=colors,
                   edgecolor='black', linewidth=1.5)
    axes[1, 1].set_ylabel('Model Size (MB)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Model Size Comparison', fontsize=13, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=15)
    for i, v in enumerate(df['Model Size (MB)']):
        axes[1, 1].text(i, v + 3, f'{v}MB', ha='center', fontweight='bold', fontsize=10)

    plt.suptitle('Comprehensive Model Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Comparison plot saved: {save_path}")


def print_comparison(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print("\n" + df.to_string(index=False))
    print("\n" + "=" * 70)

    best_accuracy_idx = df['Accuracy (%)'].idxmax()
    best_f1_idx = df['Macro F1 (%)'].idxmax()
    smallest_size_idx = df['Model Size (MB)'].idxmin()

    print("\nKey Findings:")
    print(f"  Best Accuracy:    {df.loc[best_accuracy_idx, 'Model']} "
          f"({df.loc[best_accuracy_idx, 'Accuracy (%)']:.2f}%)")
    print(f"  Best Macro F1:    {df.loc[best_f1_idx, 'Model']} "
          f"({df.loc[best_f1_idx, 'Macro F1 (%)']:.2f}%)")
    print(f"  Smallest Model:   {df.loc[smallest_size_idx, 'Model']} "
          f"({df.loc[smallest_size_idx, 'Model Size (MB)']} MB)")
    print("=" * 70)


def main():
    results = load_results()

    if not results:
        print("No evaluation results found. Please run evaluation scripts first.")
        return

    print(f"Found results for {len(results)} models: {list(results.keys())}")

    df = create_comparison_table(results)
    print_comparison(df)

    plot_comparison(df, 'results/model_comparison.png')

    comparison_file = 'results/model_comparison.json'
    with open(comparison_file, 'w') as f:
        json.dump(df.to_dict(orient='records'), f, indent=2)
    print(f"\nComparison data saved: {comparison_file}")


if __name__ == "__main__":
    main()