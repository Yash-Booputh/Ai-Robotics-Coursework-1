"""
Comprehensive Model Comparison

Compares all trained models and generates detailed comparison visualizations
and analysis for academic reporting.

Usage:
    python scripts/compare_models.py
"""

import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results():
    """Load evaluation results for all models"""
    results_path = Path('results')
    models = ['resnet34', 'resnet50', 'yolo11m', 'yolo12m']

    all_results = {}

    print("\n" + "=" * 80)
    print("MODEL COMPARISON SYSTEM")
    print("=" * 80)
    print("\nLoading evaluation results...")

    for model in models:
        # Look in model subfolder first (new format)
        result_file = results_path / model / f'test_results_{model}.json'

        # Fallback to old location if not found
        if not result_file.exists():
            result_file = results_path / f'test_results_{model}.json'

        if result_file.exists():
            with open(result_file, 'r') as f:
                all_results[model] = json.load(f)
            print(f"  Loaded: {model.upper()}")
        else:
            print(f"  Missing: {model.upper()} (file not found: {result_file})")

    return all_results


def create_comparison_table(results: dict) -> pd.DataFrame:
    """Create detailed comparison table"""
    comparison_data = {
        'Model': [],
        'Architecture': [],
        'Parameters (M)': [],
        'Model Size (MB)': [],
        'Accuracy (%)': [],
        'Macro F1 (%)': [],
        'Weighted F1 (%)': [],
        'Macro Precision (%)': [],
        'Macro Recall (%)':[],
    }

    # Model specifications
    model_specs = {
        'resnet34': {
            'architecture': 'ResNet-34',
            'parameters': 21.3,
            'size_mb': 84
        },
        'resnet50': {
            'architecture': 'ResNet-50',
            'parameters': 23.5,
            'size_mb': 98
        },
        'yolo11m': {
            'architecture': 'YOLO11m-cls',
            'parameters': 5.2,
            'size_mb': 20
        },
        'yolo12m': {
            'architecture': 'YOLO12m-cls',
            'parameters': 5.5,
            'size_mb': 22
        }
    }

    for model_name, result in results.items():
        specs = model_specs[model_name]
        overall = result['overall_metrics']

        comparison_data['Model'].append(model_name.upper())
        comparison_data['Architecture'].append(specs['architecture'])
        comparison_data['Parameters (M)'].append(specs['parameters'])
        comparison_data['Model Size (MB)'].append(specs['size_mb'])
        comparison_data['Accuracy (%)'].append(overall['test_accuracy'] * 100)
        comparison_data['Macro F1 (%)'].append(overall['macro_f1'] * 100)
        comparison_data['Weighted F1 (%)'].append(overall['weighted_f1'] * 100)
        comparison_data['Macro Precision (%)'].append(overall['macro_precision'] * 100)
        comparison_data['Macro Recall (%)'].append(overall['macro_recall'] * 100)

    return pd.DataFrame(comparison_data)


def plot_performance_metrics(df: pd.DataFrame, comparison_path: Path):
    """Plot 1: Performance Metrics (Accuracy, F1, Precision, Recall)"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Metrics Comparison', fontsize=16, fontweight='bold', y=0.995)

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    models = df['Model'].tolist()
    x = np.arange(len(models))
    width = 0.35

    # 1. Test Accuracy
    bars = axes[0, 0].bar(models, df['Accuracy (%)'], color=colors, edgecolor='black', linewidth=2)
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Test Accuracy', fontsize=13, fontweight='bold', pad=10)
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].set_ylim([0, 105])
    for i, (bar, v) in enumerate(zip(bars, df['Accuracy (%)'])):
        axes[0, 0].text(i, v + 1.5, f'{v:.2f}%', ha='center', fontweight='bold', fontsize=10)

    # 2. F1 Scores
    axes[0, 1].bar(x - width/2, df['Macro F1 (%)'], width, label='Macro F1',
                   color='#3498db', edgecolor='black', linewidth=1.5)
    axes[0, 1].bar(x + width/2, df['Weighted F1 (%)'], width, label='Weighted F1',
                   color='#e74c3c', edgecolor='black', linewidth=1.5)
    axes[0, 1].set_ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('F1-Score Comparison', fontsize=13, fontweight='bold', pad=10)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models, fontsize=11)
    axes[0, 1].legend(fontsize=11, loc='lower right')
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].set_ylim([0, 105])

    # 3. Precision vs Recall
    axes[1, 0].bar(x - width/2, df['Macro Precision (%)'], width, label='Precision',
                   color='#2ecc71', edgecolor='black', linewidth=1.5)
    axes[1, 0].bar(x + width/2, df['Macro Recall (%)'], width, label='Recall',
                   color='#9b59b6', edgecolor='black', linewidth=1.5)
    axes[1, 0].set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Precision vs Recall', fontsize=13, fontweight='bold', pad=10)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(models, fontsize=11)
    axes[1, 0].legend(fontsize=11, loc='lower right')
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].set_ylim([0, 105])

    # 4. Overall Metrics Table
    axes[1, 1].axis('off')
    table_data = []
    for i, model in enumerate(models):
        table_data.append([
            model,
            f"{df['Accuracy (%)'].iloc[i]:.2f}%",
            f"{df['Macro F1 (%)'].iloc[i]:.2f}%",
            f"{df['Macro Precision (%)'].iloc[i]:.2f}%",
            f"{df['Macro Recall (%)'].iloc[i]:.2f}%"
        ])

    table = axes[1, 1].table(cellText=table_data,
                            colLabels=['Model', 'Accuracy', 'Macro F1', 'Precision', 'Recall'],
                            cellLoc='center',
                            loc='center',
                            colWidths=[0.15, 0.18, 0.18, 0.18, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    for i in range(len(table_data) + 1):
        for j in range(5):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#3498db')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor(['#ecf0f1', '#d5dbdb'][i % 2])

    axes[1, 1].set_title('Performance Summary', fontsize=13, fontweight='bold', pad=10)

    plt.tight_layout()
    save_path = comparison_path / '1_performance_metrics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Performance metrics plot saved: {save_path.name}")


def plot_model_efficiency(df: pd.DataFrame, comparison_path: Path):
    """Plot 2: Model Size and Efficiency Analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Efficiency and Resource Analysis', fontsize=16, fontweight='bold', y=0.995)

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    models = df['Model'].tolist()

    # 1. Model Size
    bars = axes[0, 0].bar(models, df['Model Size (MB)'], color=colors, edgecolor='black', linewidth=2)
    axes[0, 0].set_ylabel('Size (MB)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Model Size Comparison', fontsize=13, fontweight='bold', pad=10)
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, (bar, v) in enumerate(zip(bars, df['Model Size (MB)'])):
        axes[0, 0].text(i, v + 2, f'{v} MB', ha='center', fontweight='bold', fontsize=10)

    # 2. Parameters Count
    bars = axes[0, 1].bar(models, df['Parameters (M)'], color=colors, edgecolor='black', linewidth=2)
    axes[0, 1].set_ylabel('Parameters (Millions)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Model Parameters', fontsize=13, fontweight='bold', pad=10)
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, (bar, v) in enumerate(zip(bars, df['Parameters (M)'])):
        axes[0, 1].text(i, v + 0.3, f'{v}M', ha='center', fontweight='bold', fontsize=10)

    # 3. Efficiency Scatter
    scatter = axes[1, 0].scatter(df['Model Size (MB)'], df['Accuracy (%)'],
                                s=df['Parameters (M)']*80, c=colors, alpha=0.6,
                                edgecolors='black', linewidth=2)
    for i, model in enumerate(models):
        axes[1, 0].annotate(model,
                           (df['Model Size (MB)'].iloc[i], df['Accuracy (%)'].iloc[i]),
                           fontsize=10, fontweight='bold', ha='center',
                           xytext=(0, 10), textcoords='offset points')
    axes[1, 0].set_xlabel('Model Size (MB)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Efficiency Analysis\n(Bubble size = Parameters)',
                        fontsize=13, fontweight='bold', pad=10)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Efficiency Bar Chart
    efficiency = df['Accuracy (%)'] / df['Model Size (MB)']
    bars = axes[1, 1].bar(models, efficiency, color=colors, edgecolor='black', linewidth=2)
    axes[1, 1].set_ylabel('Accuracy % per MB', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Efficiency Score\n(Accuracy per MB)', fontsize=13, fontweight='bold', pad=10)
    axes[1, 1].grid(axis='y', alpha=0.3)
    for i, (bar, v) in enumerate(zip(bars, efficiency)):
        axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    save_path = comparison_path / '2_efficiency_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Efficiency analysis plot saved: {save_path.name}")


def plot_radar_comparison(df: pd.DataFrame, comparison_path: Path):
    """Plot 3: Radar Chart for Overall Performance"""
    fig, ax = plt.subplots(figsize=(14, 12), subplot_kw=dict(projection='polar'))

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    models = df['Model'].tolist()

    categories = ['Accuracy', 'Macro F1', 'Precision', 'Recall']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    for i, model in enumerate(models):
        values = [
            df['Accuracy (%)'].iloc[i],
            df['Macro F1 (%)'].iloc[i],
            df['Macro Precision (%)'].iloc[i],
            df['Macro Recall (%)'].iloc[i]
        ]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2.5, label=model, color=colors[i], markersize=8)
        ax.fill(angles, values, alpha=0.2, color=colors[i])

    # Set the labels and add padding using tick_params
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', pad=20)  # Add padding to x-axis labels

    # Extend the y-axis limit to create more space
    ax.set_ylim(0, 110)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)

    ax.set_title('Overall Performance Comparison\n(Radar Chart)',
                fontsize=15, fontweight='bold', pad=40)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=12, framealpha=0.9)
    ax.grid(True, linewidth=1.2, alpha=0.7)

    # Add some padding around the plot
    plt.tight_layout(pad=2.0)

    save_path = comparison_path / '3_radar_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Radar comparison plot saved: {save_path.name}")


def plot_detailed_comparison_table(df: pd.DataFrame, comparison_path: Path):
    """Plot 4: Detailed Comparison Table"""
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data with all metrics
    table_data = []
    for i in range(len(df)):
        row = [
            df['Model'].iloc[i],
            df['Architecture'].iloc[i],
            f"{df['Parameters (M)'].iloc[i]} M",
            f"{df['Model Size (MB)'].iloc[i]} MB",
            f"{df['Accuracy (%)'].iloc[i]:.2f}%",
            f"{df['Macro F1 (%)'].iloc[i]:.2f}%",
            f"{df['Weighted F1 (%)'].iloc[i]:.2f}%",
            f"{df['Macro Precision (%)'].iloc[i]:.2f}%",
            f"{df['Macro Recall (%)'].iloc[i]:.2f}%",
        ]
        table_data.append(row)

    # Add efficiency metric
    efficiency = df['Accuracy (%)'] / df['Model Size (MB)']
    for i, row in enumerate(table_data):
        row.append(f"{efficiency.iloc[i]:.3f}")

    # Column headers
    headers = [
        'Model',
        'Architecture',
        'Parameters',
        'Size',
        'Accuracy',
        'Macro F1',
        'Weighted F1',
        'Precision',
        'Recall',
        'Efficiency\n(Acc/MB)'
    ]

    # Create table
    table = ax.table(cellText=table_data,
                     colLabels=headers,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.10, 0.12, 0.10, 0.08, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)

    # Color scheme
    colors_map = {
        'RESNET34': '#3498db',
        'RESNET50': '#e74c3c',
        'YOLO11M': '#2ecc71',
        'YOLO12M': '#f39c12'
    }

    # Style header row
    for j in range(len(headers)):
        cell = table[(0, j)]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(weight='bold', color='white', fontsize=12)
        cell.set_height(0.08)

    # Style data rows with alternating colors and highlight best values
    for i in range(len(table_data)):
        model_color = colors_map.get(df['Model'].iloc[i], '#95a5a6')

        for j in range(len(headers)):
            cell = table[(i + 1, j)]

            # Model name column - use model color
            if j == 0:
                cell.set_facecolor(model_color)
                cell.set_text_props(weight='bold', color='white', fontsize=11)
            else:
                # Alternating row colors for data
                cell.set_facecolor(['#ecf0f1', '#d5dbdb'][i % 2])
                cell.set_text_props(fontsize=11)

            cell.set_height(0.06)

    # Highlight best values in each metric column
    metric_columns = {
        4: df['Accuracy (%)'],      # Accuracy
        5: df['Macro F1 (%)'],       # Macro F1
        6: df['Weighted F1 (%)'],    # Weighted F1
        7: df['Macro Precision (%)'],# Precision
        8: df['Macro Recall (%)'],   # Recall
        9: efficiency                # Efficiency
    }

    # Columns where lower is better (size, parameters)
    lower_better = {2, 3}

    for col_idx, series in metric_columns.items():
        best_idx = series.idxmax()
        cell = table[(best_idx + 1, col_idx)]
        cell.set_facecolor('#2ecc71')
        cell.set_text_props(weight='bold', color='white')

    # Highlight smallest size and parameters
    for col_idx in [2, 3]:
        if col_idx == 2:  # Parameters
            best_idx = df['Parameters (M)'].idxmin()
        else:  # Size
            best_idx = df['Model Size (MB)'].idxmin()
        cell = table[(best_idx + 1, col_idx)]
        cell.set_facecolor('#3498db')
        cell.set_text_props(weight='bold', color='white')

    # Add title and legend
    title_text = 'Comprehensive Model Comparison Table\n'
    title_text += 'Green: Best Performance | Blue: Smallest/Most Efficient'
    plt.title(title_text, fontsize=16, fontweight='bold', pad=20)

    # Add statistics below table
    stats_text = (
        f"Statistical Summary:\n"
        f"Mean Accuracy: {df['Accuracy (%)'].mean():.2f}% | "
        f"Std Dev: {df['Accuracy (%)'].std():.2f}% | "
        f"Range: {df['Accuracy (%)'].max() - df['Accuracy (%)'].min():.2f}%\n"
        f"Mean F1: {df['Macro F1 (%)'].mean():.2f}% | "
        f"Std Dev: {df['Macro F1 (%)'].std():.2f}%"
    )
    plt.figtext(0.5, 0.05, stats_text, ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout(pad=2.0)

    save_path = comparison_path / '4_detailed_comparison_table.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Detailed comparison table saved: {save_path.name}")


def print_detailed_comparison(df: pd.DataFrame, results: dict):
    """Print detailed comparison analysis"""
    print("\n" + "=" * 80)
    print("DETAILED MODEL COMPARISON")
    print("=" * 80)

    print("\n" + "-" * 80)
    print("Model Specifications and Performance")
    print("-" * 80)
    print(df.to_string(index=False))

    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)

    # Best performers
    best_accuracy_idx = df['Accuracy (%)'].idxmax()
    best_f1_idx = df['Macro F1 (%)'].idxmax()
    smallest_size_idx = df['Model Size (MB)'].idxmin()
    fewest_params_idx = df['Parameters (M)'].idxmin()

    print("\nPerformance Leaders:")
    print("-" * 80)
    print(f"Highest Accuracy:       {df.loc[best_accuracy_idx, 'Model']:<12} "
          f"({df.loc[best_accuracy_idx, 'Accuracy (%)']:.2f}%)")
    print(f"Highest Macro F1:       {df.loc[best_f1_idx, 'Model']:<12} "
          f"({df.loc[best_f1_idx, 'Macro F1 (%)']:.2f}%)")
    print(f"Smallest Model:         {df.loc[smallest_size_idx, 'Model']:<12} "
          f"({df.loc[smallest_size_idx, 'Model Size (MB)']} MB)")
    print(f"Fewest Parameters:      {df.loc[fewest_params_idx, 'Model']:<12} "
          f"({df.loc[fewest_params_idx, 'Parameters (M)']} M)")

    # Statistical analysis
    print("\n" + "-" * 80)
    print("Statistical Analysis:")
    print("-" * 80)
    print(f"Mean Accuracy:          {df['Accuracy (%)'].mean():.2f}%")
    print(f"Accuracy Std Dev:       {df['Accuracy (%)'].std():.2f}%")
    print(f"Accuracy Range:         {df['Accuracy (%)'].max() - df['Accuracy (%)'].min():.2f}%")
    print(f"Mean Macro F1:          {df['Macro F1 (%)'].mean():.2f}%")
    print(f"F1 Std Dev:             {df['Macro F1 (%)'].std():.2f}%")

    # Efficiency analysis
    print("\n" + "-" * 80)
    print("Efficiency Analysis (Accuracy per MB):")
    print("-" * 80)
    efficiency = df['Accuracy (%)'] / df['Model Size (MB)']
    for i, model in enumerate(df['Model']):
        print(f"{model:<12} {efficiency.iloc[i]:.3f}% per MB")

    # Per-class performance variance
    print("\n" + "-" * 80)
    print("Per-Class Performance Variance:")
    print("-" * 80)

    for model_name in results.keys():
        per_class = results[model_name]['per_class_metrics']
        accuracies = [metrics['accuracy'] for metrics in per_class.values()]
        print(f"{model_name.upper():<12} "
              f"Mean: {np.mean(accuracies)*100:.2f}%  "
              f"Std: {np.std(accuracies)*100:.2f}%  "
              f"Range: {(max(accuracies)-min(accuracies))*100:.2f}%")

    print("\n" + "=" * 80)


def generate_summary_report(df: pd.DataFrame, results: dict, save_path: str):
    """Generate comprehensive text report"""
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE MODEL COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of Models: {len(df)}\n")
        f.write(f"Models Compared: {', '.join(df['Model'].tolist())}\n\n")

        f.write("=" * 80 + "\n")
        f.write("MODEL SPECIFICATIONS AND PERFORMANCE\n")
        f.write("=" * 80 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")

        f.write("=" * 80 + "\n")
        f.write("PERFORMANCE RANKINGS\n")
        f.write("=" * 80 + "\n\n")

        metrics_to_rank = ['Accuracy (%)', 'Macro F1 (%)', 'Weighted F1 (%)']
        for metric in metrics_to_rank:
            f.write(f"\n{metric}:\n")
            f.write("-" * 40 + "\n")
            sorted_df = df.sort_values(by=metric, ascending=False)
            for i, (idx, row) in enumerate(sorted_df.iterrows(), 1):
                f.write(f"{i}. {row['Model']:<12} {row[metric]:.2f}%\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("EFFICIENCY METRICS\n")
        f.write("=" * 80 + "\n\n")

        f.write("Model Size Ranking (Smallest to Largest):\n")
        f.write("-" * 40 + "\n")
        sorted_df = df.sort_values(by='Model Size (MB)')
        for i, (idx, row) in enumerate(sorted_df.iterrows(), 1):
            f.write(f"{i}. {row['Model']:<12} {row['Model Size (MB)']} MB, "
                   f"{row['Parameters (M)']} M parameters\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("KEY FINDINGS AND RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")

        best_acc_model = df.loc[df['Accuracy (%)'].idxmax(), 'Model']
        smallest_model = df.loc[df['Model Size (MB)'].idxmin(), 'Model']

        f.write(f"1. Best Overall Performance: {best_acc_model}\n")
        f.write(f"   - Highest test accuracy and F1-score\n")
        f.write(f"   - Recommended for applications prioritizing accuracy\n\n")

        f.write(f"2. Most Efficient Model: {smallest_model}\n")
        f.write(f"   - Smallest size and parameter count\n")
        f.write(f"   - Recommended for resource-constrained deployments\n\n")

        f.write("3. Performance-Efficiency Trade-off:\n")
        efficiency = df['Accuracy (%)'] / df['Model Size (MB)']
        best_efficiency_idx = efficiency.idxmax()
        f.write(f"   - Best trade-off: {df.loc[best_efficiency_idx, 'Model']}\n")
        f.write(f"   - Efficiency score: {efficiency.iloc[best_efficiency_idx]:.3f}% per MB\n\n")

        f.write("=" * 80 + "\n")

    print(f"Summary report saved: {save_path}")


def main():
    """Run comprehensive model comparison"""
    results = load_results()

    if not results:
        print("\nNo evaluation results found. Please run evaluation scripts first:")
        print("  python scripts/evaluate_resnet34.py")
        print("  python scripts/evaluate_resnet50.py")
        print("  python scripts/evaluate_yolo11m.py")
        print("  python scripts/evaluate_yolo12m.py")
        return

    if len(results) < 4:
        print(f"\nWarning: Only {len(results)}/4 models found.")
        print("Run remaining evaluation scripts for complete comparison.")

    print(f"\nFound results for {len(results)} models: {list(results.keys())}")

    # Create comparison table
    df = create_comparison_table(results)

    # Print detailed comparison
    print_detailed_comparison(df, results)

    # Create comparison folder
    comparison_path = Path('results/comparison')
    comparison_path.mkdir(parents=True, exist_ok=True)

    # Generate separate visualizations
    print("\nGenerating comparison visualizations...")
    plot_performance_metrics(df, comparison_path)
    plot_model_efficiency(df, comparison_path)
    plot_radar_comparison(df, comparison_path)
    plot_detailed_comparison_table(df, comparison_path)

    # Save comparison data
    comparison_json = comparison_path / 'model_comparison.json'
    comparison_dict = {
        'comparison_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models_compared': len(results),
        'summary_statistics': {
            'mean_accuracy': float(df['Accuracy (%)'].mean()),
            'std_accuracy': float(df['Accuracy (%)'].std()),
            'mean_f1': float(df['Macro F1 (%)'].mean()),
            'std_f1': float(df['Macro F1 (%)'].std())
        },
        'detailed_comparison': df.to_dict(orient='records')
    }

    with open(comparison_json, 'w') as f:
        json.dump(comparison_dict, f, indent=2)
    print(f"\nComparison data saved: {comparison_json}")

    # Generate summary report
    generate_summary_report(df, results, str(comparison_path / 'comparison_report.txt'))

    print("\n" + "=" * 80)
    print("MODEL COMPARISON COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - results/comparison/1_performance_metrics.png")
    print("  - results/comparison/2_efficiency_analysis.png")
    print("  - results/comparison/3_radar_comparison.png")
    print("  - results/comparison/4_detailed_comparison_table.png")
    print("  - results/comparison/model_comparison.json")
    print("  - results/comparison/comparison_report.txt")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()