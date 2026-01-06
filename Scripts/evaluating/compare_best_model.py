"""compare_best_model.py
Compare all fusion strategies using ConvNeXt-B backbone.

Loads pre-computed results from training scripts:
- Feature Fusion from train_best_model.py (Results/best_model/)
- Other strategies from train_other_strategies.py (Results/other_strategies/convnext_b/)

Generates comprehensive comparison plots and analysis for all strategies.

Output: Results/other_strategies/convnext_b/
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_all_results():
    """Load results from pre-computed JSON files."""
    # Load feature fusion (best model) - from train_best_model.py
    feature_fusion_file = project_root / "Results" / "best_model" / "metrics.json"

    if not feature_fusion_file.exists():
        print(f"\nERROR: Feature Fusion results not found at {feature_fusion_file}")
        print("Please run Scripts/training/train_best_model.py first.")
        return None

    with open(feature_fusion_file, 'r') as f:
        feature_results = json.load(f)

    # Load other strategies from train_other_strategies.py
    other_strategies_file = project_root / "Results" / "other_strategies" / "convnext_b" / "other_strategies_metrics.json"

    if not other_strategies_file.exists():
        print(f"\nERROR: Other strategies results not found at {other_strategies_file}")
        print("Please run Scripts/training/train_other_strategies.py first.")
        return None

    with open(other_strategies_file, 'r') as f:
        other_results = json.load(f)

    # Combine all results
    all_results = {
        'TOP View': {
            'test_mae': other_results['top_view']['test_mae'],
            'test_rmse': other_results['top_view']['test_rmse'],
        },
        'SIDE View': {
            'test_mae': other_results['side_view']['test_mae'],
            'test_rmse': other_results['side_view']['test_rmse'],
        },
        'Baseline': {
            'test_mae': other_results['baseline']['test_mae'],
            'test_rmse': other_results['baseline']['test_rmse'],
        },
        'Late Fusion': {
            'test_mae': other_results['late_fusion']['test_mae'],
            'test_rmse': other_results['late_fusion']['test_rmse'],
        },
        'Feature Fusion': {
            'test_mae': feature_results['test_mae'],
            'test_rmse': feature_results['test_rmse'],
        }
    }

    return all_results


def create_summary_table(results):
    """Create summary table from results."""
    summary = []

    for strategy, metrics in results.items():
        summary.append({
            "Strategy": strategy,
            "Test MAE": metrics['test_mae'],
            "Test RMSE": metrics['test_rmse'],
        })

    df = pd.DataFrame(summary)
    # Sort by Test MAE
    df = df.sort_values("Test MAE")

    return df


def plot_comparison_bar(df):
    """Plot comparison bar chart of all strategies."""
    results_dir = project_root / "Results" / "other_strategies" / "convnext_b" / "graphs"
    results_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Define colors
    colors = {
        "TOP View": "lightblue",
        "SIDE View": "lightcoral",
        "Baseline": "lightgreen",
        "Late Fusion": "lightsalmon",
        "Feature Fusion": "gold"
    }

    # Get best model (lowest Test MAE)
    best_idx = df["Test MAE"].idxmin()
    best_strategy = df.loc[best_idx, "Strategy"]

    # Plot bars
    bar_colors = [colors.get(strategy, "gray") for strategy in df["Strategy"]]
    edge_colors = ['red' if strategy == best_strategy else 'black' for strategy in df["Strategy"]]
    edge_widths = [3 if strategy == best_strategy else 1.5 for strategy in df["Strategy"]]

    bars = ax.bar(range(len(df)), df["Test MAE"],
                   color=bar_colors,
                   alpha=0.8,
                   edgecolor=edge_colors,
                   linewidth=edge_widths)

    # Customize plot
    ax.set_xlabel('Strategy', fontsize=16, fontweight='bold')
    ax.set_ylabel('Test MAE (days)', fontsize=16, fontweight='bold')
    ax.set_title('ConvNeXt-B: Comparison of All Fusion Strategies',
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["Strategy"], fontsize=14, rotation=15, ha='right')
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (idx, row) in enumerate(df.iterrows()):
        height = row["Test MAE"]
        ax.text(i, height + 0.005,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Add legend for best model
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', edgecolor='red', linewidth=2, label='Best Model')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.savefig(results_dir / "strategy_comparison.png", dpi=200, bbox_inches='tight')
    print(f"  Saved: graphs/strategy_comparison.png")
    plt.close()


def plot_grouped_comparison(df):
    """Plot grouped bar chart comparing strategies by category."""
    results_dir = project_root / "Results" / "other_strategies" / "convnext_b" / "graphs"

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Group strategies
    single_view = df[df["Strategy"].isin(["TOP View", "SIDE View"])]
    fusion_strategies = df[df["Strategy"].isin(["Baseline", "Late Fusion", "Feature Fusion"])]

    # Define colors
    colors_single = {"TOP View": "steelblue", "SIDE View": "coral"}
    colors_fusion = {"Baseline": "mediumseagreen", "Late Fusion": "orange", "Feature Fusion": "crimson"}

    x_positions = []
    bar_colors = []
    current_x = 0

    # Plot single-view strategies
    for idx, row in single_view.iterrows():
        x_positions.append(current_x)
        bar_colors.append(colors_single[row["Strategy"]])
        current_x += 1

    # Add gap
    current_x += 0.5

    # Plot fusion strategies
    for idx, row in fusion_strategies.iterrows():
        x_positions.append(current_x)
        bar_colors.append(colors_fusion[row["Strategy"]])
        current_x += 1

    # Combine data
    all_strategies = pd.concat([single_view, fusion_strategies])

    # Plot
    bars = ax.bar(x_positions, all_strategies["Test MAE"],
                   color=bar_colors,
                   alpha=0.8,
                   edgecolor='black',
                   linewidth=2)

    # Highlight best model
    best_idx = all_strategies["Test MAE"].idxmin()
    best_strategy = all_strategies.loc[best_idx, "Strategy"]
    best_position = x_positions[list(all_strategies.index).index(best_idx)]

    bars[list(all_strategies.index).index(best_idx)].set_edgecolor('red')
    bars[list(all_strategies.index).index(best_idx)].set_linewidth(4)

    # Customize
    ax.set_xlabel('Strategy', fontsize=16, fontweight='bold')
    ax.set_ylabel('Test MAE (days)', fontsize=16, fontweight='bold')
    ax.set_title('ConvNeXt-B: Single-View vs Multi-View Fusion Strategies',
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(all_strategies["Strategy"], fontsize=13, rotation=15, ha='right')
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for x, height in zip(x_positions, all_strategies["Test MAE"]):
        ax.text(x, height + 0.005,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Add category labels
    ax.text(0.5, ax.get_ylim()[1] * 0.95, 'Single-View',
            ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(len(single_view) + 1.25, ax.get_ylim()[1] * 0.95, 'Multi-View Fusion',
            ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(results_dir / "grouped_comparison.png", dpi=200, bbox_inches='tight')
    print(f"  Saved: graphs/grouped_comparison.png")
    plt.close()


def perform_statistical_analysis(df):
    """Perform statistical analysis on results."""
    results_dir = project_root / "Results" / "other_strategies" / "convnext_b"

    report = []
    report.append("="*80)
    report.append("STATISTICAL ANALYSIS: ConvNeXt-B Strategy Comparison")
    report.append("="*80)
    report.append("")

    report.append("Test Set Performance Ranking (by MAE):")
    report.append("-"*80)
    for idx, (i, row) in enumerate(df.iterrows(), 1):
        report.append(f"{idx}. {row['Strategy']:<20} MAE: {row['Test MAE']:.4f} days")
    report.append("")

    # Best model info
    best_idx = df["Test MAE"].idxmin()
    best_strategy = df.loc[best_idx, "Strategy"]
    best_mae = df.loc[best_idx, "Test MAE"]

    report.append("-"*80)
    report.append(f"Best Strategy: {best_strategy}")
    report.append(f"Best Test MAE: {best_mae:.4f} days")
    report.append("")

    # Improvements over other strategies
    report.append("Improvement of Best Model over Others:")
    report.append("-"*80)
    for idx, row in df.iterrows():
        if row["Strategy"] != best_strategy:
            improvement = ((row["Test MAE"] - best_mae) / row["Test MAE"]) * 100
            diff = row["Test MAE"] - best_mae
            report.append(f"  vs {row['Strategy']:<20} {improvement:>6.2f}% improvement ({diff:+.4f} days)")
    report.append("")

    # Single-view vs Fusion analysis
    single_view_strategies = ["TOP View", "SIDE View"]
    fusion_strategies = ["Baseline", "Late Fusion", "Feature Fusion"]

    single_view_mae = df[df["Strategy"].isin(single_view_strategies)]["Test MAE"].mean()
    fusion_mae = df[df["Strategy"].isin(fusion_strategies)]["Test MAE"].mean()

    report.append("-"*80)
    report.append("Single-View vs Multi-View Fusion:")
    report.append("-"*80)
    report.append(f"Average Single-View MAE: {single_view_mae:.4f} days")
    report.append(f"Average Fusion MAE:      {fusion_mae:.4f} days")

    if fusion_mae < single_view_mae:
        improvement = ((single_view_mae - fusion_mae) / single_view_mae) * 100
        report.append(f"Fusion improves over single-view by {improvement:.2f}%")
    else:
        improvement = ((fusion_mae - single_view_mae) / fusion_mae) * 100
        report.append(f"Single-view improves over fusion by {improvement:.2f}%")

    report.append("")
    report.append("="*80)

    # Save report
    report_file = results_dir / "analysis_report.txt"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))

    # Print to console
    print("\n" + '\n'.join(report))

    return report_file


def main():
    print("="*80)
    print("COMPARING ALL STRATEGIES: ConvNeXt-B")
    print("="*80)

    # Load pre-computed results from training scripts
    print("\nLoading pre-computed results...")
    print("  - Feature Fusion: Results/best_model/metrics.json")
    print("  - Other strategies: Results/other_strategies/convnext_b/other_strategies_metrics.json")
    print("\nNOTE: These results are from:")
    print("  1. Scripts/training/train_best_model.py")
    print("  2. Scripts/training/train_other_strategies.py")

    results = load_all_results()
    if results is None:
        return

    print("\nAll results loaded successfully!")

    # Create summary table
    df = create_summary_table(results)

    # Save summary CSV
    results_dir = project_root / "Results" / "other_strategies" / "convnext_b"
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_file = results_dir / "csv" / "summary.csv"
    csv_file.parent.mkdir(exist_ok=True)
    df.to_csv(csv_file, index=False)
    print(f"\nSaved: csv/summary.csv")

    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\n{df.to_string(index=False)}\n")

    # Generate plots
    print("="*80)
    print("GENERATING COMPARISON PLOTS")
    print("="*80 + "\n")

    plot_comparison_bar(df)
    plot_grouped_comparison(df)

    # Statistical analysis
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)

    report_file = perform_statistical_analysis(df)
    print(f"\nSaved: analysis_report.txt")

    # Print completion
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {results_dir}")
    print("\nGenerated files:")
    print("  - csv/summary.csv")
    print("  - graphs/strategy_comparison.png")
    print("  - graphs/grouped_comparison.png")
    print("  - analysis_report.txt")


if __name__ == "__main__":
    main()
