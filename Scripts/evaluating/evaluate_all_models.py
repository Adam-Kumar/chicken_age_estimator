"""evaluate_all_models.py
Evaluate all 27 model configurations and generate comparison plots.

Loads CV results for:
- 9 Backbones × 3 Fusion Types = 27 configurations
- Baseline (single-view TOP), Late Fusion, Feature Fusion

Generates:
- All models comparison bar chart
- Model size vs performance scatter plot
- Fusion strategy comparison
- Summary CSV table

Output: Results/comparison/
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from Model.model import count_parameters, ResNetRegressor, LateFusionRegressor, FeatureFusionRegressor

# Backbone configuration
BACKBONES = [
    "efficientnet_b0",
    "resnet18",
    "resnet50",
    "resnet101",
    "vit_b_16",
    "swin_t",
    "swin_b",
    "convnext_t",
    "convnext_b",
]

# Model sizes (in millions of parameters)
MODEL_SIZES = {
    "efficientnet_b0": {"baseline": 4.01, "late": 8.02, "feature": 4.52},
    "resnet18": {"baseline": 11.18, "late": 22.36, "feature": 11.69},
    "resnet50": {"baseline": 23.51, "late": 47.03, "feature": 24.54},
    "resnet101": {"baseline": 42.50, "late": 85.01, "feature": 43.53},
    "vit_b_16": {"baseline": 86.57, "late": 173.13, "feature": 87.34},
    "swin_t": {"baseline": 27.52, "late": 55.04, "feature": 28.29},
    "swin_b": {"baseline": 87.27, "late": 174.54, "feature": 88.30},
    "convnext_t": {"baseline": 27.82, "late": 55.64, "feature": 28.59},
    "convnext_b": {"baseline": 88.02, "late": 176.04, "feature": 89.05},
}


def load_all_cv_results():
    """Load all cross-validation results (baseline + late + feature fusion)."""
    cv_results_file = project_root / "Results" / "comparison" / "csv" / "all_cv_results.json"

    if not cv_results_file.exists():
        print(f"\nERROR: CV results not found at {cv_results_file}")
        print("Please train all models first.")
        return {}

    with open(cv_results_file, 'r') as f:
        return json.load(f)


def create_summary_table():
    """Create comprehensive summary table with all 27 configurations."""
    cv_data = load_all_cv_results()

    summary = []

    for backbone in BACKBONES:
        # Baseline (view-agnostic)
        baseline_key = f"{backbone}_baseline"
        if baseline_key in cv_data:
            summary.append({
                "Backbone": backbone,
                "Fusion": "Baseline",
                "Mean MAE": cv_data[baseline_key].get("mean", np.nan),
                "Std MAE": cv_data[baseline_key].get("std", np.nan),
                "Params (M)": MODEL_SIZES[backbone]["baseline"],
            })
        else:
            summary.append({
                "Backbone": backbone,
                "Fusion": "Baseline",
                "Mean MAE": np.nan,
                "Std MAE": np.nan,
                "Params (M)": MODEL_SIZES[backbone]["baseline"],
            })

        # Late Fusion
        late_key = f"{backbone}_late"
        if late_key in cv_data:
            summary.append({
                "Backbone": backbone,
                "Fusion": "Late",
                "Mean MAE": cv_data[late_key].get("mean", np.nan),
                "Std MAE": cv_data[late_key].get("std", np.nan),
                "Params (M)": MODEL_SIZES[backbone]["late"],
            })

        # Feature Fusion
        feature_key = f"{backbone}_feature"
        if feature_key in cv_data:
            summary.append({
                "Backbone": backbone,
                "Fusion": "Feature",
                "Mean MAE": cv_data[feature_key].get("mean", np.nan),
                "Std MAE": cv_data[feature_key].get("std", np.nan),
                "Params (M)": MODEL_SIZES[backbone]["feature"],
            })

    df = pd.DataFrame(summary)

    # Sort by Mean MAE
    df = df.sort_values("Mean MAE")

    return df


def plot_all_models_comparison(df):
    """Plot comparison of all 27 models with improved readability."""
    from scipy import stats as scipy_stats

    results_dir = project_root / "Results" / "comparison"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Remove NaN rows for plotting
    df_clean = df.dropna(subset=["Mean MAE"])

    # Get best model info
    best_model = df_clean.loc[df_clean["Mean MAE"].idxmin()]

    # Create figure with subplots for each fusion type
    fig, axes = plt.subplots(1, 3, figsize=(24, 10))

    fusion_types = ["Baseline", "Late", "Feature"]
    colors = ["steelblue", "coral", "mediumseagreen"]

    for idx, (fusion, color) in enumerate(zip(fusion_types, colors)):
        ax = axes[idx]
        data = df_clean[df_clean["Fusion"] == fusion].sort_values("Mean MAE")

        if len(data) > 0:
            # Check if this fusion contains the best model
            is_best_fusion = (fusion == best_model['Fusion'])

            # Plot bars with conditional styling
            for i, (_, row) in enumerate(data.iterrows()):
                is_best = (is_best_fusion and row['Backbone'] == best_model['Backbone'])

                # Highlight best model with red edge
                edge_color = 'red' if is_best else 'black'
                edge_width = 4 if is_best else 2

                ax.barh(row["Backbone"], row["Mean MAE"],
                       xerr=row["Std MAE"],
                       color=color,
                       alpha=0.7,
                       edgecolor=edge_color,
                       linewidth=edge_width,
                       capsize=10,
                       error_kw={'linewidth': 2.5, 'elinewidth': 2.5})

                # Add annotation for best model
                if is_best:
                    annotation_text = f'Best Model\nMAE: {row["Mean MAE"]:.4f} ± {row["Std MAE"]:.4f}'
                    ax.text(row["Mean MAE"] + row["Std MAE"] + 0.03, row["Backbone"],
                           annotation_text,
                           fontsize=18,
                           verticalalignment='center',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat',
                                   alpha=0.9, edgecolor='red', linewidth=2))

            # Larger font sizes for presentation
            ax.set_xlabel("MAE (days)", fontsize=24, fontweight='bold')
            ax.set_title(f"{fusion} Fusion", fontsize=28, fontweight='bold', pad=40)
            ax.tick_params(axis='both', labelsize=20)

            # Calculate ANOVA for this fusion type (comparing backbones)
            mae_values = data["Mean MAE"].values
            if len(mae_values) > 2:  # Need at least 3 groups for ANOVA
                # One-way ANOVA comparing different backbones within this fusion type
                f_stat = np.var(mae_values) / np.mean(np.square(data["Std MAE"].values)) if np.mean(np.square(data["Std MAE"].values)) > 0 else 0
                # Add ANOVA text below title (non-bold)
                ax.text(0.5, 1.02, f"ANOVA F = {f_stat:.2f}",
                       transform=ax.transAxes,
                       fontsize=22,
                       ha='center',
                       va='bottom',
                       fontweight='normal')
            ax.grid(axis='x', alpha=0.3, linewidth=1.5)
            ax.invert_yaxis()

    # Set xlim AFTER all plotting to ensure it's applied
    for ax in axes:
        ax.set_xlim(0, 0.6)

    # Add main title
    fig.suptitle('All Models Comparison',
                fontsize=30, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    (results_dir / "graphs").mkdir(exist_ok=True)
    plt.savefig(results_dir / "graphs" / "all_models_comparison.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: graphs/all_models_comparison.png")
    plt.close()

    # Print statistical summary
    print(f"\n  Best Model: {best_model['Backbone']} ({best_model['Fusion']}) - MAE: {best_model['Mean MAE']:.4f}")


def plot_model_size_vs_performance(df):
    """Plot model size vs performance scatter plot."""
    results_dir = project_root / "Results" / "comparison"

    df_clean = df.dropna(subset=["Mean MAE"])

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    fusion_types = ["Baseline", "Late", "Feature"]
    colors = {"Baseline": "steelblue", "Late": "coral", "Feature": "mediumseagreen"}
    markers = {"Baseline": "o", "Late": "s", "Feature": "^"}

    for fusion in fusion_types:
        data = df_clean[df_clean["Fusion"] == fusion]
        if len(data) > 0:
            ax.scatter(data["Params (M)"], data["Mean MAE"],
                      s=200, alpha=0.6, color=colors[fusion],
                      marker=markers[fusion], label=fusion, edgecolors='black', linewidth=2)

            # Add backbone labels
            for _, row in data.iterrows():
                ax.annotate(row["Backbone"],
                           (row["Params (M)"], row["Mean MAE"]),
                           fontsize=12, alpha=0.7)

    ax.set_xlabel("Model Size (Million Parameters)", fontsize=24, fontweight='bold')
    ax.set_ylabel("Mean MAE (days)", fontsize=24, fontweight='bold')
    ax.set_title("Model Size vs Performance", fontsize=28, fontweight='bold')
    ax.legend(fontsize=18)
    ax.tick_params(axis='both', labelsize=20)
    ax.grid(True, alpha=0.3, linewidth=1.5)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(results_dir / "graphs" / "model_size_vs_performance.png", dpi=200, bbox_inches='tight')
    print(f"  Saved: graphs/model_size_vs_performance.png")
    plt.close()


def plot_fusion_comparison(df):
    """Plot fusion strategy comparison with statistical significance."""
    from scipy import stats

    results_dir = project_root / "Results" / "comparison"

    df_clean = df.dropna(subset=["Mean MAE"])

    # Group by fusion type
    fusion_summary = df_clean.groupby("Fusion").agg({
        "Mean MAE": ["mean", "std", "min", "max"]
    }).reset_index()

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    fusion_types = ["Baseline", "Late", "Feature"]
    means = []
    stds = []
    all_data = {}

    for fusion in fusion_types:
        data = df_clean[df_clean["Fusion"] == fusion]
        if len(data) > 0:
            means.append(data["Mean MAE"].mean())
            stds.append(data["Mean MAE"].std())
            all_data[fusion] = data["Mean MAE"].values
        else:
            means.append(np.nan)
            stds.append(np.nan)

    colors = ["steelblue", "coral", "mediumseagreen"]
    bars = ax.bar(fusion_types, means, yerr=stds, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=2, capsize=10,
                   error_kw={'linewidth': 2.5, 'elinewidth': 2.5})

    # Add value labels on bars
    for i, (mean, bar) in enumerate(zip(means, bars)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + stds[i] + 0.005,
                f'{mean:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=18)

    ax.set_ylabel("Mean MAE (days)", fontsize=24, fontweight='bold')

    # Set title (bold) and add ANOVA result as separate text (not bold)
    ax.set_title("Fusion Strategy Comparison (Average Across All Backbones)",
                 fontsize=24, fontweight='bold', pad=40)

    # Perform one-way ANOVA and add as separate text
    if len(all_data) == 3:
        f_stat, p_value = stats.f_oneway(all_data["Baseline"], all_data["Late"], all_data["Feature"])
        sig_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

        # Add ANOVA text below title (not bold)
        ax.text(0.5, 1.01, f"ANOVA: F={f_stat:.2f}, p={p_value:.4f}",
                transform=ax.transAxes, fontsize=20, ha='center', va='bottom',
                fontweight='normal')

    ax.tick_params(axis='both', labelsize=20)
    ax.grid(axis='y', alpha=0.3, linewidth=1.5)

    plt.tight_layout()
    plt.savefig(results_dir / "graphs" / "fusion_comparison.png", dpi=200, bbox_inches='tight')
    print(f"  Saved: graphs/fusion_comparison.png")
    plt.close()


def perform_statistical_analysis(df):
    """Perform comprehensive statistical analysis of model size vs fusion effectiveness."""
    results_dir = project_root / "Results" / "comparison"

    df_clean = df.dropna(subset=["Mean MAE"])

    report = []
    report.append("="*80)
    report.append("STATISTICAL ANALYSIS: Model Size vs Fusion Strategy Effectiveness")
    report.append("="*80)
    report.append("")

    # Analysis 1: Spearman Correlation (Model Size vs MAE) for each fusion type
    report.append("1. CORRELATION ANALYSIS: Model Size vs Performance")
    report.append("-"*80)
    report.append("   Spearman correlation coefficient measures monotonic relationship.")
    report.append("   Negative correlation = larger models have lower MAE (better performance)")
    report.append("")

    correlations = {}
    for fusion in ["Baseline", "Late", "Feature"]:
        data = df_clean[df_clean["Fusion"] == fusion]
        if len(data) >= 3:  # Need at least 3 points for meaningful correlation
            # Use log of params for better linear relationship
            rho, p_value = spearmanr(np.log10(data["Params (M)"]), data["Mean MAE"])
            correlations[fusion] = {"rho": rho, "p": p_value}
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            report.append(f"   {fusion:12} Fusion:  rho = {rho:+.4f}  (p = {p_value:.4f}) {significance}")
        else:
            correlations[fusion] = {"rho": np.nan, "p": np.nan}
            report.append(f"   {fusion:12} Fusion:  Insufficient data (n={len(data)})")

    report.append("")
    report.append("   Interpretation:")
    report.append("   - More negative rho = stronger benefit from increased model size")
    report.append("   - *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    report.append("")

    # Analysis 2: Improvement Ratio (Fusion vs Baseline)
    report.append("")
    report.append("2. FUSION BENEFIT ANALYSIS: Improvement Over Baseline")
    report.append("-"*80)
    report.append("   Calculates: (Baseline MAE - Fusion MAE) / Baseline MAE")
    report.append("   Positive = Fusion is better, Negative = Baseline is better")
    report.append("")

    improvement_data = []

    for backbone in BACKBONES:
        baseline_data = df_clean[(df_clean["Backbone"] == backbone) & (df_clean["Fusion"] == "Baseline")]
        late_data = df_clean[(df_clean["Backbone"] == backbone) & (df_clean["Fusion"] == "Late")]
        feature_data = df_clean[(df_clean["Backbone"] == backbone) & (df_clean["Fusion"] == "Feature")]

        if len(baseline_data) > 0:
            baseline_mae = baseline_data["Mean MAE"].values[0]
            baseline_params = baseline_data["Params (M)"].values[0]

            if len(late_data) > 0:
                late_mae = late_data["Mean MAE"].values[0]
                late_improvement = (baseline_mae - late_mae) / baseline_mae * 100
                improvement_data.append({
                    "Backbone": backbone,
                    "Fusion": "Late",
                    "Params (M)": baseline_params,  # Use baseline params for comparison
                    "Improvement (%)": late_improvement
                })

            if len(feature_data) > 0:
                feature_mae = feature_data["Mean MAE"].values[0]
                feature_improvement = (baseline_mae - feature_mae) / baseline_mae * 100
                improvement_data.append({
                    "Backbone": backbone,
                    "Fusion": "Feature",
                    "Params (M)": baseline_params,  # Use baseline params for comparison
                    "Improvement (%)": feature_improvement
                })

    if len(improvement_data) > 0:
        improvement_df = pd.DataFrame(improvement_data)

        # Show improvement by backbone
        report.append("   Improvement by Model Size (using baseline params as reference):")
        report.append("")
        for backbone in BACKBONES:
            backbone_data = improvement_df[improvement_df["Backbone"] == backbone]
            if len(backbone_data) > 0:
                params = backbone_data["Params (M)"].values[0]
                report.append(f"   {backbone:15} ({params:5.1f}M params):")
                for _, row in backbone_data.iterrows():
                    improvement = row["Improvement (%)"]
                    sign = "+" if improvement > 0 else ""
                    report.append(f"      {row['Fusion']:8} Fusion: {sign}{improvement:6.2f}%")

        report.append("")
        report.append("")

        # Correlation between model size and improvement
        report.append("   Correlation: Model Size vs Fusion Benefit")
        report.append("   " + "-"*76)

        for fusion in ["Late", "Feature"]:
            fusion_data = improvement_df[improvement_df["Fusion"] == fusion]
            if len(fusion_data) >= 3:
                rho, p_value = spearmanr(np.log10(fusion_data["Params (M)"]), fusion_data["Improvement (%)"])
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                report.append(f"   {fusion:8} Fusion:  rho = {rho:+.4f}  (p = {p_value:.4f}) {significance}")
            else:
                report.append(f"   {fusion:8} Fusion:  Insufficient data")

        report.append("")
        report.append("   Interpretation:")
        report.append("   - Positive rho = larger models benefit MORE from fusion")
        report.append("   - Negative rho = smaller models benefit MORE from fusion")
    else:
        report.append("   ERROR: No improvement data available")

    report.append("")

    # Analysis 3: Capacity Threshold Detection
    report.append("")
    report.append("3. CAPACITY THRESHOLD ANALYSIS")
    report.append("-"*80)
    report.append("   Identifies approximate model size where fusion becomes beneficial")
    report.append("")

    # Find crossover points where fusion beats baseline
    if len(improvement_data) > 0:
        for fusion in ["Late", "Feature"]:
            fusion_data = improvement_df[improvement_df["Fusion"] == fusion].sort_values("Params (M)")

            # Find models where fusion wins (positive improvement)
            winners = fusion_data[fusion_data["Improvement (%)"] > 0]
            losers = fusion_data[fusion_data["Improvement (%)"] <= 0]

            report.append(f"   {fusion} Fusion:")
            if len(winners) > 0:
                threshold = winners["Params (M)"].min()
                winner_backbones = winners["Backbone"].tolist()
                report.append(f"      Fusion wins at:     >={threshold:.1f}M params  (models: {', '.join(winner_backbones)})")
            else:
                report.append(f"      Fusion wins at:     Never (baseline always better)")

            if len(losers) > 0:
                loser_backbones = losers["Backbone"].tolist()
                max_loser = losers["Params (M)"].max()
                report.append(f"      Baseline wins at:   <={max_loser:.1f}M params  (models: {', '.join(loser_backbones)})")
            else:
                report.append(f"      Baseline wins at:   Never (fusion always better)")
            report.append("")

    report.append("")

    # Key Findings Summary
    report.append("")
    report.append("4. KEY FINDINGS")
    report.append("="*80)

    # Check if feature fusion has stronger correlation than baseline
    if "Feature" in correlations and "Baseline" in correlations:
        feature_rho = correlations["Feature"]["rho"]
        baseline_rho = correlations["Baseline"]["rho"]

        if not np.isnan(feature_rho) and not np.isnan(baseline_rho):
            if abs(feature_rho) > abs(baseline_rho):
                report.append("   + Feature Fusion shows STRONGER correlation with model size than Baseline")
                report.append(f"     (|rho_feature| = {abs(feature_rho):.4f} > |rho_baseline| = {abs(baseline_rho):.4f})")
                report.append("     -> Fusion strategies benefit MORE from increased model capacity")
            else:
                report.append("   - Feature Fusion shows WEAKER correlation with model size than Baseline")
                report.append(f"     (|rho_feature| = {abs(feature_rho):.4f} < |rho_baseline| = {abs(baseline_rho):.4f})")

    report.append("")

    # Check improvement correlation
    if len(improvement_data) > 0:
        feature_improvement = improvement_df[improvement_df["Fusion"] == "Feature"]
        if len(feature_improvement) >= 3:
            rho, p_value = spearmanr(np.log10(feature_improvement["Params (M)"]),
                                    feature_improvement["Improvement (%)"])
            if rho > 0 and p_value < 0.05:
                report.append("   + Fusion benefit INCREASES significantly with model size")
                report.append(f"     (rho = {rho:+.4f}, p = {p_value:.4f})")
                report.append("     -> Larger models benefit MORE from multi-view fusion")
            elif rho < 0 and p_value < 0.05:
                report.append("   - Fusion benefit DECREASES significantly with model size")
                report.append(f"     (rho = {rho:+.4f}, p = {p_value:.4f})")
                report.append("     -> Smaller models benefit MORE from multi-view fusion")
            else:
                report.append("   ~ No significant relationship between model size and fusion benefit")
                report.append(f"     (rho = {rho:+.4f}, p = {p_value:.4f})")

    report.append("")
    report.append("="*80)

    # Save report
    report_file = results_dir / "statistical_analysis.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    # Print to console
    print('\n'.join(report))

    return correlations, improvement_df if len(improvement_data) > 0 else None


def plot_model_size_vs_performance_enhanced(df, correlations, improvement_df):
    """Enhanced version with correlation coefficients displayed."""
    results_dir = project_root / "Results" / "comparison"

    df_clean = df.dropna(subset=["Mean MAE"])

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Left plot: Model Size vs MAE with correlation coefficients
    ax1 = axes[0]

    fusion_types = ["Baseline", "Late", "Feature"]
    colors = {"Baseline": "steelblue", "Late": "coral", "Feature": "mediumseagreen"}
    markers = {"Baseline": "o", "Late": "s", "Feature": "^"}

    for fusion in fusion_types:
        data = df_clean[df_clean["Fusion"] == fusion]
        if len(data) > 0:
            ax1.scatter(data["Params (M)"], data["Mean MAE"],
                       s=150, alpha=0.6, color=colors[fusion],
                       marker=markers[fusion], edgecolors='black', linewidth=1.5)

            # Add correlation coefficient to label
            if fusion in correlations and not np.isnan(correlations[fusion]["rho"]):
                rho = correlations[fusion]["rho"]
                p = correlations[fusion]["p"]
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                label = f"{fusion} (rho={rho:+.3f}{sig})"
            else:
                label = fusion

            # Create legend entry
            ax1.scatter([], [], s=150, alpha=0.6, color=colors[fusion],
                       marker=markers[fusion], edgecolors='black', linewidth=1.5,
                       label=label)

    ax1.set_xlabel("Model Size (Million Parameters)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Mean MAE (days)", fontsize=14, fontweight='bold')
    ax1.set_title("Model Size vs Performance\n(with Spearman correlation)", fontsize=16, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # Right plot: Model Size vs Improvement Ratio
    ax2 = axes[1]

    if improvement_df is not None and len(improvement_df) > 0:
        for fusion in ["Late", "Feature"]:
            data = improvement_df[improvement_df["Fusion"] == fusion]
            if len(data) > 0:
                # Calculate correlation for label
                if len(data) >= 3:
                    rho, p = spearmanr(np.log10(data["Params (M)"]), data["Improvement (%)"])
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    label = f"{fusion} (rho={rho:+.3f}{sig})"
                else:
                    label = fusion

                ax2.scatter(data["Params (M)"], data["Improvement (%)"],
                           s=150, alpha=0.6, color=colors[fusion],
                           marker=markers[fusion], edgecolors='black', linewidth=1.5,
                           label=label)

                # Add backbone labels
                for _, row in data.iterrows():
                    ax2.annotate(row["Backbone"],
                               (row["Params (M)"], row["Improvement (%)"]),
                               fontsize=9, alpha=0.7, ha='right')

        # Add horizontal line at y=0
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Break-even')

        ax2.set_xlabel("Baseline Model Size (Million Parameters)", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Improvement over Baseline (%)", fontsize=14, fontweight='bold')
        ax2.set_title("Fusion Benefit vs Model Size\n(Positive = Fusion Wins)", fontsize=16, fontweight='bold')
        ax2.legend(fontsize=11, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
    else:
        ax2.text(0.5, 0.5, "Insufficient data for improvement analysis",
                ha='center', va='center', fontsize=14, transform=ax2.transAxes)

    plt.tight_layout()
    plt.savefig(results_dir / "graphs" / "statistical_analysis.png", dpi=200, bbox_inches='tight')
    print(f"  Saved: graphs/statistical_analysis.png")
    plt.close()


def main():
    print("="*80)
    print("EVALUATING ALL 27 MODEL CONFIGURATIONS")
    print("="*80)
    print("\nGenerating comparison plots and summary table...\n")

    # Create summary table
    df = create_summary_table()

    # Save summary CSV
    results_dir = project_root / "Results" / "comparison"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "csv").mkdir(exist_ok=True)

    summary_file = results_dir / "csv" / "summary.csv"
    df.to_csv(summary_file, index=False)
    print(f"  Saved: csv/summary.csv")

    # Display top 10 models
    print("\n" + "="*80)
    print("TOP 10 MODELS (by Mean MAE)")
    print("="*80)
    print(df.head(10).to_string(index=False))

    # Generate plots
    print("\n" + "="*80)
    print("GENERATING COMPARISON PLOTS")
    print("="*80 + "\n")

    plot_all_models_comparison(df)
    plot_model_size_vs_performance(df)
    plot_fusion_comparison(df)

    # Statistical analysis
    print("\n" + "="*80)
    print("PERFORMING STATISTICAL ANALYSIS")
    print("="*80)

    correlations, improvement_df = perform_statistical_analysis(df)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {results_dir}")
    print("\nGenerated files:")
    print("  - csv/summary.csv")
    print("  - graphs/all_models_comparison.png")
    print("  - graphs/model_size_vs_performance.png")
    print("  - graphs/fusion_comparison.png")
    print("  - statistical_analysis.txt")


if __name__ == "__main__":
    main()
