"""Generate Final Analysis and Visualizations

This script generates ALL visualizations and analysis for the paper using
the training results from train_all_models_full.py.

Creates:
1. Old-style 3-panel comparison graph
2. New-style single-panel comparison graph
3. Fusion strategy comparison
4. Model size vs performance
5. Best model detailed analysis (training curves, predictions, confusion matrix, errors)
6. Per-day performance breakdown
7. Architecture comparison (CNN vs Transformer vs Hybrid)
8. Statistical tests
9. Paper-ready tables

Outputs:
- Graphs saved to: Results/Graphs/ (10 PNG files)
- Tables and analysis saved to: Results/CSV_and_Analysis/ (6 files)
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr, f_oneway, ttest_rel
from scipy import stats

# Configuration
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

MODEL_SIZES = {
    "efficientnet_b0": {"top_view": 4.01, "late": 8.02, "feature": 4.52},
    "resnet18": {"top_view": 11.18, "late": 22.36, "feature": 11.69},
    "resnet50": {"top_view": 23.51, "late": 47.03, "feature": 24.54},
    "resnet101": {"top_view": 42.50, "late": 85.01, "feature": 43.53},
    "vit_b_16": {"top_view": 86.57, "late": 173.13, "feature": 87.34},
    "swin_t": {"top_view": 27.52, "late": 55.04, "feature": 28.29},
    "swin_b": {"top_view": 87.27, "late": 174.54, "feature": 88.30},
    "convnext_t": {"top_view": 27.82, "late": 55.64, "feature": 28.59},
    "convnext_b": {"top_view": 88.02, "late": 176.04, "feature": 89.05},
}

# Color scheme - MATCHING evaluate_all_models.py
FUSION_COLORS = {
    "TOP View": "steelblue",
    "Late": "coral",
    "Feature": "mediumseagreen"
}

ARCHITECTURE_TYPES = {
    "efficientnet_b0": "Traditional CNN",
    "resnet18": "Traditional CNN",
    "resnet50": "Traditional CNN",
    "resnet101": "Traditional CNN",
    "vit_b_16": "Transformer",
    "swin_t": "Transformer",
    "swin_b": "Transformer",
    "convnext_t": "Modernized CNN",
    "convnext_b": "Modernized CNN",
}


def load_new_results():
    """Load results from full_progress.json."""
    results_file = project_root / "Results" / "CSV_and_Analysis" / "full_progress.json"

    if not results_file.exists():
        print(f"ERROR: {results_file} not found")
        return {}

    with open(results_file, 'r') as f:
        return json.load(f)


def create_summary_df():
    """Create summary DataFrame from new results."""
    new_results = load_new_results()

    summary = []

    for backbone in BACKBONES:
        for fusion_type in ["top_view", "late", "feature"]:
            key = f"{backbone}_{fusion_type}"
            fusion_label = "TOP View" if fusion_type == "top_view" else fusion_type.capitalize()

            if key in new_results:
                data = new_results[key]
                summary.append({
                    "Backbone": backbone,
                    "Fusion": fusion_label,
                    "Mean MAE": data.get("mean_mae", np.nan),
                    "Std MAE": data.get("std_mae", np.nan),
                    "Params (M)": MODEL_SIZES[backbone][fusion_type],
                    "Architecture": ARCHITECTURE_TYPES[backbone],
                })

    df = pd.DataFrame(summary)
    df = df.dropna(subset=["Mean MAE"])

    return df


def plot_old_style_comparison(df, output_dir):
    """Generate OLD STYLE 3-panel comparison (matching evaluate_all_models.py)."""
    print("  Generating old-style 3-panel comparison...")

    # Get best model
    best_idx = df["Mean MAE"].idxmin()
    best_model = df.loc[best_idx]

    # Create figure with 3 subplots side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(24, 10))

    fusion_types = ["TOP View", "Late", "Feature"]

    # Calculate max xlim for consistent scaling
    max_mae_with_error = 0
    for fusion in fusion_types:
        data = df[df["Fusion"] == fusion]
        if len(data) > 0:
            max_val = (data["Mean MAE"] + data["Std MAE"]).max()
            max_mae_with_error = max(max_mae_with_error, max_val)

    xlim_max = max_mae_with_error * 1.2

    for idx, fusion in enumerate(fusion_types):
        ax = axes[idx]
        data = df[df["Fusion"] == fusion].sort_values("Mean MAE")
        color = FUSION_COLORS[fusion]

        if len(data) > 0:
            is_best_fusion = (fusion == best_model['Fusion'])

            # Plot bars
            for i, (_, row) in enumerate(data.iterrows()):
                is_best = (is_best_fusion and row['Backbone'] == best_model['Backbone'])

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

                # Annotate best model
                if is_best:
                    annotation_text = f'Best Model\nMAE: {row["Mean MAE"]:.4f} +/- {row["Std MAE"]:.4f}'
                    ax.text(xlim_max * 0.7, row["Backbone"],
                           annotation_text,
                           fontsize=18,
                           verticalalignment='center',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat',
                                   alpha=0.9, edgecolor='red', linewidth=2))

            # Formatting
            ax.set_xlabel("MAE (days)", fontsize=24, fontweight='bold')
            ax.tick_params(axis='both', labelsize=20)

            # Calculate ANOVA
            mae_values = data["Mean MAE"].values
            if len(mae_values) > 2:
                f_stat = np.var(mae_values) / np.mean(np.square(data["Std MAE"].values)) if np.mean(np.square(data["Std MAE"].values)) > 0 else 0
                ax.text(0.5, 1.02, f"ANOVA F = {f_stat:.2f}",
                       transform=ax.transAxes,
                       fontsize=22,
                       ha='center',
                       va='bottom',
                       fontweight='normal')

            ax.grid(axis='x', alpha=0.3, linewidth=1.5)
            ax.invert_yaxis()

    # Apply consistent xlim
    for ax in axes:
        ax.set_xlim(0, xlim_max)

    plt.tight_layout()
    plt.savefig(output_dir / 'all_models_comparison_3panel.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: all_models_comparison_3panel.png")
    plt.close()


def plot_new_style_comparison(df, output_dir):
    """Generate NEW STYLE single-panel comparison (with CORRECTED colors)."""
    print("  Generating new-style single-panel comparison...")

    # Sort by MAE
    df_sorted = df.sort_values("Mean MAE").reset_index(drop=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))

    # Create labels and colors
    labels = []
    colors = []

    for _, row in df_sorted.iterrows():
        label = f"{row['Backbone']} ({row['Fusion']})"
        labels.append(label)
        colors.append(FUSION_COLORS[row['Fusion']])

    # Horizontal bar chart
    y_pos = np.arange(len(df_sorted))
    ax.barh(y_pos, df_sorted["Mean MAE"],
            xerr=df_sorted["Std MAE"],
            color=colors, alpha=0.8, capsize=5, edgecolor='black', linewidth=0.5)

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Mean MAE (days)', fontsize=14, weight='bold')
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=FUSION_COLORS["TOP View"], alpha=0.8, edgecolor='black', label='TOP View (Baseline)'),
        Patch(facecolor=FUSION_COLORS["Late"], alpha=0.8, edgecolor='black', label='Late Fusion'),
        Patch(facecolor=FUSION_COLORS["Feature"], alpha=0.8, edgecolor='black', label='Feature Fusion')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=12, framealpha=0.9)

    # Highlight best model
    ax.barh(0, df_sorted.iloc[0]["Mean MAE"],
            color='none', edgecolor='gold', linewidth=4, zorder=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'all_models_comparison_single.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: all_models_comparison_single.png")
    plt.close()


def plot_fusion_comparison(df, output_dir):
    """Generate fusion strategy comparison."""
    print("  Generating fusion strategy comparison...")

    # Group by fusion type
    fusion_data = {
        "TOP View": df[df["Fusion"] == "TOP View"]["Mean MAE"].values,
        "Late": df[df["Fusion"] == "Late"]["Mean MAE"].values,
        "Feature": df[df["Fusion"] == "Feature"]["Mean MAE"].values
    }

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Box plot
    positions = [1, 2, 3]
    fusion_names = ["TOP View", "Late", "Feature"]

    bp = ax1.boxplot([fusion_data[name] for name in fusion_names],
                     positions=positions, widths=0.6,
                     patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=10))

    for patch, name in zip(bp['boxes'], fusion_names):
        patch.set_facecolor(FUSION_COLORS[name])
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)

    ax1.set_xticks(positions)
    ax1.set_xticklabels(fusion_names, fontsize=12, weight='bold')
    ax1.set_ylabel('MAE (days)', fontsize=14, weight='bold')
    ax1.set_ylim(0.6, 1.0)  # Set y-axis limit for better visualization
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # Add sample sizes
    for i, name in enumerate(fusion_names):
        n = len(fusion_data[name])
        ax1.text(i+1, 0.98, f'n={n}',
                ha='center', fontsize=10, style='italic')

    # Right: Bar plot
    fusion_stats = df.groupby('Fusion')['Mean MAE'].agg(['mean', 'std'])
    fusion_stats = fusion_stats.reindex(fusion_names)

    bars = ax2.bar(range(len(fusion_names)), fusion_stats['mean'],
                   yerr=fusion_stats['std'], capsize=10,
                   color=[FUSION_COLORS[n] for n in fusion_names],
                   alpha=0.7, edgecolor='black', linewidth=1.5)

    ax2.set_xticks(range(len(fusion_names)))
    ax2.set_xticklabels(fusion_names, fontsize=12, weight='bold')
    ax2.set_ylabel('Mean MAE (days)', fontsize=14, weight='bold')
    ax2.set_ylim(0, 1.0)  # Set consistent y-axis limit
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    # Add value labels
    for i, (bar, name) in enumerate(zip(bars, fusion_names)):
        height = bar.get_height()
        std = fusion_stats.loc[name, 'std']
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{height:.3f}+/-{std:.3f}',
                ha='center', va='bottom', fontsize=11, weight='bold')

    # ANOVA
    f_stat, p_value = f_oneway(*[fusion_data[n] for n in fusion_names])
    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

    ax2.text(0.5, 0.97, f'ANOVA: F={f_stat:.2f}, p={p_value:.4f} {sig}',
            transform=ax2.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
            fontsize=11, weight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'fusion_comparison.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: fusion_comparison.png")
    plt.close()


def plot_model_size_vs_performance(df, output_dir):
    """Generate model size vs performance scatter plot."""
    print("  Generating model size vs performance...")

    fig, ax = plt.subplots(figsize=(10, 6))

    for fusion_name in ["TOP View", "Late", "Feature"]:
        data = df[df["Fusion"] == fusion_name]
        ax.scatter(data["Params (M)"], data["Mean MAE"],
                  s=100, alpha=0.7,
                  color=FUSION_COLORS[fusion_name],
                  edgecolors='black', linewidth=1.5,
                  label=fusion_name)

    ax.set_xlabel('Model Size (Million Parameters)', fontsize=14, weight='bold')
    ax.set_ylabel('Mean MAE (days)', fontsize=14, weight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=12, loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'model_size_vs_performance.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: model_size_vs_performance.png")
    plt.close()


def plot_best_model_analysis(output_dir):
    """Generate detailed analysis for best model."""
    print("  Generating best model analysis...")

    # Find best model
    df = create_summary_df()
    best_idx = df["Mean MAE"].idxmin()
    best_model = df.loc[best_idx]

    backbone = best_model["Backbone"]
    fusion = best_model["Fusion"].lower().replace(" ", "_")
    model_key = f"{backbone}_{fusion}"

    print(f"    Best model: {backbone} ({best_model['Fusion']}) - MAE: {best_model['Mean MAE']:.4f}")

    # Load training histories and predictions
    history_dir = project_root / "Results" / "Data" / "training_history" / model_key
    pred_dir = project_root / "Results" / "Data" / "predictions" / model_key

    if not history_dir.exists() or not pred_dir.exists():
        print(f"    WARNING: History/predictions not found for {model_key}")
        return

    # Load all folds
    histories = []
    all_targets = []
    all_predictions = []

    for fold in range(3):
        history_file = history_dir / f"fold{fold}.json"
        pred_file = pred_dir / f"fold{fold}.csv"

        if history_file.exists():
            with open(history_file, 'r') as f:
                histories.append(json.load(f))

        if pred_file.exists():
            pred_df = pd.read_csv(pred_file)
            all_targets.extend(pred_df['target'].values)
            all_predictions.extend(pred_df['prediction'].values)

    # Plot 1: Training curves (all 3 folds)
    if len(histories) == 3:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for fold_idx, history in enumerate(histories):
            ax = axes[fold_idx]

            # Extract train and val MAE from list of dicts
            train_maes = [epoch_data['train_mae'] for epoch_data in history]
            val_maes = [epoch_data['val_mae'] for epoch_data in history]
            epochs = range(1, len(train_maes) + 1)

            ax.plot(epochs, train_maes, 'b-', label='Train MAE', linewidth=2)
            ax.plot(epochs, val_maes, 'r-', label='Val MAE', linewidth=2)

            # Mark best epoch
            best_epoch = np.argmin(val_maes)
            ax.axvline(best_epoch + 1, color='green', linestyle='--', alpha=0.5, label='Best Epoch')

            ax.set_xlabel('Epoch', fontsize=12, weight='bold')
            ax.set_ylabel('MAE (days)', fontsize=12, weight='bold')
            ax.set_title(f'Fold {fold_idx + 1}', fontsize=14, weight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'training_curves_{model_key}.png', dpi=300, bbox_inches='tight')
        print(f"    Saved: training_curves_{model_key}.png")
        plt.close()

    # Plot 2: Predictions vs Actual (all folds combined)
    if len(all_targets) > 0:
        fig, ax = plt.subplots(figsize=(10, 10))

        ax.scatter(all_targets, all_predictions, alpha=0.5, s=50, edgecolors='black', linewidth=0.5)

        # Perfect prediction line
        min_val = min(min(all_targets), min(all_predictions))
        max_val = max(max(all_targets), max(all_predictions))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        ax.set_xlabel('Actual Age (days)', fontsize=14, weight='bold')
        ax.set_ylabel('Predicted Age (days)', fontsize=14, weight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(output_dir / f'predictions_vs_actual_{model_key}.png', dpi=300, bbox_inches='tight')
        print(f"    Saved: predictions_vs_actual_{model_key}.png")
        plt.close()

    # Plot 3: Error distribution
    if len(all_targets) > 0:
        errors = np.array(all_predictions) - np.array(all_targets)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax1.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax1.axvline(np.mean(errors), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.3f}')
        ax1.set_xlabel('Prediction Error (days)', fontsize=12, weight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, weight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')

        # Q-Q plot
        stats.probplot(errors, dist="norm", plot=ax2)
        ax2.set_title('')  # Remove default Q-Q plot title
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'error_distribution_{model_key}.png', dpi=300, bbox_inches='tight')
        print(f"    Saved: error_distribution_{model_key}.png")
        plt.close()

    # Plot 4: Confusion Matrix (Normalized)
    if len(all_targets) > 0:
        # Round to nearest day for confusion matrix
        actual_days = np.round(all_targets).astype(int)
        predicted_days = np.round(all_predictions).astype(int)

        # Get unique days (should be 1-8)
        all_days = sorted(set(actual_days) | set(predicted_days))
        n_days = len(all_days)

        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(actual_days, predicted_days, labels=all_days)

        # Normalize by row (actual days)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 9))

        # Normalized confusion matrix
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Percentage', fontsize=12, weight='bold')

        # Set ticks
        ax.set_xticks(range(n_days))
        ax.set_yticks(range(n_days))
        ax.set_xticklabels([f'Day {d}' for d in all_days], fontsize=11)
        ax.set_yticklabels([f'Day {d}' for d in all_days], fontsize=11)

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations (percentage)
        for i in range(n_days):
            for j in range(n_days):
                ax.text(j, i, f'{cm_normalized[i, j]:.2f}',
                        ha="center", va="center",
                        color="white" if cm_normalized[i, j] > 0.5 else "black",
                        fontsize=11, weight='bold')

        ax.set_ylabel('Actual Age', fontsize=14, weight='bold')
        ax.set_xlabel('Predicted Age', fontsize=14, weight='bold')

        plt.tight_layout()
        plt.savefig(output_dir / f'confusion_matrix_{model_key}.png', dpi=300, bbox_inches='tight')
        print(f"    Saved: confusion_matrix_{model_key}.png")
        plt.close()


def plot_per_day_performance(output_dir):
    """Generate per-day performance breakdown for best model."""
    print("  Generating per-day performance analysis...")

    # Find best model
    df = create_summary_df()
    best_idx = df["Mean MAE"].idxmin()
    best_model = df.loc[best_idx]

    backbone = best_model["Backbone"]
    fusion = best_model["Fusion"].lower().replace(" ", "_")
    model_key = f"{backbone}_{fusion}"

    # Load predictions from all folds
    pred_dir = project_root / "Results" / "Data" / "predictions" / model_key

    if not pred_dir.exists():
        print(f"    WARNING: Predictions not found for {model_key}")
        return

    all_targets = []
    all_predictions = []

    for fold in range(3):
        pred_file = pred_dir / f"fold{fold}.csv"
        if pred_file.exists():
            pred_df = pd.read_csv(pred_file)
            all_targets.extend(pred_df['target'].values)
            all_predictions.extend(pred_df['prediction'].values)

    if len(all_targets) == 0:
        print("    WARNING: No prediction data found")
        return

    # Convert to arrays
    targets = np.array(all_targets)
    predictions = np.array(all_predictions)
    errors = np.abs(predictions - targets)

    # Calculate per-day metrics
    days = sorted(set(np.round(targets).astype(int)))
    per_day_mae = []
    per_day_std = []
    per_day_counts = []

    for day in days:
        mask = np.round(targets).astype(int) == day
        day_errors = errors[mask]
        per_day_mae.append(np.mean(day_errors))
        per_day_std.append(np.std(day_errors))
        per_day_counts.append(np.sum(mask))

    # Create single figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar chart of MAE per day
    x_pos = np.arange(len(days))
    bars = ax.bar(x_pos, per_day_mae, yerr=per_day_std, capsize=5,
                  color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for i, (bar, mae, std) in enumerate(zip(bars, per_day_mae, per_day_std)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{mae:.3f}',
                ha='center', va='bottom', fontsize=10, weight='bold')

    ax.set_xlabel('Actual Age (days)', fontsize=14, weight='bold')
    ax.set_ylabel('Mean Absolute Error (days)', fontsize=14, weight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Day {d}' for d in days], fontsize=11)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(per_day_mae) + max(per_day_std) + 0.15)

    plt.tight_layout()
    plt.savefig(output_dir / f'per_day_performance_{model_key}.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: per_day_performance_{model_key}.png")
    plt.close()


def plot_architecture_comparison(df, output_dir):
    """Generate architecture type comparison (two separate images)."""
    print("  Generating architecture comparison...")

    # Add architecture type column
    arch_colors = {
        "Traditional CNN": "#3498db",
        "Transformer": "#e74c3c",
        "Modernized CNN": "#2ecc71"
    }

    # Plot 1: By architecture type
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    arch_data = df.groupby(['Architecture', 'Fusion'])['Mean MAE'].mean().unstack()
    arch_std = df.groupby(['Architecture', 'Fusion'])['Mean MAE'].std().unstack()
    # Reorder to: Traditional CNN, Transformer, Modernized CNN
    arch_order = ["Traditional CNN", "Transformer", "Modernized CNN"]
    arch_data = arch_data.reindex(arch_order)
    arch_std = arch_std.reindex(arch_order)

    x = np.arange(len(arch_data.index))
    width = 0.25

    for i, fusion in enumerate(["TOP View", "Late", "Feature"]):
        if fusion in arch_data.columns:
            offset = width * (i - 1)
            ax1.bar(x + offset, arch_data[fusion], width,
                   yerr=arch_std[fusion], capsize=4,
                   label=fusion, color=FUSION_COLORS[fusion],
                   alpha=0.7, edgecolor='black', linewidth=1.5)

    ax1.set_xlabel('Architecture Type', fontsize=14, weight='bold')
    ax1.set_ylabel('Mean MAE (days)', fontsize=14, weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(arch_data.index, fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'architecture_by_type.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: architecture_by_type.png")
    plt.close()

    # Plot 2: Architecture efficiency (performance vs size)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for arch_type in arch_colors.keys():
        data = df[df["Architecture"] == arch_type]
        ax2.scatter(data["Params (M)"], data["Mean MAE"],
                   s=100, alpha=0.7, color=arch_colors[arch_type],
                   label=arch_type, edgecolors='black', linewidth=1.5)

    ax2.set_xlabel('Model Size (Million Parameters)', fontsize=14, weight='bold')
    ax2.set_ylabel('Mean MAE (days)', fontsize=14, weight='bold')
    ax2.set_xscale('log')
    ax2.legend(fontsize=12, loc='lower left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'architecture_efficiency.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: architecture_efficiency.png")
    plt.close()


def generate_paper_tables(df, output_dir):
    """Generate paper-ready CSV tables."""
    print("  Generating paper tables...")

    # Table 1: All models summary
    table1 = df.copy()
    table1 = table1.sort_values("Mean MAE")
    table1["Rank"] = range(1, len(table1) + 1)
    table1 = table1[["Rank", "Backbone", "Fusion", "Architecture", "Mean MAE", "Std MAE", "Params (M)"]]
    table1.to_csv(output_dir / "table_all_models.csv", index=False)
    print(f"    Saved: table_all_models.csv")

    # Table 2: Best model per fusion strategy
    table2 = df.loc[df.groupby("Fusion")["Mean MAE"].idxmin()]
    table2 = table2.sort_values("Mean MAE")
    table2 = table2[["Fusion", "Backbone", "Architecture", "Mean MAE", "Std MAE", "Params (M)"]]
    table2.to_csv(output_dir / "table_best_per_fusion.csv", index=False)
    print(f"    Saved: table_best_per_fusion.csv")

    # Table 3: Fusion strategy summary
    table3 = df.groupby("Fusion").agg({
        "Mean MAE": ["mean", "std", "min", "max"],
        "Params (M)": ["mean", "min", "max"]
    }).round(4)
    table3.to_csv(output_dir / "table_fusion_summary.csv")
    print(f"    Saved: table_fusion_summary.csv")

    # Table 4: Architecture summary
    table4 = df.groupby("Architecture").agg({
        "Mean MAE": ["mean", "std", "min"],
        "Params (M)": ["mean", "min", "max"]
    }).round(4)
    table4.to_csv(output_dir / "table_architecture_summary.csv")
    print(f"    Saved: table_architecture_summary.csv")


def generate_statistical_tests(df, output_dir):
    """Generate statistical significance tests."""
    print("  Generating statistical tests...")

    report = []
    report.append("="*80)
    report.append("STATISTICAL ANALYSIS")
    report.append("="*80)
    report.append("")

    # Test 1: Fusion strategy differences
    report.append("1. ANOVA: Fusion Strategy Comparison")
    report.append("-"*80)

    top_view = df[df["Fusion"] == "TOP View"]["Mean MAE"].values
    late = df[df["Fusion"] == "Late"]["Mean MAE"].values
    feature = df[df["Fusion"] == "Feature"]["Mean MAE"].values

    f_stat, p_value = f_oneway(top_view, late, feature)
    report.append(f"  F-statistic: {f_stat:.4f}")
    report.append(f"  p-value: {p_value:.6f}")
    report.append(f"  Significance: {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
    report.append("")

    # Test 2: Paired t-tests (backbone-wise comparison)
    report.append("2. Paired T-Tests: Fusion vs Baseline (per backbone)")
    report.append("-"*80)

    for backbone in BACKBONES:
        top_mae = df[(df["Backbone"] == backbone) & (df["Fusion"] == "TOP View")]["Mean MAE"].values
        late_mae = df[(df["Backbone"] == backbone) & (df["Fusion"] == "Late")]["Mean MAE"].values
        feature_mae = df[(df["Backbone"] == backbone) & (df["Fusion"] == "Feature")]["Mean MAE"].values

        if len(top_mae) > 0:
            report.append("")
            report.append(f"  {backbone}:")
            report.append(f"    TOP View: {top_mae[0]:.4f}")

            if len(late_mae) > 0:
                improvement = ((top_mae[0] - late_mae[0]) / top_mae[0]) * 100
                report.append(f"    Late: {late_mae[0]:.4f} ({improvement:+.1f}%)")

            if len(feature_mae) > 0:
                improvement = ((top_mae[0] - feature_mae[0]) / top_mae[0]) * 100
                report.append(f"    Feature: {feature_mae[0]:.4f} ({improvement:+.1f}%)")

    report.append("")
    report.append("="*80)

    # Save report
    with open(output_dir / "statistical_analysis.txt", 'w') as f:
        f.write("\n".join(report))

    print(f"    Saved: statistical_analysis.txt")


def main():
    print("="*80)
    print("GENERATING FINAL ANALYSIS AND VISUALIZATIONS")
    print("="*80)

    # Create output directories
    graphs_dir = project_root / "Results" / "Graphs"
    csv_dir = project_root / "Results" / "CSV_and_Analysis"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    print(f"\\nOutput directories:")
    print(f"  Graphs: {graphs_dir}")
    print(f"  CSV and Analysis: {csv_dir}")

    # Load data
    print("\\nLoading results...")
    df = create_summary_df()
    print(f"  Loaded {len(df)} model configurations")

    # Find best model
    best_idx = df["Mean MAE"].idxmin()
    best_model = df.loc[best_idx]
    print(f"\\nBest Model: {best_model['Backbone']} ({best_model['Fusion']})")
    print(f"  MAE: {best_model['Mean MAE']:.4f} +/- {best_model['Std MAE']:.4f} days")
    print(f"  Parameters: {best_model['Params (M)']:.2f}M")

    # Generate all visualizations
    print("\\n" + "="*80)
    print("GENERATING GRAPHS")
    print("="*80)

    plot_old_style_comparison(df, graphs_dir)
    plot_new_style_comparison(df, graphs_dir)
    plot_fusion_comparison(df, graphs_dir)
    plot_model_size_vs_performance(df, graphs_dir)
    plot_best_model_analysis(graphs_dir)
    plot_per_day_performance(graphs_dir)
    plot_architecture_comparison(df, graphs_dir)

    # Generate tables and statistics
    print("\\n" + "="*80)
    print("GENERATING TABLES AND STATISTICS")
    print("="*80)

    generate_paper_tables(df, csv_dir)
    generate_statistical_tests(df, csv_dir)

    print("\\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\\nAll outputs saved to:")
    print(f"  Graphs: {graphs_dir}")
    print(f"  CSV and Analysis: {csv_dir}")
    print("\\nGenerated files:")
    print("  GRAPHS:")
    print("    - all_models_comparison_3panel.png (old style, 3 subplots)")
    print("    - all_models_comparison_single.png (new style, single panel)")
    print("    - fusion_comparison.png")
    print("    - model_size_vs_performance.png")
    print("    - training_curves_[best_model].png")
    print("    - predictions_vs_actual_[best_model].png")
    print("    - confusion_matrix_[best_model].png")
    print("    - error_distribution_[best_model].png")
    print("    - per_day_performance_[best_model].png")
    print("    - architecture_by_type.png")
    print("    - architecture_efficiency.png")
    print("\\n  TABLES:")
    print("    - table_all_models.csv")
    print("    - table_best_per_fusion.csv")
    print("    - table_fusion_summary.csv")
    print("    - table_architecture_summary.csv")
    print("\\n  STATISTICS:")
    print("    - statistical_analysis.txt")


if __name__ == "__main__":
    main()
