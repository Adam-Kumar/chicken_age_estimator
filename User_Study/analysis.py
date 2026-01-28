"""analysis.py
Statistical analysis script for chicken age estimation user study.

Generates:
- Confusion matrices (ViT-B/16 Feature Fusion, Human Pre/Post)
- Bar graph comparison (ViT-B/16 vs Human)
- Individual participant performance
- Metrics CSV
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score, confusion_matrix

# =========================
# CONFIG
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / 'Results'
CSV_FILE = SCRIPT_DIR / 'Chicken Decay Estimation Study .csv'  # Updated filename with space
PRE_CALIBRATION_GT = [4, 4, 1, 1, 6, 6, 3, 7, 5, 2]
POST_CALIBRATION_GT = [7, 7, 3, 3, 1, 1, 4, 6, 5, 2]

RESULTS_DIR.mkdir(exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_FILE)
pre_cols = df.columns[5:15]
post_cols = df.columns[16:26]

pre_predictions = df[pre_cols].apply(pd.to_numeric, errors='coerce').values
post_predictions = df[post_cols].apply(pd.to_numeric, errors='coerce').values
pre_gt = np.array(PRE_CALIBRATION_GT)
post_gt = np.array(POST_CALIBRATION_GT)
n_participants = len(df)

print(f"Loaded {n_participants} participants from user study")

# =========================
# METRIC FUNCTIONS
# =========================
def calculate_mae(preds, gt): return np.mean(np.abs(preds - gt))
def calculate_rmse(preds, gt): return np.sqrt(np.mean((preds - gt)**2))
def calculate_accuracy(preds, gt, tol=0): return np.mean(np.abs(preds - gt) <= tol)
def calculate_r2(preds, gt): return r2_score(gt, preds)
def calculate_r(preds, gt): return stats.pearsonr(preds, gt)[0]

# =========================
# PER PARTICIPANT METRICS
# =========================
participant_results = []
for i in range(n_participants):
    pre_mae = calculate_mae(pre_predictions[i], pre_gt)
    post_mae = calculate_mae(post_predictions[i], post_gt)
    participant_results.append({
        'participant_id': i+1,
        'pre_mae': pre_mae,
        'post_mae': post_mae,
        'improvement': pre_mae - post_mae
    })
results_df = pd.DataFrame(participant_results)

# Save detailed participant results
results_df.to_csv(RESULTS_DIR / 'participant_detailed_results.csv', index=False)
print(f"Saved participant_detailed_results.csv ({n_participants} participants)")

all_pre_predictions = pre_predictions.flatten()
all_post_predictions = post_predictions.flatten()
all_pre_gt = np.tile(pre_gt, n_participants)
all_post_gt = np.tile(post_gt, n_participants)

# =========================
# LOAD VIT-B/16 FEATURE FUSION PREDICTIONS
# =========================
# Load all 3 folds and combine
vit_predictions_dir = SCRIPT_DIR.parent / 'Results' / 'Data' / 'predictions' / 'vit_b_16_feature'

all_vit_preds = []
all_vit_targets = []

for fold in range(3):
    fold_file = vit_predictions_dir / f'fold{fold}.csv'
    if fold_file.exists():
        fold_df = pd.read_csv(fold_file)
        all_vit_preds.extend(fold_df['prediction'].values)
        all_vit_targets.extend(fold_df['target'].values)
    else:
        print(f"Warning: {fold_file} not found")

vit_preds = np.array(all_vit_preds)
vit_targets = np.array(all_vit_targets)

print(f"Loaded ViT-B/16 Feature Fusion predictions: {len(vit_preds)} samples")

# =========================
# CALCULATE METRICS CSV
# =========================
metrics = []
datasets = [
    ('ViT-B/16 Feature (Best Model)', vit_preds, vit_targets),
    ('Pre-Calibration', all_pre_predictions, all_pre_gt),
    ('Post-Calibration', all_post_predictions, all_post_gt)
]

for name, preds, gt in datasets:
    mae = calculate_mae(preds, gt)
    rmse = calculate_rmse(preds, gt)
    r2 = calculate_r2(preds, gt)
    r = calculate_r(preds, gt)
    acc = calculate_accuracy(np.round(preds), gt)
    metrics.append({'dataset': name, 'MAE': mae, 'RMSE': rmse, 'R²': r2, 'R': r, 'Accuracy': acc})

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(RESULTS_DIR / 'metrics_summary.csv', index=False)
print("Saved metrics_summary.csv")
print("\nMetrics Summary:")
print(metrics_df.to_string(index=False))

def plot_confusion_matrix_comparison(model_preds, model_targets,
                                     pre_predictions, pre_gt,
                                     post_predictions, post_gt,
                                     results_dir):
    """Plot 3 normalized confusion matrices side by side with standardized color scale."""

    # Round predictions to nearest integer
    model_preds_rounded = np.round(model_preds).astype(int)
    model_targets_int = model_targets.astype(int)
    pre_preds = np.round(pre_predictions.flatten()).astype(int)
    pre_targets = np.tile(pre_gt, len(pre_predictions))
    post_preds = np.round(post_predictions.flatten()).astype(int)
    post_targets = np.tile(post_gt, len(post_predictions))

    labels = [1, 2, 3, 4, 5, 6, 7]

    # Compute normalized confusion matrices
    def normalized_cm(targets, preds):
        cm = confusion_matrix(targets, preds, labels=labels)
        return cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    # Order: Pre-calibration, Post-calibration, ViT model
    cms = [normalized_cm(pre_targets, pre_preds),
           normalized_cm(post_targets, post_preds),
           normalized_cm(model_targets_int, model_preds_rounded)]
    titles = ['Human (Pre-Calibration)', 'Human (Post-Calibration)', 'ViT-B/16 Feature Fusion']

    # Determine global min/max for color normalization (0–1)
    vmin, vmax = 0, 1

    fig, axes = plt.subplots(1, 3, figsize=(26, 8))
    plt.subplots_adjust(wspace=0.3)

    for i, ax in enumerate(axes):
        sns.heatmap(cms[i], annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    ax=ax, square=True, linewidths=0, linecolor=None,
                    vmin=vmin, vmax=vmax, cbar=False)
        ax.set_title(titles[i], fontsize=24, fontweight='bold', pad=15)
        ax.set_xlabel('Predicted Age (days)', fontsize=24, fontweight='bold')
        ax.set_ylabel('Actual Age (days)', fontsize=24, fontweight='bold')
        ax.tick_params(axis='both', labelsize=20)

    # Add single colorbar to the right
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    norm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    norm.set_array([])
    cbar = fig.colorbar(norm, cax=cbar_ax)
    cbar.set_label('Proportion', fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    plt.tight_layout(rect=[0, 0, 0.91, 1.0])
    plt.savefig(results_dir / '1_confusion_matrix_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved 1_confusion_matrix_comparison.png")
    plt.close()

# Call the function
plot_confusion_matrix_comparison(vit_preds, vit_targets,
                                 pre_predictions, pre_gt,
                                 post_predictions, post_gt,
                                 RESULTS_DIR)

# =========================
# BAR GRAPH COMPARISON
# =========================
best_human = results_df['post_mae'].min()
worst_human = results_df['post_mae'].max()
avg_human_pre = results_df['pre_mae'].mean()
avg_human_post = results_df['post_mae'].mean()

bar_labels = ['ViT-B/16\nFeature\n(Best Model)', 'Human\nPre-Cal\n(Average)',
              'Human\nPost-Cal\n(Average)']
bar_values = [
    0.660,  # ViT-B/16 MAE from model comparison results
    avg_human_pre,
    avg_human_post
]

fig, ax = plt.subplots(figsize=(10,7))
bars = ax.bar(bar_labels, bar_values,
              color=['#1f77b4','#ff7f0e','#2ca02c'])
ax.set_ylabel('Mean Absolute Error (Days)', fontsize=24, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, height, f'{height:.3f}',
            ha='center', va='bottom', fontsize=18, fontweight='bold')

ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=20)
plt.tight_layout()
plt.savefig(RESULTS_DIR / '2_bar_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved 2_bar_comparison.png")

# =========================
# INDIVIDUAL PARTICIPANT PERFORMANCE
# =========================
fig, ax = plt.subplots(figsize=(16,6))
x = np.arange(len(results_df))
width = 0.35
ax.bar(x - width/2, results_df['pre_mae'], width, label='Pre-Calibration', color='coral', alpha=0.8)
ax.bar(x + width/2, results_df['post_mae'], width, label='Post-Calibration', color='lightgreen', alpha=0.8)

# Add model performance line
model_mae = calculate_mae(vit_preds, vit_targets)
ax.axhline(y=model_mae, color='blue', linestyle='--', linewidth=2,
           label=f'ViT-B/16 Feature (MAE: {model_mae:.3f})')

ax.set_xlabel('Participant ID', fontsize=24, fontweight='bold')
ax.set_ylabel('MAE (Days)', fontsize=24, fontweight='bold')
ax.set_xticks(x[::5])  # Show every 5th participant to avoid crowding
ax.set_xticklabels(results_df['participant_id'][::5])
ax.legend(fontsize=18, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
ax.tick_params(axis='both', labelsize=18)
plt.tight_layout()
plt.savefig(RESULTS_DIR / '3_individual_participant_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved 3_individual_participant_performance.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nResults for {n_participants} participants:")
print(f"  Average Pre-Calibration MAE:  {avg_human_pre:.3f} days")
print(f"  Average Post-Calibration MAE: {avg_human_post:.3f} days")
print(f"  Best Human (Post-Cal):        {best_human:.3f} days")
print(f"  Worst Human (Post-Cal):       {worst_human:.3f} days")
print(f"  ViT-B/16 Feature Model MAE:   {model_mae:.3f} days")
print(f"\nImprovement: {avg_human_pre - avg_human_post:.3f} days ({((avg_human_pre - avg_human_post)/avg_human_pre*100):.1f}%)")
print("="*80)
