"""analysis.py
Statistical analysis script for chicken age estimation user study.

Generates:
- Confusion matrices (ConvNeXt-T Feature Fusion, Human Pre/Post)
- Bar graph comparison (ConvNeXt vs Human)
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
CSV_FILE = SCRIPT_DIR / 'Chicken Decay Estimation Study.csv'
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

all_pre_predictions = pre_predictions.flatten()
all_post_predictions = post_predictions.flatten()
all_pre_gt = np.tile(pre_gt, n_participants)
all_post_gt = np.tile(post_gt, n_participants)

# =========================
# LOAD CONVNEXT PREDICTIONS
# =========================
convnext_csv_file = SCRIPT_DIR.parent / 'Results' / 'convnext_t' / 'csv' / 'predictions.csv'
convnext_df = pd.read_csv(convnext_csv_file)

convnext_preds = convnext_df['predicted_age'].values
convnext_targets = convnext_df['actual_age'].values

# =========================
# CALCULATE METRICS CSV
# =========================
metrics = []
datasets = [
    ('Best Model', convnext_preds, convnext_targets),
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

def plot_confusion_matrix_comparison(convnext_preds, convnext_targets,
                                     pre_predictions, pre_gt,
                                     post_predictions, post_gt,
                                     results_dir):
    """Plot 3 normalized confusion matrices side by side with standardized color scale."""

    # Round predictions to nearest integer
    conv_preds = np.round(convnext_preds).astype(int)
    conv_targets = convnext_targets.astype(int)
    pre_preds = np.round(pre_predictions.flatten()).astype(int)
    pre_targets = np.tile(pre_gt, len(pre_predictions))
    post_preds = np.round(post_predictions.flatten()).astype(int)
    post_targets = np.tile(post_gt, len(post_predictions))

    labels = [1, 2, 3, 4, 5, 6, 7]

    # Compute normalized confusion matrices
    def normalized_cm(targets, preds):
        cm = confusion_matrix(targets, preds, labels=labels)
        return cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    cms = [normalized_cm(conv_targets, conv_preds),
           normalized_cm(pre_targets, pre_preds),
           normalized_cm(post_targets, post_preds)]

    # Compute accuracies
    accuracies = [np.mean(conv_preds == conv_targets),
                  np.mean(pre_preds == pre_targets),
                  np.mean(post_preds == post_targets)]

    titles = ['Best Model', 'Pre-Calibration', 'Post-Calibration']

    # Determine global min/max for color normalization (0–1)
    vmin, vmax = 0, 1

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle('Confusion Matrix Comparison', fontsize=30, fontweight='bold')

    for i, ax in enumerate(axes):
        sns.heatmap(cms[i], annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    ax=ax, square=True, linewidths=0, linecolor=None,
                    vmin=vmin, vmax=vmax, cbar=False)

        ax.set_title(f"{titles[i]}\nAccuracy: {accuracies[i]*100:.1f}%", fontsize=28, fontweight='bold')
        ax.set_xlabel('Predicted Age (days)', fontsize=24, fontweight='bold')
        ax.set_ylabel('Actual Age (days)', fontsize=24, fontweight='bold')
        ax.tick_params(axis='both', labelsize=20)

    # Add single colorbar to the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    norm.set_array([])
    cbar = fig.colorbar(norm, cax=cbar_ax)
    cbar.set_label('Proportion', fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(results_dir / '1_confusion_matrix_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved 1_confusion_matrix_comparison.png")
    plt.close()

# Call the function
plot_confusion_matrix_comparison(convnext_preds, convnext_targets,
                                 pre_predictions, pre_gt,
                                 post_predictions, post_gt,
                                 RESULTS_DIR)

# =========================
# BAR GRAPH COMPARISON
# =========================
best_human = results_df['post_mae'].min()
worst_human = results_df['post_mae'].max()
bar_labels = ['Best Model', 'Pre-Calibration', 'Post-Calibration', 'Best Human', 'Worst Human']
bar_values = [
    calculate_mae(convnext_preds, convnext_targets),
    calculate_mae(all_pre_predictions, all_pre_gt),
    calculate_mae(all_post_predictions, all_post_gt),
    best_human,
    worst_human
]

fig, ax = plt.subplots(figsize=(12,7))
bars = ax.bar(bar_labels, bar_values,
              color=['#1f77b4','#ff7f0e','#ffbb78','#2ca02c','#d62728'])
ax.set_ylabel('Mean Absolute Error (Days)', fontsize=24, fontweight='bold')
ax.set_title('Best Model vs Human Performance', fontsize=28, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, height, f'{height:.3f}', ha='center', va='bottom', fontsize=20)

ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.tight_layout()
plt.savefig(RESULTS_DIR / '2_bar_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved 2_bar_comparison.png")

# =========================
# INDIVIDUAL PARTICIPANT PERFORMANCE
# =========================
fig, ax = plt.subplots(figsize=(12,6))
x = np.arange(len(results_df))
width = 0.35
ax.bar(x - width/2, results_df['pre_mae'], width, label='Pre-Calibration', color='coral')
ax.bar(x + width/2, results_df['post_mae'], width, label='Post-Calibration', color='lightgreen')
ax.set_xlabel('Participant ID', fontsize=24, fontweight='bold')
ax.set_ylabel('MAE (Days)', fontsize=24, fontweight='bold')
ax.set_title('Individual Participant Performance', fontsize=28, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(results_df['participant_id'])
ax.legend(fontsize=20)
ax.grid(True, alpha=0.3, axis='y')
ax.tick_params(axis='both', labelsize=20)
plt.tight_layout()
plt.savefig(RESULTS_DIR / '3_individual_participant_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved 3_individual_participant_performance.png")

print("\nANALYSIS COMPLETE!")
