"""analysis.py
Statistical analysis script for chicken age estimation user study.

Analyzes human performance on chicken drumette age estimation before and after
calibration, comparing results with the ResNet-50 model. Generates comprehensive
metrics, statistical tests, and visualizations.

Usage:
    Run this script from any directory:

    ```
    python User_Study/analysis.py
    ```

    Or from within the User_Study directory:

    ```
    cd User_Study
    python analysis.py
    ```

Inputs:
- Chicken Decay Estimation Study.csv (user study responses)
- pre_survey_images.txt (ground truth for pre-calibration images)
- post_survey_images.txt (ground truth for post-calibration images)

Outputs (saved to Results/ folder):
- 1_predictions_vs_groundtruth.png
- 2_model_vs_human_comparison.png (includes all 3 model architectures)
- 3_individual_participant_performance.png
- participant_detailed_results.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# ============================================
# CONFIGURATION - Update these as needed
# ============================================
CSV_FILE = SCRIPT_DIR / 'Chicken Decay Estimation Study.csv'
RESULTS_DIR = SCRIPT_DIR / 'Results'

# Model performance (Test MAE in days)
MODEL_MAE_BASELINE = 0.490
MODEL_MAE_FEATURE_FUSION = 0.465
MODEL_MAE_LATE_FUSION = 0.430

# Ground truth for pre and post calibration
PRE_CALIBRATION_GT = [4, 4, 1, 1, 6, 6, 3, 7, 5, 2]
POST_CALIBRATION_GT = [7, 7, 3, 3, 1, 1, 4, 6, 5, 2]

# Create Results directory if it doesn't exist
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================
# LOAD AND PREPARE DATA
# ============================================
print("Loading data...")
df = pd.read_csv(CSV_FILE)

# Get column indices for predictions
# Pre-calibration: columns 6-15 (index 5-14) - 10 prediction columns
# Note: Column 5 (index 4) is "How often do you cook chicken?" which we skip
# Column 16 (index 15) is confidence question, which we skip
# Post-calibration: columns 17-26 (index 16-25) - 10 prediction columns
pre_cols = df.columns[5:15]
post_cols = df.columns[16:26]

print(f"Total participants: {len(df)}")
print(f"Pre-calibration columns: {len(pre_cols)}")
print(f"Post-calibration columns: {len(post_cols)}")

# ============================================
# EXTRACT PREDICTIONS
# ============================================
# Convert predictions to numeric values (in case they're stored as strings)
pre_predictions = df[pre_cols].apply(pd.to_numeric, errors='coerce').values  # Shape: (n_participants, 10)
post_predictions = df[post_cols].apply(pd.to_numeric, errors='coerce').values  # Shape: (n_participants, 10)

# Convert to numpy arrays
pre_gt = np.array(PRE_CALIBRATION_GT)
post_gt = np.array(POST_CALIBRATION_GT)

# ============================================
# CALCULATE METRICS
# ============================================
def calculate_mae(predictions, ground_truth):
    """Calculate Mean Absolute Error.

    Args:
        predictions: Array of predicted values
        ground_truth: Array of true values

    Returns:
        Mean absolute error as a float
    """
    return np.mean(np.abs(predictions - ground_truth))

def calculate_rmse(predictions, ground_truth):
    """Calculate Root Mean Squared Error.

    Args:
        predictions: Array of predicted values
        ground_truth: Array of true values

    Returns:
        Root mean squared error as a float
    """
    return np.sqrt(np.mean((predictions - ground_truth) ** 2))

def calculate_accuracy(predictions, ground_truth, tolerance=0):
    """Calculate accuracy within tolerance.

    Args:
        predictions: Array of predicted values
        ground_truth: Array of true values
        tolerance: Maximum allowed error to count as correct (default: 0 for exact match)

    Returns:
        Accuracy as a float between 0 and 1
    """
    return np.mean(np.abs(predictions - ground_truth) <= tolerance)

# Per-participant metrics
n_participants = len(df)
participant_results = []

for i in range(n_participants):
    pre_mae = calculate_mae(pre_predictions[i], pre_gt)
    pre_rmse = calculate_rmse(pre_predictions[i], pre_gt)
    pre_acc_exact = calculate_accuracy(pre_predictions[i], pre_gt, tolerance=0)
    pre_acc_1day = calculate_accuracy(pre_predictions[i], pre_gt, tolerance=1)
    
    post_mae = calculate_mae(post_predictions[i], post_gt)
    post_rmse = calculate_rmse(post_predictions[i], post_gt)
    post_acc_exact = calculate_accuracy(post_predictions[i], post_gt, tolerance=0)
    post_acc_1day = calculate_accuracy(post_predictions[i], post_gt, tolerance=1)
    
    participant_results.append({
        'participant_id': i + 1,
        'pre_mae': pre_mae,
        'pre_rmse': pre_rmse,
        'pre_acc_exact': pre_acc_exact,
        'pre_acc_1day': pre_acc_1day,
        'post_mae': post_mae,
        'post_rmse': post_rmse,
        'post_acc_exact': post_acc_exact,
        'post_acc_1day': post_acc_1day,
        'improvement': pre_mae - post_mae
    })

results_df = pd.DataFrame(participant_results)

# Overall metrics
print("\n" + "="*60)
print("OVERALL PERFORMANCE METRICS")
print("="*60)

# Pre-calibration
all_pre_predictions = pre_predictions.flatten()
all_pre_gt = np.tile(pre_gt, n_participants)
overall_pre_mae = calculate_mae(all_pre_predictions, all_pre_gt)
overall_pre_rmse = calculate_rmse(all_pre_predictions, all_pre_gt)
overall_pre_acc_exact = calculate_accuracy(all_pre_predictions, all_pre_gt, 0)
overall_pre_acc_1day = calculate_accuracy(all_pre_predictions, all_pre_gt, 1)

print(f"\nPRE-CALIBRATION:")
print(f"  MAE: {overall_pre_mae:.3f} days")
print(f"  RMSE: {overall_pre_rmse:.3f} days")
print(f"  Exact Accuracy: {overall_pre_acc_exact*100:.1f}%")
print(f"  Within ±1 day: {overall_pre_acc_1day*100:.1f}%")

# Post-calibration
all_post_predictions = post_predictions.flatten()
all_post_gt = np.tile(post_gt, n_participants)
overall_post_mae = calculate_mae(all_post_predictions, all_post_gt)
overall_post_rmse = calculate_rmse(all_post_predictions, all_post_gt)
overall_post_acc_exact = calculate_accuracy(all_post_predictions, all_post_gt, 0)
overall_post_acc_1day = calculate_accuracy(all_post_predictions, all_post_gt, 1)

print(f"\nPOST-CALIBRATION:")
print(f"  MAE: {overall_post_mae:.3f} days")
print(f"  RMSE: {overall_post_rmse:.3f} days")
print(f"  Exact Accuracy: {overall_post_acc_exact*100:.1f}%")
print(f"  Within ±1 day: {overall_post_acc_1day*100:.1f}%")

print(f"\nIMPROVEMENT AFTER CALIBRATION:")
print(f"  MAE reduction: {overall_pre_mae - overall_post_mae:.3f} days ({(overall_pre_mae - overall_post_mae)/overall_pre_mae*100:.1f}%)")

# Statistical test
t_stat, p_value = stats.ttest_rel(results_df['pre_mae'], results_df['post_mae'])
print(f"  Paired t-test p-value: {p_value:.4f} {'(significant)' if p_value < 0.05 else '(not significant)'}")

print(f"\nMODEL vs HUMAN COMPARISON:")
print(f"  Baseline Model MAE: {MODEL_MAE_BASELINE:.3f} days")
print(f"  Feature Fusion Model MAE: {MODEL_MAE_FEATURE_FUSION:.3f} days")
print(f"  Late Fusion Model MAE: {MODEL_MAE_LATE_FUSION:.3f} days")
print(f"  Human (pre-calib) MAE: {overall_pre_mae:.3f} days")
print(f"  Human (post-calib) MAE: {overall_post_mae:.3f} days")
print(f"  Best human MAE: {results_df['post_mae'].min():.3f} days")
print(f"  Worst human MAE: {results_df['post_mae'].max():.3f} days")

# ============================================
# INTER-RATER RELIABILITY
# ============================================
print(f"\nINTER-RATER RELIABILITY:")

def calculate_icc(predictions):
    """Calculate Intraclass Correlation Coefficient (ICC).

    Uses ICC(2,1) - two-way random effects model with absolute agreement
    for a single rater. Measures inter-rater reliability.

    Args:
        predictions: 2D array of shape (n_raters, n_items)

    Returns:
        ICC value as a float. Values closer to 1 indicate higher agreement.
        - < 0.5: Poor agreement
        - 0.5-0.75: Moderate agreement
        - 0.75-0.9: Good agreement
        - > 0.9: Excellent agreement
    """
    n_raters = predictions.shape[0]
    n_items = predictions.shape[1]

    # Calculate between-item variance
    item_means = np.mean(predictions, axis=0)
    grand_mean = np.mean(predictions)
    between_item_var = np.sum((item_means - grand_mean)**2) * n_raters / (n_items - 1)

    # Calculate within-item variance
    within_item_var = np.sum((predictions - item_means)**2) / (n_items * (n_raters - 1))

    # ICC(2,1)
    icc = (between_item_var - within_item_var) / (between_item_var + (n_raters - 1) * within_item_var)
    return icc

pre_icc = calculate_icc(pre_predictions)
post_icc = calculate_icc(post_predictions)

print(f"  Pre-calibration ICC: {pre_icc:.3f}")
print(f"  Post-calibration ICC: {post_icc:.3f}")

# Standard deviation across raters
pre_std = np.mean(np.std(pre_predictions, axis=0))
post_std = np.mean(np.std(post_predictions, axis=0))
print(f"  Pre-calibration avg std: {pre_std:.3f} days")
print(f"  Post-calibration avg std: {post_std:.3f} days")

# ============================================
# PERFORMANCE BY DECAY DAY
# ============================================
print(f"\nPERFORMANCE BY ACTUAL DECAY DAY:")

all_days = np.concatenate([pre_gt, post_gt])
all_predictions_combined = np.concatenate([all_pre_predictions, all_post_predictions])
all_gt_combined = np.concatenate([all_pre_gt, all_post_gt])

for day in range(1, 8):
    mask = all_gt_combined == day
    if np.sum(mask) > 0:
        day_mae = calculate_mae(all_predictions_combined[mask], all_gt_combined[mask])
        day_std = np.std(all_predictions_combined[mask])
        print(f"  Day {day}: MAE = {day_mae:.3f}, Std = {day_std:.3f} (n={np.sum(mask)})")

# ============================================
# VISUALIZATIONS
# ============================================
print("\nGenerating visualizations...")

# Figure 1: Scatter plots - Predictions vs Ground Truth
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pre-calibration
axes[0].scatter(all_pre_gt, all_pre_predictions, alpha=0.5, s=50)
axes[0].plot([0, 7], [0, 7], 'r--', linewidth=2, label='Perfect prediction')
axes[0].set_xlabel('Ground Truth (Days)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Human Prediction (Days)', fontsize=12, fontweight='bold')
axes[0].set_title(f'Pre-Calibration\nMAE: {overall_pre_mae:.3f} days', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(-0.5, 7.5)
axes[0].set_ylim(-0.5, 7.5)

# Post-calibration
axes[1].scatter(all_post_gt, all_post_predictions, alpha=0.5, s=50, color='green')
axes[1].plot([0, 7], [0, 7], 'r--', linewidth=2, label='Perfect prediction')
axes[1].set_xlabel('Ground Truth (Days)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Human Prediction (Days)', fontsize=12, fontweight='bold')
axes[1].set_title(f'Post-Calibration\nMAE: {overall_post_mae:.3f} days', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(-0.5, 7.5)
axes[1].set_ylim(-0.5, 7.5)

plt.tight_layout()
plt.savefig(RESULTS_DIR / '1_predictions_vs_groundtruth.png', dpi=300, bbox_inches='tight')
print("  Saved: Results/1_predictions_vs_groundtruth.png")

# Figure 2: Model vs Human Comparison - ALL 3 MODELS
fig, ax = plt.subplots(figsize=(12, 7))
comparison_data = {
    'Baseline\n(Single View)': MODEL_MAE_BASELINE,
    'Feature Fusion\n(Multi-View)': MODEL_MAE_FEATURE_FUSION,
    'Late Fusion\n(Multi-View)': MODEL_MAE_LATE_FUSION,
    'Human\n(Pre-Calib)': overall_pre_mae,
    'Human\n(Post-Calib)': overall_post_mae,
    'Best\nHuman': results_df['post_mae'].min(),
    'Worst\nHuman': results_df['post_mae'].max()
}

bars = ax.bar(comparison_data.keys(), comparison_data.values(),
               color=['#1f77b4', '#17becf', '#2ca02c', '#ff7f0e', '#ffbb78', '#d62728', '#9467bd'])
ax.set_ylabel('Mean Absolute Error (Days)', fontsize=12, fontweight='bold')
ax.set_title('Model vs Human Performance Comparison', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

# Add a horizontal line to separate models from humans
ax.axvline(x=2.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
ax.text(1.5, ax.get_ylim()[1] * 0.95, 'Models', ha='center', fontsize=11, fontweight='bold', color='gray')
ax.text(4.5, ax.get_ylim()[1] * 0.95, 'Humans', ha='center', fontsize=11, fontweight='bold', color='gray')

plt.tight_layout()
plt.savefig(RESULTS_DIR / '2_model_vs_human_comparison.png', dpi=300, bbox_inches='tight')
print("  Saved: Results/2_model_vs_human_comparison.png")

# Figure 3: Pre vs Post Calibration for Each Participant
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(results_df))
width = 0.35

bars1 = ax.bar(x - width/2, results_df['pre_mae'], width, label='Pre-Calibration', color='coral')
bars2 = ax.bar(x + width/2, results_df['post_mae'], width, label='Post-Calibration', color='lightgreen')

ax.set_xlabel('Participant ID', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Absolute Error (Days)', fontsize=12, fontweight='bold')
ax.set_title('Individual Participant Performance: Pre vs Post Calibration', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(results_df['participant_id'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(RESULTS_DIR / '3_individual_participant_performance.png', dpi=300, bbox_inches='tight')
print("  Saved: Results/3_individual_participant_performance.png")

# ============================================
# SAVE DETAILED RESULTS
# ============================================
results_df.to_csv(RESULTS_DIR / 'participant_detailed_results.csv', index=False)
print("\n  Saved: Results/participant_detailed_results.csv")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print(f"\nGenerated files in Results/ folder:")
print("  - 1_predictions_vs_groundtruth.png")
print("  - 2_model_vs_human_comparison.png (includes all 3 models)")
print("  - 3_individual_participant_performance.png")
print("  - participant_detailed_results.csv")