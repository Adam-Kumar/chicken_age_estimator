"""train_baseline.py
Training script for the Baseline (ResNetRegressor) model - single-view chicken age regression.

This model uses a single ResNet50 backbone to predict chicken drumette age from individual
images (TOP or SIDE views processed independently).

Architecture:
    Input: Single image → ResNet50 backbone → FC layer → Age prediction (1-7 days)

Usage:
    Basic training with default parameters:
        python Model/train_baseline.py

    Custom parameters:
        python Model/train_baseline.py --epochs 50 --batch_size 16 --lr 5e-5

Outputs:
    - Results/checkpoints/best_baseline.pth - Best model checkpoint
    - Results/training_curves/train_val_mae_baseline.png - Training curve

Default Parameters:
    - Epochs: 30
    - Batch size: 32
    - Learning rate: 1e-4
    - Weight decay: 1e-2
    - Seed: 42
"""

import subprocess
import sys
from pathlib import Path

# Get project root and training script path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_SCRIPT = Path(__file__).resolve().parent.parent / "train_core.py"

def main():
    print("=" * 80)
    print("TRAINING BASELINE MODEL (ResNetRegressor - Single View)")
    print("=" * 80)
    print("\nArchitecture: ResNet50 backbone for single-view age prediction")
    print("Input: Individual images (TOP or SIDE views)")
    print("Parameters: ~23.5M trainable parameters\n")

    # Build command to call main training script
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--model_type", "baseline",
    ] + sys.argv[1:]  # Forward any additional arguments

    # Run training
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("[SUCCESS] Baseline model training complete!")
        print("=" * 80)
        print("\nOutputs:")
        print("  - Checkpoint: Model/checkpoints/best_baseline.pth")
        print("  - Training curve: Results/training_curves/train_val_mae_baseline.png")
        print("\nNext steps:")
        print("  - Evaluate: python Model/Evaluating/evaluate_baseline.py")
        print("  - Compare all models: python Model/Evaluating/evaluate_all_models.py")
    else:
        print("\n" + "=" * 80)
        print("[FAILED] Training failed. Check error messages above.")
        print("=" * 80)

    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
