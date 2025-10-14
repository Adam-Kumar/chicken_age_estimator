"""train_latefusion.py
Training script for the Late Fusion model - multi-view chicken age regression.

This model uses two separate ResNet50 backbones (one for TOP view, one for SIDE view)
that make independent predictions, which are then averaged to produce the final prediction.

Architecture:
    TOP image → ResNet50 → prediction_top ──┐
                                              ├─→ average → final prediction
    SIDE image → ResNet50 → prediction_side ─┘

Usage:
    Basic training with default parameters:
        python Model/train_latefusion.py

    Custom parameters:
        python Model/train_latefusion.py --epochs 50 --batch_size 16 --lr 5e-5

Outputs:
    - Model/checkpoints/best_late_fusion.pth - Best model checkpoint
    - Results/training_curves/train_val_mae_late_fusion.png - Training curve

Default Parameters:
    - Epochs: 30
    - Batch size: 32
    - Learning rate: 1e-4
    - Weight decay: 1e-2
    - Seed: 42

Note:
    This model requires paired TOP and SIDE view images of the same chicken piece.
    It typically outperforms single-view and feature fusion on small datasets due to
    its simple ensemble-like averaging strategy.
"""

import subprocess
import sys
from pathlib import Path

# Get project root and training script path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_SCRIPT = Path(__file__).resolve().parent.parent / "train_core.py"

def main():
    print("=" * 80)
    print("TRAINING LATE FUSION MODEL (Multi-View Prediction Averaging)")
    print("=" * 80)
    print("\nArchitecture: Two ResNet50 backbones with prediction averaging")
    print("Input: Paired images (TOP + SIDE views)")
    print("Parameters: ~47.0M trainable parameters\n")

    # Build command to call main training script
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--model_type", "late_fusion",
    ] + sys.argv[1:]  # Forward any additional arguments

    # Run training
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("[SUCCESS] Late Fusion model training complete!")
        print("=" * 80)
        print("\nOutputs:")
        print("  - Checkpoint: Model/checkpoints/best_late_fusion.pth")
        print("  - Training curve: Results/training_curves/train_val_mae_late_fusion.png")
        print("\nNext steps:")
        print("  - Evaluate: python Model/Evaluating/evaluate_latefusion.py --model_type late_fusion")
        print("  - Compare all models: python Model/Evaluating/evaluate_all_models.py")
    else:
        print("\n" + "=" * 80)
        print("[FAILED] Training failed. Check error messages above.")
        print("=" * 80)

    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
