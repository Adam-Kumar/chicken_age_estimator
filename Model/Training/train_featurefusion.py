"""train_featurefusion.py
Training script for the Feature Fusion model - multi-view chicken age regression.

This model uses two ResNet50 backbones to extract features from TOP and SIDE views,
concatenates them, and uses a fusion head to learn cross-view interactions before
making the final age prediction.

Architecture:
    TOP image → ResNet50 → features_top (2048) ──┐
                                                   ├─→ concatenate (4096) → FC(512) → FC(1) → prediction
    SIDE image → ResNet50 → features_side (2048) ─┘

Usage:
    Basic training with default parameters:
        python Model/train_featurefusion.py

    Custom parameters:
        python Model/train_featurefusion.py --epochs 50 --batch_size 16 --lr 5e-5

Outputs:
    - Model/checkpoints/best_feature_fusion.pth - Best model checkpoint
    - Results/training_curves/train_val_mae_feature_fusion.png - Training curve

Default Parameters:
    - Epochs: 30
    - Batch size: 32
    - Learning rate: 1e-4
    - Weight decay: 1e-2
    - Seed: 42

Note:
    This model requires paired TOP and SIDE view images of the same chicken piece.
    It learns to fuse features at the feature level, allowing more complex interactions
    between views, but may overfit on small datasets compared to late fusion.
"""

import subprocess
import sys
from pathlib import Path

# Get project root and training script path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_SCRIPT = Path(__file__).resolve().parent.parent / "train_core.py"

def main():
    print("=" * 80)
    print("TRAINING FEATURE FUSION MODEL (Multi-View Feature Concatenation)")
    print("=" * 80)
    print("\nArchitecture: Two ResNet50 backbones with feature-level fusion")
    print("Input: Paired images (TOP + SIDE views)")
    print("Parameters: ~49.1M trainable parameters\n")

    # Build command to call main training script
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--model_type", "feature_fusion",
    ] + sys.argv[1:]  # Forward any additional arguments

    # Run training
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("[SUCCESS] Feature Fusion model training complete!")
        print("=" * 80)
        print("\nOutputs:")
        print("  - Checkpoint: Model/checkpoints/best_feature_fusion.pth")
        print("  - Training curve: Results/training_curves/train_val_mae_feature_fusion.png")
        print("\nNext steps:")
        print("  - Evaluate: python Model/Evaluating/evaluate_featurefusion.py --model_type feature_fusion")
        print("  - Compare all models: python Model/Evaluating/evaluate_all_models.py")
    else:
        print("\n" + "=" * 80)
        print("[FAILED] Training failed. Check error messages above.")
        print("=" * 80)

    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
