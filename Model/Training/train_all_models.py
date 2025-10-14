"""train_all_models.py
Automated script to train all 3 chicken age regression models sequentially.

This script trains all model architectures in sequence:
1. Baseline (ResNetRegressor) - Single-view model using TOP or SIDE images
2. Late Fusion (LateFusionRegressor) - Averages predictions from TOP and SIDE views
3. Feature Fusion (FeatureFusionRegressor) - Concatenates features from both views

Usage:
    Train all models with default parameters:
        python Model/train_all_models.py

Outputs:
    For each model:
        - Results/checkpoints/best_{model_type}.pth - Best model checkpoint
        - Results/training_curves/train_val_mae_{model_type}.png - Training curve

Training Configuration:
    Customize parameters by editing the CONFIG dictionary below:
    - epochs: 30 (number of training epochs)
    - batch_size: 32 (images per batch)
    - lr: 1e-4 (learning rate)
    - weight_decay: 1e-2 (L2 regularization)
    - seed: 42 (random seed for reproducibility)

After Training:
    Evaluate and compare all models:
        python Model/evaluate_all_models.py

    Results will be saved to:
        Results/comparison/metrics_comparison.csv
        Results/comparison/scatter_comparison.png

Estimated Time:
    - Baseline: 20-40 minutes (depending on GPU)
    - Late Fusion: 30-60 minutes
    - Feature Fusion: 30-60 minutes
    - Total: 1.5-3 hours
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

# Training configuration
CONFIG = {
    "epochs": 30,
    "batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "seed": 42,
}

# Model types to train (in order)
MODELS = ["baseline", "late_fusion", "feature_fusion"]

# Path to individual training scripts
TRAIN_SCRIPTS = {
    "baseline": Path(__file__).parent / "train_baseline.py",
    "late_fusion": Path(__file__).parent / "train_latefusion.py",
    "feature_fusion": Path(__file__).parent / "train_featurefusion.py",
}


def run_training(model_type: str, config: dict) -> bool:
    """Run training for a specific model type using its dedicated training script."""
    print("\n" + "=" * 80)
    print(f"TRAINING: {model_type.upper().replace('_', ' ')}")
    print("=" * 80 + "\n")

    # Build command with training script and parameters
    train_script = TRAIN_SCRIPTS[model_type]
    cmd = [
        sys.executable,
        str(train_script),
        "--epochs", str(config["epochs"]),
        "--batch_size", str(config["batch_size"]),
        "--lr", str(config["lr"]),
        "--weight_decay", str(config["weight_decay"]),
        "--seed", str(config["seed"]),
    ]

    start_time = time.time()

    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"\n[SUCCESS] {model_type} training completed in {elapsed/60:.1f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] {model_type} training failed with error code {e.returncode}")
        return False


def main():
    print("=" * 80)
    print("TRAINING ALL MODELS FOR CHICKEN AGE REGRESSION")
    print("=" * 80)
    print(f"\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print(f"\nModels to train: {', '.join(MODELS)}")
    print()

    results = {}
    total_start = time.time()

    for model_type in MODELS:
        success = run_training(model_type, CONFIG)
        results[model_type] = success

        if not success:
            print(f"\nWarning: {model_type} training failed. Continuing with next model...")

    total_elapsed = time.time() - total_start

    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    for model_type, success in results.items():
        status = "[SUCCESS]" if success else "[FAILED]"
        print(f"  {model_type:20s}: {status}")
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes")
    print("=" * 80)

    # Check if all succeeded
    if all(results.values()):
        print("\n[SUCCESS] All models trained successfully!")
        print("\nOutputs saved to:")
        print("  - Checkpoints: Model/checkpoints/")
        print("  - Training curves: Results/training_curves/")
        print("\nNext steps:")
        print("  1. Evaluate and compare models:")
        print("     python Model/Evaluating/evaluate_all_models.py")
        print("  2. Check comparison results:")
        print("     Results/comparison/")
        return 0
    else:
        print("\n[FAILED] Some models failed to train. Check logs above.")
        print("\nTo retry individual models, use:")
        for model_type, success in results.items():
            if not success:
                print(f"  python Model/Training/train_{model_type}.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
