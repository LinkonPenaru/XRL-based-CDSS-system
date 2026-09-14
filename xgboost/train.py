"""
Training script for XGBoost Baseline on MIMIC-IV Type 2 Diabetes Cohort.

Fits a multi-class gradient boosted classifier on patient state vectors
to predict clinician monitoring actions.
"""

from pathlib import Path
import sys
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from preprocessing.pipeline import split_by_patient
from rl.environment import StateConstructor, generate_offline_transitions
from xgboost.model import XGBoostAgent


def train_xgboost(
    test_size: float = 0.2,
    seed: int = 42,
    save_path: Path = None,
):
    if save_path is None:
        save_path = Path(__file__).resolve().parent / "xgboost_model.json"

    data_path = config.PROCESSED_DATA_DIR / "trajectory_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Trajectory data not found at {data_path}")

    print("=" * 60)
    print("[*] TRAINING XGBOOST BASELINE MODEL")
    print("=" * 60)

    print(f"[*] Loading patient trajectories from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"[*] Total records: {len(df)}, unique patients: {df['subject_id'].nunique()}")

    # Strict patient-level partition
    train_df, test_df = split_by_patient(df, test_size=test_size, seed=seed)
    print(f"[*] Train patients: {train_df['subject_id'].nunique()} | Test patients: {test_df['subject_id'].nunique()}")

    # Build offline transitions
    sc = StateConstructor()
    print(f"[*] Constructing training state vectors (state_dim={sc.state_dim})...")
    X_train, y_train, r_train, _, _, _ = generate_offline_transitions(train_df, sc)
    print(f"[*] Training transitions: {X_train.shape[0]}")
    print(f"[*] Target action distribution: {pd.Series(y_train).value_counts().to_dict()}")

    # Initialize and train
    agent = XGBoostAgent(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        num_class=config.NUM_ACTIONS,
        random_state=seed,
    )
    print("[*] Fitting XGBoost classifier...")
    agent.fit(X_train, y_train)

    # Save artifact
    agent.save_model(save_path)
    print(f"[OK] XGBoost model saved successfully to: {save_path}")

    # Quick train set accuracy check
    train_preds = agent.predict(X_train)
    train_acc = (train_preds == y_train).mean() * 100
    print(f"[OK] Training Action Match Accuracy: {train_acc:.2f}%")
    print("=" * 60)
    return agent


if __name__ == "__main__":
    train_xgboost()
