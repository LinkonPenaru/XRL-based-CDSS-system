"""
Test and Accuracy Evaluation Script for Discrete BCQ Baseline.

Evaluates:
1. Action concordance accuracy against Gold-Standard ADA Clinical Guidelines.
2. Action concordance accuracy against historical clinician behavior.
3. Per-class Precision, Recall, and F1-Scores.
4. Early Deterioration Sensitivity & Alarm Fatigue False-Alarm Rate.
5. Mean Cumulative Discounted Trajectory Return.
6. Bellman Temporal Difference (TD) Loss.
7. Confusion matrix visualization.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from bcq.model import DiscreteBCQNetwork
from preprocessing.pipeline import split_by_patient
from rl.cql import clinical_rule_policy
from rl.environment import (
    MonitoringAction,
    StateConstructor,
    compute_monitoring_reward,
    infer_behavior_action,
)


def evaluate_bcq_accuracy(
    model_path: Path = None,
    test_size: float = 0.2,
    seed: int = 42,
    save_plot: bool = True,
):
    if model_path is None:
        model_path = Path(__file__).resolve().parent / "bcq_model.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Run train.py first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sc = StateConstructor()
    model = DiscreteBCQNetwork(state_dim=sc.state_dim, num_actions=config.NUM_ACTIONS)
    model.load_checkpoint(model_path, device=device)

    data_path = config.PROCESSED_DATA_DIR / "trajectory_data.csv"
    df = pd.read_csv(data_path)
    train_df, test_df = split_by_patient(df, test_size=test_size, seed=seed)

    gamma = config.RL_CONFIG.get("gamma", 0.99)
    results = {}

    for split_name, split_df in [("Train", train_df), ("Test", test_df)]:
        y_true_clinical = []
        y_behavior = []
        y_pred = []
        trajectory_returns = []
        td_errors = []

        deteriorating_total = 0
        early_detected = 0
        stable_total = 0
        false_alarms = 0

        grouped = split_df.groupby("subject_id")
        for _, group in grouped:
            records = group.sort_values(by="chartdate").reset_index(drop=True)
            T = len(records)
            traj_ret = 0.0

            for t in range(T):
                curr_row = records.iloc[t]
                next_row = records.iloc[t + 1] if t < T - 1 else None

                a_clinical = clinical_rule_policy(curr_row)
                a_behav = infer_behavior_action(curr_row, next_row)

                s_vec = sc.extract_state_vector(curr_row)
                a_pred = model.select_action(s_vec, deterministic=True)

                r_step = compute_monitoring_reward(curr_row, a_pred, next_row)
                traj_ret += (gamma ** t) * r_step

                # TD error computation
                if next_row is not None:
                    s_tensor = torch.tensor(s_vec, dtype=torch.float32, device=device).unsqueeze(0)
                    ns_vec = sc.extract_state_vector(next_row)
                    ns_tensor = torch.tensor(ns_vec, dtype=torch.float32, device=device).unsqueeze(0)

                    with torch.no_grad():
                        q1_val = model.q_net(s_tensor)[0][0, a_pred].item()
                        tq1, tq2 = model.target_q_net(ns_tensor)
                        max_next_q = torch.max(torch.min(tq1, tq2)).item()
                        y_target = r_step + gamma * max_next_q
                        td_errors.append((q1_val - y_target) ** 2)

                y_true_clinical.append(a_clinical)
                y_behavior.append(a_behav)
                y_pred.append(a_pred)

                # Clinical safety metrics
                status = curr_row.get("overall_trajectory_status", "stable")
                if status in ["deteriorating", "rapidly_deteriorating"]:
                    deteriorating_total += 1
                    if a_pred in [
                        MonitoringAction.INCREASED_MONITORING,
                        MonitoringAction.EARLY_RISK_ALERT,
                        MonitoringAction.CLINICAL_EVALUATION,
                    ]:
                        early_detected += 1
                elif status == "stable":
                    stable_total += 1
                    if a_pred in [
                        MonitoringAction.EARLY_RISK_ALERT,
                        MonitoringAction.CLINICAL_EVALUATION,
                    ]:
                        false_alarms += 1

            trajectory_returns.append(traj_ret)

        acc_clinical = accuracy_score(y_true_clinical, y_pred)
        acc_behavior = accuracy_score(y_behavior, y_pred)
        f1_macro = f1_score(y_true_clinical, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true_clinical, y_pred, average="weighted", zero_division=0)

        sensitivity = (early_detected / deteriorating_total * 100) if deteriorating_total > 0 else 0.0
        false_alarm_rate = (false_alarms / stable_total * 100) if stable_total > 0 else 0.0
        mean_return = float(np.mean(trajectory_returns))
        mean_td = float(np.mean(td_errors)) if td_errors else 0.0

        cm = confusion_matrix(y_true_clinical, y_pred, labels=list(range(4)))
        report = classification_report(
            y_true_clinical,
            y_pred,
            target_names=[config.ACTION_NAMES[i] for i in range(4)],
            output_dict=True,
            zero_division=0,
        )

        results[split_name] = {
            "accuracy_guideline": acc_clinical,
            "accuracy_behavior": acc_behavior,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "sensitivity": sensitivity,
            "false_alarm_rate": false_alarm_rate,
            "mean_return": mean_return,
            "mean_td_error": mean_td,
            "confusion_matrix": cm,
            "report": report,
        }

    test_res = results["Test"]
    train_res = results["Train"]
    print("\n" + "=" * 65)
    print("      DISCRETE BCQ PERFORMANCE EVALUATION (TEST SET)")
    print("=" * 65)
    print(f"  * Clinical Guideline Accuracy : {test_res['accuracy_guideline'] * 100:.2f}%  (Train: {train_res['accuracy_guideline'] * 100:.2f}%)")
    print(f"  * Clinician Action Match      : {test_res['accuracy_behavior'] * 100:.2f}%  (Train: {train_res['accuracy_behavior'] * 100:.2f}%)")
    print(f"  * Macro F1-Score              : {test_res['f1_macro']:.4f}  (Train: {train_res['f1_macro']:.4f})")
    print(f"  * Weighted F1-Score           : {test_res['f1_weighted']:.4f}")
    print(f"  * Deterioration Sensitivity   : {test_res['sensitivity']:.2f}%")
    print(f"  * False Alarm Rate            : {test_res['false_alarm_rate']:.2f}%")
    print(f"  * Mean Trajectory Return (V)  : {test_res['mean_return']:.2f}")
    print(f"  * Mean Bellman TD Error       : {test_res['mean_td_error']:.4f}")
    print("-" * 65)
    print("  Per-Action Breakdown (Test Set):")
    for a_idx in range(4):
        name = config.ACTION_NAMES[a_idx]
        stats = test_res["report"][name]
        print(f"    - {name:<22}: Precision={stats['precision']:.2f} | Recall={stats['recall']:.2f} | F1={stats['f1-score']:.2f}")
    print("=" * 65)

    if save_plot:
        plot_path = Path(__file__).resolve().parent / "bcq_confusion_matrix.png"
        fig, ax = plt.subplots(figsize=(6, 5))
        labels = [config.ACTION_NAMES[i].replace("_", " ") for i in range(4)]
        sns.heatmap(
            test_res["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Oranges",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_title(f"BCQ Test Confusion Matrix\nAccuracy: {test_res['accuracy_guideline'] * 100:.1f}% | Macro F1: {test_res['f1_macro']:.2f}")
        ax.set_xlabel("Predicted Action")
        ax.set_ylabel("ADA Guideline Action")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[OK] Confusion matrix plot saved to: {plot_path}")

    return results


if __name__ == "__main__":
    evaluate_bcq_accuracy()
