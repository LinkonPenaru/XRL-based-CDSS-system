"""
Unified Multi-Model Benchmark Evaluator for Type 2 Diabetes CDSS.

Evaluates all models on the identical patient-level test split:
- Historical Clinician Practice
- XGBoost Supervised Baseline
- Behavioral Cloning (BC) Deep Imitation Baseline
- Discrete BCQ Offline RL Baseline
- Discrete CQL Primary RL Agent
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from preprocessing.pipeline import split_by_patient
from rl.cql import clinical_rule_policy, DiscreteCQLNetwork
from rl.environment import (
    MonitoringAction,
    StateConstructor,
    compute_monitoring_reward,
    infer_behavior_action,
)
from xgboost.model import XGBoostAgent
from bc.model import BehavioralCloningPolicy
from bcq.model import DiscreteBCQNetwork


def load_all_models(
    state_dim: int,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Loads all available model checkpoints."""
    models = {}

    # 1. XGBoost
    xgb_path = PROJECT_ROOT / "xgboost" / "xgboost_model.json"
    if xgb_path.exists():
        xgb_agent = XGBoostAgent(num_class=config.NUM_ACTIONS)
        xgb_agent.load_model(xgb_path)
        models["XGBoost"] = {
            "agent": xgb_agent,
            "type": "Supervised ML",
            "predict_fn": lambda s: xgb_agent.select_action(s, deterministic=True),
        }
    else:
        print(f"[!] XGBoost checkpoint not found at {xgb_path}")

    # 2. Behavioral Cloning (BC)
    bc_path = PROJECT_ROOT / "bc" / "bc_model.pt"
    if bc_path.exists():
        bc_model = BehavioralCloningPolicy(state_dim=state_dim, num_actions=config.NUM_ACTIONS)
        bc_model.load_checkpoint(bc_path, device=device)
        models["Behavioral Cloning (BC)"] = {
            "agent": bc_model,
            "type": "Imitation Learning",
            "predict_fn": lambda s: bc_model.select_action(s, deterministic=True),
        }
    else:
        print(f"[!] BC checkpoint not found at {bc_path}")

    # 3. Discrete BCQ
    bcq_path = PROJECT_ROOT / "bcq" / "bcq_model.pt"
    if bcq_path.exists():
        bcq_model = DiscreteBCQNetwork(state_dim=state_dim, num_actions=config.NUM_ACTIONS)
        bcq_model.load_checkpoint(bcq_path, device=device)
        models["Discrete BCQ"] = {
            "agent": bcq_model,
            "type": "Offline RL (Constraint)",
            "predict_fn": lambda s: bcq_model.select_action(s, deterministic=True),
        }
    else:
        print(f"[!] BCQ checkpoint not found at {bcq_path}")

    # 4. Discrete CQL (Ours)
    cql_path = config.RL_AGENT_DIR / "cql_model.pt"
    if cql_path.exists():
        checkpoint = torch.load(cql_path, map_location=device)
        cql_model = DiscreteCQLNetwork(state_dim=state_dim, num_actions=config.NUM_ACTIONS).to(device)
        cql_model.load_state_dict(checkpoint["model_state_dict"])
        cql_model.eval()
        models["Discrete CQL (Ours)"] = {
            "agent": cql_model,
            "type": "Offline RL (Conservative Q)",
            "predict_fn": lambda s: cql_model.select_action(
                torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0),
                deterministic=True,
            ),
        }
    else:
        print(f"[!] CQL checkpoint not found at {cql_path}")

    return models


def evaluate_policy_on_split(
    name: str,
    predict_fn: Any,
    test_df: pd.DataFrame,
    sc: StateConstructor,
    gamma: float = 0.99,
) -> Dict[str, Any]:
    """Evaluates a single model or heuristic policy on patient trajectories."""
    y_true_clinical = []
    y_behavior = []
    y_pred = []
    trajectory_returns = []

    deteriorating_total = 0
    early_detected = 0
    stable_total = 0
    false_alarms = 0

    grouped = test_df.groupby("subject_id")
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
            a_pred = predict_fn(curr_row, s_vec, a_behav)

            r_step = compute_monitoring_reward(curr_row, a_pred, next_row)
            traj_ret += (gamma ** t) * r_step

            y_true_clinical.append(a_clinical)
            y_behavior.append(a_behav)
            y_pred.append(a_pred)

            # Safety metrics
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

    acc_guideline = accuracy_score(y_true_clinical, y_pred) * 100.0
    acc_behavior = accuracy_score(y_behavior, y_pred) * 100.0
    f1_macro = f1_score(y_true_clinical, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true_clinical, y_pred, average="weighted", zero_division=0)

    sensitivity = (early_detected / deteriorating_total * 100.0) if deteriorating_total > 0 else 0.0
    false_alarm_rate = (false_alarms / stable_total * 100.0) if stable_total > 0 else 0.0
    mean_return = float(np.mean(trajectory_returns))

    cm = confusion_matrix(y_true_clinical, y_pred, labels=list(range(4)))
    report = classification_report(
        y_true_clinical,
        y_pred,
        target_names=[config.ACTION_NAMES[i] for i in range(4)],
        output_dict=True,
        zero_division=0,
    )

    return {
        "model_name": name,
        "accuracy_guideline": acc_guideline,
        "accuracy_behavior": acc_behavior,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "sensitivity": sensitivity,
        "false_alarm_rate": false_alarm_rate,
        "mean_return": mean_return,
        "confusion_matrix": cm,
        "report": report,
        "y_true": y_true_clinical,
        "y_pred": y_pred,
    }


def run_full_benchmark(
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Runs end-to-end benchmark across all models and baseline policies."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sc = StateConstructor()

    data_path = config.PROCESSED_DATA_DIR / "trajectory_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Trajectory data not found at {data_path}")

    df = pd.read_csv(data_path)
    _, test_df = split_by_patient(df, test_size=test_size, seed=seed)

    models_dict = load_all_models(state_dim=sc.state_dim, device=device)

    all_results = {}
    table_rows = []

    # 1. Historical Clinician Policy Baseline
    clinician_res = evaluate_policy_on_split(
        name="Historical Clinician",
        predict_fn=lambda row, s, a_behav: a_behav,
        test_df=test_df,
        sc=sc,
    )
    all_results["Historical Clinician"] = clinician_res
    table_rows.append({
        "Model": "Historical Clinician",
        "Category": "Empirical Baseline",
        "Guideline Acc (%)": round(clinician_res["accuracy_guideline"], 2),
        "Clinician Match (%)": 100.0,
        "Macro F1": round(clinician_res["f1_macro"], 4),
        "Sensitivity (%)": round(clinician_res["sensitivity"], 2),
        "False Alarm (%)": round(clinician_res["false_alarm_rate"], 2),
        "Mean Return (V)": round(clinician_res["mean_return"], 2),
    })

    # 2. Machine Learning & RL Models
    for name, info in models_dict.items():
        res = evaluate_policy_on_split(
            name=name,
            predict_fn=lambda row, s, a_b, fn=info["predict_fn"]: fn(s),
            test_df=test_df,
            sc=sc,
        )
        all_results[name] = res
        table_rows.append({
            "Model": name,
            "Category": info["type"],
            "Guideline Acc (%)": round(res["accuracy_guideline"], 2),
            "Clinician Match (%)": round(res["accuracy_behavior"], 2),
            "Macro F1": round(res["f1_macro"], 4),
            "Sensitivity (%)": round(res["sensitivity"], 2),
            "False Alarm (%)": round(res["false_alarm_rate"], 2),
            "Mean Return (V)": round(res["mean_return"], 2),
        })

    summary_df = pd.DataFrame(table_rows)
    return summary_df, all_results
