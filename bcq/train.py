"""
Training script for Discrete BCQ Baseline on MIMIC-IV Type 2 Diabetes Cohort.

Executes offline Batch-Constrained Q-learning with:
1. Supervised Behavior Network distillation.
2. Twin Q-Networks with soft target Polyak updates.
3. Batch action space filtering constraint.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from bcq.model import DiscreteBCQNetwork
from preprocessing.pipeline import split_by_patient
from rl.environment import StateConstructor, generate_offline_transitions


def train_bcq(
    epochs: int = 35,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 3e-4,
    polyak_tau: float = 0.005,
    threshold: float = 0.3,
    test_size: float = 0.2,
    seed: int = 42,
    save_path: Path = None,
):
    if save_path is None:
        save_path = Path(__file__).resolve().parent / "bcq_model.pt"

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = config.PROCESSED_DATA_DIR / "trajectory_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Trajectory data not found at {data_path}")

    print("=" * 60)
    print("[*] TRAINING DISCRETE BCQ (OFFLINE RL) BASELINE")
    print(f"[*] Device: {device} | Epochs: {epochs} | Threshold Tau: {threshold}")
    print("=" * 60)

    df = pd.read_csv(data_path)
    train_df, val_df = split_by_patient(df, test_size=test_size, seed=seed)

    sc = StateConstructor()
    print(f"[*] Constructing offline transitions (state_dim={sc.state_dim})...")
    s_tr, a_tr, r_tr, ns_tr, d_tr, _ = generate_offline_transitions(train_df, sc)
    s_val, a_val, r_val, ns_val, d_val, _ = generate_offline_transitions(val_df, sc)

    train_dataset = TensorDataset(
        torch.tensor(s_tr, dtype=torch.float32),
        torch.tensor(a_tr, dtype=torch.long),
        torch.tensor(r_tr, dtype=torch.float32),
        torch.tensor(ns_tr, dtype=torch.float32),
        torch.tensor(d_tr, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    bcq = DiscreteBCQNetwork(
        state_dim=sc.state_dim,
        num_actions=config.NUM_ACTIONS,
        hidden_dim=128,
        threshold=threshold,
    ).to(device)

    optimizer_q = optim.Adam(bcq.q_net.parameters(), lr=lr)
    optimizer_b = optim.Adam(bcq.behavior_net.parameters(), lr=lr)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        bcq.train()
        total_q_loss = 0.0
        total_b_loss = 0.0
        batches = 0

        for states, actions, rewards, next_states, dones in train_loader:
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)

            # -------------------------------------------------------------
            # 1. Train Behavior Network (Imitation / Generative Model)
            # -------------------------------------------------------------
            b_logits = bcq.behavior_net(states)
            b_loss = criterion_ce(b_logits, actions)

            optimizer_b.zero_grad()
            b_loss.backward()
            optimizer_b.step()

            # -------------------------------------------------------------
            # 2. Train Twin Q-Networks with Batch-Constrained Double Q
            # -------------------------------------------------------------
            with torch.no_grad():
                # Get behavior policy probabilities on next state
                next_b_logits = bcq.behavior_net(next_states)
                next_b_probs = F.softmax(next_b_logits, dim=-1)
                max_next_b = torch.max(next_b_probs, dim=-1, keepdim=True)[0]
                valid_mask = (next_b_probs / (max_next_b + 1e-8)) >= threshold

                # Target Q estimation on next states
                target_q1, target_q2 = bcq.target_q_net(next_states)
                min_target_q = torch.min(target_q1, target_q2)

                # Mask out out-of-distribution actions
                masked_next_q = min_target_q.clone()
                masked_next_q[~valid_mask] = -1e9

                max_next_q_vals = torch.max(masked_next_q, dim=-1)[0]
                target_y = rewards + (1.0 - dones) * gamma * max_next_q_vals

            curr_q1, curr_q2 = bcq.q_net(states)
            q1_selected = curr_q1.gather(1, actions.unsqueeze(1)).squeeze(1)
            q2_selected = curr_q2.gather(1, actions.unsqueeze(1)).squeeze(1)

            q_loss = criterion_mse(q1_selected, target_y) + criterion_mse(q2_selected, target_y)

            optimizer_q.zero_grad()
            q_loss.backward()
            optimizer_q.step()

            # Polyak soft target updates
            bcq.soft_update_targets(tau=polyak_tau)

            total_q_loss += q_loss.item()
            total_b_loss += b_loss.item()
            batches += 1

        avg_q_loss = total_q_loss / max(batches, 1)
        avg_b_loss = total_b_loss / max(batches, 1)

        # Validation Bellman check
        bcq.eval()
        with torch.no_grad():
            s_v = torch.tensor(s_val, dtype=torch.float32, device=device)
            a_v = torch.tensor(a_val, dtype=torch.long, device=device)
            r_v = torch.tensor(r_val, dtype=torch.float32, device=device)
            ns_v = torch.tensor(ns_val, dtype=torch.float32, device=device)
            d_v = torch.tensor(d_val, dtype=torch.float32, device=device)

            val_q1, _ = bcq.q_net(s_v)
            val_q1_sel = val_q1.gather(1, a_v.unsqueeze(1)).squeeze(1)
            val_tq1, val_tq2 = bcq.target_q_net(ns_v)
            val_target_y = r_v + (1.0 - d_v) * gamma * torch.max(torch.min(val_tq1, val_tq2), dim=-1)[0]
            val_bellman_loss = F.mse_loss(val_q1_sel, val_target_y).item()

        if val_bellman_loss < best_val_loss:
            best_val_loss = val_bellman_loss
            bcq.save_checkpoint(save_path)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Q Loss: {avg_q_loss:.4f} | Behavior Loss: {avg_b_loss:.4f} | Val Bellman MSE: {val_bellman_loss:.4f}")

    print(f"[OK] Training complete. Best Val Bellman MSE: {best_val_loss:.4f}")
    print(f"[OK] BCQ model checkpoint saved to: {save_path}")
    print("=" * 60)
    return bcq


if __name__ == "__main__":
    train_bcq()
