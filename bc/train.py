"""
Training script for Behavioral Cloning (BC) Baseline.

Fits a deep supervised policy network on historical clinician actions
using Cross-Entropy Loss with patient-level cross-validation.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from bc.model import BehavioralCloningPolicy
from preprocessing.pipeline import split_by_patient
from rl.environment import StateConstructor, generate_offline_transitions


def train_bc(
    epochs: int = 35,
    batch_size: int = 64,
    lr: float = 3e-4,
    test_size: float = 0.2,
    seed: int = 42,
    save_path: Path = None,
):
    if save_path is None:
        save_path = Path(__file__).resolve().parent / "bc_model.pt"

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = config.PROCESSED_DATA_DIR / "trajectory_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Trajectory data not found at {data_path}")

    print("=" * 60)
    print("[*] TRAINING BEHAVIORAL CLONING (BC) BASELINE")
    print(f"[*] Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}")
    print("=" * 60)

    df = pd.read_csv(data_path)
    train_df, val_df = split_by_patient(df, test_size=test_size, seed=seed)

    sc = StateConstructor()
    print(f"[*] Extracting state-action pairs (state_dim={sc.state_dim})...")
    X_train, y_train, _, _, _, _ = generate_offline_transitions(train_df, sc)
    X_val, y_val, _, _, _, _ = generate_offline_transitions(val_df, sc)

    train_tensor_x = torch.tensor(X_train, dtype=torch.float32)
    train_tensor_y = torch.tensor(y_train, dtype=torch.long)
    val_tensor_x = torch.tensor(X_val, dtype=torch.float32)
    val_tensor_y = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(train_tensor_x, train_tensor_y),
        batch_size=batch_size,
        shuffle=True,
    )

    model = BehavioralCloningPolicy(
        state_dim=sc.state_dim,
        num_actions=config.NUM_ACTIONS,
        hidden_dim=128,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            preds = torch.argmax(logits, dim=-1)
            correct_train += (preds == batch_y).sum().item()
            total_train += len(batch_y)

        train_loss = total_loss / total_train
        train_acc = (correct_train / total_train) * 100

        # Evaluate on validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_tensor_x.to(device))
            val_loss = criterion(val_logits, val_tensor_y.to(device)).item()
            val_preds = torch.argmax(val_logits, dim=-1)
            val_acc = (val_preds == val_tensor_y.to(device)).float().mean().item() * 100

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_checkpoint(save_path)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}%")

    print(f"[OK] Training complete. Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"[OK] Model checkpoint saved to: {save_path}")
    print("=" * 60)
    return model


if __name__ == "__main__":
    train_bc()
