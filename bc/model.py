"""
Behavioral Cloning (BC) Deep Policy Network for Type 2 Diabetes CDSS.

Implements an imitation learning Multi-Layer Perceptron (MLP) trained
via supervised cross-entropy loss directly on clinician monitoring actions.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


class BehavioralCloningPolicy(nn.Module):
    """
    MLP Behavioral Cloning Policy matching the discrete CQL architecture.
    Maps state s (dim=56) to action logits over {0, 1, 2, 3}.
    """

    def __init__(
        self,
        state_dim: int = 56,
        num_actions: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Outputs unnormalized action logits."""
        return self.net(state)

    def select_action(
        self,
        state: Union[np.ndarray, torch.Tensor, list],
        deterministic: bool = True,
    ) -> int:
        """
        Inference interface compatible with RL agents.
        Accepts numpy array, tensor, or list.
        """
        self.eval()
        device = next(self.parameters()).device

        if isinstance(state, (np.ndarray, list)):
            s_t = torch.tensor(np.asarray(state, dtype=np.float32), device=device)
        else:
            s_t = state.to(device)

        if s_t.dim() == 1:
            s_t = s_t.unsqueeze(0)

        with torch.no_grad():
            logits = self.net(s_t)
            if deterministic:
                action = int(torch.argmax(logits, dim=-1).item())
            else:
                probs = torch.softmax(logits, dim=-1)
                action = int(torch.multinomial(probs, 1).item())
        return action

    def get_action_probabilities(self, state: torch.Tensor) -> torch.Tensor:
        """Returns softmax action probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.net(state)
            return torch.softmax(logits, dim=-1)

    def save_checkpoint(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
            "state_dim": self.state_dim,
            "num_actions": self.num_actions,
        }, path)

    def load_checkpoint(self, path: Union[str, Path], device: str = "cpu"):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.to(device)
        self.eval()
