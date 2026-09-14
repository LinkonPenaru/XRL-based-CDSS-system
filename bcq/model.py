"""
Discrete Batch-Constrained Deep Q-Learning (BCQ) Architecture.

Based on Fujimoto et al. (ICML 2019 / NeurIPS):
- Twin Q-Networks (Q1, Q2) to prevent maximization bias.
- Behavior Generative Policy G(a|s) to model the data distribution.
- Threshold constraint tau to eliminate out-of-distribution actions.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwinQNet(nn.Module):
    """Twin Q-Network mapping state s to Q-values for all discrete actions."""

    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state), self.q2(state)


class BehaviorNet(nn.Module):
    """Generative/Imitation network estimating behavior policy probability distribution pi_b(a|s)."""

    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DiscreteBCQNetwork(nn.Module):
    """
    Consolidated Discrete BCQ agent:
    - Twin Q network + delayed target networks.
    - Behavior network with threshold parameter tau.
    """

    def __init__(
        self,
        state_dim: int = 56,
        num_actions: int = 4,
        hidden_dim: int = 128,
        threshold: float = 0.3,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.threshold = threshold

        self.q_net = TwinQNet(state_dim, num_actions, hidden_dim)
        self.target_q_net = TwinQNet(state_dim, num_actions, hidden_dim)
        self.behavior_net = BehaviorNet(state_dim, num_actions, hidden_dim)

        self.hard_update_targets()

    def hard_update_targets(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def soft_update_targets(self, tau: float = 0.005):
        for target_p, p in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_p.data.copy_(tau * p.data + (1.0 - tau) * target_p.data)

    def select_action(
        self,
        state: Union[np.ndarray, torch.Tensor, list],
        deterministic: bool = True,
    ) -> int:
        """
        Constrained action selection:
        Finds actions where pi_b(a|s) / max_b(pi_b(b|s)) >= tau,
        then picks the action with highest Q1 value among allowed actions.
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
            q1, _ = self.q_net(s_t)
            b_logits = self.behavior_net(s_t)
            b_probs = F.softmax(b_logits, dim=-1)

            # BCQ constraint mask
            max_prob = torch.max(b_probs, dim=-1, keepdim=True)[0]
            valid_mask = (b_probs / (max_prob + 1e-8)) >= self.threshold

            # Mask out invalid actions by setting Q-value to -inf
            masked_q = q1.clone()
            masked_q[~valid_mask] = -1e9

            if deterministic:
                action = int(torch.argmax(masked_q, dim=-1).item())
            else:
                masked_probs = F.softmax(masked_q, dim=-1)
                action = int(torch.multinomial(masked_probs, 1).item())

        return action

    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            s = state.to(device)
            if s.dim() == 1:
                s = s.unsqueeze(0)
            q1, _ = self.q_net(s)
        return q1.squeeze(0)

    def save_checkpoint(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
            "state_dim": self.state_dim,
            "num_actions": self.num_actions,
            "threshold": self.threshold,
        }, path)

    def load_checkpoint(self, path: Union[str, Path], device: str = "cpu"):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.to(device)
        self.eval()


# Alias for unified agent naming
DiscreteBCQAgent = DiscreteBCQNetwork
