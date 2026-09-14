"""
Discrete Batch-Constrained Deep Q-Learning (BCQ) package for Type 2 Diabetes CDSS.

Offline Reinforcement Learning with explicit behavior action space constraint.
"""

from .model import DiscreteBCQNetwork, DiscreteBCQAgent

__all__ = ["DiscreteBCQNetwork", "DiscreteBCQAgent"]
