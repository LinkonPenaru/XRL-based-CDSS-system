"""
Behavioral Cloning (BC) baseline package for Type 2 Diabetes CDSS.
Imitation learning via supervised cross-entropy policy distillation.
"""

from .model import BehavioralCloningPolicy

__all__ = ["BehavioralCloningPolicy"]
