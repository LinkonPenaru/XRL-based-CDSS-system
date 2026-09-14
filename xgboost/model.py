"""
XGBoost Supervised Baseline Model for Type 2 Diabetes CDSS.

Provides a multi-class gradient boosted decision tree classifier predicting
clinical monitoring actions from patient state vectors.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import sys

# Ensure site-packages xgboost is loaded
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except (ImportError, AttributeError):
    # If shaded by local directory, load from site-packages
    import importlib.util
    for p in sys.path:
        cand = Path(p) / "xgboost" / "__init__.py"
        if cand.exists() and "site-packages" in str(cand):
            spec = importlib.util.spec_from_file_location("real_xgb", cand)
            xgb = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(xgb)
            XGBClassifier = xgb.XGBClassifier
            break


class XGBoostAgent:
    """
    XGBoost classifier baseline wrapped for RL policy evaluation.
    Maps patient state s (dim=56) to monitoring action a in {0, 1, 2, 3}.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        num_class: int = 4,
        random_state: int = 42,
    ):
        self.num_class = num_class
        self.random_state = random_state
        self.clf = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="multi:softprob",
            num_class=num_class,
            random_state=random_state,
            eval_metric="mlogloss",
            tree_method="hist",
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostAgent":
        """Fits the gradient boosted tree classifier on state-action pairs."""
        self.clf.fit(X, y)
        self.is_fitted = True
        return self

    def select_action(self, state: Union[np.ndarray, list], deterministic: bool = True) -> int:
        """
        Policy interface matching RL agents.
        Accepts state vector of shape (state_dim,) or (1, state_dim).
        """
        arr = np.asarray(state, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        probs = self.clf.predict_proba(arr)[0]
        if deterministic:
            return int(np.argmax(probs))
        else:
            return int(np.random.choice(self.num_class, p=probs))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Batch action predictions."""
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Action probability distributions over all classes."""
        return self.clf.predict_proba(X)

    def save_model(self, file_path: Union[str, Path]):
        """Saves trained model to JSON format."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.clf.save_model(str(file_path))

    def load_model(self, file_path: Union[str, Path]):
        """Loads trained model from JSON format."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {file_path}")
        self.clf.load_model(str(file_path))
        self.is_fitted = True
