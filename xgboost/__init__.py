"""
XGBoost baseline package for Type 2 Diabetes CDSS.

Transparently integrates the standard xgboost C++ library while exposing
the local XGBoostAgent wrapper.
"""

from pathlib import Path
import sys
import importlib.util

# Ensure root directory and parent are preserved
_this_dir = Path(__file__).resolve().parent

# Find site-packages xgboost if top-level import occurs
_site_xgboost = None
for p in sys.path:
    cand = Path(p) / "xgboost" / "__init__.py"
    if cand.exists() and str(cand.parent.resolve()) != str(_this_dir):
        spec = importlib.util.spec_from_file_location("_real_xgboost", cand)
        if spec and spec.loader:
            _site_xgboost = importlib.util.module_from_spec(spec)
            sys.modules["_real_xgboost"] = _site_xgboost
            spec.loader.exec_module(_site_xgboost)
            for k, v in _site_xgboost.__dict__.items():
                if not k.startswith("__"):
                    globals()[k] = v
            break

from .model import XGBoostAgent

__all__ = ["XGBoostAgent"]
