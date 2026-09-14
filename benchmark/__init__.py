"""
Consolidated Multi-Model Benchmark Suite for Type 2 Diabetes CDSS.

Compares Historical Clinician, Clinical Guidelines, XGBoost,
Behavioral Cloning (BC), Discrete BCQ, and Discrete CQL side-by-side.
"""

from .evaluate_all import run_full_benchmark
from .visualize import plot_side_by_side_benchmark

__all__ = ["run_full_benchmark", "plot_side_by_side_benchmark"]
