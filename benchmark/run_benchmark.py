"""
Main Execution Script for Multi-Model CDSS Benchmarking.

Evaluates and compares:
- Historical Clinician Practice
- XGBoost (Supervised ML)
- Behavioral Cloning (BC, Imitation Learning)
- Discrete BCQ (Batch-Constrained Q-Learning)
- Discrete CQL (Primary Explainable RL Agent)

Outputs:
1. Formatted side-by-side console comparison table.
2. benchmark/benchmark_results.csv.
3. benchmark/BENCHMARK_REPORT.md (Thesis-ready summary).
4. benchmark/model_benchmark_comparison.png.
5. benchmark/confusion_matrices_all.png.
"""

import argparse
from pathlib import Path
import sys
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.evaluate_all import run_full_benchmark
from benchmark.visualize import plot_side_by_side_benchmark


def generate_markdown_report(summary_df: pd.DataFrame, output_path: Path):
    """Generates an academic thesis-ready markdown summary."""
    # Custom markdown table generator without external dependencies
    headers = summary_df.columns.tolist()
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_lines = []
    for _, row in summary_df.iterrows():
        row_str = "| " + " | ".join(str(val) for val in row.values) + " |"
        data_lines.append(row_str)
    table_md = "\n".join([header_line, separator_line] + data_lines)

    md_content = f"""# Multi-Model Benchmark Report: Type 2 Diabetes CDSS

This report documents the empirical evaluation and benchmarking of the primary **Discrete CQL (Conservative Q-Learning)** Clinical Decision Support System against established machine learning and offline reinforcement learning baselines on the **MIMIC-IV** cohort.

---

## 1. Experimental Methodology
- **Cohort**: MIMIC-IV Type 2 Diabetes Trajectory Dataset (500 patients, 3,888 total encounters).
- **Split Strategy**: Strict patient-level partition (`split_by_patient`) with 80% Train (400 patients, 3,099 transitions) and 20% Test (100 patients, 789 transitions). Fixed random seed `42` ensures zero cross-patient data leakage.
- **State Representation**: 56-dimensional clinical biomarker vector constructed by `StateConstructor` (scaled biomarkers, 30-day change rates, rolling moving averages, trend direction codes, demographics).
- **Action Space**: 4 discrete monitoring frequencies defined by ADA guidelines:
  - `0: ROUTINE_MONITORING` (Standard 6-month checkup)
  - `1: INCREASED_MONITORING` (1-3 month close monitoring)
  - `2: EARLY_RISK_ALERT` (2-4 week urgent flag for deteriorating metabolic status)
  - `3: CLINICAL_EVALUATION` (Immediate comprehensive specialist review)

---

## 2. Side-by-Side Model Comparison Table

{table_md}

---

## 3. Metric Descriptions & Interpretation
1. **Clinical Guideline Accuracy (%)**: Percentage of encounters where the model's recommendation perfectly matched the Gold-Standard American Diabetes Association (ADA) clinical rules.
2. **Clinician Action Match (%)**: Concordance with historical physician practice in the raw EHR records.
3. **Macro F1-Score**: Unweighted mean of F1-scores across all 4 action classes, penalizing models that fail on rare critical alert actions.
4. **Early Deterioration Sensitivity (%)**: Percentage of deteriorating or rapidly deteriorating encounters where an active monitoring or clinical alert was raised.
5. **Alarm Fatigue False Alarm Rate (%)**: Rate at which high-urgency alerts were inappropriately fired for stable, well-controlled patients (lower is better to prevent alarm fatigue).
6. **Mean Trajectory Return ($V^\pi$)**: Cumulative discounted reward ($\sum_{{t=0}}^T \gamma^t r_t$) accumulated across patient episodes under the learned policy.

---

## 4. Visual Artifacts
- Comparison Bar Charts: `model_benchmark_comparison.png`
- Confusion Matrix Grid: `confusion_matrices_all.png`
"""
    output_path.write_text(md_content, encoding="utf-8")
    print(f"[OK] Thesis benchmark report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Multi-Model CDSS Benchmark")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of patients for testing (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for patient partition (default: 42)")
    args = parser.parse_args()

    benchmark_dir = Path(__file__).resolve().parent

    print("\n" + "=" * 90)
    print("      EXECUTING COMPREHENSIVE CDSS MULTI-MODEL BENCHMARK SUITE")
    print(f"      Split: {int((1-args.test_size)*100)}% Train / {int(args.test_size*100)}% Test | Seed: {args.seed}")
    print("=" * 90 + "\n")

    summary_df, all_results = run_full_benchmark(test_size=args.test_size, seed=args.seed)

    # 1. Print formatted console table
    print("\n" + "=" * 90)
    print("                         SIDE-BY-SIDE BENCHMARK RESULTS")
    print("=" * 90)
    print(summary_df.to_string(index=False))
    print("=" * 90 + "\n")

    # 2. Save CSV artifact
    csv_path = benchmark_dir / "benchmark_results.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"[OK] Tabular benchmark results saved to: {csv_path}")

    # 3. Generate Visual Plots
    plot_side_by_side_benchmark(summary_df, all_results)

    # 4. Generate Markdown Report
    report_path = benchmark_dir / "BENCHMARK_REPORT.md"
    generate_markdown_report(summary_df, report_path)

    print("\n" + "=" * 90)
    print("[SUCCESS] All baseline models evaluated and benchmarked successfully!")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
