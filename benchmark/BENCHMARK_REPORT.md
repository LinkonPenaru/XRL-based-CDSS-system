# Multi-Model Benchmark Report: Type 2 Diabetes CDSS

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

| Model | Category | Guideline Acc (%) | Clinician Match (%) | Macro F1 | Sensitivity (%) | False Alarm (%) | Mean Return (V) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Historical Clinician | Empirical Baseline | 58.81 | 100.0 | 0.6724 | 100.0 | 0.0 | 13.55 |
| XGBoost | Supervised ML | 56.65 | 72.62 | 0.6055 | 99.06 | 0.0 | 18.07 |
| Behavioral Cloning (BC) | Imitation Learning | 57.41 | 70.09 | 0.5955 | 96.71 | 1.07 | 18.18 |
| Discrete BCQ | Offline RL (Constraint) | 52.34 | 61.98 | 0.475 | 95.77 | 11.81 | 16.17 |
| Discrete CQL (Ours) | Offline RL (Conservative Q) | 53.49 | 62.48 | 0.4921 | 95.31 | 12.34 | 16.02 |

---

## 3. Metric Descriptions & Interpretation
1. **Clinical Guideline Accuracy (%)**: Percentage of encounters where the model's recommendation perfectly matched the Gold-Standard American Diabetes Association (ADA) clinical rules.
2. **Clinician Action Match (%)**: Concordance with historical physician practice in the raw EHR records.
3. **Macro F1-Score**: Unweighted mean of F1-scores across all 4 action classes, penalizing models that fail on rare critical alert actions.
4. **Early Deterioration Sensitivity (%)**: Percentage of deteriorating or rapidly deteriorating encounters where an active monitoring or clinical alert was raised.
5. **Alarm Fatigue False Alarm Rate (%)**: Rate at which high-urgency alerts were inappropriately fired for stable, well-controlled patients (lower is better to prevent alarm fatigue).
6. **Mean Trajectory Return ($V^\pi$)**: Cumulative discounted reward ($\sum_{t=0}^T \gamma^t r_t$) accumulated across patient episodes under the learned policy.

---

## 4. Visual Artifacts
- Comparison Bar Charts: `model_benchmark_comparison.png`
- Confusion Matrix Grid: `confusion_matrices_all.png`
