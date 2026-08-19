from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

CAL_FILE = (
    PROJECT_DIR
    / "baseline_results"
    / "calibration_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "calibration_analysis"
)

PATIENT_FILE = (
    OUTPUT_DIR
    / "patient_calibration_residuals.csv"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "calibration_heterogeneity_report.txt"
)

TARGET = "target_future"
PREDICTION = "prediction"


# ============================================================
# HELPER
# ============================================================

def header(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ============================================================
# STEP 1 - LOAD CALIBRATION PREDICTIONS
# ============================================================

header("STEP 1: LOAD CALIBRATION PREDICTIONS")

if not CAL_FILE.exists():
    raise FileNotFoundError(
        f"Calibration prediction file not found:\n{CAL_FILE}"
    )

cal = pd.read_csv(CAL_FILE)

print(f"Calibration samples : {len(cal)}")
print(f"Columns             : {len(cal.columns)}")

required = [
    "patient_id",
    "time",
    TARGET,
    PREDICTION,
]

for column in required:
    if column not in cal.columns:
        raise ValueError(
            f"Required column missing: {column}"
        )

print("Required columns : PASS")


# ============================================================
# STEP 2 - BASIC VALIDATION
# ============================================================

header("STEP 2: BASIC CALIBRATION VALIDATION")

print(
    f"Patients : {cal['patient_id'].nunique()}"
)

print(
    f"Time range : "
    f"{cal['time'].min()} - {cal['time'].max()}"
)

missing = cal[required].isna().sum()

print("\nMissing values:")

print(missing.to_string())

if missing.sum() != 0:
    raise ValueError(
        "Missing values detected in required columns."
    )

print("\nMissing-value check : PASS")


# ============================================================
# STEP 3 - VERIFY FIVE CALIBRATION OBSERVATIONS PER PATIENT
# ============================================================

header("STEP 3: PATIENT CALIBRATION COVERAGE")

patient_counts = (
    cal
    .groupby("patient_id")
    .size()
)

print(
    f"Minimum observations per patient : "
    f"{patient_counts.min()}"
)

print(
    f"Maximum observations per patient : "
    f"{patient_counts.max()}"
)

print(
    f"Mean observations per patient    : "
    f"{patient_counts.mean():.2f}"
)

if not (patient_counts == 5).all():
    print(
        "\nWARNING: Not every patient has exactly "
        "5 calibration observations."
    )
else:
    print(
        "\nExactly 5 calibration observations "
        "per patient : PASS"
    )


# ============================================================
# STEP 4 - CALCULATE CALIBRATION RESIDUALS
# ============================================================

header("STEP 4: CALCULATE CALIBRATION RESIDUALS")

cal = cal.copy()

cal["residual"] = (
    cal[TARGET] - cal[PREDICTION]
)

cal["absolute_residual"] = np.abs(
    cal["residual"]
)

cal["squared_residual"] = (
    cal["residual"] ** 2
)

print(
    "Residual calculation : PASS"
)

print(
    f"Mean absolute residual : "
    f"{cal['absolute_residual'].mean():.6f}"
)

print(
    f"Median absolute residual : "
    f"{cal['absolute_residual'].median():.6f}"
)

print(
    f"Maximum absolute residual : "
    f"{cal['absolute_residual'].max():.6f}"
)


# ============================================================
# STEP 5 - PATIENT-WISE CALIBRATION RESIDUAL SUMMARY
# ============================================================

header(
    "STEP 5: PATIENT-WISE CALIBRATION "
    "RESIDUAL SUMMARY"
)

def q50(x):
    return np.quantile(x, 0.50)

def q90(x):
    return np.quantile(x, 0.90)

def q95(x):
    return np.quantile(x, 0.95)


patient_summary = (
    cal
    .groupby("patient_id")
    .agg(
        n_calibration=("patient_id", "size"),

        mean_residual=(
            "residual",
            "mean"
        ),

        mean_absolute_residual=(
            "absolute_residual",
            "mean"
        ),

        rmse=(
            "squared_residual",
            lambda x: np.sqrt(np.mean(x))
        ),

        median_absolute_residual=(
            "absolute_residual",
            q50
        ),

        q90_absolute_residual=(
            "absolute_residual",
            q90
        ),

        q95_absolute_residual=(
            "absolute_residual",
            q95
        ),

        max_absolute_residual=(
            "absolute_residual",
            "max"
        ),
    )
    .reset_index()
)


print(
    f"Patients evaluated : "
    f"{len(patient_summary)}"
)


# ============================================================
# STEP 6 - DISTRIBUTION OF PATIENT ERROR
# ============================================================

header("STEP 6: PATIENT ERROR HETEROGENEITY")

metrics = [
    "mean_absolute_residual",
    "rmse",
    "median_absolute_residual",
    "q90_absolute_residual",
    "q95_absolute_residual",
    "max_absolute_residual",
]

for metric in metrics:

    print(f"\n{metric}")

    print(
        f"  Mean   : "
        f"{patient_summary[metric].mean():.6f}"
    )

    print(
        f"  Median : "
        f"{patient_summary[metric].median():.6f}"
    )

    print(
        f"  Min    : "
        f"{patient_summary[metric].min():.6f}"
    )

    print(
        f"  Max    : "
        f"{patient_summary[metric].max():.6f}"
    )


# ============================================================
# STEP 7 - PATIENT-SPECIFIC 90% RESIDUAL THRESHOLDS
# ============================================================

header(
    "STEP 7: PATIENT-SPECIFIC CALIBRATION "
    "QUANTILES"
)

global_q90 = np.quantile(
    cal["absolute_residual"],
    0.90
)

global_q95 = np.quantile(
    cal["absolute_residual"],
    0.95
)

print(
    f"Global calibration q90 : "
    f"{global_q90:.6f}"
)

print(
    f"Global calibration q95 : "
    f"{global_q95:.6f}"
)

print()

print(
    "Patient q90 range:"
)

print(
    f"Minimum : "
    f"{patient_summary['q90_absolute_residual'].min():.6f}"
)

print(
    f"Maximum : "
    f"{patient_summary['q90_absolute_residual'].max():.6f}"
)

print()

print(
    "Patient q95 range:"
)

print(
    f"Minimum : "
    f"{patient_summary['q95_absolute_residual'].min():.6f}"
)

print(
    f"Maximum : "
    f"{patient_summary['q95_absolute_residual'].max():.6f}"
)


# ============================================================
# STEP 8 - VARIABILITY RATIOS
# ============================================================

header("STEP 8: HETEROGENEITY RATIOS")

for metric in metrics:

    minimum = patient_summary[metric].min()
    maximum = patient_summary[metric].max()

    if minimum > 0:
        ratio = maximum / minimum
    else:
        ratio = np.inf

    print(
        f"{metric} max/min ratio : "
        f"{ratio:.4f}"
    )


# ============================================================
# STEP 9 - PATIENT RANKING
# ============================================================

header(
    "STEP 9: HIGHEST CALIBRATION ERROR PATIENTS"
)

worst = (
    patient_summary
    .sort_values(
        "mean_absolute_residual",
        ascending=False
    )
    .head(15)
)

display_columns = [
    "patient_id",
    "n_calibration",
    "mean_absolute_residual",
    "rmse",
    "q90_absolute_residual",
    "q95_absolute_residual",
    "max_absolute_residual",
]

print(
    worst[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# STEP 10 - LOWEST CALIBRATION ERROR PATIENTS
# ============================================================

header(
    "STEP 10: LOWEST CALIBRATION ERROR PATIENTS"
)

best = (
    patient_summary
    .sort_values(
        "mean_absolute_residual",
        ascending=True
    )
    .head(15)
)

print(
    best[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# STEP 11 - SAVE PATIENT SUMMARY
# ============================================================

header("STEP 11: SAVE CALIBRATION ANALYSIS")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

patient_summary.to_csv(
    PATIENT_FILE,
    index=False
)

print(
    f"Patient calibration summary saved to:\n"
    f"{PATIENT_FILE}"
)


# ============================================================
# STEP 12 - SAVE REPORT
# ============================================================

report = f"""
CALIBRATION RESIDUAL HETEROGENEITY AUDIT
========================================

Calibration samples:
{len(cal)}

Patients:
{len(patient_summary)}

Observations per patient:
{patient_counts.min()} to {patient_counts.max()}


GLOBAL RESIDUAL STATISTICS
--------------------------

Mean absolute residual:
{cal['absolute_residual'].mean():.10f}

Median absolute residual:
{cal['absolute_residual'].median():.10f}

Maximum absolute residual:
{cal['absolute_residual'].max():.10f}


GLOBAL QUANTILES
----------------

Global q90:
{global_q90:.10f}

Global q95:
{global_q95:.10f}


PATIENT q90
-----------

Minimum:
{patient_summary['q90_absolute_residual'].min():.10f}

Maximum:
{patient_summary['q90_absolute_residual'].max():.10f}


PATIENT q95
-----------

Minimum:
{patient_summary['q95_absolute_residual'].min():.10f}

Maximum:
{patient_summary['q95_absolute_residual'].max():.10f}


PATIENT ERROR HETEROGENEITY
---------------------------

Mean patient MAE:
{patient_summary['mean_absolute_residual'].mean():.10f}

Median patient MAE:
{patient_summary['mean_absolute_residual'].median():.10f}

Minimum patient MAE:
{patient_summary['mean_absolute_residual'].min():.10f}

Maximum patient MAE:
{patient_summary['mean_absolute_residual'].max():.10f}


HIGHEST ERROR PATIENTS
----------------------

{worst[display_columns].to_string(index=False)}


LOWEST ERROR PATIENTS
---------------------

{best[display_columns].to_string(index=False)}
"""

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

print(
    f"Report saved to:\n{REPORT_FILE}"
)


# ============================================================
# FINAL RESULT
# ============================================================

header("FINAL RESULT")

print(
    "CALIBRATION HETEROGENEITY AUDIT : SUCCESS"
)

print()

print(
    f"Patients analyzed : "
    f"{len(patient_summary)}"
)

print(
    f"Global q90 : "
    f"{global_q90:.6f}"
)

print(
    f"Patient q90 minimum : "
    f"{patient_summary['q90_absolute_residual'].min():.6f}"
)

print(
    f"Patient q90 maximum : "
    f"{patient_summary['q90_absolute_residual'].max():.6f}"
)

print()

print(
    "READY FOR PATIENT-ADAPTIVE CALIBRATION DESIGN"
)