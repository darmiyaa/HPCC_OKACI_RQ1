from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

SCP_FILE = (
    PROJECT_DIR
    / "conformal_results"
    / "scp_test_intervals.csv"
)

ACI_FILE = (
    PROJECT_DIR
    / "aci_results"
    / "aci_test_intervals.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "patient_analysis"
)

PATIENT_FILE = (
    OUTPUT_DIR
    / "patient_error_coverage.csv"
)

TIME_FILE = (
    OUTPUT_DIR
    / "patient_time_coverage.csv"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "patient_coverage_report.txt"
)

TARGET = "target_future"


# ============================================================
# HELPER
# ============================================================

def header(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ============================================================
# STEP 1 - LOAD RESULTS
# ============================================================

header("STEP 1: LOAD SCP AND ACI RESULTS")

if not SCP_FILE.exists():
    raise FileNotFoundError(
        f"SCP result not found:\n{SCP_FILE}"
    )

if not ACI_FILE.exists():
    raise FileNotFoundError(
        f"ACI result not found:\n{ACI_FILE}"
    )

scp = pd.read_csv(SCP_FILE)
aci = pd.read_csv(ACI_FILE)

print(
    f"SCP samples : {len(scp)}"
)

print(
    f"ACI samples : {len(aci)}"
)


# ============================================================
# STEP 2 - VERIFY SAMPLE ALIGNMENT
# ============================================================

header("STEP 2: VERIFY SCP / ACI ALIGNMENT")

key_columns = [
    "patient_id",
    "time",
    TARGET,
    "prediction",
]

for column in key_columns:

    if column not in scp.columns:
        raise ValueError(
            f"SCP missing column: {column}"
        )

    if column not in aci.columns:
        raise ValueError(
            f"ACI missing column: {column}"
        )


scp_keys = set(
    zip(
        scp["patient_id"],
        scp["time"]
    )
)

aci_keys = set(
    zip(
        aci["patient_id"],
        aci["time"]
    )
)

if scp_keys != aci_keys:
    raise AssertionError(
        "SCP and ACI patient/time keys do not match."
    )

# Align both results to the same patient/time order
# before comparing them.

key_columns = [
    "patient_id",
    "time"
]

scp = scp.sort_values(
    key_columns
).reset_index(drop=True)

aci = aci.sort_values(
    key_columns
).reset_index(drop=True)

print(
    "Patient/time key set : PASS"
)

print(
    "SCP/ACI chronological alignment : PASS"
)

if not np.allclose(
    scp[TARGET].to_numpy(float),
    aci[TARGET].to_numpy(float)
):
    raise AssertionError(
        "SCP and ACI targets differ."
    )

if not np.allclose(
    scp["prediction"].to_numpy(float),
    aci["prediction"].to_numpy(float)
):
    raise AssertionError(
        "SCP and ACI predictions differ."
    )

print(
    "Patient/time alignment : PASS"
)

print(
    "Target alignment       : PASS"
)

print(
    "Prediction alignment   : PASS"
)


# ============================================================
# STEP 3 - CALCULATE COMMON PREDICTION ERROR
# ============================================================

header("STEP 3: CALCULATE PATIENT-WISE PREDICTION ERROR")

analysis = scp[
    [
        "patient_id",
        "time",
        TARGET,
        "prediction",
        "covered",
        "interval_width",
    ]
].copy()

analysis["absolute_error"] = np.abs(
    analysis[TARGET]
    - analysis["prediction"]
)

analysis["squared_error"] = (
    analysis[TARGET]
    - analysis["prediction"]
) ** 2

analysis["aci_covered"] = (
    aci["covered"].to_numpy()
)

analysis["aci_width"] = (
    aci["interval_width"].to_numpy()
)

analysis["q_used"] = (
    aci["q_used"].to_numpy()
)


# ============================================================
# STEP 4 - PATIENT-WISE SUMMARY
# ============================================================

header("STEP 4: CALCULATE PATIENT-WISE SUMMARY")

patient_summary = (
    analysis
    .groupby("patient_id")
    .agg(
        n_test=("patient_id", "size"),

        mean_absolute_error=(
            "absolute_error",
            "mean"
        ),

        rmse=(
            "squared_error",
            lambda x: np.sqrt(np.mean(x))
        ),

        max_absolute_error=(
            "absolute_error",
            "max"
        ),

        mean_prediction=(
            "prediction",
            "mean"
        ),

        mean_target=(
            TARGET,
            "mean"
        ),

        scp_covered=(
            "covered",
            "sum"
        ),

        scp_coverage=(
            "covered",
            "mean"
        ),

        aci_covered=(
            "aci_covered",
            "sum"
        ),

        aci_coverage=(
            "aci_covered",
            "mean"
        ),

        scp_mean_width=(
            "interval_width",
            "mean"
        ),

        aci_mean_width=(
            "aci_width",
            "mean"
        ),
    )
    .reset_index()
)


patient_summary["coverage_difference"] = (
    patient_summary["aci_coverage"]
    - patient_summary["scp_coverage"]
)

patient_summary["scp_missed"] = (
    patient_summary["n_test"]
    - patient_summary["scp_covered"]
)

patient_summary["aci_missed"] = (
    patient_summary["n_test"]
    - patient_summary["aci_covered"]
)


print(
    f"Patients evaluated : "
    f"{len(patient_summary)}"
)

print()

print(
    "SCP coverage statistics:"
)

print(
    f"Mean   : "
    f"{patient_summary['scp_coverage'].mean():.4f}"
)

print(
    f"Median : "
    f"{patient_summary['scp_coverage'].median():.4f}"
)

print(
    f"Minimum: "
    f"{patient_summary['scp_coverage'].min():.4f}"
)

print(
    f"Maximum: "
    f"{patient_summary['scp_coverage'].max():.4f}"
)

print()

print(
    "ACI coverage statistics:"
)

print(
    f"Mean   : "
    f"{patient_summary['aci_coverage'].mean():.4f}"
)

print(
    f"Median : "
    f"{patient_summary['aci_coverage'].median():.4f}"
)

print(
    f"Minimum: "
    f"{patient_summary['aci_coverage'].min():.4f}"
)

print(
    f"Maximum: "
    f"{patient_summary['aci_coverage'].max():.4f}"
)


# ============================================================
# STEP 5 - IDENTIFY LOW-COVERAGE PATIENTS
# ============================================================

header("STEP 5: IDENTIFY LOW-COVERAGE PATIENTS")

low_scp = patient_summary[
    patient_summary["scp_coverage"] < 0.90
].copy()

low_aci = patient_summary[
    patient_summary["aci_coverage"] < 0.90
].copy()

print(
    f"SCP patients below 90% : "
    f"{len(low_scp)}"
)

print(
    f"ACI patients below 90% : "
    f"{len(low_aci)}"
)


# ============================================================
# STEP 6 - PATIENT-WISE ERROR DISTRIBUTION
# ============================================================

header("STEP 6: ERROR HETEROGENEITY")

print(
    "Patient MAE statistics:"
)

print(
    f"Mean   : "
    f"{patient_summary['mean_absolute_error'].mean():.6f}"
)

print(
    f"Median : "
    f"{patient_summary['mean_absolute_error'].median():.6f}"
)

print(
    f"Minimum: "
    f"{patient_summary['mean_absolute_error'].min():.6f}"
)

print(
    f"Maximum: "
    f"{patient_summary['mean_absolute_error'].max():.6f}"
)

print()

print(
    "Patient RMSE statistics:"
)

print(
    f"Mean   : "
    f"{patient_summary['rmse'].mean():.6f}"
)

print(
    f"Median : "
    f"{patient_summary['rmse'].median():.6f}"
)

print(
    f"Minimum: "
    f"{patient_summary['rmse'].min():.6f}"
)

print(
    f"Maximum: "
    f"{patient_summary['rmse'].max():.6f}"
)


# ============================================================
# STEP 7 - TIME × PATIENT COVERAGE
# ============================================================

header("STEP 7: TIME-WISE PATIENT COVERAGE")

time_patient = (
    analysis
    .groupby(
        [
            "patient_id",
            "time"
        ]
    )
    .agg(
        target=(TARGET, "first"),
        prediction=("prediction", "first"),
        absolute_error=("absolute_error", "first"),
        scp_covered=("covered", "first"),
        aci_covered=("aci_covered", "first"),
        scp_width=("interval_width", "first"),
        aci_width=("aci_width", "first"),
        q_used=("q_used", "first"),
    )
    .reset_index()
)

print(
    "Time-wise aggregate coverage:"
)

time_summary = (
    time_patient
    .groupby("time")
    .agg(
        n=("patient_id", "size"),

        scp_coverage=(
            "scp_covered",
            "mean"
        ),

        aci_coverage=(
            "aci_covered",
            "mean"
        ),

        mean_error=(
            "absolute_error",
            "mean"
        ),

        mean_scp_width=(
            "scp_width",
            "mean"
        ),

        mean_aci_width=(
            "aci_width",
            "mean"
        ),
    )
    .reset_index()
)

print(
    time_summary.to_string(
        index=False
    )
)


# ============================================================
# STEP 8 - COVERAGE VS ERROR RELATIONSHIP
# ============================================================

header("STEP 8: COVERAGE / ERROR RELATIONSHIP")

coverage_error_correlation = (
    patient_summary[
        [
            "scp_coverage",
            "mean_absolute_error"
        ]
    ]
    .corr()
    .iloc[0, 1]
)

print(
    "Correlation between SCP patient coverage"
)

print(
    "and patient MAE:"
)

print(
    f"{coverage_error_correlation:.6f}"
)


# ============================================================
# STEP 9 - WORST COVERED PATIENTS
# ============================================================

header("STEP 9: LOWEST SCP COVERAGE PATIENTS")

worst = (
    patient_summary
    .sort_values(
        [
            "scp_coverage",
            "mean_absolute_error"
        ],
        ascending=[
            True,
            False
        ]
    )
    .head(15)
)

display_columns = [
    "patient_id",
    "n_test",
    "scp_coverage",
    "aci_coverage",
    "mean_absolute_error",
    "rmse",
    "max_absolute_error",
    "scp_mean_width",
    "aci_mean_width",
]

print(
    worst[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# STEP 10 - BEST COVERED PATIENTS
# ============================================================

header("STEP 10: HIGHEST SCP COVERAGE PATIENTS")

best = (
    patient_summary
    .sort_values(
        [
            "scp_coverage",
            "mean_absolute_error"
        ],
        ascending=[
            False,
            True
        ]
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
# STEP 11 - SAVE RESULTS
# ============================================================

header("STEP 11: SAVE PATIENT ANALYSIS")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

patient_summary.to_csv(
    PATIENT_FILE,
    index=False
)

time_summary.to_csv(
    TIME_FILE,
    index=False
)


# ============================================================
# STEP 12 - SAVE REPORT
# ============================================================

report = f"""
PATIENT-WISE CONFORMAL COVERAGE ANALYSIS
========================================

Test samples:
{len(analysis)}

Patients:
{len(patient_summary)}

SCP
---

Mean patient coverage:
{patient_summary['scp_coverage'].mean():.10f}

Median patient coverage:
{patient_summary['scp_coverage'].median():.10f}

Minimum patient coverage:
{patient_summary['scp_coverage'].min():.10f}

Maximum patient coverage:
{patient_summary['scp_coverage'].max():.10f}

Patients below 90%:
{len(low_scp)}


ACI
---

Mean patient coverage:
{patient_summary['aci_coverage'].mean():.10f}

Median patient coverage:
{patient_summary['aci_coverage'].median():.10f}

Minimum patient coverage:
{patient_summary['aci_coverage'].min():.10f}

Maximum patient coverage:
{patient_summary['aci_coverage'].max():.10f}

Patients below 90%:
{len(low_aci)}


ERROR HETEROGENEITY
-------------------

Mean patient MAE:
{patient_summary['mean_absolute_error'].mean():.10f}

Median patient MAE:
{patient_summary['mean_absolute_error'].median():.10f}

Minimum patient MAE:
{patient_summary['mean_absolute_error'].min():.10f}

Maximum patient MAE:
{patient_summary['mean_absolute_error'].max():.10f}

Mean patient RMSE:
{patient_summary['rmse'].mean():.10f}

Median patient RMSE:
{patient_summary['rmse'].median():.10f}

Minimum patient RMSE:
{patient_summary['rmse'].min():.10f}

Maximum patient RMSE:
{patient_summary['rmse'].max():.10f}


COVERAGE / ERROR CORRELATION
----------------------------

Correlation:
{coverage_error_correlation:.10f}


TIME-WISE SUMMARY
-----------------

{time_summary.to_string(index=False)}


LOWEST COVERAGE PATIENTS
------------------------

{worst[display_columns].to_string(index=False)}
"""


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(report)


print(
    f"Patient summary saved to:\n{PATIENT_FILE}"
)

print(
    f"Time summary saved to:\n{TIME_FILE}"
)

print(
    f"Report saved to:\n{REPORT_FILE}"
)


# ============================================================
# FINAL RESULT
# ============================================================

header("FINAL RESULT")

print(
    "PATIENT COVERAGE ANALYSIS : SUCCESS"
)

print()
print(
    f"Patients analyzed : "
    f"{len(patient_summary)}"
)

print(
    f"SCP mean coverage : "
    f"{patient_summary['scp_coverage'].mean():.2%}"
)

print(
    f"SCP minimum coverage : "
    f"{patient_summary['scp_coverage'].min():.2%}"
)

print(
    f"ACI mean coverage : "
    f"{patient_summary['aci_coverage'].mean():.2%}"
)

print(
    f"ACI minimum coverage : "
    f"{patient_summary['aci_coverage'].min():.2%}"
)

print()
print(
    "READY FOR PATIENT-ADAPTIVE DESIGN"
)