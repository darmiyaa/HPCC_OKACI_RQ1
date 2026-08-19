from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent

CAL_FILE = PROJECT_DIR / "baseline_results" / "calibration_predictions.csv"
TEST_FILE = PROJECT_DIR / "baseline_results" / "test_predictions.csv"
SCP_FILE = PROJECT_DIR / "conformal_results" / "scp_test_intervals.csv"

OUT_DIR = PROJECT_DIR / "stability_analysis"
OUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("CALIBRATION -> TEST STABILITY ANALYSIS")
print("=" * 70)

# ================================================================
# STEP 1: LOAD DATA
# ================================================================
print("\nSTEP 1: LOAD CALIBRATION AND TEST DATA")

cal = pd.read_csv(CAL_FILE)
test = pd.read_csv(TEST_FILE)
scp = pd.read_csv(SCP_FILE)

print(f"Calibration samples : {len(cal)}")
print(f"Test samples        : {len(test)}")
print(f"SCP samples         : {len(scp)}")

required_cal = {
    "patient_id",
    "time",
    "target_future",
    "prediction",
}

required_test = {
    "patient_id",
    "time",
    "target_future",
    "prediction",
}

required_scp = {
    "patient_id",
    "time",
    "target_future",
    "prediction",
}

for col in required_cal:
    if col not in cal.columns:
        raise AssertionError(f"Missing calibration column: {col}")

for col in required_test:
    if col not in test.columns:
        raise AssertionError(f"Missing test column: {col}")

for col in required_scp:
    if col not in scp.columns:
        raise AssertionError(f"Missing SCP column: {col}")

print("Required columns : PASS")


# ================================================================
# STEP 2: BASIC VALIDATION
# ================================================================
print("\nSTEP 2: BASIC VALIDATION")

for name, df in [
    ("CALIBRATION", cal),
    ("TEST", test),
    ("SCP", scp),
]:
    if df[["patient_id", "time"]].isnull().any().any():
        raise AssertionError(f"{name} contains missing patient/time values")

    if df.duplicated(["patient_id", "time"]).any():
        raise AssertionError(f"{name} contains duplicate patient-time samples")

print("Missing values : PASS")
print("Duplicate patient-time samples : PASS")


# ================================================================
# STEP 3: CALIBRATION RESIDUALS
# ================================================================
print("\nSTEP 3: CALCULATE CALIBRATION ERROR")

cal = cal.copy()
test = test.copy()
scp = scp.copy()

cal["absolute_error"] = (
    cal["target_future"] - cal["prediction"]
).abs()

cal["squared_error"] = (
    cal["target_future"] - cal["prediction"]
) ** 2

test["absolute_error"] = (
    test["target_future"] - test["prediction"]
).abs()

test["squared_error"] = (
    test["target_future"] - test["prediction"]
) ** 2

print("Calibration residual calculation : PASS")


# ================================================================
# STEP 4: PATIENT-WISE CALIBRATION SUMMARY
# ================================================================
print("\nSTEP 4: PATIENT-WISE CALIBRATION SUMMARY")

cal_summary = (
    cal.groupby("patient_id")
    .agg(
        n_calibration=("absolute_error", "size"),
        cal_mae=("absolute_error", "mean"),
        cal_rmse=("squared_error", lambda x: np.sqrt(np.mean(x))),
        cal_q90=("absolute_error", lambda x: np.quantile(x, 0.90)),
        cal_q95=("absolute_error", lambda x: np.quantile(x, 0.95)),
        cal_max_error=("absolute_error", "max"),
    )
    .reset_index()
)

print(f"Patients in calibration : {len(cal_summary)}")


# ================================================================
# STEP 5: PATIENT-WISE TEST ERROR
# ================================================================
print("\nSTEP 5: PATIENT-WISE TEST ERROR")

test_summary = (
    test.groupby("patient_id")
    .agg(
        n_test=("absolute_error", "size"),
        test_mae=("absolute_error", "mean"),
        test_rmse=("squared_error", lambda x: np.sqrt(np.mean(x))),
        test_max_error=("absolute_error", "max"),
    )
    .reset_index()
)

print(f"Patients in test : {len(test_summary)}")


# ================================================================
# STEP 6: TEST COVERAGE
# ================================================================
print("\nSTEP 6: PATIENT-WISE TEST COVERAGE")

scp = scp.copy()

if "covered" not in scp.columns:
    lower_candidates = [
        "lower",
        "lower_bound",
        "prediction_lower",
        "interval_lower",
    ]

    upper_candidates = [
        "upper",
        "upper_bound",
        "prediction_upper",
        "interval_upper",
    ]

    lower_col = next(
        (c for c in lower_candidates if c in scp.columns),
        None,
    )

    upper_col = next(
        (c for c in upper_candidates if c in scp.columns),
        None,
    )

    if lower_col is None or upper_col is None:
        raise AssertionError(
            "Could not identify SCP lower/upper interval columns."
        )

    scp["covered"] = (
        (scp["target_future"] >= scp[lower_col])
        & (scp["target_future"] <= scp[upper_col])
    )

coverage_summary = (
    scp.groupby("patient_id")
    .agg(
        n_coverage=("covered", "size"),
        test_coverage=("covered", "mean"),
    )
    .reset_index()
)

print(f"Patients with coverage information : {len(coverage_summary)}")


# ================================================================
# STEP 7: MERGE PATIENT INFORMATION
# ================================================================
print("\nSTEP 7: MERGE CALIBRATION AND TEST INFORMATION")

summary = cal_summary.merge(
    test_summary,
    on="patient_id",
    how="inner",
)

summary = summary.merge(
    coverage_summary,
    on="patient_id",
    how="inner",
)

if len(summary) != 100:
    raise AssertionError(
        f"Expected 100 patients after merge, found {len(summary)}"
    )

print(f"Patients in final analysis : {len(summary)}")
print("Patient alignment : PASS")


# ================================================================
# STEP 8: CORRELATIONS
# ================================================================
print("\nSTEP 8: CALCULATE CALIBRATION -> TEST RELATIONSHIPS")

correlations = {
    "cal_MAE_vs_test_MAE":
        summary["cal_mae"].corr(summary["test_mae"]),

    "cal_RMSE_vs_test_RMSE":
        summary["cal_rmse"].corr(summary["test_rmse"]),

    "cal_q90_vs_test_MAE":
        summary["cal_q90"].corr(summary["test_mae"]),

    "cal_q95_vs_test_MAE":
        summary["cal_q95"].corr(summary["test_mae"]),

    "cal_q90_vs_test_coverage":
        summary["cal_q90"].corr(summary["test_coverage"]),

    "cal_MAE_vs_test_coverage":
        summary["cal_mae"].corr(summary["test_coverage"]),
}

for name, value in correlations.items():
    print(f"{name:35s}: {value:+.6f}")


# ================================================================
# STEP 9: CALIBRATION ERROR HETEROGENEITY
# ================================================================
print("\nSTEP 9: CALIBRATION / TEST HETEROGENEITY")

print(
    f"Calibration MAE range : "
    f"{summary['cal_mae'].min():.6f} - "
    f"{summary['cal_mae'].max():.6f}"
)

print(
    f"Test MAE range        : "
    f"{summary['test_mae'].min():.6f} - "
    f"{summary['test_mae'].max():.6f}"
)

print(
    f"Calibration q90 range : "
    f"{summary['cal_q90'].min():.6f} - "
    f"{summary['cal_q90'].max():.6f}"
)

print(
    f"Test coverage range   : "
    f"{summary['test_coverage'].min():.6f} - "
    f"{summary['test_coverage'].max():.6f}"
)


# ================================================================
# STEP 10: TOP DIFFICULT PATIENTS
# ================================================================
print("\nSTEP 10: HIGHEST CALIBRATION ERROR PATIENTS")

top_cal = summary.sort_values(
    "cal_mae",
    ascending=False
).head(15)

print(
    top_cal[
        [
            "patient_id",
            "cal_mae",
            "test_mae",
            "cal_q90",
            "test_coverage",
        ]
    ].to_string(index=False)
)


# ================================================================
# STEP 11: SAVE PATIENT STABILITY TABLE
# ================================================================
print("\nSTEP 11: SAVE PATIENT STABILITY TABLE")

patient_file = OUT_DIR / "patient_calibration_test_stability.csv"

summary.to_csv(
    patient_file,
    index=False,
)

print(f"Saved to:\n{patient_file}")


# ================================================================
# STEP 12: SAVE CORRELATION REPORT
# ================================================================
print("\nSTEP 12: SAVE STABILITY REPORT")

report_file = OUT_DIR / "calibration_test_stability_report.txt"

with open(report_file, "w", encoding="utf-8") as f:

    f.write("CALIBRATION -> TEST STABILITY ANALYSIS\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Patients analyzed : {len(summary)}\n")
    f.write(f"Calibration samples : {len(cal)}\n")
    f.write(f"Test samples : {len(test)}\n\n")

    f.write("CORRELATIONS\n")
    f.write("-" * 70 + "\n")

    for name, value in correlations.items():
        f.write(f"{name}: {value:.6f}\n")

    f.write("\nRANGES\n")
    f.write("-" * 70 + "\n")

    f.write(
        f"Calibration MAE: "
        f"{summary['cal_mae'].min():.6f} - "
        f"{summary['cal_mae'].max():.6f}\n"
    )

    f.write(
        f"Test MAE: "
        f"{summary['test_mae'].min():.6f} - "
        f"{summary['test_mae'].max():.6f}\n"
    )

    f.write(
        f"Calibration q90: "
        f"{summary['cal_q90'].min():.6f} - "
        f"{summary['cal_q90'].max():.6f}\n"
    )

    f.write(
        f"Test coverage: "
        f"{summary['test_coverage'].min():.6f} - "
        f"{summary['test_coverage'].max():.6f}\n"
    )

print(f"Report saved to:\n{report_file}")


# ================================================================
# FINAL RESULT
# ================================================================
print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)

print("CALIBRATION -> TEST STABILITY ANALYSIS : SUCCESS")

print(f"\nPatients analyzed : {len(summary)}")

print(
    f"Calibration MAE -> Test MAE correlation : "
    f"{correlations['cal_MAE_vs_test_MAE']:+.6f}"
)

print(
    f"Calibration q90 -> Test MAE correlation : "
    f"{correlations['cal_q90_vs_test_MAE']:+.6f}"
)

print(
    f"Calibration q90 -> Test coverage correlation : "
    f"{correlations['cal_q90_vs_test_coverage']:+.6f}"
)

print("\nREADY TO DECIDE PATIENT-ADAPTIVE CALIBRATION DESIGN")