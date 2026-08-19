from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

SPLIT_DIR = PROJECT_DIR / "temporal_split"
BASELINE_DIR = PROJECT_DIR / "baseline_results"
OUTPUT_DIR = PROJECT_DIR / "conformal_results"

CAL_FILE = SPLIT_DIR / "calibration.csv"
TEST_FILE = SPLIT_DIR / "test.csv"

CAL_PRED_FILE = BASELINE_DIR / "calibration_predictions.csv"
TEST_PRED_FILE = BASELINE_DIR / "test_predictions.csv"

INTERVAL_FILE = OUTPUT_DIR / "scp_test_intervals.csv"
METRICS_FILE = OUTPUT_DIR / "scp_metrics.txt"

TARGET = "target_future"

ALPHA = 0.10
NOMINAL_COVERAGE = 1.0 - ALPHA


# ============================================================
# HELPER
# ============================================================

def header(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ============================================================
# STEP 1 - LOAD CALIBRATION AND TEST DATA
# ============================================================

header("STEP 1: LOAD CALIBRATION AND TEST DATA")

if not CAL_FILE.exists():
    raise FileNotFoundError(
        f"Calibration file not found:\n{CAL_FILE}"
    )

if not TEST_FILE.exists():
    raise FileNotFoundError(
        f"Test file not found:\n{TEST_FILE}"
    )

if not CAL_PRED_FILE.exists():
    raise FileNotFoundError(
        f"Calibration predictions not found:\n{CAL_PRED_FILE}"
    )

if not TEST_PRED_FILE.exists():
    raise FileNotFoundError(
        f"Test predictions not found:\n{TEST_PRED_FILE}"
    )


calibration = pd.read_csv(CAL_FILE)
test = pd.read_csv(TEST_FILE)

cal_pred = pd.read_csv(CAL_PRED_FILE)
test_pred = pd.read_csv(TEST_PRED_FILE)


print(f"Calibration samples : {len(calibration)}")
print(f"Test samples        : {len(test)}")


# ============================================================
# STEP 2 - VERIFY ALIGNMENT
# ============================================================

header("STEP 2: VERIFY PREDICTION ALIGNMENT")

required_columns = [
    "patient_id",
    "time",
    TARGET,
    "prediction",
]

for column in required_columns:

    if column not in cal_pred.columns:
        raise ValueError(
            f"Missing calibration prediction column: {column}"
        )

    if column not in test_pred.columns:
        raise ValueError(
            f"Missing test prediction column: {column}"
        )


if len(calibration) != len(cal_pred):
    raise AssertionError(
        "Calibration data and predictions have different lengths."
    )

if len(test) != len(test_pred):
    raise AssertionError(
        "Test data and predictions have different lengths."
    )


# Verify patient/time ordering.

cal_keys_data = list(
    zip(
        calibration["patient_id"],
        calibration["time"]
    )
)

cal_keys_pred = list(
    zip(
        cal_pred["patient_id"],
        cal_pred["time"]
    )
)

test_keys_data = list(
    zip(
        test["patient_id"],
        test["time"]
    )
)

test_keys_pred = list(
    zip(
        test_pred["patient_id"],
        test_pred["time"]
    )
)


if cal_keys_data != cal_keys_pred:
    raise AssertionError(
        "Calibration patient/time ordering does not match predictions."
    )

if test_keys_data != test_keys_pred:
    raise AssertionError(
        "Test patient/time ordering does not match predictions."
    )


print("Calibration alignment : PASS")
print("Test alignment        : PASS")


# ============================================================
# STEP 3 - CALIBRATION NONCONFORMITY SCORES
# ============================================================

header("STEP 3: CALCULATE CALIBRATION SCORES")

y_cal = cal_pred[TARGET].to_numpy(
    dtype=float
)

pred_cal = cal_pred["prediction"].to_numpy(
    dtype=float
)

cal_scores = np.abs(
    y_cal - pred_cal
)

n_cal = len(cal_scores)

print(
    f"Number of calibration scores : {n_cal}"
)

print(
    f"Minimum score : {cal_scores.min():.6f}"
)

print(
    f"Median score  : {np.median(cal_scores):.6f}"
)

print(
    f"Maximum score : {cal_scores.max():.6f}"
)


# ============================================================
# STEP 4 - FINITE-SAMPLE CONFORMAL QUANTILE
# ============================================================

header("STEP 4: CALCULATE FINITE-SAMPLE CONFORMAL QUANTILE")

# Split conformal finite-sample quantile:
#
# k = ceil((n + 1) * (1 - alpha))
#
# q is the k-th smallest calibration score.
#
# With n=500 and alpha=0.10:
#
# k = ceil(501 * 0.90) = 451

k = int(
    np.ceil(
        (n_cal + 1) * NOMINAL_COVERAGE
    )
)

if k > n_cal:
    k = n_cal

sorted_scores = np.sort(
    cal_scores
)

q = sorted_scores[k - 1]

print(
    f"Nominal coverage : {NOMINAL_COVERAGE:.2%}"
)

print(
    f"Alpha            : {ALPHA:.2f}"
)

print(
    f"Calibration n    : {n_cal}"
)

print(
    f"Quantile rank k  : {k}"
)

print(
    f"Conformal q      : {q:.6f}"
)


# ============================================================
# STEP 5 - CONSTRUCT TEST INTERVALS
# ============================================================

header("STEP 5: CONSTRUCT TEST PREDICTION INTERVALS")

y_test = test_pred[TARGET].to_numpy(
    dtype=float
)

pred_test = test_pred["prediction"].to_numpy(
    dtype=float
)

lower = pred_test - q
upper = pred_test + q

interval_width = upper - lower


# ============================================================
# STEP 6 - COVERAGE
# ============================================================

header("STEP 6: CALCULATE TEST COVERAGE")

covered = (
    (y_test >= lower) &
    (y_test <= upper)
)

n_test = len(y_test)

n_covered = int(
    covered.sum()
)

n_missed = int(
    n_test - n_covered
)

empirical_coverage = (
    n_covered / n_test
)

coverage_error = (
    empirical_coverage
    - NOMINAL_COVERAGE
)

mean_width = float(
    np.mean(interval_width)
)

median_width = float(
    np.median(interval_width)
)

min_width = float(
    np.min(interval_width)
)

max_width = float(
    np.max(interval_width)
)


print(
    f"Test samples       : {n_test}"
)

print(
    f"Covered             : {n_covered}"
)

print(
    f"Missed              : {n_missed}"
)

print(
    f"Empirical coverage  : {empirical_coverage:.4f}"
)

print(
    f"Nominal coverage    : {NOMINAL_COVERAGE:.4f}"
)

print(
    f"Coverage error      : {coverage_error:+.4f}"
)

print(
    f"Mean interval width : {mean_width:.6f}"
)

print(
    f"Median width        : {median_width:.6f}"
)

print(
    f"Minimum width       : {min_width:.6f}"
)

print(
    f"Maximum width       : {max_width:.6f}"
)


# ============================================================
# STEP 7 - PATIENT-WISE COVERAGE
# ============================================================

header("STEP 7: PATIENT-WISE COVERAGE")

patient_results = []

test_ids = test_pred["patient_id"].to_numpy()

for patient_id in sorted(
    pd.unique(test_ids)
):

    mask = (
        test_ids == patient_id
    )

    patient_n = int(
        mask.sum()
    )

    patient_covered = int(
        covered[mask].sum()
    )

    patient_coverage = (
        patient_covered / patient_n
    )

    patient_results.append(
        {
            "patient_id": patient_id,
            "n_test": patient_n,
            "covered": patient_covered,
            "coverage": patient_coverage,
        }
    )


patient_df = pd.DataFrame(
    patient_results
)

print(
    f"Patients evaluated : {len(patient_df)}"
)

print(
    f"Mean patient coverage : "
    f"{patient_df['coverage'].mean():.4f}"
)

print(
    f"Minimum patient coverage : "
    f"{patient_df['coverage'].min():.4f}"
)

print(
    f"Maximum patient coverage : "
    f"{patient_df['coverage'].max():.4f}"
)


# ============================================================
# STEP 8 - TIME-WISE COVERAGE
# ============================================================

header("STEP 8: TIME-WISE COVERAGE")

time_results = []

test_times = test_pred["time"].to_numpy()

for time_value in sorted(
    pd.unique(test_times)
):

    mask = (
        test_times == time_value
    )

    time_n = int(
        mask.sum()
    )

    time_covered = int(
        covered[mask].sum()
    )

    time_coverage = (
        time_covered / time_n
    )

    time_results.append(
        {
            "time": time_value,
            "n_test": time_n,
            "covered": time_covered,
            "coverage": time_coverage,
        }
    )


time_df = pd.DataFrame(
    time_results
)

print(
    time_df.to_string(
        index=False
    )
)


# ============================================================
# STEP 9 - SAVE TEST INTERVALS
# ============================================================

header("STEP 9: SAVE TEST INTERVALS")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

interval_df = test_pred[
    [
        "patient_id",
        "time",
        TARGET,
        "prediction",
    ]
].copy()

interval_df["lower"] = lower

interval_df["upper"] = upper

interval_df["interval_width"] = interval_width

interval_df["covered"] = covered

interval_df.to_csv(
    INTERVAL_FILE,
    index=False
)

print(
    f"Intervals saved to:\n{INTERVAL_FILE}"
)


# ============================================================
# STEP 10 - SAVE METRICS
# ============================================================

header("STEP 10: SAVE SCP METRICS")

metrics_text = f"""
RQ1 SPLIT CONFORMAL PREDICTION
==============================

Nominal coverage:
{NOMINAL_COVERAGE:.4f}

Alpha:
{ALPHA:.4f}

Calibration samples:
{n_cal}

Finite-sample quantile rank:
{k}

Conformal quantile q:
{q:.10f}


TEST RESULTS
------------

Test samples:
{n_test}

Covered:
{n_covered}

Missed:
{n_missed}

Empirical coverage:
{empirical_coverage:.10f}

Coverage error:
{coverage_error:.10f}

Mean interval width:
{mean_width:.10f}

Median interval width:
{median_width:.10f}

Minimum interval width:
{min_width:.10f}

Maximum interval width:
{max_width:.10f}


PATIENT-WISE COVERAGE
---------------------

Number of patients:
{len(patient_df)}

Mean patient coverage:
{patient_df['coverage'].mean():.10f}

Minimum patient coverage:
{patient_df['coverage'].min():.10f}

Maximum patient coverage:
{patient_df['coverage'].max():.10f}


TIME-WISE COVERAGE
------------------

{time_df.to_string(index=False)}
"""


with open(
    METRICS_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(metrics_text)


print(
    f"Metrics saved to:\n{METRICS_FILE}"
)


# ============================================================
# FINAL RESULT
# ============================================================

header("FINAL RESULT")

print(
    "SPLIT CONFORMAL PREDICTION : SUCCESS"
)

print()
print(
    f"Nominal coverage   : "
    f"{NOMINAL_COVERAGE:.2%}"
)

print(
    f"Empirical coverage : "
    f"{empirical_coverage:.2%}"
)

print(
    f"Coverage error     : "
    f"{coverage_error:+.2%}"
)

print(
    f"Mean interval width: "
    f"{mean_width:.6f}"
)

print()
print(
    "Calibration data used only for q : PASS"
)

print(
    "Test data used only for evaluation : PASS"
)

print()
print(
    "READY FOR ACI EXPERIMENT"
)