from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

SPLIT_DIR = PROJECT_DIR / "temporal_split"
BASELINE_DIR = PROJECT_DIR / "baseline_results"
OUTPUT_DIR = PROJECT_DIR / "aci_results"

CAL_FILE = SPLIT_DIR / "calibration.csv"
TEST_FILE = SPLIT_DIR / "test.csv"

CAL_PRED_FILE = BASELINE_DIR / "calibration_predictions.csv"
TEST_PRED_FILE = BASELINE_DIR / "test_predictions.csv"

OUTPUT_FILE = OUTPUT_DIR / "aci_test_intervals.csv"
METRICS_FILE = OUTPUT_DIR / "aci_metrics.txt"

TARGET = "target_future"

ALPHA = 0.10

# ACI learning rate.
# This controls how quickly q responds to recent
# under-coverage / over-coverage.
GAMMA = 0.01


# ============================================================
# HELPER
# ============================================================

def header(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ============================================================
# STEP 1 - LOAD DATA
# ============================================================

header("STEP 1: LOAD CALIBRATION AND TEST PREDICTIONS")

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

required = [
    "patient_id",
    "time",
    TARGET,
    "prediction",
]

for column in required:

    if column not in cal_pred.columns:
        raise ValueError(
            f"Missing calibration column: {column}"
        )

    if column not in test_pred.columns:
        raise ValueError(
            f"Missing test column: {column}"
        )


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
        "Calibration data/prediction alignment failed."
    )

if test_keys_data != test_keys_pred:
    raise AssertionError(
        "Test data/prediction alignment failed."
    )

print("Calibration alignment : PASS")
print("Test alignment        : PASS")


# ============================================================
# STEP 3 - INITIAL CALIBRATION QUANTILE
# ============================================================

header("STEP 3: INITIALIZE ACI FROM CALIBRATION DATA")

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

k = int(
    np.ceil(
        (n_cal + 1) * (1.0 - ALPHA)
    )
)

k = min(k, n_cal)

q_initial = np.sort(cal_scores)[k - 1]

print(
    f"Calibration samples : {n_cal}"
)

print(
    f"Nominal coverage    : {(1.0 - ALPHA):.2%}"
)

print(
    f"Initial quantile rank: {k}"
)

print(
    f"Initial q            : {q_initial:.6f}"
)


# ============================================================
# STEP 4 - PREPARE TEST DATA
# ============================================================

header("STEP 4: PREPARE CHRONOLOGICAL TEST SEQUENCE")

aci_test = test_pred[
    [
        "patient_id",
        "time",
        TARGET,
        "prediction",
    ]
].copy()

aci_test = aci_test.sort_values(
    ["time", "patient_id"]
).reset_index(drop=True)

unique_times = sorted(
    aci_test["time"].unique()
)

print(
    f"Test times: {unique_times}"
)

print(
    "Processing rule:"
)

print(
    "Predict entire time t first, "
    "then update q using outcomes from time t."
)


# ============================================================
# STEP 5 - RUN ACI
# ============================================================

header("STEP 5: RUN ADAPTIVE CONFORMAL INFERENCE")

q = float(q_initial)

records = []

time_summary = []

all_covered = []

all_widths = []

for time_value in unique_times:

    current = aci_test[
        aci_test["time"] == time_value
    ].copy()

    predictions = current[
        "prediction"
    ].to_numpy(dtype=float)

    actual = current[
        TARGET
    ].to_numpy(dtype=float)

    # --------------------------------------------------------
    # PREDICTION STEP
    # --------------------------------------------------------

    lower = predictions - q
    upper = predictions + q

    covered = (
        (actual >= lower) &
        (actual <= upper)
    )

    errors = np.abs(
        actual - predictions
    )

    width = upper - lower

    # Store results BEFORE updating q.
    for index in range(len(current)):

        records.append(
            {
                "patient_id":
                    current.iloc[index]["patient_id"],

                "time":
                    time_value,

                TARGET:
                    actual[index],

                "prediction":
                    predictions[index],

                "q_used":
                    q,

                "lower":
                    lower[index],

                "upper":
                    upper[index],

                "interval_width":
                    width[index],

                "absolute_error":
                    errors[index],

                "covered":
                    bool(covered[index]),
            }
        )

    # --------------------------------------------------------
    # COVERAGE BEFORE UPDATE
    # --------------------------------------------------------

    time_covered = int(
        covered.sum()
    )

    time_n = len(current)

    time_coverage = (
        time_covered / time_n
    )

    all_covered.extend(
        covered.tolist()
    )

    all_widths.extend(
        width.tolist()
    )

    # --------------------------------------------------------
    # ACI UPDATE
    # --------------------------------------------------------
    #
    # Update after all predictions at time t have been made.
    #
    # q_{t+1} =
    # q_t + gamma * (coverage_error)
    #
    # For each batch:
    #
    # if coverage is below target:
    #     increase q
    #
    # if coverage is above target:
    #     decrease q
    #
    # This is a batch/time-level implementation designed
    # specifically to avoid same-time leakage.
    # --------------------------------------------------------

    coverage_error = (
        (1.0 - ALPHA)
        - time_coverage
    )

    q_old = q

    q = q + GAMMA * coverage_error

    # q must remain positive.
    q = max(
        q,
        1e-8
    )

    time_summary.append(
        {
            "time": time_value,
            "n": time_n,
            "covered": time_covered,
            "coverage": time_coverage,
            "q_before_update": q_old,
            "coverage_error": coverage_error,
            "q_after_update": q,
            "mean_width": float(
                np.mean(width)
            ),
        }
    )

    print()
    print(
        f"TIME {time_value}"
    )

    print(
        f"Samples             : {time_n}"
    )

    print(
        f"Covered             : {time_covered}"
    )

    print(
        f"Coverage            : "
        f"{time_coverage:.4f}"
    )

    print(
        f"q before update     : "
        f"{q_old:.6f}"
    )

    print(
        f"q after update      : "
        f"{q:.6f}"
    )

    print(
        f"Mean interval width : "
        f"{np.mean(width):.6f}"
    )


# ============================================================
# STEP 6 - OVERALL METRICS
# ============================================================

header("STEP 6: CALCULATE OVERALL ACI METRICS")

aci_result = pd.DataFrame(
    records
)

nominal_coverage = 1.0 - ALPHA

n_test = len(aci_result)

n_covered = int(
    aci_result["covered"].sum()
)

n_missed = (
    n_test - n_covered
)

empirical_coverage = (
    n_covered / n_test
)

coverage_error = (
    empirical_coverage
    - nominal_coverage
)

mean_width = float(
    aci_result["interval_width"].mean()
)

median_width = float(
    aci_result["interval_width"].median()
)

min_width = float(
    aci_result["interval_width"].min()
)

max_width = float(
    aci_result["interval_width"].max()
)

mae = float(
    aci_result["absolute_error"].mean()
)

rmse = float(
    np.sqrt(
        np.mean(
            aci_result["absolute_error"] ** 2
        )
    )
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
    f"Nominal coverage    : "
    f"{nominal_coverage:.4f}"
)

print(
    f"Empirical coverage  : "
    f"{empirical_coverage:.4f}"
)

print(
    f"Coverage error      : "
    f"{coverage_error:+.4f}"
)

print(
    f"Mean interval width : "
    f"{mean_width:.6f}"
)

print(
    f"Median width        : "
    f"{median_width:.6f}"
)

print(
    f"Minimum width       : "
    f"{min_width:.6f}"
)

print(
    f"Maximum width       : "
    f"{max_width:.6f}"
)

print(
    f"MAE                 : "
    f"{mae:.6f}"
)

print(
    f"RMSE                : "
    f"{rmse:.6f}"
)


# ============================================================
# STEP 7 - PATIENT-WISE COVERAGE
# ============================================================

header("STEP 7: PATIENT-WISE ACI COVERAGE")

patient_summary = (
    aci_result
    .groupby("patient_id")
    .agg(
        n_test=("covered", "size"),
        covered=("covered", "sum"),
        coverage=("covered", "mean"),
        mean_width=("interval_width", "mean"),
    )
    .reset_index()
)

print(
    f"Patients evaluated : "
    f"{len(patient_summary)}"
)

print(
    f"Mean patient coverage : "
    f"{patient_summary['coverage'].mean():.4f}"
)

print(
    f"Minimum patient coverage : "
    f"{patient_summary['coverage'].min():.4f}"
)

print(
    f"Maximum patient coverage : "
    f"{patient_summary['coverage'].max():.4f}"
)


# ============================================================
# STEP 8 - SAVE RESULTS
# ============================================================

header("STEP 8: SAVE ACI RESULTS")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

aci_result.to_csv(
    OUTPUT_FILE,
    index=False
)

time_df = pd.DataFrame(
    time_summary
)

time_file = OUTPUT_DIR / "aci_time_summary.csv"

time_df.to_csv(
    time_file,
    index=False
)

patient_file = OUTPUT_DIR / "aci_patient_summary.csv"

patient_summary.to_csv(
    patient_file,
    index=False
)


metrics_text = f"""
RQ1 ADAPTIVE CONFORMAL INFERENCE
================================

Nominal coverage:
{nominal_coverage:.10f}

Alpha:
{ALPHA:.10f}

ACI gamma:
{GAMMA:.10f}

Calibration samples:
{n_cal}

Initial q:
{q_initial:.10f}

Final q:
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

MAE:
{mae:.10f}

RMSE:
{rmse:.10f}


PATIENT-WISE COVERAGE
---------------------

Patients:
{len(patient_summary)}

Mean patient coverage:
{patient_summary['coverage'].mean():.10f}

Minimum patient coverage:
{patient_summary['coverage'].min():.10f}

Maximum patient coverage:
{patient_summary['coverage'].max():.10f}


TIME-WISE RESULTS
-----------------

{time_df.to_string(index=False)}
"""


with open(
    METRICS_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(metrics_text)


print(
    f"Test intervals saved:\n{OUTPUT_FILE}"
)

print(
    f"Time summary saved:\n{time_file}"
)

print(
    f"Patient summary saved:\n{patient_file}"
)

print(
    f"Metrics saved:\n{METRICS_FILE}"
)


# ============================================================
# FINAL RESULT
# ============================================================

header("FINAL RESULT")

print(
    "ACI EXPERIMENT : SUCCESS"
)

print()
print(
    f"Nominal coverage   : "
    f"{nominal_coverage:.2%}"
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

print(
    f"Initial q          : "
    f"{q_initial:.6f}"
)

print(
    f"Final q            : "
    f"{q:.6f}"
)

print()
print(
    "Same-time prediction leakage : NONE"
)

print(
    "Test outcomes used only after prediction : PASS"
)

print()
print(
    "READY FOR SCP vs ACI COMPARISON"
)