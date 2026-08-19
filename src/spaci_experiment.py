from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent

CAL_FILE = PROJECT_DIR / "baseline_results" / "calibration_predictions.csv"
TEST_FILE = PROJECT_DIR / "baseline_results" / "test_predictions.csv"

OUT_DIR = PROJECT_DIR / "spaci_results"
OUT_DIR.mkdir(exist_ok=True)

ALPHA = 0.10
LAMBDA = 0.25
LEARNING_RATE = 0.002

print("=" * 70)
print("SPACI: SHRUNK PATIENT-ADAPTIVE CONFORMAL INFERENCE")
print("=" * 70)

# ================================================================
# STEP 1: LOAD DATA
# ================================================================
print("\nSTEP 1: LOAD CALIBRATION AND TEST PREDICTIONS")

cal = pd.read_csv(CAL_FILE)
test = pd.read_csv(TEST_FILE)

print(f"Calibration samples : {len(cal)}")
print(f"Test samples        : {len(test)}")

required = {
    "patient_id",
    "time",
    "target_future",
    "prediction",
}

for col in required:
    if col not in cal.columns:
        raise AssertionError(f"Missing calibration column: {col}")
    if col not in test.columns:
        raise AssertionError(f"Missing test column: {col}")

print("Required columns : PASS")


# ================================================================
# STEP 2: BASIC VALIDATION
# ================================================================
print("\nSTEP 2: VALIDATE TEMPORAL STRUCTURE")

if cal["patient_id"].isnull().any():
    raise AssertionError("Missing calibration patient IDs")

if test["patient_id"].isnull().any():
    raise AssertionError("Missing test patient IDs")

if cal.duplicated(["patient_id", "time"]).any():
    raise AssertionError("Duplicate calibration patient-time samples")

if test.duplicated(["patient_id", "time"]).any():
    raise AssertionError("Duplicate test patient-time samples")

cal_times = sorted(cal["time"].unique())
test_times = sorted(test["time"].unique())

print(f"Calibration times : {cal_times}")
print(f"Test times        : {test_times}")

if max(cal_times) >= min(test_times):
    raise AssertionError("Temporal leakage detected")

print("Calibration < Test : PASS")


# ================================================================
# STEP 3: CALIBRATION RESIDUALS
# ================================================================
print("\nSTEP 3: CALCULATE CALIBRATION RESIDUALS")

cal = cal.copy()
test = test.copy()

cal["absolute_error"] = (
    cal["target_future"] - cal["prediction"]
).abs()

test["absolute_error"] = (
    test["target_future"] - test["prediction"]
).abs()

print("Residual calculation : PASS")


# ================================================================
# STEP 4: GLOBAL CALIBRATION QUANTILE
# ================================================================
print("\nSTEP 4: CALCULATE GLOBAL CONFORMAL QUANTILE")

n_cal = len(cal)

k = int(np.ceil((n_cal + 1) * (1 - ALPHA)))

k = min(max(k, 1), n_cal)

global_sorted = np.sort(cal["absolute_error"].values)

q_global = global_sorted[k - 1]

print(f"Calibration n : {n_cal}")
print(f"Alpha         : {ALPHA}")
print(f"Quantile rank : {k}")
print(f"Global q      : {q_global:.6f}")


# ================================================================
# STEP 5: PATIENT-SPECIFIC CALIBRATION QUANTILES
# ================================================================
print("\nSTEP 5: CALCULATE PATIENT-SPECIFIC CALIBRATION QUANTILES")

patient_q = {}

for patient_id, group in cal.groupby("patient_id"):

    errors = np.sort(group["absolute_error"].values)

    n_i = len(errors)

    k_i = int(np.ceil((n_i + 1) * (1 - ALPHA)))
    k_i = min(max(k_i, 1), n_i)

    q_local = errors[k_i - 1]

    patient_q[patient_id] = q_local

print(f"Patients calibrated : {len(patient_q)}")

print(
    f"Patient q range     : "
    f"{min(patient_q.values()):.6f} - "
    f"{max(patient_q.values()):.6f}"
)


# ================================================================
# STEP 6: SHRINK PATIENT QUANTILES
# ================================================================
print("\nSTEP 6: APPLY SHRINKAGE")

print(f"Shrinkage lambda : {LAMBDA:.2f}")

initial_q = {}

for patient_id, q_local in patient_q.items():

    q_initial = (
        (1 - LAMBDA) * q_global
        + LAMBDA * q_local
    )

    initial_q[patient_id] = q_initial

print(
    f"Initial SPACI q range : "
    f"{min(initial_q.values()):.6f} - "
    f"{max(initial_q.values()):.6f}"
)


# ================================================================
# STEP 7: PREPARE CHRONOLOGICAL TEST DATA
# ================================================================
print("\nSTEP 7: PREPARE CHRONOLOGICAL TEST SEQUENCE")

test = test.sort_values(
    ["time", "patient_id"]
).reset_index(drop=True)

print(f"Test times : {sorted(test['time'].unique().tolist())}")

print(
    "Processing rule:\n"
    "Predict entire time t first, then update patient q "
    "using outcomes from time t."
)


# ================================================================
# STEP 8: INITIALIZE PATIENT STATES
# ================================================================
print("\nSTEP 8: INITIALIZE PATIENT STATES")

q_state = dict(initial_q)

for patient_id in test["patient_id"].unique():

    if patient_id not in q_state:
        raise AssertionError(
            f"No calibration state for patient {patient_id}"
        )

print("Patient state initialization : PASS")


# ================================================================
# STEP 9: RUN SPACI
# ================================================================
print("\nSTEP 9: RUN SPACI")

results = []
time_summary = []

for time_value in sorted(test["time"].unique()):

    current = test[
        test["time"] == time_value
    ].copy()

    q_before = {
        pid: q_state[pid]
        for pid in current["patient_id"]
    }

    # ------------------------------------------------------------
    # PREDICTION PHASE
    # ------------------------------------------------------------

    current["q_before"] = current["patient_id"].map(q_before)

    current["lower"] = (
        current["prediction"]
        - current["q_before"]
    )

    current["upper"] = (
        current["prediction"]
        + current["q_before"]
    )

    current["covered"] = (
        (current["target_future"] >= current["lower"])
        &
        (current["target_future"] <= current["upper"])
    )

    current["interval_width"] = (
        current["upper"] - current["lower"]
    )

    current["absolute_error"] = (
        current["target_future"]
        - current["prediction"]
    ).abs()

    covered_count = int(current["covered"].sum())

    coverage = (
        covered_count / len(current)
    )

    mean_width = current["interval_width"].mean()

    print(f"\nTIME {time_value}")
    print(f"Samples             : {len(current)}")
    print(f"Covered             : {covered_count}")
    print(f"Coverage            : {coverage:.4f}")
    print(
        f"Mean interval width : {mean_width:.6f}"
    )

    # ------------------------------------------------------------
    # UPDATE PHASE
    # ------------------------------------------------------------

    q_after = {}

    for _, row in current.iterrows():

        patient_id = row["patient_id"]

        q_old = q_state[patient_id]

        error = row["absolute_error"]

        miss = 1.0 if error > q_old else 0.0

        # A conservative ACI-style patient update.
        #
        # If the interval misses, increase q.
        # If the interval covers, decrease q slightly.
        #
        # alpha = 0.10

        update = LEARNING_RATE * (
            miss - ALPHA
        )

        q_new = q_old + update

        # Safety constraint: q cannot become negative.
        q_new = max(q_new, 0.0)

        q_after[patient_id] = q_new

        q_state[patient_id] = q_new

    print(
        f"Mean q before update: "
        f"{current['q_before'].mean():.6f}"
    )

    print(
        f"Mean q after update : "
        f"{np.mean(list(q_after.values())):.6f}"
    )

    current["q_after"] = current["patient_id"].map(
        q_after
    )

    results.append(current)

    time_summary.append(
        {
            "time": time_value,
            "n": len(current),
            "covered": covered_count,
            "coverage": coverage,
            "mean_width": mean_width,
            "mean_q_before": current["q_before"].mean(),
            "mean_q_after": current["q_after"].mean(),
        }
    )


# ================================================================
# STEP 10: COMBINE RESULTS
# ================================================================
print("\nSTEP 10: COMBINE SPACI RESULTS")

spaci = pd.concat(
    results,
    ignore_index=True
)

time_df = pd.DataFrame(time_summary)

print(f"SPACI test samples : {len(spaci)}")


# ================================================================
# STEP 11: OVERALL METRICS
# ================================================================
print("\nSTEP 11: CALCULATE OVERALL SPACI METRICS")

covered_total = int(spaci["covered"].sum())

overall_coverage = (
    covered_total / len(spaci)
)

coverage_error = (
    overall_coverage - (1 - ALPHA)
)

mean_width = spaci["interval_width"].mean()
median_width = spaci["interval_width"].median()

mae = spaci["absolute_error"].mean()

rmse = np.sqrt(
    np.mean(
        (
            spaci["target_future"]
            - spaci["prediction"]
        ) ** 2
    )
)

print(f"Test samples       : {len(spaci)}")
print(f"Covered             : {covered_total}")
print(f"Missed              : {len(spaci) - covered_total}")
print(f"Nominal coverage    : {1 - ALPHA:.4f}")
print(f"Empirical coverage  : {overall_coverage:.4f}")
print(f"Coverage error      : {coverage_error:+.4f}")
print(f"Mean interval width : {mean_width:.6f}")
print(f"Median width        : {median_width:.6f}")
print(f"MAE                 : {mae:.6f}")
print(f"RMSE                : {rmse:.6f}")


# ================================================================
# STEP 12: PATIENT-WISE COVERAGE
# ================================================================
print("\nSTEP 12: PATIENT-WISE SPACI COVERAGE")

patient_summary = (
    spaci.groupby("patient_id")
    .agg(
        n_test=("covered", "size"),
        coverage=("covered", "mean"),
        mean_width=("interval_width", "mean"),
        mean_absolute_error=("absolute_error", "mean"),
        rmse=(
            "absolute_error",
            lambda x: np.sqrt(np.mean(x ** 2))
        ),
    )
    .reset_index()
)

print(
    f"Patients evaluated : "
    f"{len(patient_summary)}"
)

print(
    f"Mean coverage      : "
    f"{patient_summary['coverage'].mean():.4f}"
)

print(
    f"Median coverage    : "
    f"{patient_summary['coverage'].median():.4f}"
)

print(
    f"Minimum coverage   : "
    f"{patient_summary['coverage'].min():.4f}"
)

print(
    f"Maximum coverage   : "
    f"{patient_summary['coverage'].max():.4f}"
)

below_nominal = (
    patient_summary["coverage"] < (1 - ALPHA)
).sum()

print(
    f"Patients below 90%: "
    f"{below_nominal}"
)


# ================================================================
# STEP 13: COVERAGE DISPERSION
# ================================================================
print("\nSTEP 13: PATIENT COVERAGE DISPERSION")

coverage_std = patient_summary["coverage"].std()
coverage_mad = (
    patient_summary["coverage"]
    - patient_summary["coverage"].median()
).abs().median()

print(
    f"Coverage standard deviation : "
    f"{coverage_std:.6f}"
)

print(
    f"Coverage median absolute deviation : "
    f"{coverage_mad:.6f}"
)


# ================================================================
# STEP 14: SAVE RESULTS
# ================================================================
print("\nSTEP 14: SAVE SPACI RESULTS")

interval_file = OUT_DIR / "spaci_test_intervals.csv"
time_file = OUT_DIR / "spaci_time_summary.csv"
patient_file = OUT_DIR / "spaci_patient_summary.csv"
metrics_file = OUT_DIR / "spaci_metrics.txt"

spaci.to_csv(
    interval_file,
    index=False
)

time_df.to_csv(
    time_file,
    index=False
)

patient_summary.to_csv(
    patient_file,
    index=False
)

with open(metrics_file, "w", encoding="utf-8") as f:

    f.write(
        "SPACI EXPERIMENT\n"
        + "=" * 70
        + "\n\n"
    )

    f.write(f"Alpha: {ALPHA}\n")
    f.write(f"Lambda: {LAMBDA}\n")
    f.write(f"Learning rate: {LEARNING_RATE}\n")
    f.write(f"Global q: {q_global:.6f}\n\n")

    f.write(
        f"Test samples: {len(spaci)}\n"
    )

    f.write(
        f"Covered: {covered_total}\n"
    )

    f.write(
        f"Empirical coverage: "
        f"{overall_coverage:.6f}\n"
    )

    f.write(
        f"Coverage error: "
        f"{coverage_error:+.6f}\n"
    )

    f.write(
        f"Mean interval width: "
        f"{mean_width:.6f}\n"
    )

    f.write(
        f"Median interval width: "
        f"{median_width:.6f}\n"
    )

    f.write(
        f"MAE: {mae:.6f}\n"
    )

    f.write(
        f"RMSE: {rmse:.6f}\n"
    )

    f.write(
        f"Mean patient coverage: "
        f"{patient_summary['coverage'].mean():.6f}\n"
    )

    f.write(
        f"Minimum patient coverage: "
        f"{patient_summary['coverage'].min():.6f}\n"
    )

    f.write(
        f"Maximum patient coverage: "
        f"{patient_summary['coverage'].max():.6f}\n"
    )

    f.write(
        f"Patients below 90%: "
        f"{below_nominal}\n"
    )

print(f"Intervals saved to:\n{interval_file}")
print(f"Time summary saved to:\n{time_file}")
print(f"Patient summary saved to:\n{patient_file}")
print(f"Metrics saved to:\n{metrics_file}")


# ================================================================
# FINAL
# ================================================================
print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)

print("SPACI EXPERIMENT : SUCCESS")

print(
    f"\nNominal coverage   : "
    f"{1 - ALPHA:.2%}"
)

print(
    f"Empirical coverage : "
    f"{overall_coverage:.2%}"
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
    f"Patients below 90%: "
    f"{below_nominal}"
)

print(
    "\nPrediction-before-update rule : PASS"
)

print(
    "Calibration/test temporal separation : PASS"
)

print(
    "\nREADY FOR SCP vs ACI vs SPACI COMPARISON"
)