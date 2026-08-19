"""
RQ1 - Dataset Validation

Validates the synthetic longitudinal patient-panel dataset
before any prediction or conformal calibration is performed.
"""

import pandas as pd
from pathlib import Path


# ============================================================
# 1. LOCATE DATASET
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "rq1_synthetic_panel.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)


print("\n==========================================")
print("RQ1 DATASET VALIDATION")
print("==========================================")

print(f"\nDataset path:")
print(DATA_FILE)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# 3. CHECK NUMBER OF PATIENTS
# ============================================================

n_patients = df["patient_id"].nunique()

print("\n1. Number of unique patients")
print("------------------------------------------")
print(f"Patients found : {n_patients}")
print("Expected       : 100")

assert n_patients == 100, "ERROR: Expected 100 patients."


# ============================================================
# 4. CHECK OBSERVATIONS PER PATIENT
# ============================================================

observations_per_patient = (
    df.groupby("patient_id")
    .size()
)

print("\n2. Observations per patient")
print("------------------------------------------")

print(
    observations_per_patient
    .value_counts()
    .sort_index()
)

assert (
    observations_per_patient == 20
).all(), "ERROR: Every patient must have exactly 20 observations."


# ============================================================
# 5. CHECK TIME RANGE
# ============================================================

time_min = df["time"].min()
time_max = df["time"].max()

print("\n3. Time range")
print("------------------------------------------")
print(f"Minimum time : {time_min}")
print(f"Maximum time : {time_max}")

assert time_min == 1
assert time_max == 20


# ============================================================
# 6. CHECK DUPLICATE PATIENT-TIME PAIRS
# ============================================================

duplicates = df.duplicated(
    subset=["patient_id", "time"]
).sum()

print("\n4. Duplicate patient-time observations")
print("------------------------------------------")
print(f"Duplicates found : {duplicates}")

assert duplicates == 0, (
    "ERROR: Duplicate patient-time combinations found."
)


# ============================================================
# 7. CHECK MISSING VALUES
# ============================================================

missing_values = df.isnull().sum()

total_missing = missing_values.sum()

print("\n5. Missing values")
print("------------------------------------------")

print(missing_values)

print(f"\nTotal missing values : {total_missing}")

assert total_missing == 0, (
    "ERROR: Missing values found."
)


# ============================================================
# 8. CHECK BASELINE FEATURES
# ============================================================

baseline_features = [
    "baseline_severity",
    "baseline_sleep",
    "baseline_activity",
    "baseline_stress"
]

print("\n6. Baseline feature consistency")
print("------------------------------------------")

for feature in baseline_features:

    unique_counts = (
        df.groupby("patient_id")[feature]
        .nunique()
    )

    max_unique = unique_counts.max()

    print(
        f"{feature:20s} "
        f"maximum unique values/patient: {max_unique}"
    )

    assert max_unique == 1, (
        f"ERROR: {feature} changes within a patient."
    )


# ============================================================
# 9. CHECK TIME-VARYING FEATURES
# ============================================================

dynamic_features = [
    "sleep",
    "activity",
    "stress",
    "mood"
]

print("\n7. Time-varying feature check")
print("------------------------------------------")

for feature in dynamic_features:

    unique_counts = (
        df.groupby("patient_id")[feature]
        .nunique()
    )

    patients_with_variation = (
        (unique_counts > 1).sum()
    )

    print(
        f"{feature:12s} "
        f"patients with temporal variation: "
        f"{patients_with_variation}/100"
    )


# ============================================================
# 10. CHECK TARGET VARIATION
# ============================================================

print("\n8. Target statistics")
print("------------------------------------------")

print(df["target"].describe())


assert df["target"].nunique() > 1, (
    "ERROR: Target has no variation."
)


# ============================================================
# 11. FINAL RESULT
# ============================================================

print("\n==========================================")
print("VALIDATION SUCCESSFUL")
print("==========================================")

print(
    "\nThe synthetic patient-panel dataset "
    "passed all structural validation checks."
)

print("\nReady for RQ1-Step 03:")
print(
    "Chronological train / calibration / test split."
)

print("==========================================\n")