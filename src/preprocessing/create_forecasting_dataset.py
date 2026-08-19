"""
RQ1 - One-Step-Ahead Forecasting Dataset

Transforms the longitudinal panel into a supervised
one-step-ahead prediction problem:

X(i,t) -> Y(i,t+1)
"""

import pandas as pd
from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "rq1_synthetic_panel.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "rq1_one_step_forecasting.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("\n==========================================")
print("RQ1 ONE-STEP-AHEAD DATASET")
print("==========================================")

print(f"\nOriginal dataset shape: {df.shape}")


# ============================================================
# 3. SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    ["patient_id", "time"]
).reset_index(drop=True)


# ============================================================
# 4. CREATE NEXT-TIME TARGET
# ============================================================

df["target_next"] = (
    df.groupby("patient_id")["target"]
    .shift(-1)
)


# ============================================================
# 5. REMOVE LAST OBSERVATION OF EACH PATIENT
# ============================================================

forecast_df = df.dropna(
    subset=["target_next"]
).copy()


# ============================================================
# 6. RENAME CURRENT TARGET
# ============================================================

forecast_df = forecast_df.rename(
    columns={
        "target": "target_current",
        "target_next": "target_future"
    }
)


# ============================================================
# 7. SELECT VARIABLES
# ============================================================

columns = [
    "patient_id",
    "time",

    # Baseline features
    "baseline_severity",
    "baseline_sleep",
    "baseline_activity",
    "baseline_stress",

    # Current time-varying features
    "sleep",
    "activity",
    "stress",
    "mood",

    # Current and future target
    "target_current",
    "target_future"
]

forecast_df = forecast_df[columns]


# ============================================================
# 8. SAVE DATASET
# ============================================================

forecast_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 9. VALIDATION
# ============================================================

expected_rows = 100 * 19

print(f"\nExpected forecasting samples: {expected_rows}")
print(f"Actual forecasting samples  : {len(forecast_df)}")

assert len(forecast_df) == expected_rows


# Check each patient has 19 forecasting samples

samples_per_patient = (
    forecast_df
    .groupby("patient_id")
    .size()
)

assert (
    samples_per_patient == 19
).all()


# Check final time is 19

assert forecast_df["time"].max() == 19


# Check there are no missing future targets

assert forecast_df["target_future"].isnull().sum() == 0


# ============================================================
# 10. DISPLAY EXAMPLE
# ============================================================

print("\nFirst 10 forecasting observations:")
print(
    forecast_df.head(10).to_string(index=False)
)


# ============================================================
# 11. FINAL RESULT
# ============================================================

print("\n==========================================")
print("FORECASTING DATASET CREATED SUCCESSFULLY")
print("==========================================")

print(f"\nDataset shape: {forecast_df.shape}")

print("\nEach row now represents:")
print("X(i,t)  ->  Y(i,t+1)")

print("\nSaved to:")
print(OUTPUT_FILE)

print("==========================================\n")