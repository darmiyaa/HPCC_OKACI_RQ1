"""
RQ1 - Synthetic Patient Panel Data Generator

Purpose:
    Generate a synthetic longitudinal patient-panel dataset
    for testing the HPCC-OKACI research framework.

Initial experiment:
    - 100 patients
    - 20 observations per patient
    - 4 baseline features
    - 4 time-varying features
    - 1 continuous target
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# 1. EXPERIMENT SETTINGS
# ============================================================

RANDOM_SEED = 42
N_PATIENTS = 100
OBSERVATIONS_PER_PATIENT = 20

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# 2. OUTPUT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "rq1_synthetic_panel.csv"


# ============================================================
# 3. GENERATE PATIENT BASELINE CHARACTERISTICS
# ============================================================

patients = []

for i in range(N_PATIENTS):

    patient_id = f"P{i + 1:03d}"

    baseline_severity = rng.uniform(0.2, 0.8)
    baseline_sleep = rng.uniform(0.3, 0.9)
    baseline_activity = rng.uniform(0.2, 0.9)
    baseline_stress = rng.uniform(0.2, 0.9)

    patients.append({
        "patient_id": patient_id,
        "baseline_severity": baseline_severity,
        "baseline_sleep": baseline_sleep,
        "baseline_activity": baseline_activity,
        "baseline_stress": baseline_stress
    })


baseline_df = pd.DataFrame(patients)


# ============================================================
# 4. GENERATE LONGITUDINAL DATA
# ============================================================

records = []

for _, patient in baseline_df.iterrows():

    patient_id = patient["patient_id"]

    severity = patient["baseline_severity"]
    base_sleep = patient["baseline_sleep"]
    base_activity = patient["baseline_activity"]
    base_stress = patient["baseline_stress"]

    # Patient-specific effect
    patient_effect = rng.normal(0, 0.10)

    for t in range(1, OBSERVATIONS_PER_PATIENT + 1):

        # Small temporal variation
        sleep = (
            base_sleep
            + 0.05 * np.sin(t / 3)
            + rng.normal(0, 0.04)
        )

        activity = (
            base_activity
            + 0.05 * np.cos(t / 4)
            + rng.normal(0, 0.04)
        )

        stress = (
            base_stress
            + 0.04 * np.sin(t / 2)
            + rng.normal(0, 0.04)
        )

        mood = (
            0.6
            - 0.35 * severity
            + 0.20 * sleep
            - 0.15 * stress
            + rng.normal(0, 0.05)
        )

        # Keep generated values in [0, 1]
        sleep = np.clip(sleep, 0, 1)
        activity = np.clip(activity, 0, 1)
        stress = np.clip(stress, 0, 1)
        mood = np.clip(mood, 0, 1)

        # Future symptom/risk target
        target = (
            0.40 * stress
            - 0.30 * sleep
            - 0.20 * activity
            + 0.20 * mood
            + 0.30 * severity
            + patient_effect
            + rng.normal(0, 0.08)
        )

        records.append({
            "patient_id": patient_id,
            "time": t,

            # Baseline features
            "baseline_severity": severity,
            "baseline_sleep": base_sleep,
            "baseline_activity": base_activity,
            "baseline_stress": base_stress,

            # Time-varying features
            "sleep": sleep,
            "activity": activity,
            "stress": stress,
            "mood": mood,

            # Prediction target
            "target": target
        })


# ============================================================
# 5. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(records)


# ============================================================
# 6. SAVE DATASET
# ============================================================

df.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# 7. BASIC VALIDATION
# ============================================================

print("\n==========================================")
print("RQ1 SYNTHETIC DATASET CREATED")
print("==========================================")

print(f"Number of patients       : {df['patient_id'].nunique()}")
print(f"Total observations       : {len(df)}")
print(f"Observations per patient : {df.groupby('patient_id').size().unique()}")

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nFirst 10 observations:")
print(df.head(10))

print("\nDataset saved to:")
print(OUTPUT_FILE)

print("==========================================\n")