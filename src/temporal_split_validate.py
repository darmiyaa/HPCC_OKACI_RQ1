from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_DIR / "data" / "processed" / "rq1_one_step_forecasting.csv"

OUTPUT_DIR = PROJECT_DIR / "temporal_split"

TRAIN_FILE = OUTPUT_DIR / "train.csv"
CALIBRATION_FILE = OUTPUT_DIR / "calibration.csv"
TEST_FILE = OUTPUT_DIR / "test.csv"

TRAIN_START = 1
TRAIN_END = 10

CAL_START = 11
CAL_END = 15

TEST_START = 16
TEST_END = 19

EXPECTED_TRAIN = 1000
EXPECTED_CAL = 500
EXPECTED_TEST = 400


def find_column(df, candidates, description):

    lower_map = {
        str(column).lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    raise ValueError(
        f"Could not identify {description} column. "
        f"Available columns: {list(df.columns)}"
    )


def header(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

header("STEP 1: LOAD ONE-STEP-AHEAD DATA")

print(f"Looking for:\n{INPUT_FILE}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{INPUT_FILE}\n\n"
        "Check the filename of your existing one-step-ahead dataset."
    )

df = pd.read_csv(INPUT_FILE)

print(f"Input rows    : {len(df)}")
print(f"Input columns : {len(df.columns)}")

print("\nColumns:")
for column in df.columns:
    print(f"  {column}")


# ============================================================
# 2. IDENTIFY PATIENT AND TIME COLUMNS
# ============================================================

header("STEP 2: IDENTIFY PATIENT AND TIME COLUMNS")

patient_col = find_column(
    df,
    [
        "patient_id",
        "patient",
        "subject_id",
        "subject",
        "id"
    ],
    "patient identifier"
)

time_col = find_column(
    df,
    [
        "time",
        "time_index",
        "t",
        "timestamp",
        "visit",
        "visit_time"
    ],
    "time"
)

print(f"Patient column : {patient_col}")
print(f"Time column    : {time_col}")


# ============================================================
# 3. BASIC VALIDATION
# ============================================================

header("STEP 3: BASIC VALIDATION")

if df[patient_col].isna().any():
    raise ValueError("Missing patient IDs detected.")

if df[time_col].isna().any():
    raise ValueError("Missing time values detected.")

df[time_col] = pd.to_numeric(
    df[time_col],
    errors="raise"
)

print("Missing patient IDs : PASS")
print("Missing time values : PASS")
print("Numeric time        : PASS")

print(
    f"Number of patients  : "
    f"{df[patient_col].nunique()}"
)


# ============================================================
# 4. PATIENT-WISE TEMPORAL ORDER
# ============================================================

header("STEP 4: PATIENT-WISE TEMPORAL ORDER")

ordering_errors = []

for patient_id, group in df.groupby(patient_col):

    times = group[time_col].tolist()

    if times != sorted(times):
        ordering_errors.append(patient_id)

if ordering_errors:

    print("FAIL")

    for patient in ordering_errors:
        print(f"Patient {patient}")

    raise ValueError(
        "Patient-wise temporal ordering failed."
    )

print("Patient-wise temporal ordering : PASS")


# ============================================================
# 5. DUPLICATE PATIENT-TIME CHECK
# ============================================================

header("STEP 5: DUPLICATE CHECK")

duplicates = df.duplicated(
    subset=[patient_col, time_col],
    keep=False
)

if duplicates.any():

    print(
        df.loc[
            duplicates,
            [patient_col, time_col]
        ].to_string(index=False)
    )

    raise ValueError(
        "Duplicate patient-time samples detected."
    )

print("Duplicate patient-time samples : PASS")


# ============================================================
# 6. TEMPORAL SPLIT
# ============================================================

header("STEP 6: CREATE TEMPORAL SPLITS")

train = df[
    (df[time_col] >= TRAIN_START) &
    (df[time_col] <= TRAIN_END)
].copy()

calibration = df[
    (df[time_col] >= CAL_START) &
    (df[time_col] <= CAL_END)
].copy()

test = df[
    (df[time_col] >= TEST_START) &
    (df[time_col] <= TEST_END)
].copy()


# ============================================================
# 7. DISPLAY SPLITS
# ============================================================

header("STEP 7: SPLIT INFORMATION")

print(
    f"TRAINING     : {len(train)} samples "
    f"| time {train[time_col].min()}-"
    f"{train[time_col].max()}"
)

print(
    f"CALIBRATION  : {len(calibration)} samples "
    f"| time {calibration[time_col].min()}-"
    f"{calibration[time_col].max()}"
)

print(
    f"TEST         : {len(test)} samples "
    f"| time {test[time_col].min()}-"
    f"{test[time_col].max()}"
)


# ============================================================
# 8. TEMPORAL LEAKAGE CHECK
# ============================================================

header("STEP 8: TEMPORAL LEAKAGE CHECK")

max_train = train[time_col].max()
min_cal = calibration[time_col].min()

max_cal = calibration[time_col].max()
min_test = test[time_col].min()

print(f"Maximum training time    : {max_train}")
print(f"Minimum calibration time : {min_cal}")
print(f"Maximum calibration time : {max_cal}")
print(f"Minimum test time        : {min_test}")

if not max_train < min_cal:
    raise AssertionError(
        "LEAKAGE: Training overlaps calibration."
    )

if not max_cal < min_test:
    raise AssertionError(
        "LEAKAGE: Calibration overlaps test."
    )

print()
print("Training < Calibration < Test : PASS")


# ============================================================
# 9. SAMPLE OVERLAP CHECK
# ============================================================

header("STEP 9: SAMPLE OVERLAP CHECK")

train_keys = set(
    zip(
        train[patient_col],
        train[time_col]
    )
)

cal_keys = set(
    zip(
        calibration[patient_col],
        calibration[time_col]
    )
)

test_keys = set(
    zip(
        test[patient_col],
        test[time_col]
    )
)

if train_keys & cal_keys:
    raise AssertionError(
        "Training/calibration sample overlap detected."
    )

if train_keys & test_keys:
    raise AssertionError(
        "Training/test sample overlap detected."
    )

if cal_keys & test_keys:
    raise AssertionError(
        "Calibration/test sample overlap detected."
    )

print("Training vs Calibration : PASS")
print("Training vs Test        : PASS")
print("Calibration vs Test     : PASS")


# ============================================================
# 10. PATIENT COVERAGE
# ============================================================

header("STEP 10: PATIENT COVERAGE")

train_patients = set(
    train[patient_col].unique()
)

cal_patients = set(
    calibration[patient_col].unique()
)

test_patients = set(
    test[patient_col].unique()
)

print(
    f"Training patients    : {len(train_patients)}"
)

print(
    f"Calibration patients : {len(cal_patients)}"
)

print(
    f"Test patients        : {len(test_patients)}"
)

if train_patients != cal_patients:
    print(
        "WARNING: Training and calibration "
        "patient sets differ."
    )

if train_patients != test_patients:
    print(
        "WARNING: Training and test "
        "patient sets differ."
    )


# ============================================================
# 11. EXPECTED COUNTS
# ============================================================

header("STEP 11: VERIFY 1000 / 500 / 400")

print(
    f"Training     : {len(train)} "
    f"(expected {EXPECTED_TRAIN})"
)

print(
    f"Calibration  : {len(calibration)} "
    f"(expected {EXPECTED_CAL})"
)

print(
    f"Test         : {len(test)} "
    f"(expected {EXPECTED_TEST})"
)

if len(train) != EXPECTED_TRAIN:
    raise AssertionError(
        f"Training count mismatch: "
        f"{len(train)} != {EXPECTED_TRAIN}"
    )

if len(calibration) != EXPECTED_CAL:
    raise AssertionError(
        f"Calibration count mismatch: "
        f"{len(calibration)} != {EXPECTED_CAL}"
    )

if len(test) != EXPECTED_TEST:
    raise AssertionError(
        f"Test count mismatch: "
        f"{len(test)} != {EXPECTED_TEST}"
    )

print()
print("Expected 1000 / 500 / 400 : PASS")


# ============================================================
# 12. SAVE SPLITS
# ============================================================

header("STEP 12: SAVE SPLITS")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

train.to_csv(
    TRAIN_FILE,
    index=False
)

calibration.to_csv(
    CALIBRATION_FILE,
    index=False
)

test.to_csv(
    TEST_FILE,
    index=False
)

print(f"Training file    : {TRAIN_FILE}")
print(f"Calibration file : {CALIBRATION_FILE}")
print(f"Test file        : {TEST_FILE}")


# ============================================================
# FINAL RESULT
# ============================================================

header("FINAL RESULT")

print("TEMPORAL SPLIT VALIDATION : SUCCESS")
print()
print(f"Training samples    : {len(train)}")
print(f"Calibration samples : {len(calibration)}")
print(f"Test samples        : {len(test)}")
print()
print("Temporal ordering : PASS")
print("Duplicate check   : PASS")
print("Overlap check     : PASS")
print("Leakage check     : PASS")
print("Count check       : PASS")
print()
print("READY FOR CALIBRATION")