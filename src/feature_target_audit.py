from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

SPLIT_DIR = PROJECT_DIR / "temporal_split"

TRAIN_FILE = SPLIT_DIR / "train.csv"
CAL_FILE = SPLIT_DIR / "calibration.csv"
TEST_FILE = SPLIT_DIR / "test.csv"


# ============================================================
# EXPECTED DESIGN
# ============================================================

PATIENT_COL = "patient_id"
TIME_COL = "time"

TARGET_COL = "target_future"

# Variables known to be available at prediction time.
# We deliberately exclude IDs, time, current target, and
# baseline variables initially from the predictive feature set.
CANDIDATE_FEATURES = [
    "sleep",
    "activity",
    "stress",
    "mood",
]


# Variables that must NOT enter X.
FORBIDDEN_FEATURES = [
    "patient_id",
    "time",
    "target_future",
]


# ============================================================
# HELPER
# ============================================================

def header(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ============================================================
# STEP 1 - LOAD SPLITS
# ============================================================

header("STEP 1: LOAD TEMPORAL SPLITS")

if not TRAIN_FILE.exists():
    raise FileNotFoundError(
        f"Training file not found:\n{TRAIN_FILE}"
    )

if not CAL_FILE.exists():
    raise FileNotFoundError(
        f"Calibration file not found:\n{CAL_FILE}"
    )

if not TEST_FILE.exists():
    raise FileNotFoundError(
        f"Test file not found:\n{TEST_FILE}"
    )


train = pd.read_csv(TRAIN_FILE)
calibration = pd.read_csv(CAL_FILE)
test = pd.read_csv(TEST_FILE)


print(f"Training samples    : {len(train)}")
print(f"Calibration samples : {len(calibration)}")
print(f"Test samples        : {len(test)}")


# ============================================================
# STEP 2 - COLUMN CONSISTENCY
# ============================================================

header("STEP 2: CHECK COLUMN CONSISTENCY")

train_columns = set(train.columns)
cal_columns = set(calibration.columns)
test_columns = set(test.columns)

if train_columns != cal_columns:
    raise AssertionError(
        "Training and calibration columns differ."
    )

if train_columns != test_columns:
    raise AssertionError(
        "Training and test columns differ."
    )

print("Training vs Calibration columns : PASS")
print("Training vs Test columns        : PASS")


# ============================================================
# STEP 3 - REQUIRED COLUMNS
# ============================================================

header("STEP 3: CHECK REQUIRED COLUMNS")

required_columns = {
    PATIENT_COL,
    TIME_COL,
    TARGET_COL,
}

missing = required_columns - train_columns

if missing:
    raise AssertionError(
        f"Missing required columns: {missing}"
    )

print("patient_id       : PRESENT")
print("time             : PRESENT")
print("target_future    : PRESENT")
print("Required columns : PASS")


# ============================================================
# STEP 4 - TARGET AUDIT
# ============================================================

header("STEP 4: TARGET AUDIT")

print(f"Prediction target: {TARGET_COL}")

for name, dataset in [
    ("TRAIN", train),
    ("CALIBRATION", calibration),
    ("TEST", test),
]:

    missing_target = dataset[TARGET_COL].isna().sum()

    print(
        f"{name:<12} missing targets : "
        f"{missing_target}"
    )

    if missing_target > 0:
        raise AssertionError(
            f"{name} contains missing target values."
        )

print("\nTarget missing-value check : PASS")


# ============================================================
# STEP 5 - TARGET FUTURE MUST NOT BE A FEATURE
# ============================================================

header("STEP 5: FUTURE-TARGET LEAKAGE CHECK")

if TARGET_COL in CANDIDATE_FEATURES:
    raise AssertionError(
        "LEAKAGE: target_future is included in features."
    )

print(
    "target_future in feature list : NO"
)

print(
    "Future-target leakage          : PASS"
)


# ============================================================
# STEP 6 - FORBIDDEN FEATURES
# ============================================================

header("STEP 6: FORBIDDEN FEATURE CHECK")

for forbidden in FORBIDDEN_FEATURES:

    if forbidden in CANDIDATE_FEATURES:

        raise AssertionError(
            f"Forbidden feature included: {forbidden}"
        )

    print(
        f"{forbidden:<20} : EXCLUDED"
    )

print("\nForbidden-feature check : PASS")


# ============================================================
# STEP 7 - FEATURE EXISTENCE
# ============================================================

header("STEP 7: FEATURE EXISTENCE CHECK")

missing_features = [
    feature
    for feature in CANDIDATE_FEATURES
    if feature not in train_columns
]

if missing_features:

    raise AssertionError(
        f"Feature columns not found: "
        f"{missing_features}"
    )

for feature in CANDIDATE_FEATURES:

    print(
        f"{feature:<15} : PRESENT"
    )

print("\nFeature existence check : PASS")


# ============================================================
# STEP 8 - FEATURE CONSISTENCY
# ============================================================

header("STEP 8: FEATURE CONSISTENCY")

for feature in CANDIDATE_FEATURES:

    train_dtype = train[feature].dtype
    cal_dtype = calibration[feature].dtype
    test_dtype = test[feature].dtype

    print(
        f"{feature:<15} "
        f"train={train_dtype} "
        f"cal={cal_dtype} "
        f"test={test_dtype}"
    )

    if not (
        train_dtype == cal_dtype == test_dtype
    ):

        raise AssertionError(
            f"Dtype mismatch for feature: {feature}"
        )

print("\nFeature dtype consistency : PASS")


# ============================================================
# STEP 9 - MISSING FEATURE VALUES
# ============================================================

header("STEP 9: FEATURE MISSING-VALUE AUDIT")

for feature in CANDIDATE_FEATURES:

    train_missing = train[feature].isna().sum()
    cal_missing = calibration[feature].isna().sum()
    test_missing = test[feature].isna().sum()

    print(
        f"{feature:<15} "
        f"train={train_missing} "
        f"cal={cal_missing} "
        f"test={test_missing}"
    )

print("\nMissing-value audit completed.")


# ============================================================
# STEP 10 - CHECK TARGET IS DIFFERENT FROM FEATURES
# ============================================================

header("STEP 10: FEATURE/TARGET SEPARATION")

feature_set = set(CANDIDATE_FEATURES)

if TARGET_COL in feature_set:

    raise AssertionError(
        "Target appears in feature set."
    )

if feature_set & {TARGET_COL}:

    raise AssertionError(
        "Feature/target overlap detected."
    )

print(
    "Feature-target overlap : NONE"
)

print(
    "Feature/target separation : PASS"
)


# ============================================================
# STEP 11 - PRINT FINAL FEATURE SET
# ============================================================

header("STEP 11: FINAL MODEL INPUT DESIGN")

print("X features:")

for i, feature in enumerate(
    CANDIDATE_FEATURES,
    start=1
):

    print(
        f"  X{i} = {feature}"
    )

print()
print(
    f"Y = {TARGET_COL}"
)


# ============================================================
# STEP 12 - SAVE AUDIT REPORT
# ============================================================

header("STEP 12: SAVE AUDIT REPORT")

REPORT_DIR = PROJECT_DIR / "reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_FILE = REPORT_DIR / "feature_target_audit.txt"

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "RQ1 FEATURE / TARGET AUDIT\n"
    )

    f.write(
        "==========================\n\n"
    )

    f.write(
        f"Target: {TARGET_COL}\n\n"
    )

    f.write(
        "Features:\n"
    )

    for feature in CANDIDATE_FEATURES:

        f.write(
            f"- {feature}\n"
        )

    f.write(
        "\nExcluded variables:\n"
    )

    for forbidden in FORBIDDEN_FEATURES:

        f.write(
            f"- {forbidden}\n"
        )

    f.write(
        "\nFeature-target separation: PASS\n"
    )

    f.write(
        "Future-target leakage: PASS\n"
    )


print(
    f"Audit report saved to:\n{REPORT_FILE}"
)


# ============================================================
# FINAL RESULT
# ============================================================

header("FINAL RESULT")

print(
    "FEATURE / TARGET AUDIT : SUCCESS"
)

print()
print(
    "Target : target_future"
)

print(
    "Features:"
)

for feature in CANDIDATE_FEATURES:

    print(
        f"  - {feature}"
    )

print()
print(
    "target_future excluded from X : PASS"
)

print(
    "Feature/target separation      : PASS"
)

print()
print(
    "READY FOR BASELINE MODEL"
)