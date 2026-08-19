from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

SPLIT_DIR = PROJECT_DIR / "temporal_split"

TRAIN_FILE = SPLIT_DIR / "train.csv"
CAL_FILE = SPLIT_DIR / "calibration.csv"
TEST_FILE = SPLIT_DIR / "test.csv"

OUTPUT_DIR = PROJECT_DIR / "baseline_results"

TRAIN_PRED_FILE = OUTPUT_DIR / "train_predictions.csv"
CAL_PRED_FILE = OUTPUT_DIR / "calibration_predictions.csv"
TEST_PRED_FILE = OUTPUT_DIR / "test_predictions.csv"
METRICS_FILE = OUTPUT_DIR / "baseline_metrics.txt"


# ============================================================
# MODEL DESIGN
# ============================================================

FEATURES = [
    "sleep",
    "activity",
    "stress",
    "mood",
]

TARGET = "target_future"

RANDOM_STATE = 42


# ============================================================
# HELPER
# ============================================================

def header(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    return mae, rmse


# ============================================================
# STEP 1 - LOAD DATA
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
# STEP 2 - VERIFY FEATURES
# ============================================================

header("STEP 2: VERIFY MODEL FEATURES")

for feature in FEATURES:

    if feature not in train.columns:
        raise ValueError(
            f"Missing feature: {feature}"
        )

    if feature not in calibration.columns:
        raise ValueError(
            f"Missing calibration feature: {feature}"
        )

    if feature not in test.columns:
        raise ValueError(
            f"Missing test feature: {feature}"
        )

    print(f"{feature:<15} : PASS")


if TARGET not in train.columns:
    raise ValueError(
        f"Target not found: {TARGET}"
    )

print(f"{TARGET:<15} : PASS")


# ============================================================
# STEP 3 - CREATE X AND Y
# ============================================================

header("STEP 3: CREATE FEATURES AND TARGET")

X_train = train[FEATURES]
y_train = train[TARGET]

X_cal = calibration[FEATURES]
y_cal = calibration[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]


print(
    f"X_train shape : {X_train.shape}"
)

print(
    f"y_train shape : {y_train.shape}"
)

print(
    f"X_cal shape   : {X_cal.shape}"
)

print(
    f"y_cal shape   : {y_cal.shape}"
)

print(
    f"X_test shape  : {X_test.shape}"
)

print(
    f"y_test shape  : {y_test.shape}"
)


# ============================================================
# STEP 4 - TRAIN BASELINE MODEL
# ============================================================

header("STEP 4: TRAIN RANDOM FOREST BASELINE")

model = RandomForestRegressor(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("Model training : COMPLETE")


# ============================================================
# STEP 5 - PREDICTIONS
# ============================================================

header("STEP 5: GENERATE PREDICTIONS")

train_pred = model.predict(
    X_train
)

cal_pred = model.predict(
    X_cal
)

test_pred = model.predict(
    X_test
)

print(
    f"Training predictions    : {len(train_pred)}"
)

print(
    f"Calibration predictions : {len(cal_pred)}"
)

print(
    f"Test predictions        : {len(test_pred)}"
)


# ============================================================
# STEP 6 - BASELINE METRICS
# ============================================================

header("STEP 6: CALCULATE BASELINE METRICS")

train_mae, train_rmse = calculate_metrics(
    y_train,
    train_pred
)

cal_mae, cal_rmse = calculate_metrics(
    y_cal,
    cal_pred
)

test_mae, test_rmse = calculate_metrics(
    y_test,
    test_pred
)


print("\nTRAINING")
print(f"MAE  : {train_mae:.6f}")
print(f"RMSE : {train_rmse:.6f}")

print("\nCALIBRATION")
print(f"MAE  : {cal_mae:.6f}")
print(f"RMSE : {cal_rmse:.6f}")

print("\nTEST")
print(f"MAE  : {test_mae:.6f}")
print(f"RMSE : {test_rmse:.6f}")


# ============================================================
# STEP 7 - SAVE PREDICTIONS
# ============================================================

header("STEP 7: SAVE PREDICTIONS")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


train_output = train[
    ["patient_id", "time", TARGET]
].copy()

train_output["prediction"] = train_pred

train_output["absolute_error"] = np.abs(
    train_output[TARGET]
    - train_output["prediction"]
)


cal_output = calibration[
    ["patient_id", "time", TARGET]
].copy()

cal_output["prediction"] = cal_pred

cal_output["absolute_error"] = np.abs(
    cal_output[TARGET]
    - cal_output["prediction"]
)


test_output = test[
    ["patient_id", "time", TARGET]
].copy()

test_output["prediction"] = test_pred

test_output["absolute_error"] = np.abs(
    test_output[TARGET]
    - test_output["prediction"]
)


train_output.to_csv(
    TRAIN_PRED_FILE,
    index=False
)

cal_output.to_csv(
    CAL_PRED_FILE,
    index=False
)

test_output.to_csv(
    TEST_PRED_FILE,
    index=False
)


print(
    f"Training predictions saved:\n{TRAIN_PRED_FILE}"
)

print(
    f"Calibration predictions saved:\n{CAL_PRED_FILE}"
)

print(
    f"Test predictions saved:\n{TEST_PRED_FILE}"
)


# ============================================================
# STEP 8 - SAVE METRICS
# ============================================================

metrics_text = f"""
RQ1 BASELINE RANDOM FOREST RESULTS
===================================

Features:
{", ".join(FEATURES)}

Target:
{TARGET}

Model:
RandomForestRegressor

n_estimators:
300

random_state:
{RANDOM_STATE}


TRAINING
--------
Samples: {len(train)}
MAE: {train_mae:.6f}
RMSE: {train_rmse:.6f}


CALIBRATION
-----------
Samples: {len(calibration)}
MAE: {cal_mae:.6f}
RMSE: {cal_rmse:.6f}


TEST
----
Samples: {len(test)}
MAE: {test_mae:.6f}
RMSE: {test_rmse:.6f}
"""


with open(
    METRICS_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(metrics_text)


print(
    f"\nMetrics saved:\n{METRICS_FILE}"
)


# ============================================================
# FINAL RESULT
# ============================================================

header("FINAL RESULT")

print("BASELINE MODEL : SUCCESS")

print()
print(
    f"Training MAE    : {train_mae:.6f}"
)

print(
    f"Calibration MAE : {cal_mae:.6f}"
)

print(
    f"Test MAE        : {test_mae:.6f}"
)

print()
print(
    f"Training RMSE    : {train_rmse:.6f}"
)

print(
    f"Calibration RMSE : {cal_rmse:.6f}"
)

print(
    f"Test RMSE        : {test_rmse:.6f}"
)

print()
print("READY FOR CONFORMAL CALIBRATION")