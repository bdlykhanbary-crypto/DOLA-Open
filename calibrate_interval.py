from pathlib import Path
import os
import numpy as np
import xgboost as xgb

BASE = str(Path(__file__).resolve().parent)
DATA = BASE + "/data/"
MODELS = BASE + "/models/"

def load_predictions(part):
    d = xgb.DMatrix(DATA + part + "_30d.libsvm?format=libsvm")
    actual = d.get_label()
    preds = {}

    for alpha in [10, 90]:
        model = xgb.Booster()
        model.load_model(MODELS + f"usd_q{alpha}.json")
        preds[alpha] = model.predict(
            d, iteration_range=(0, model.best_iteration + 1)
        )

    return actual, preds[10], preds[90]

# Calibration on validation
y_val, lo_val, hi_val = load_predictions("validation")

scores = np.maximum.reduce([
    lo_val - y_val,
    y_val - hi_val,
    np.zeros_like(y_val)
])

n = len(scores)
level = min(1.0, np.ceil((n + 1) * 0.80) / n)
delta = np.quantile(scores, level, method="higher")

print("Calibration samples:", n)
print("Expansion delta:", round(float(delta), 6))

# Final evaluation on untouched test
y_test, lo_test, hi_test = load_predictions("test")

lo_cal = lo_test - delta
hi_cal = hi_test + delta

coverage = np.mean((y_test >= lo_cal) & (y_test <= hi_cal))
width = np.mean(hi_cal - lo_cal)

print("\nCALIBRATED TEST INTERVAL")
print("Coverage:", round(float(coverage * 100), 2), "%")
print("Mean width:", round(float(width), 6))
print("Test samples:", len(y_test))
