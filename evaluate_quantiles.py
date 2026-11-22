from pathlib import Path
import os
import numpy as np
import xgboost as xgb

BASE = str(Path(__file__).resolve().parent)
DATA = BASE + "/data/"
MODELS = BASE + "/models/"

dtest = xgb.DMatrix(DATA + "test_30d.libsvm?format=libsvm")
actual = dtest.get_label()

predictions = {}

for alpha in [10, 50, 90]:
    model = xgb.Booster()
    model.load_model(MODELS + f"usd_q{alpha}.json")
    predictions[alpha] = model.predict(
        dtest,
        iteration_range=(0, model.best_iteration + 1)
    )

q10 = predictions[10]
q50 = predictions[50]
q90 = predictions[90]

def pinball(y, p, alpha):
    error = y - p
    return np.mean(np.maximum(alpha * error, (alpha - 1) * error))

print("TEST QUANTILE METRICS")
for alpha in [10, 50, 90]:
    score = pinball(actual, predictions[alpha], alpha / 100)
    print(f"Q{alpha} pinball loss:", round(float(score), 6))

crossings = np.sum(q10 > q90)
coverage = np.mean((actual >= q10) & (actual <= q90))
width = np.mean(q90 - q10)

print("\nPREDICTION INTERVAL")
print("Q10 > Q90 crossings:", int(crossings))
print("Raw interval coverage:", round(float(coverage * 100), 2), "%")
print("Mean interval width (log return):", round(float(width), 6))
print("Median prediction MAE:", round(float(np.mean(np.abs(actual-q50))), 6))
print("Test samples:", len(actual))
