from pathlib import Path
import os
import numpy as np
import xgboost as xgb

BASE = str(Path(__file__).resolve().parent)
DATA = BASE + "/data/"
MODEL = BASE + "/models/usd_baseline.json"

model = xgb.Booster()
model.load_model(MODEL)

dval = xgb.DMatrix(DATA + "validation_30d.libsvm?format=libsvm")
actual = dval.get_label()
pred = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
baseline = np.zeros_like(actual)

def metrics(name, y, p):
    print(name)
    print("  MAE :", round(float(np.mean(np.abs(y-p))), 6))
    print("  RMSE:", round(float(np.sqrt(np.mean((y-p)**2))), 6))

metrics("XGBoost", actual, pred)
metrics("No-change baseline", actual, baseline)

improvement = 100 * (
    1 - np.sqrt(np.mean((actual-pred)**2)) /
    np.sqrt(np.mean(actual**2))
)
print("\nRMSE improvement vs baseline:", round(float(improvement), 2), "%")
