from pathlib import Path
import os
import numpy as np
import xgboost as xgb

BASE = str(Path(__file__).resolve().parent)
DATA = BASE + "/data/"
MODEL = BASE + "/models/usd_baseline.json"

model = xgb.Booster()
model.load_model(MODEL)

dtest = xgb.DMatrix(DATA + "test_30d.libsvm?format=libsvm")
actual = dtest.get_label()
pred = model.predict(
    dtest,
    iteration_range=(0, model.best_iteration + 1)
)
baseline = np.zeros_like(actual)

for name, p in [
    ("XGBoost", pred),
    ("No-change baseline", baseline)
]:
    mae = np.mean(np.abs(actual - p))
    rmse = np.sqrt(np.mean((actual - p) ** 2))
    print(name)
    print("  MAE :", round(float(mae), 6))
    print("  RMSE:", round(float(rmse), 6))

base_rmse = np.sqrt(np.mean(actual ** 2))
model_rmse = np.sqrt(np.mean((actual - pred) ** 2))
improvement = 100 * (1 - model_rmse / base_rmse)

print("\nTest RMSE improvement:", round(float(improvement), 2), "%")
print("Test samples:", len(actual))
