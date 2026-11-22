from pathlib import Path
import os
import numpy as np
import pandas as pd
import xgboost as xgb

BASE = str(Path(__file__).resolve().parent)
DATA = BASE + "/data/"
MODEL = BASE + "/models/usd_baseline.json"

df = pd.read_csv(DATA + "test_purged.csv", parse_dates=["date", "target_date"])

dtest = xgb.DMatrix(DATA + "test_30d.libsvm?format=libsvm")

model = xgb.Booster()
model.load_model(MODEL)

pred = model.predict(
    dtest,
    iteration_range=(0, model.best_iteration + 1)
)

actual = df["target_return_30d"].to_numpy()
dates = df["date"].to_numpy()
targets = df["target_date"].to_numpy()

# انتخاب تاریخ‌هایی که افق هدفشان با هم هم‌پوشانی ندارد
selected = []
next_allowed = None

for i in range(len(df)):
    if next_allowed is None or dates[i] >= next_allowed:
        selected.append(i)
        next_allowed = targets[i]

idx = np.array(selected, dtype=int)

y = actual[idx]
p = pred[idx]
baseline = np.zeros_like(y)

def report(name, truth, estimate):
    mae = np.mean(np.abs(truth - estimate))
    rmse = np.sqrt(np.mean((truth - estimate) ** 2))
    direction = np.mean(np.sign(truth) == np.sign(estimate)) * 100

    print(name)
    print("MAE:", round(float(mae), 6))
    print("RMSE:", round(float(rmse), 6))
    print("Direction accuracy:", round(float(direction), 2), "%")

print("NON-OVERLAPPING TEST BACKTEST")
print("Selected samples:", len(idx))
print("First date:", df.iloc[idx[0]]["date"].date())
print("Last date:", df.iloc[idx[-1]]["date"].date())

report("XGBoost", y, p)
report("No-change baseline", y, baseline)

improvement = 100 * (
    1 - np.sqrt(np.mean((y-p)**2)) /
    np.sqrt(np.mean(y**2))
)

print("RMSE improvement:", round(float(improvement), 2), "%")

out = df.iloc[idx][["date", "target_date", "target_return_30d"]].copy()
out["prediction"] = p
out["absolute_error"] = np.abs(y-p)
out.to_csv(BASE + "/results/nonoverlap_backtest.csv", index=False)

print("Report saved: results/nonoverlap_backtest.csv")
