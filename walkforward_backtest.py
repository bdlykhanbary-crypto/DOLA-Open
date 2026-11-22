from pathlib import Path
import os
import numpy as np
import pandas as pd
import xgboost as xgb

BASE = str(Path(__file__).resolve().parent)
DATA = BASE + "/data/"
RESULTS = BASE + "/results/"
os.makedirs(RESULTS, exist_ok=True)

features = [
    "return_1", "return_2", "return_3", "return_5",
    "return_7", "return_14", "return_21", "return_30",
    "return_45", "return_63", "volatility_7", "volatility_14",
    "volatility_30", "volatility_60", "daily_range",
    "open_gap", "close_position", "weekday", "month"
]

df = pd.read_csv(DATA + "usd_features_30d.csv")
test = pd.read_csv(DATA + "test_purged.csv")

df["date"] = pd.to_datetime(df["date"])
df["target_date"] = pd.to_datetime(df["target_date"])
test["date"] = pd.to_datetime(test["date"])
test["target_date"] = pd.to_datetime(test["target_date"])

# انتخاب تاریخ‌های آزمون با فاصله حداقل 30 روز
selected = []
next_allowed = None

for _, row in test.iterrows():
    if next_allowed is None or row["date"] >= next_allowed:
        selected.append(row)
        next_allowed = row["target_date"]

params = {
    "objective": "reg:squarederror",
    "max_depth": 3,
    "eta": 0.03,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 5,
    "reg_lambda": 5,
    "nthread": 1,
    "seed": 42
}

def write_libsvm(frame, path, include_label=True):
    with open(path, "w") as f:
        for _, row in frame.iterrows():
            values = row[features].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("Non-finite feature detected")
            label = float(row["target_return_30d"]) if include_label else 0.0
            line = [f"{label:.12g}"]
            line += [f"{i+1}:{v:.12g}" for i, v in enumerate(values)]
            f.write(" ".join(line) + "\n")

results = []

for number, point in enumerate(selected, 1):
    origin = point["date"]

    # فقط برچسب‌هایی که پیش از تاریخ پیش‌بینی محقق شده‌اند
    train = df[
        (df["date"] < origin) &
        (df["target_date"] < origin)
    ].copy()

    current = df[df["date"] == origin].copy()

    if len(train) < 500 or len(current) != 1:
        print("Skipped:", origin.date(), "train:", len(train))
        continue

    train_path = RESULTS + "wf_train.libsvm"
    pred_path = RESULTS + "wf_predict.libsvm"

    write_libsvm(train, train_path, True)
    write_libsvm(current, pred_path, False)

    dtrain = xgb.DMatrix(train_path + "?format=libsvm")
    dpred = xgb.DMatrix(pred_path + "?format=libsvm")

    model = xgb.train(params, dtrain, num_boost_round=10)
    prediction = float(model.predict(dpred)[0])

    actual = float(point["target_return_30d"])

    results.append({
        "date": origin.date(),
        "target_date": point["target_date"].date(),
        "train_rows": len(train),
        "actual_return": actual,
        "predicted_return": prediction,
        "absolute_error": abs(actual - prediction)
    })

    print(
        f"{number}/{len(selected)}",
        origin.date(),
        "train:", len(train),
        "prediction:", round(prediction, 5),
        "actual:", round(actual, 5)
    )

out = pd.DataFrame(results)

if len(out) == 0:
    raise RuntimeError("No valid walk-forward predictions")

y = out["actual_return"].to_numpy()
p = out["predicted_return"].to_numpy()

mae = np.mean(np.abs(y-p))
rmse = np.sqrt(np.mean((y-p)**2))
base_mae = np.mean(np.abs(y))
base_rmse = np.sqrt(np.mean(y**2))
direction = np.mean(np.sign(y) == np.sign(p)) * 100

print("\nWALK-FORWARD RESULTS")
print("Predictions:", len(out))
print("MAE:", round(float(mae), 6))
print("RMSE:", round(float(rmse), 6))
print("Direction accuracy:", round(float(direction), 2), "%")
print("No-change MAE:", round(float(base_mae), 6))
print("No-change RMSE:", round(float(base_rmse), 6))
print("RMSE improvement:", round(float(100*(1-rmse/base_rmse)), 2), "%")

out.to_csv(RESULTS + "walkforward_backtest.csv", index=False)
print("Saved: results/walkforward_backtest.csv")
