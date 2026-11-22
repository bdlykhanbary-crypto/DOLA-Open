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
val = pd.read_csv(DATA + "validation_purged.csv")

for frame in (df, val):
    frame["date"] = pd.to_datetime(frame["date"])
    frame["target_date"] = pd.to_datetime(frame["target_date"])

selected = []
next_allowed = None

for _, row in val.iterrows():
    if next_allowed is None or row["date"] >= next_allowed:
        selected.append(row)
        next_allowed = row["target_date"]

candidates = [
    (2, 0.03, 5),
    (3, 0.03, 5),
    (4, 0.03, 5),
    (3, 0.05, 5),
    (3, 0.03, 10),
    (2, 0.05, 3)
]

def write_libsvm(frame, path):
    with open(path, "w") as f:
        for _, row in frame.iterrows():
            values = row[features].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("Invalid feature")
            label = float(row["target_return_30d"])
            parts = [f"{label:.12g}"]
            parts += [
                f"{i+1}:{v:.12g}"
                for i, v in enumerate(values)
            ]
            f.write(" ".join(parts) + "\n")

results = []

for ci, (depth, eta, child) in enumerate(candidates, 1):
    predictions = []
    actuals = []

    print(
        f"\nCandidate {ci}/{len(candidates)}:",
        "depth", depth, "eta", eta, "min_child", child
    )

    for j, point in enumerate(selected, 1):
        origin = point["date"]

        train = df[
            (df["date"] < origin) &
            (df["target_date"] < origin)
        ].copy()

        current = df[df["date"] == origin].copy()

        if len(train) < 500 or len(current) != 1:
            continue

        train_path = RESULTS + "tune_train.libsvm"
        pred_path = RESULTS + "tune_pred.libsvm"

        write_libsvm(train, train_path)
        write_libsvm(current, pred_path)

        dtrain = xgb.DMatrix(train_path + "?format=libsvm")
        dpred = xgb.DMatrix(pred_path + "?format=libsvm")

        params = {
            "objective": "reg:squarederror",
            "max_depth": depth,
            "eta": eta,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": child,
            "reg_lambda": 5,
            "nthread": 1,
            "seed": 42
        }

        model = xgb.train(params, dtrain, num_boost_round=10)
        predictions.append(float(model.predict(dpred)[0]))
        actuals.append(float(point["target_return_30d"]))

    y = np.array(actuals)
    p = np.array(predictions)

    mae = float(np.mean(np.abs(y-p)))
    rmse = float(np.sqrt(np.mean((y-p)**2)))

    results.append({
        "max_depth": depth,
        "eta": eta,
        "min_child_weight": child,
        "samples": len(y),
        "MAE": mae,
        "RMSE": rmse
    })

    print("Samples:", len(y), "MAE:", round(mae, 6),
          "RMSE:", round(rmse, 6))

out = pd.DataFrame(results).sort_values("RMSE")
out.to_csv(RESULTS + "xgboost_tuning.csv", index=False)

print("\nRANKING BY VALIDATION RMSE")
print(out.to_string(index=False))
print("\nSaved: results/xgboost_tuning.csv")
