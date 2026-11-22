from pathlib import Path
import os
import json
import xgboost as xgb

BASE = str(Path(__file__).resolve().parent)
DATA = BASE + "/data/"
MODELS = BASE + "/models/"
os.makedirs(MODELS, exist_ok=True)

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 3,
    "eta": 0.03,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 5,
    "reg_lambda": 5,
    "nthread": 1,
    "seed": 42
}

dtrain = xgb.DMatrix(DATA + "train_30d.libsvm?format=libsvm")
dval = xgb.DMatrix(DATA + "validation_30d.libsvm?format=libsvm")

model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=[(dtrain, "train"), (dval, "validation")],
    early_stopping_rounds=40,
    verbose_eval=50
)

path = MODELS + "usd_baseline.json"
model.save_model(path)

print("\nTraining completed.")
print("Best iteration:", model.best_iteration)
print("Best validation RMSE:", model.best_score)
print("Model saved:", path)
