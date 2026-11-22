from pathlib import Path
import os
import xgboost as xgb

BASE = str(Path(__file__).resolve().parent)
DATA = BASE + "/data/"
MODELS = BASE + "/models/"
os.makedirs(MODELS, exist_ok=True)

dtrain = xgb.DMatrix(DATA + "train_30d.libsvm?format=libsvm")
dval = xgb.DMatrix(DATA + "validation_30d.libsvm?format=libsvm")

for alpha in [0.1, 0.5, 0.9]:
    print(f"\nTraining quantile alpha={alpha}")

    params = {
        "objective": "reg:quantileerror",
        "quantile_alpha": alpha,
        "eval_metric": "quantile",
        "max_depth": 3,
        "eta": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
        "reg_lambda": 5,
        "tree_method": "hist",
        "nthread": 1,
        "seed": 42
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=40,
        verbose_eval=50
    )

    path = MODELS + f"usd_q{int(alpha*100)}.json"
    model.save_model(path)

    print("Best iteration:", model.best_iteration)
    print("Best validation score:", model.best_score)
    print("Saved:", path)

print("\nAll quantile models trained.")
