from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "sample_market_data.csv"
OUT = ROOT / "data" / "usd_features_30d.csv"

df = pd.read_csv(SOURCE, parse_dates=["date"])
df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

for c in ["open", "high", "low", "close"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["date", "close"])
df = df[df["close"] > 0].reset_index(drop=True)

logp = np.log(df["close"])
ret = logp.diff()

features = pd.DataFrame(index=df.index)

# گذشته قیمت و بازده
for lag in [1, 2, 3, 5, 7, 14, 21, 30, 45, 63]:
    features[f"return_{lag}"] = logp.diff(lag)

# نوسان گذشته
for window in [7, 14, 30, 60]:
    features[f"volatility_{window}"] = ret.rolling(window).std()

# دامنه و موقعیت بسته‌شدن کندل
features["daily_range"] = np.log(df["high"] / df["low"])
features["open_gap"] = np.log(df["open"] / df["close"].shift(1))
features["close_position"] = (
    (df["close"] - df["low"]) /
    (df["high"] - df["low"]).replace(0, np.nan)
)

# ویژگی‌های تقویمی
features["weekday"] = df["date"].dt.weekday
features["month"] = df["date"].dt.month

# اولین مشاهده در روز 30 یا بعد از آن
dates = df["date"].to_numpy(dtype="datetime64[ns]")
rows = []

for i in range(len(df)):
    target_date = dates[i] + np.timedelta64(30, "D")
    j = int(np.searchsorted(dates, target_date, side="left"))

    if j >= len(df) or j <= i:
        continue

    actual_days = int(
        (dates[j] - dates[i]) / np.timedelta64(1, "D")
    )

    # حذف فاصله‌های غیرعادی
    if actual_days > 37:
        continue

    row = features.iloc[i].to_dict()
    row.update({
        "date": df.loc[i, "date"],
        "target_date": df.loc[j, "date"],
        "calendar_days": actual_days,
        "close_irr": float(df.loc[i, "close"]),
        "close_toman": float(df.loc[i, "close"]) / 10,
        "target_close_irr": float(df.loc[j, "close"]),
        "target_close_toman": float(df.loc[j, "close"]) / 10,
        "target_return_30d": float(logp.iloc[j] - logp.iloc[i]),
    })
    rows.append(row)

result = pd.DataFrame(rows)
feature_cols = list(features.columns)

result = result.replace([np.inf, -np.inf], np.nan)
result = result.dropna(subset=feature_cols + ["target_return_30d"])
result = result.reset_index(drop=True)

result.to_csv(OUT, index=False)

print("SAVED:", OUT)
print("SAMPLES:", len(result))
print("FEATURE COUNT:", len(feature_cols))
print("FEATURES:", feature_cols)
print("DATE RANGE:", result["date"].min(), "to", result["date"].max())
print("TARGET HORIZON DAYS:")
print(result["calendar_days"].value_counts().sort_index().to_string())
print("\nLAST 5 SAMPLES:")
print(result[[
    "date", "target_date", "calendar_days",
    "close_toman", "target_close_toman",
    "target_return_30d"
]].tail().to_string(index=False))

# تقسیم زمانی، بدون مخلوط‌کردن گذشته و آینده
n = len(result)
a = int(n * 0.70)
b = int(n * 0.85)

print("\nTEMPORAL SPLITS:")
print("TRAIN:", n and result.iloc[0]["date"], "to", result.iloc[a-1]["date"])
print("VALIDATION:", result.iloc[a]["date"], "to", result.iloc[b-1]["date"])
print("TEST:", result.iloc[b]["date"], "to", result.iloc[-1]["date"])
