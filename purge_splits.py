from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
src = root / "data" / "usd_features_30d.csv"
df = pd.read_csv(src, parse_dates=["date", "target_date"])

n = len(df)
a = int(n * 0.70)
b = int(n * 0.85)

val_start = df.iloc[a]["date"]
test_start = df.iloc[b]["date"]

# هدف آموزش باید قبل از شروع اعتبارسنجی باشد
train = df.iloc[:a].copy()
train = train[train["target_date"] < val_start]

# هدف اعتبارسنجی باید قبل از شروع آزمون باشد
valid = df.iloc[a:b].copy()
valid = valid[valid["target_date"] < test_start]

test = df.iloc[b:].copy()

out = root / "data"
train.to_csv(out / "train_purged.csv", index=False)
valid.to_csv(out / "validation_purged.csv", index=False)
test.to_csv(out / "test_purged.csv", index=False)

for name, part in [
    ("TRAIN", train),
    ("VALIDATION", valid),
    ("TEST", test)
]:
    print("\n", name)
    print("SAMPLES:", len(part))
    print("DATE:", part["date"].min(), "to", part["date"].max())
    print("LAST TARGET DATE:", part["target_date"].max())

print("\nLEAKAGE CHECK:")
print("TRAIN TARGET < VALID START:",
      bool(train["target_date"].max() < val_start))
print("VALID TARGET < TEST START:",
      bool(valid["target_date"].max() < test_start))
print("\nFILES SAVED IN:", out)
