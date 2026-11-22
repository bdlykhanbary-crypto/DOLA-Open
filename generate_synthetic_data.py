from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

rng = np.random.default_rng(42)
n = 2200

dates = pd.date_range("2018-01-01", periods=n, freq="D")
returns = rng.normal(0.0004, 0.012, n)
close = 100000 * np.exp(np.cumsum(returns))

open_price = close * np.exp(rng.normal(0, 0.003, n))
spread = rng.uniform(0.001, 0.012, n)

df = pd.DataFrame({
    "date": dates,
    "open": open_price,
    "high": np.maximum(open_price, close) * (1 + spread),
    "low": np.minimum(open_price, close) * (1 - spread),
    "close": close,
})

out = DATA / "sample_market_data.csv"
df.to_csv(out, index=False, date_format="%Y-%m-%d")

print("Synthetic sample created:", out)
print("Rows:", len(df))
print("Date range:", df.date.min().date(), "to", df.date.max().date())
print("NOTICE: This is simulated data, not real market prices.")
