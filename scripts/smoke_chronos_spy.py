"""E1 smoke test: Chronos-2 zero-shot forecast on SPY."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.ml.models.chronos import ChronosForecaster

df = pd.read_csv("data/raw/SPY_daily.csv", parse_dates=True, index_col=0).dropna()
close = df["Close"]
print(f"SPY data: {len(close)} bars, {close.index[0]} to {close.index[-1]}")
print(f"Last close: {close.iloc[-1]:.2f}")

print("Loading chronos-2 (120M params)...")
fc = ChronosForecaster(model_size="chronos-2", device="cpu", prediction_length=21)

print("Generating 21-day forecast...")
result = fc.predict(close, num_samples=100)

mean_fc = result["mean"]
median_fc = result["median"]
q10 = result["q10"]
q90 = result["q90"]

print(f"Forecast horizon: {mean_fc.index[0]} to {mean_fc.index[-1]}")
print(f"Last price: {close.iloc[-1]:.2f}")
print(f"Mean forecast end: {mean_fc.iloc[-1]:.2f}")
print(f"Expected return: {(mean_fc.iloc[-1] / close.iloc[-1] - 1) * 100:.2f}%")
print(f"Median forecast end: {median_fc.iloc[-1]:.2f}")
print(f"90% CI: [{q10.iloc[-1]:.2f}, {q90.iloc[-1]:.2f}]")
print("SUCCESS: chronos-2 zero-shot SPY forecast complete")
