#!/usr/bin/env python3
# Downloads daily EUR/USD spot (Yahoo Finance: EURUSD=X) and writes a tidy CSV.
# Columns: Date, close

import os
import pandas as pd
import yfinance as yf

DATA_DIR = "data"
OUT_CSV  = os.path.join(DATA_DIR, "eurusd.csv")

TICKER = "EURUSD=X"
START  = "2019-01-01"
END    = None  # to today

os.makedirs(DATA_DIR, exist_ok=True)

print("Downloading EUR/USD…")
df = yf.download(TICKER, start=START, end=END, progress=False)
if df.empty:
    raise SystemExit("No data returned for EURUSD=X.")

out = df[["Close"]].rename(columns={"Close": "close"}).reset_index()  # Date, close
out.to_csv(OUT_CSV, index=False)

print(f"Wrote {len(out)} rows to {OUT_CSV} "
      f"({out['Date'].min().date()} → {out['Date'].max().date()})")
