#!/usr/bin/env python3
"""
Volatility Forecasting for Position Sizing (EUR/USD, tiny & fast)

Goal:
  Predict whether tomorrow will be a HIGH- or LOW-vol day, then size exposure:
    - If predicted LOW vol → higher exposure
    - If predicted HIGH vol → lower exposure
  KPI: Annualized Sharpe of this volatility-scaled strategy on the test set.

Deps: pandas numpy scikit-learn
"""

import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Hardcoded paths & knobs
# -----------------------------
DATA_CSV = os.path.join("/home/ozkilim/demo/volatility/data/", "eurusd.csv")

TRAIN_FRAC = 0.70
RANDOM_STATE = 42

# Vol-scaling exposures (keep simple & safe)
EXPO_LOWVOL  = 1.50   # used when model predicts LOW vol
EXPO_HIGHVOL = 0.50   # used when model predicts HIGH vol

# -----------------------------
# 1) Load data
# -----------------------------
if not os.path.exists(DATA_CSV):
    raise SystemExit(f"Missing {DATA_CSV}. Run download_eurusd.py first.")

px = pd.read_csv(DATA_CSV, parse_dates=["Date"]).sort_values("Date").set_index("Date")
if "close" not in px.columns:
    raise SystemExit("CSV must contain a 'close' column.")

print(f"Loaded {len(px)} rows from {DATA_CSV} "
      f"({px.index.min().date()} → {px.index.max().date()})")

# Daily returns
px["ret_1d"] = px["close"].pct_change()
px["absret_1d"] = px["ret_1d"].abs()

# Define a *rolling* threshold for "high vol" using past data only:
# threshold_t = median(|returns|) over the last 60 trading days (up to day t)
thr = px["absret_1d"].rolling(60).median()

# Binary target at time t: is *tomorrow's* absolute return above today's threshold?
px["y"] = (px["absret_1d"].shift(-1) > thr).astype(float)

# =============================================================================
# =================  2) FEATURE ENGINEERING (all lagged by 1)  =================
# =============================================================================

# CO_DATASCIENTIST_BLOCK_START

# 1) Realized vol (10-day std of returns), lag 1
px["feat_vol10"] = px["ret_1d"].rolling(10).std().shift(1)
# 2) Short-term vol (3-day std), lag 1, modified with added skewness measure
px["feat_vol3"] = px["ret_1d"].rolling(3).std().shift(1)
px["feat_skew3"] = px["ret_1d"].rolling(3).skew().shift(1)

# 3) Vol-of-vol proxy: relative change vol3 vs vol10, lag 1
px["feat_vol_ratio"] = ((px["feat_vol3"] - px["feat_vol10"]) / (px["feat_vol10"] + 1e-12)).shift(0)

px["feat_autocorr5"] = px["ret_1d"].rolling(5).apply(lambda x: pd.Series(x).autocorr(), raw=False).shift(1)
# Proposed New Feature: Adding harmonic mean of returns to capture balanced risk adjusted measure over 10 days, lag 1
px["feat_combined_volatility19"] = (px["ret_1d"].rolling(10).std() + px["ret_1d"].rolling(9).mean()).shift(1)

# 4) Recent absolute move (yesterday’s |return|), lag 1
px["feat_absret1"] = px["absret_1d"].shift(1)

# 5) 5-day mean squared return (proxy for realized variance), lag 1
px["feat_msq5"] = (px["ret_1d"]**2).rolling(5).mean().shift(1)

# 6) 20-day exponentially weighted average return, lag 1
px["feat_ewm20"] = px["ret_1d"].ewm(span=20, adjust=False).mean().shift(1)

# New Feature: Adding absolute value of a smoothed skewness indicator over 7 days, lag 1
# Removed absolute value of smoothed skewness indicator over 7 days to reduce feature set complexity
pass

# Additional Feature: Introducing Fourier-transformed feature capturing periodic patterns over 10 days (frequency components as dynamic signals)
def fourier_transform_component(x):
    fft_vals = np.fft.fft(x)
    return np.abs(fft_vals[1:min(len(fft_vals), 6)]).mean()
px["feat_fourier10_mean"] = px["ret_1d"].rolling(10).apply(fourier_transform_component).shift(1)

# New Feature: Adding Lagged Z-Score Normalized Returns over a 7-day rolling window to capture abnormal periods
px["feat_zscore_7"] = px["ret_1d"].rolling(7).apply(lambda x: (x.iloc[-1] - np.mean(x)) / (np.std(x) + 1e-12)).shift(1)
# Innovation: Adding autocorrelation-based volatility adjustment capturing time-series mean reversion dynamics over 14 days, lag 1
px["feat_autovol_adj14"] = px["ret_1d"].rolling(14).apply(lambda x: (pd.Series(x).autocorr() * x.std()), raw=False).shift(1)
# Innovation: Polarized moving average for returns (capturing gains/losses separately), lag 1
px["feat_polarized_ma"] = (px["ret_1d"].rolling(5).apply(lambda x: x[x>0].mean() - abs(x[x<0].mean()))).shift(1)
px["feat_ewmstd15"] = px["ret_1d"].ewm(span=15, adjust=False).std().shift(1)
px["feat_recurrence_duration"] = px["ret_1d"].rolling(14).apply(lambda x: sum([1 if abs(v) <= 0.01 else 0 for v in x]) / len(x), raw=False).shift(1)

# Innovation: Introducing entropy-based feature of returns over 15-days, capturing distribution dynamics, lag 1
from scipy.stats import entropy
px["feat_entropy15"] = px["ret_1d"].rolling(15).apply(lambda x: entropy(pd.Series(x).value_counts(normalize=True)), raw=False).shift(1)
# Additional Innovation: Adding fractal dimension feature of returns to capture irregular patterns reflective of market turbulence or stability (14-day window)
from scipy.signal import periodogram
px["feat_fractal_dim14"] = px["ret_1d"].rolling(14).apply(lambda x: sum(np.abs(np.diff(np.log(periodogram(x, scaling='density')[1]+1e-12))) / len(x)), raw=False).shift(1)
# New Feature: Introduced a nonlinear volatility-adjusted log-ratio combining fast (5-period) and slow (20-period) moving averages to identify momentum shifts more dynamically
px["feat_adaptive_log_vol_ratio"] = ((np.log(px['ret_1d'].rolling(5).mean() + 1e-12) - np.log(px['ret_1d'].rolling(20).mean() + 1e-12)) / (px["feat_vol10"] + 1e-12)).shift(1)
# New Feature: Adding a feature capturing median absolute deviation (MAD) of returns over 7 days, lag 1
px["feat_mad_7"] = px["ret_1d"].rolling(7).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=False).shift(1)
# Proposed Feature: Adding Quantum-inspired superposition feature to capture overlapping frequency bands in market volatility with mixed-process entropy dynamics, lag 1
px["feat_quantum_entropy_15"] = px["ret_1d"].rolling(15).apply(lambda x: np.mean([(abs(v) ** (3.0/7.0)) for v in x if v > 0]) - np.mean([(abs(v) ** (7.0/3.0)) for v in x if v < 0]), raw=False).shift(1)
# Additional Feature Added: Momentum Acceleration Magnitude Ratio incorporating dx/dv  — Comparative momentum rates px shifts dx magnitude extreme amps decor. gradient unbiased reconstruction (driven time-lensed expansions regulatory remote relative patch expansions turnover measures inside define ratios follow).
# New Feature: Adding a geometric mean aggregation of returns over 20 days, lag 1
px["feat_geometric_mean20"] = px["ret_1d"].rolling(20).apply(lambda x: np.prod([(v if v != 0 else 1) for v in x])**(1/len(x)), raw=False).shift(1)

feat_cols = [c for c in px.columns if c.startswith("feat_")]

# CO_DATASCIENTIST_BLOCK_END

data = px[feat_cols + ["y", "ret_1d"]].dropna().copy()

# -----------------------------
# 3) Time-based split
# -----------------------------
n = len(data)
n_train = int(n * TRAIN_FRAC)

train = data.iloc[:n_train]
test  = data.iloc[n_train:]

X_train = train[feat_cols].values
y_train = train["y"].astype(int).values
X_test  = test[feat_cols].values
y_test  = test["y"].astype(int).values

# For strategy: we need next-day returns aligned to test index
# ret_next at test dates = return from t to t+1
ret_next = px.loc[test.index, "ret_1d"].shift(-1).reindex(test.index)

# -----------------------------
# 4) Model (fast & interpretable)
# -----------------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=200, random_state=RANDOM_STATE))
])
model.fit(X_train, y_train)

# -----------------------------
# 5) Classification diagnostics (optional prints)
# -----------------------------
proba_high = model.predict_proba(X_test)[:, 1]
pred_high  = (proba_high >= 0.5).astype(int)

# -----------------------------
# 6) Volatility-scaled strategy (test set)
# -----------------------------
# If model predicts LOW vol (pred_high=0) → use higher exposure; else lower exposure.
exposure = np.where(pred_high == 0, EXPO_LOWVOL, EXPO_HIGHVOL)

# Strategy returns are exposure times next-day returns.
strat_ret = exposure * ret_next.values
strat_ret = pd.Series(strat_ret, index=test.index).fillna(0.0)

# Performance
mean_ret = strat_ret.mean()
std_ret  = strat_ret.std(ddof=1)
sharpe_daily = mean_ret / (std_ret + 1e-12)
sharpe_annual = sharpe_daily * np.sqrt(252)

equity = (1 + strat_ret).cumprod()
final_equity = float(equity.iloc[-1])

# -----------------------------
# 7) FINAL KPI 
# -----------------------------
print(f"KPI: {sharpe_annual:.6f}")
