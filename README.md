# Volatility Forecasting for Position Sizing (EUR/USD)

Welcome to a practical, fast, and actionable template for volatility forecasting—so you can scale positions intelligently in EUR/USD.

Instead of the impossible—predicting where the price will go—this framework forecasts **risk**:  
> **Will tomorrow be a high-vol or low-vol day?**

## Overview

- **Objective:** Predict "high-vol" vs. "low-vol" days for EUR/USD, and use this information to size your trading positions dynamically.
- **Why?** Because professionals manage risk, not just direction. Volatility regime calls = better position sizing.

---

## Pipeline Overview

### 1. Data Preparation
- Download daily EUR/USD spot prices from Yahoo Finance (`EURUSD=X`)
- Compute daily returns and absolute returns (foundation for volatility metrics)
- Chronological split: 70% train, 30% test

### 2. Feature Engineering
The feature set is simple and robust:
- 5-day realized volatility (rolling std of returns)
- 20-day realized volatility (medium-term)
- Vol-of-vol proxy (5-day rolling std of absolute returns)
- Yesterday’s absolute return (immediate risk)
- 20-day rolling variance proxy

### 3. Target Variable
- Binary variable: 1 = high-vol day, 0 = low-vol day  
- Threshold: 70th percentile of next-day absolute returns (about 30% high-vol, 70% low-vol)

### 4. Model
- Logistic Regression: fast, interpretable, plus standard scaler (with scikit-learn Pipeline)

### 5. Position Sizing Strategy
- If **LOW vol** predicted → 1.5x exposure (higher risk)
- If **HIGH vol** predicted → 0.5x exposure (defensive)
- Strategy returns = Exposure × Next-day returns

### 6. Evaluation
- **Primary KPI:** Annualized Sharpe ratio of the strategy
- **Secondary:** Classification metrics—confusion matrix, precision, recall

---

## Why This Matters

The final metric is the annualized Sharpe of this size-adjusted strategy on out-of-sample data—real-world, PnL-centric evaluation.  
This is how many quant hedge funds judge whether volatility forecasts are actionable.

### Extending the Baseline
Want to go further? Add:
- Options-implied volatility
- Macro indicators
- More advanced models (ensemble methods, neural networks)

This baseline already extracts usable signals from simple volatility patterns. Start here and build up.

---

## Quickstart

1. **Run the pipeline:**  
   (the code in `baseline.py` will do everything, just be sure `eurusd.csv` is present)

2. **Interpret the output:**  
   - Sharpe ratio (main KPI)
   - Classification metrics

3. **Tweak, extend, improve as needed**

---

Questions, ideas, or contributions? Pull requests are welcome.

<!--
python make_diff_roll.py /home/ozkilim/demo/volatility/current_runs --gif final_focused.gif --mp4 final_focused.mp4 --start-marker "# CO_DATASCIENTIST_BLOCK_START" --end-marker "# CO_DATASCIENTIST_BLOCK_END"
-->
