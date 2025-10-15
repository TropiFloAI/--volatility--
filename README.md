Volatility Forecasting for Position Sizing (EUR/USD)

Instead of predicting where price goes, this task forecasts risk: will tomorrow be a high-vol or low-vol day for EUR/USD? 

The approach is deliberately simple and practical:

1. Data Preparation:
   - Downloads daily EUR/USD spot prices from Yahoo Finance (EURUSD=X)
   - Calculates daily returns and absolute returns as the foundation for volatility metrics
   - Splits data into training (70%) and test sets chronologically

2. Feature Engineering:
   - Short-term realized volatility (5-day rolling standard deviation of returns)
   - Medium-term realized volatility (20-day rolling standard deviation)
   - Volatility-of-volatility proxy (5-day rolling standard deviation of absolute returns)
   - Yesterday's absolute return (immediate risk signal)
   - Simple variance proxy (20-day rolling variance of returns)
   
3. Target Variable:
   - Binary classification: 1 for high-vol days, 0 for low-vol days
   - Threshold set at the 70th percentile of next-day absolute returns
   - This creates a realistic imbalance (30% high-vol, 70% low-vol days)

4. Model:
   - Logistic regression with standard scaling preprocessing
   - Fast, interpretable, and robust for financial time series
   - Uses scikit-learn pipeline for clean feature transformation

5. Position Sizing Strategy:
   - When model predicts LOW volatility → use 1.5x exposure (higher risk tolerance)
   - When model predicts HIGH volatility → use 0.5x exposure (defensive positioning)
   - Strategy returns = exposure × next-day returns

6. Evaluation:
   - Primary KPI: Annualized Sharpe ratio of the volatility-scaled strategy
   - This matches how institutional traders evaluate volatility forecasting systems
   - Also provides classification metrics (confusion matrix, precision/recall)

The final KPI is the annualized Sharpe of this scaled strategy on the test set (printed as a single line), which is exactly how many hedge funds judge the usefulness of volatility forecasts for risk targeting.

This framework can be easily extended with more sophisticated features (options-implied volatility, macro indicators) or models (ensemble methods, neural networks), but the baseline demonstrates that even simple volatility patterns contain actionable information for position sizing.





<!-- python make_diff_roll.py /home/ozkilim/demo/volatility/current_runs --gif final_focused.gif --mp4 final_focused.mp4 --start-marker "# CO_DATASCIENTIST_BLOCK_START" --end-marker "# CO_DATASCIENTIST_BLOCK_END" -->
