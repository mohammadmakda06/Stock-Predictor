# Stock Market Predictor

Random Forest and Gradient Boosting models that predict short-term stock
price **trend direction** (up vs. down/flat over the next N trading days)
from engineered technical features.

## Stack
Python, NumPy, pandas, scikit-learn, matplotlib

## What it does
- Loads 10+ years of daily OHLCV price data (real CSV or synthetic demo data)
- Engineers features from price, volume, RSI, MACD, and moving averages
- Trains Random Forest and Gradient Boosting classifiers
- Tunes hyperparameters with `GridSearchCV` + `TimeSeriesSplit` (no lookahead leakage)
- Evaluates with a chronological train/test split (accuracy, precision, recall, F1, confusion matrix)
- Reports feature importances and saves a results plot

## Project structure
```
stock_market_predictor/
├── main.py                 # CLI entry point / pipeline orchestration
├── requirements.txt
├── src/
│   ├── data_utils.py        # data loading (CSV or synthetic OHLCV generator)
│   ├── features.py           # RSI, MACD, moving averages, volume features, target
│   └── train.py              # model tuning + evaluation
```

## Usage

```bash
pip install -r requirements.txt

# Run on synthetic 10+ year demo data (no internet required)
python main.py

# Run on real historical data
python -c "import yfinance as yf; yf.download('AAPL', start='2013-01-01').to_csv('aapl.csv')"
python main.py --csv aapl.csv --horizon 5
```

`--horizon` controls how many trading days ahead the model predicts the
trend direction for (default: 5).

## Features engineered
- **Moving averages**: SMA/EMA at 5/10/20/50/200-day windows, plus
  price-relative-to-MA and MA crossover signals
- **RSI**: 7-day and 14-day
- **MACD**: MACD line, signal line, histogram
- **Volume**: relative volume vs. 20-day average, volume % change
- **Volatility & momentum**: rolling volatility, 1/5/10-day lagged returns,
  high-low range, close position within the day's range

## Methodology notes
- Train/test split is **chronological** (not shuffled) since shuffling a
  time series leaks future information into training.
- Hyperparameter search uses `TimeSeriesSplit` cross-validation for the
  same reason.
- The synthetic data generator (used when no CSV is supplied) models
  volatility clustering and regime-switching drift so engineered
  indicators behave realistically — useful for testing the pipeline
  without needing network/API access to a real data provider.

## Possible extensions
- Add sentiment or macro features (news, earnings surprises, rates)
- Predict multi-class trend (strong up / flat / strong down)
- Walk-forward backtesting with a trading strategy layer and Sharpe ratio
- Ensemble RF + GB predictions
