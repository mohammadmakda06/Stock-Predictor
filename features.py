"""
features.py
------------
Feature engineering for short-term stock trend prediction.

Engineers predictive features from historical price + volume data:
- Simple & exponential moving averages (trend)
- RSI (momentum / overbought-oversold)
- MACD + signal line + histogram (trend + momentum)
- Volume-based features (relative volume, volume change)
- Rolling volatility and lagged returns

Target: binary short-term trend direction (1 = price up over the next
`horizon` trading days, 0 = down/flat).
"""

import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def engineer_features(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Builds the full feature matrix + target column from raw OHLCV data.

    Parameters
    ----------
    df : DataFrame with columns Open, High, Low, Close, Volume, indexed by Date
    horizon : int, number of trading days ahead to predict trend direction for
    """
    out = df.copy()
    close = out["Close"]
    volume = out["Volume"]

    # --- Moving averages ---
    for w in (5, 10, 20, 50, 200):
        out[f"sma_{w}"] = close.rolling(w).mean()
        out[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()
        # price relative to MA (normalized, scale-free)
        out[f"close_to_sma_{w}"] = close / out[f"sma_{w}"] - 1

    # --- Moving average crossover signals ---
    out["sma_5_20_cross"] = out["sma_5"] - out["sma_20"]
    out["sma_20_50_cross"] = out["sma_20"] - out["sma_50"]
    out["sma_50_200_cross"] = out["sma_50"] - out["sma_200"]

    # --- RSI ---
    out["rsi_14"] = compute_rsi(close, 14)
    out["rsi_7"] = compute_rsi(close, 7)

    # --- MACD ---
    macd_line, signal_line, hist = compute_macd(close)
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist

    # --- Volume features ---
    out["volume_sma_20"] = volume.rolling(20).mean()
    out["volume_rel_20"] = volume / out["volume_sma_20"] - 1
    out["volume_change"] = volume.pct_change()

    # --- Volatility / lagged returns ---
    out["return_1d"] = close.pct_change(1)
    out["return_5d"] = close.pct_change(5)
    out["return_10d"] = close.pct_change(10)
    out["volatility_10d"] = out["return_1d"].rolling(10).std()
    out["volatility_20d"] = out["return_1d"].rolling(20).std()

    # --- High/Low range features ---
    out["hl_range"] = (out["High"] - out["Low"]) / close
    out["close_position"] = (close - out["Low"]) / (out["High"] - out["Low"]).replace(0, np.nan)

    # --- Target: short-term trend direction ---
    future_return = close.shift(-horizon) / close - 1
    out["target"] = (future_return > 0).astype(int)

    feature_cols = [c for c in out.columns if c not in
                    ("Open", "High", "Low", "Close", "Volume", "target")]

    out = out.dropna(subset=feature_cols + ["target"])
    return out, feature_cols


if __name__ == "__main__":
    from data_utils import load_price_data
    df = load_price_data()
    feat_df, cols = engineer_features(df)
    print(f"Feature matrix: {feat_df.shape}, {len(cols)} features")
    print(feat_df[cols + ["target"]].head())
