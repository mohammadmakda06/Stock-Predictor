"""
data_utils.py
-------------
Loads historical OHLCV price data for the Stock Market Predictor project.

Two modes:
1. Real data: pass a CSV path with columns [Date, Open, High, Low, Close, Volume]
   (e.g. exported from yfinance: yfinance.download(ticker, start=..., end=...))
2. Synthetic data: if no CSV is available (e.g. no network access), generates
   a realistic 10+ year daily OHLCV series using a geometric Brownian motion
   model with volatility clustering, so the full pipeline can be built,
   tested, and demoed end-to-end without an internet connection.

To use real data locally:
    import yfinance as yf
    df = yf.download("AAPL", start="2013-01-01", end="2026-01-01")
    df.to_csv("aapl.csv")
    load_price_data("aapl.csv")
"""

import numpy as np
import pandas as pd


def generate_synthetic_ohlcv(
    n_days: int = 2600,  # ~10.4 years of trading days
    start_price: float = 100.0,
    annual_drift: float = 0.08,
    annual_vol: float = 0.22,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates a realistic daily OHLCV series with GARCH-like volatility
    clustering and occasional trend/momentum regimes, so engineered features
    (RSI, MACD, moving averages) behave realistically."""
    rng = np.random.default_rng(seed)

    dt = 1 / 252
    mu = annual_drift
    sigma_base = annual_vol

    # Volatility clustering: sigma follows a slow mean-reverting AR(1) process
    sigma_t = np.zeros(n_days)
    sigma_t[0] = sigma_base
    for t in range(1, n_days):
        shock = rng.normal(0, 0.02)
        sigma_t[t] = np.clip(
            sigma_base + 0.9 * (sigma_t[t - 1] - sigma_base) + shock, 0.08, 0.55
        )

    # Regime-switching drift to create trend momentum for the model to learn
    regime_len = 40
    n_regimes = n_days // regime_len + 2
    regime_drifts = rng.choice([mu * 2.5, mu * 0.5, -mu * 1.5], size=n_regimes,
                               p=[0.4, 0.35, 0.25])
    daily_drift = np.repeat(regime_drifts, regime_len)[:n_days]

    returns = (daily_drift - 0.5 * sigma_t ** 2) * dt + sigma_t * np.sqrt(dt) * rng.standard_normal(n_days)
    close = start_price * np.exp(np.cumsum(returns))

    # Derive Open/High/Low around Close with intraday noise
    intraday_range = close * sigma_t * np.sqrt(dt) * rng.uniform(0.5, 1.5, n_days)
    open_ = close * (1 + rng.normal(0, 0.003, n_days))
    high = np.maximum(open_, close) + np.abs(intraday_range) * rng.uniform(0.2, 0.6, n_days)
    low = np.minimum(open_, close) - np.abs(intraday_range) * rng.uniform(0.2, 0.6, n_days)

    # Volume: baseline + spikes correlated with |returns| (higher volume on big moves)
    base_volume = rng.lognormal(mean=14.5, sigma=0.3, size=n_days)
    volume = base_volume * (1 + 4 * np.abs(returns) / sigma_t.mean())

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    df = pd.DataFrame({
        "Date": dates,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume.astype(np.int64),
    })
    df.set_index("Date", inplace=True)
    return df


def load_price_data(csv_path: str | None = None) -> pd.DataFrame:
    """Loads OHLCV data from CSV if given, otherwise falls back to a
    synthetic 10+ year series.

    Handles two CSV shapes:
    1. Plain format:      Date,Open,High,Low,Close,Volume
    2. yfinance's export, which writes a 3-row header
       (Price,Close,High,Low,Open,Volume / Ticker,AAPL,... / Date,,,,,)
       followed by the actual data rows.
    """
    if csv_path:
        df = pd.read_csv(csv_path)

        # Detect yfinance's multi-row header: first column is "Price" and
        # the first couple of data rows are "Ticker"/"Date" label rows.
        first_col = df.columns[0]
        if first_col != "Date":
            df = df.rename(columns={first_col: "Date"})
            # Drop any leading rows that aren't real dates (Ticker/Date rows)
            df = df[pd.to_datetime(df["Date"], errors="coerce").notna()].copy()

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")

        # Ensure numeric dtypes (yfinance's export can leave them as strings
        # after the header rows are mixed in)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[["Open", "High", "Low", "Close", "Volume"]].sort_index()
        df = df.dropna()
        return df
    return generate_synthetic_ohlcv()


if __name__ == "__main__":
    df = load_price_data()
    print(df.head())
    print(f"\n{len(df)} trading days ({df.index[0].date()} to {df.index[-1].date()})")