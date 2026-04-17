"""
Module 1: Data & Feature Engineering
=====================================
Handles OHLCV ingestion, cleaning, normalisation, and feature construction.
All features are point-in-time correct (no look-ahead bias).
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ── Universe Definitions ──────────────────────────────────────────────────────

UNIVERSES = {
    "nifty50": [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "KOTAKBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS",
        "BAJFINANCE.NS", "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
        "WIPRO.NS", "TITAN.NS", "NESTLEIND.NS", "ULTRACEMCO.NS", "SUNPHARMA.NS",
    ],
    "sp500_sample": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
        "JPM", "JNJ", "V", "PG", "XOM", "UNH", "MA", "HD", "CVX", "MRK",
        "ABBV", "PFE",
    ],
    "crypto": [
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD",
        "AVAX-USD", "DOT-USD", "MATIC-USD", "LINK-USD", "UNI-USD",
    ],
}


# ── Data Ingestion ────────────────────────────────────────────────────────────

class DataIngester:
    """Downloads and caches OHLCV data via yfinance."""

    def __init__(self, universe: str = "sp500_sample"):
        self.tickers = UNIVERSES.get(universe, UNIVERSES["sp500_sample"])
        self._cache: dict[str, pd.DataFrame] = {}

    def fetch(self, start: str, end: str, interval: str = "1d") -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV for all tickers. Returns {ticker: df} mapping.
        Applies split/dividend adjustments automatically.
        """
        raw = yf.download(
            self.tickers,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
        )

        data = {}
        for ticker in self.tickers:
            try:
                if len(self.tickers) == 1:
                    df = raw.copy()
                else:
                    df = raw[ticker].copy()

                df.columns = [c.lower() for c in df.columns]
                df = df.dropna(subset=["close"])
                df = self._quality_checks(df, ticker)
                data[ticker] = df
            except Exception as e:
                print(f"  [WARN] {ticker}: {e}")

        self._cache = data
        print(f"  [DATA] Fetched {len(data)}/{len(self.tickers)} tickers | "
              f"{start} → {end}")
        return data

    def _quality_checks(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Remove outliers, fill stale prices, handle missing values."""
        # Forward-fill up to 3 consecutive missing values (weekends/holidays)
        df = df.ffill(limit=3)

        # Remove rows where OHLCV is completely zero
        df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]

        # Flag and clip extreme single-day returns (>50% in one day → suspect)
        returns = df["close"].pct_change()
        extreme = returns.abs() > 0.5
        if extreme.sum() > 0:
            print(f"  [QC] {ticker}: {extreme.sum()} extreme return(s) clipped")
        df.loc[extreme, "close"] = df["close"].shift(1)[extreme]

        # Ensure high >= low, high >= close, low <= close
        df["high"] = df[["high", "close", "open"]].max(axis=1)
        df["low"] = df[["low", "close", "open"]].min(axis=1)

        return df

    def get_panel(self, field: str = "close") -> pd.DataFrame:
        """Return a wide panel DataFrame: dates × tickers."""
        if not self._cache:
            raise RuntimeError("Call .fetch() first")
        return pd.DataFrame({t: df[field] for t, df in self._cache.items()})


# ── Feature Engineering ───────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Builds a point-in-time-correct feature matrix for a single ticker's OHLCV.
    All indicators use only past data (no look-ahead).
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def build_all(self) -> pd.DataFrame:
        """Return full feature DataFrame."""
        f = self.df.copy()
        f = self._price_features(f)
        f = self._momentum_features(f)
        f = self._volatility_features(f)
        f = self._volume_features(f)
        f = self._technical_indicators(f)
        # Drop initial NaN rows caused by rolling windows
        f = f.dropna()
        return f

    # ── Price / Return Features ──────────────────────────────────────────────

    def _price_features(self, f: pd.DataFrame) -> pd.DataFrame:
        for h in [1, 5, 10, 21, 63]:
            f[f"ret_{h}d"] = f["close"].pct_change(h)
        f["log_ret_1d"] = np.log(f["close"] / f["close"].shift(1))
        f["overnight_gap"] = (f["open"] - f["close"].shift(1)) / f["close"].shift(1)
        f["intraday_range"] = (f["high"] - f["low"]) / f["close"].shift(1)
        f["close_location"] = (f["close"] - f["low"]) / (f["high"] - f["low"] + 1e-8)
        return f

    # ── Momentum Features ────────────────────────────────────────────────────

    def _momentum_features(self, f: pd.DataFrame) -> pd.DataFrame:
        # Cross-sectional momentum proxies (time-series)
        for w in [21, 63, 126, 252]:
            f[f"mom_{w}d"] = f["close"].pct_change(w)

        # Rate of change
        f["roc_10"] = f["close"].pct_change(10)

        # 52-week high proximity
        f["pct_from_52w_high"] = f["close"] / f["close"].rolling(252).max() - 1
        f["pct_from_52w_low"] = f["close"] / f["close"].rolling(252).min() - 1

        return f

    # ── Volatility Features ──────────────────────────────────────────────────

    def _volatility_features(self, f: pd.DataFrame) -> pd.DataFrame:
        log_ret = np.log(f["close"] / f["close"].shift(1))
        for w in [5, 21, 63]:
            f[f"vol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)

        # Volatility regime: short vol / long vol ratio
        f["vol_regime"] = f["vol_5d"] / (f["vol_63d"] + 1e-8)

        # Parkinson volatility (uses high/low, more efficient)
        f["vol_park_21d"] = (
            np.log(f["high"] / f["low"]) ** 2 / (4 * np.log(2))
        ).rolling(21).mean().apply(np.sqrt) * np.sqrt(252)

        return f

    # ── Volume Features ──────────────────────────────────────────────────────

    def _volume_features(self, f: pd.DataFrame) -> pd.DataFrame:
        # VWAP deviation
        vwap = (f["close"] * f["volume"]).rolling(21).sum() / (
            f["volume"].rolling(21).sum() + 1e-8
        )
        f["vwap_dev"] = (f["close"] - vwap) / vwap

        # Abnormal volume
        f["vol_ratio_20d"] = f["volume"] / (f["volume"].rolling(20).mean() + 1e-8)

        # On-Balance Volume normalised
        obv = (np.sign(f["close"].diff()) * f["volume"]).cumsum()
        f["obv_norm"] = (obv - obv.rolling(21).mean()) / (obv.rolling(21).std() + 1e-8)

        # Volume momentum
        f["vol_mom_5d"] = f["volume"].pct_change(5)

        return f

    # ── Technical Indicators ─────────────────────────────────────────────────

    def _technical_indicators(self, f: pd.DataFrame) -> pd.DataFrame:
        close = f["close"]
        high  = f["high"]
        low   = f["low"]

        # RSI (14)
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / (loss + 1e-8)
        f["rsi_14"] = 100 - 100 / (1 + rs)

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line   = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        f["macd"]        = macd_line
        f["macd_signal"] = signal_line
        f["macd_hist"]   = macd_line - signal_line

        # Bollinger Bands (20, 2σ)
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        f["bb_upper"] = bb_mid + 2 * bb_std
        f["bb_lower"] = bb_mid - 2 * bb_std
        f["bb_pct_b"] = (close - f["bb_lower"]) / (
            f["bb_upper"] - f["bb_lower"] + 1e-8
        )
        f["bb_width"] = (f["bb_upper"] - f["bb_lower"]) / (bb_mid + 1e-8)

        # ADX (14) — directional strength
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.ewm(span=14, adjust=False).mean()

        dm_plus  = (high.diff()).clip(lower=0)
        dm_minus = (-low.diff()).clip(lower=0)
        di_plus  = 100 * dm_plus.ewm(span=14, adjust=False).mean() / (atr14 + 1e-8)
        di_minus = 100 * dm_minus.ewm(span=14, adjust=False).mean() / (atr14 + 1e-8)
        dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-8)
        f["adx_14"] = dx.ewm(span=14, adjust=False).mean()

        # Stochastic %K, %D
        low14  = low.rolling(14).min()
        high14 = high.rolling(14).max()
        f["stoch_k"] = 100 * (close - low14) / (high14 - low14 + 1e-8)
        f["stoch_d"] = f["stoch_k"].rolling(3).mean()

        # ATR (normalised)
        f["atr_pct"] = atr14 / close

        return f


# ── Cross-Asset Features ──────────────────────────────────────────────────────

def compute_cross_asset_features(
    panel: pd.DataFrame,
    market_ticker: str = "^GSPC",
    lookback: int = 63,
) -> pd.DataFrame:
    """
    Compute beta, correlation to market, and relative strength.
    panel: wide DataFrame of close prices (dates × tickers).
    Returns: MultiIndex DataFrame of cross-asset features.
    """
    mkt_data = yf.download(market_ticker, start=panel.index[0],
                           end=panel.index[-1], auto_adjust=True,
                           progress=False)["Close"]

    features = {}
    mkt_ret = mkt_data.pct_change().reindex(panel.index)

    for ticker in panel.columns:
        ret = panel[ticker].pct_change()
        # Rolling beta
        cov  = ret.rolling(lookback).cov(mkt_ret)
        var  = mkt_ret.rolling(lookback).var()
        beta = cov / (var + 1e-8)
        # Rolling correlation
        corr = ret.rolling(lookback).corr(mkt_ret)
        # Relative strength vs market
        rs = (1 + ret).rolling(lookback).apply(np.prod) / \
             (1 + mkt_ret).rolling(lookback).apply(np.prod)

        features[ticker] = pd.DataFrame({
            "beta": beta,
            "mkt_corr": corr,
            "relative_strength": rs,
        }, index=panel.index)

    return features


if __name__ == "__main__":
    # Quick smoke test
    ingester = DataIngester("sp500_sample")
    data = ingester.fetch("2021-01-01", "2024-12-31")
    ticker = list(data.keys())[0]
    fe = FeatureEngineer(data[ticker])
    features = fe.build_all()
    print(f"\n[Feature Matrix] {ticker}: {features.shape}")
    print(features.tail(3).to_string())









































































































































































































































































































































































































































































































































































































































































































































































