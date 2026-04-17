"""
Module 5: Backtesting & Evaluation Engine  (SHARPE-OPTIMISED)
==============================================================
Key improvements over baseline:
  1. Volatility targeting  → cuts unnecessary drawdowns
  2. Regime filter         → flattens in crisis, avoids big losses
  3. Weekly rebalance      → 5x less friction, lower TC drag
  4. Composite signal      → lower noise via diversification
  5. Tighter position caps → higher conviction, better IS
  6. Drawdown circuit breaker → hard stop at -15% (was -20%)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Optional
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# TRANSACTION COST MODEL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TransactionCostModel:
    commission_pct:      float = 0.0005   # 5 bps per side
    bid_ask_spread:      float = 0.0010   # 10 bps half-spread
    slippage_vol_mult:   float = 0.10     # 10% of daily vol
    market_impact_coeff: float = 0.10     # sqrt-impact coefficient
    adv_fraction:        float = 0.01     # order = 1% of ADV

    def total_cost(self, turnover: float, volatility: float = 0.02,
                   adv_fraction: float = None) -> float:
        if adv_fraction is None:
            adv_fraction = self.adv_fraction
        commission    = self.commission_pct
        spread_cost   = self.bid_ask_spread / 2
        slippage      = self.slippage_vol_mult * volatility
        market_impact = self.market_impact_coeff * volatility * np.sqrt(adv_fraction)
        return (commission + spread_cost + slippage + market_impact) * turnover


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

class PerformanceAnalytics:
    """Computes comprehensive performance metrics from a returns series."""

    def __init__(self, returns: pd.Series, benchmark_returns: pd.Series = None):
        self.returns   = returns.dropna()
        self.benchmark = benchmark_returns
        self.rf_annual = 0.05
        self.rf_daily  = (1 + self.rf_annual) ** (1 / 252) - 1

    def annualised_return(self) -> float:
        total   = (1 + self.returns).prod()
        n_years = len(self.returns) / 252
        return float(total ** (1 / n_years) - 1) if n_years > 0 else 0.0

    def annualised_volatility(self) -> float:
        return float(self.returns.std() * np.sqrt(252))

    def sharpe_ratio(self) -> float:
        excess = self.returns - self.rf_daily
        if excess.std() < 1e-8:
            return 0.0
        return float(excess.mean() / excess.std() * np.sqrt(252))

    def sortino_ratio(self) -> float:
        excess   = self.returns - self.rf_daily
        downside = excess[excess < 0]
        if len(downside) < 2 or downside.std() < 1e-8:
            return 0.0
        return float(excess.mean() / downside.std() * np.sqrt(252))

    def max_drawdown(self) -> float:
        cum      = (1 + self.returns).cumprod()
        roll_max = cum.cummax()
        dd       = (cum - roll_max) / roll_max
        return float(dd.min())

    def calmar_ratio(self) -> float:
        mdd = abs(self.max_drawdown())
        return float(self.annualised_return() / mdd) if mdd > 1e-8 else 0.0

    def win_rate(self) -> float:
        return float((self.returns > 0).mean())

    def profit_factor(self) -> float:
        gains  = self.returns[self.returns > 0].sum()
        losses = abs(self.returns[self.returns < 0].sum())
        return float(gains / losses) if losses > 0 else float('inf')

    def var_95(self) -> float:
        return float(-np.percentile(self.returns, 5))

    def cvar_95(self) -> float:
        cutoff = np.percentile(self.returns, 5)
        tail   = self.returns[self.returns <= cutoff]
        return float(-tail.mean()) if len(tail) > 0 else 0.0

    def information_ratio(self) -> float:
        if self.benchmark is None:
            return np.nan
        common = self.returns.index.intersection(self.benchmark.index)
        excess = self.returns[common] - self.benchmark[common]
        if excess.std() < 1e-8:
            return 0.0
        return float(excess.mean() / excess.std() * np.sqrt(252))

    def tail_ratio(self) -> float:
        return abs(np.percentile(self.returns, 95) /
                   (np.percentile(self.returns, 5) + 1e-8))

    def all_metrics(self) -> dict:
        return {
            "annual_return":  round(self.annualised_return(), 4),
            "annual_vol":     round(self.annualised_volatility(), 4),
            "sharpe_ratio":   round(self.sharpe_ratio(), 4),
            "sortino_ratio":  round(self.sortino_ratio(), 4),
            "calmar_ratio":   round(self.calmar_ratio(), 4),
            "max_drawdown":   round(self.max_drawdown(), 4),
            "win_rate":       round(self.win_rate(), 4),
            "profit_factor":  round(self.profit_factor(), 4),
            "var_95":         round(self.var_95(), 4),
            "cvar_95":        round(self.cvar_95(), 4),
            "info_ratio":     round(self.information_ratio(), 4)
                              if self.benchmark is not None else None,
            "tail_ratio":     round(self.tail_ratio(), 4),
        }

    def print_summary(self):
        m = self.all_metrics()
        print("=" * 50)
        print("  PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"  Annual Return:     {m['annual_return']:>8.2%}")
        print(f"  Annual Volatility: {m['annual_vol']:>8.2%}")
        print(f"  Sharpe Ratio:      {m['sharpe_ratio']:>8.3f}")
        print(f"  Sortino Ratio:     {m['sortino_ratio']:>8.3f}")
        print(f"  Calmar Ratio:      {m['calmar_ratio']:>8.3f}")
        print(f"  Max Drawdown:      {m['max_drawdown']:>8.2%}")
        print(f"  Win Rate:          {m['win_rate']:>8.2%}")
        print(f"  Profit Factor:     {m['profit_factor']:>8.3f}")
        print(f"  VaR 95%:           {m['var_95']:>8.4f}")
        print(f"  CVaR 95%:          {m['cvar_95']:>8.4f}")
        print("=" * 50)


# ─────────────────────────────────────────────────────────────────────────────
# VOLATILITY TARGETING  (NEW — FIX #1)
# ─────────────────────────────────────────────────────────────────────────────

def volatility_scale(
    weights: pd.Series,
    price_panel: pd.DataFrame,
    date: pd.Timestamp,
    target_vol: float = 0.10,       # 10% annualised target
    lookback:   int   = 21,         # 21-day realised vol window
    max_scale:  float = 1.5,        # never lever more than 1.5×
) -> pd.Series:
    """
    Scale portfolio weights so that expected realised vol ≈ target_vol.
    This is the single biggest Sharpe improvement lever.
    """
    hist    = price_panel.loc[:date].tail(lookback + 1)
    ret     = hist.pct_change().dropna()

    if len(ret) < 5:
        return weights

    common  = weights.index.intersection(ret.columns)
    w       = weights.reindex(common, fill_value=0.0).values
    cov     = ret[common].cov().values * 252          # annualised covariance
    port_var = float(w @ cov @ w)
    port_vol = np.sqrt(max(port_var, 1e-10))

    scalar  = min(target_vol / port_vol, max_scale)
    return weights * scalar


# ─────────────────────────────────────────────────────────────────────────────
# REGIME FILTER  (NEW — FIX #2)
# ─────────────────────────────────────────────────────────────────────────────

def detect_crisis_regime(
    price_panel: pd.DataFrame,
    date: pd.Timestamp,
    vol_window:    int   = 21,
    vol_threshold: float = 0.25,    # annualised vol above this = crisis
    dd_threshold:  float = 0.08,    # 8% drawdown from 63-day high = crisis
) -> bool:
    """
    Returns True if market is in a crisis regime → caller should flatten.
    Uses two independent signals so both must trigger (reduces false positives).
    """
    hist  = price_panel.loc[:date]
    mkt   = hist.mean(axis=1)       # equal-weight market proxy

    # Signal 1: short-term vol spike
    ret       = mkt.pct_change().dropna()
    if len(ret) < vol_window:
        return False
    realised_vol = ret.tail(vol_window).std() * np.sqrt(252)
    vol_crisis   = realised_vol > vol_threshold

    # Signal 2: market in meaningful drawdown
    recent    = mkt.tail(63)
    peak      = recent.max()
    current   = recent.iloc[-1]
    drawdown  = (current - peak) / peak
    dd_crisis = drawdown < -dd_threshold

    # Both must trigger to avoid false flattening in normal corrections
    return bool(vol_crisis and dd_crisis)


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE  (IMPROVED)
# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Event-driven backtesting engine with:
      - Volatility targeting
      - Regime-based flattening
      - Tighter position limits (8% vs 10%)
      - Hard drawdown circuit breaker at -15%
      - Weekly rebalance default (lower TC drag)
    """

    def __init__(
        self,
        price_panel:     pd.DataFrame,
        signal_fn:       Callable,
        tc_model:        TransactionCostModel = None,
        initial_capital: float = 1_000_000,
        rebalance_freq:  str   = "W",          # ← changed default to Weekly
        target_vol:      float = 0.10,          # NEW: vol target
        max_position:    float = 0.08,          # NEW: tighter cap (was 0.10)
        dd_halt:         float = 0.15,          # NEW: halt at -15% (was -20%)
        use_vol_target:  bool  = True,          # toggle vol targeting
        use_regime_filter: bool = True,         # toggle regime filter
    ):
        self.prices         = price_panel
        self.signal_fn      = signal_fn
        self.tc             = tc_model or TransactionCostModel()
        self.capital        = initial_capital
        self.freq           = rebalance_freq
        self.target_vol     = target_vol
        self.max_position   = max_position
        self.dd_halt        = dd_halt
        self.use_vol_target = use_vol_target
        self.use_regime     = use_regime_filter

        self.equity_curve:    list[dict]                   = []
        self.trade_log:       list[dict]                   = []
        self.weights_history: dict[pd.Timestamp, pd.Series] = {}

        # internal state
        self._halted    = False
        self._hwm       = None          # high-water mark

    # ── helpers ──────────────────────────────────────────────────────────────

    def _should_rebalance(self, date: pd.Timestamp) -> bool:
        if self.freq == "D":
            return True
        elif self.freq == "W":
            return date.dayofweek == 4          # Friday
        elif self.freq == "M":
            return date.is_month_end
        return False

    def _check_drawdown_halt(self, nav: float) -> bool:
        """Hard stop: halt trading if drawdown from HWM > dd_halt."""
        if self._hwm is None:
            self._hwm = nav
        if nav > self._hwm:
            self._hwm = nav
        dd = (nav - self._hwm) / self._hwm
        return dd < -self.dd_halt

    # ── main run ─────────────────────────────────────────────────────────────

    def run(
        self,
        start:       str,
        end:         str,
        feature_data = None,          # optional pre-computed feature dict
        warmup_days: int = 252,
    ) -> dict:

        dates = self.prices.loc[start:end].index
        print(f"\n[BACKTEST] {start} → {end} | {len(dates)} trading days")
        print(f"           vol_target={self.target_vol:.0%}  "
              f"max_pos={self.max_position:.0%}  "
              f"freq={self.freq}  "
              f"dd_halt={self.dd_halt:.0%}")

        nav             = float(self.capital)
        current_weights = pd.Series(dtype=float)
        prev_prices     = None

        for i, date in enumerate(dates):
            prices_today = self.prices.loc[date]

            # ── 1. DAILY P&L ─────────────────────────────────────────────
            if prev_prices is not None and not current_weights.empty:
                common = (current_weights.index
                          .intersection(prices_today.index)
                          .intersection(prev_prices.index))
                rets   = (prices_today[common] / prev_prices[common] - 1).fillna(0)
                pnl    = float((current_weights.reindex(common, fill_value=0) * rets).sum())
                nav   *= (1 + pnl)
            else:
                pnl = 0.0

            # ── 2. CIRCUIT BREAKER ───────────────────────────────────────
            if self._check_drawdown_halt(nav):
                if not self._halted:
                    print(f"  [HALT] Drawdown circuit breaker triggered on {date.date()}")
                    self._halted    = True
                    current_weights = pd.Series(dtype=float)
            else:
                self._halted = False   # reset once NAV recovers

            turnover = 0.0

            # ── 3. REBALANCE ─────────────────────────────────────────────
            if (i >= warmup_days
                    and self._should_rebalance(date)
                    and not self._halted):

                # 3a. Regime filter — flatten if crisis
                if self.use_regime and detect_crisis_regime(self.prices, date):
                    if not current_weights.empty:
                        print(f"  [REGIME] Crisis detected {date.date()} — flattening")
                    current_weights = pd.Series(dtype=float)

                else:
                    try:
                        # 3b. Get signal
                        new_weights = self.signal_fn(date, feature_data
                                                     if feature_data is not None
                                                     else self.prices.loc[:date])

                        if new_weights is not None and not new_weights.empty:

                            # 3c. Tighter position cap
                            new_weights = new_weights.clip(
                                -self.max_position, self.max_position
                            )

                            # 3d. Normalise (dollar-neutral: longs=1, shorts=-1)
                            longs  = new_weights[new_weights > 0]
                            shorts = new_weights[new_weights < 0]
                            if longs.sum() > 0:
                                new_weights[new_weights > 0] = longs / longs.sum()
                            if shorts.sum() < 0:
                                new_weights[new_weights < 0] = (
                                    shorts / shorts.abs().sum() * -1
                                )

                            # 3e. VOLATILITY TARGETING  ← key improvement
                            if self.use_vol_target:
                                new_weights = volatility_scale(
                                    new_weights, self.prices, date,
                                    target_vol=self.target_vol,
                                )

                            # 3f. Compute turnover & transaction costs
                            all_t    = new_weights.index.union(current_weights.index)
                            n_w      = new_weights.reindex(all_t, fill_value=0)
                            c_w      = current_weights.reindex(all_t, fill_value=0)
                            turnover = float((n_w - c_w).abs().sum() / 2)

                            hist_ret = self.prices.pct_change().tail(21)
                            vol_est  = float(hist_ret.std().mean())
                            tc_cost  = self.tc.total_cost(turnover, vol_est)
                            nav     *= (1 - tc_cost)

                            # 3g. Trade log
                            delta = n_w - c_w
                            for ticker, dw in delta.items():
                                if abs(dw) > 1e-4:
                                    self.trade_log.append({
                                        "date":          date,
                                        "ticker":        ticker,
                                        "weight_change": round(dw, 4),
                                        "price":         prices_today.get(ticker, np.nan),
                                        "tc_cost":       round(tc_cost * abs(dw), 6),
                                    })

                            current_weights = new_weights.copy()
                            self.weights_history[date] = current_weights.copy()

                    except Exception as e:
                        pass   # keep previous weights on signal failure

            # ── 4. RECORD ────────────────────────────────────────────────
            self.equity_curve.append({
                "date":         date,
                "nav":          nav,
                "daily_return": pnl,
                "turnover":     turnover,
                "halted":       self._halted,
            })
            prev_prices = prices_today.copy()

        print(f"[BACKTEST] Done | Final NAV: ${nav:,.0f} | "
              f"Trades: {len(self.trade_log)}")
        return self._compile_results()

    def _compile_results(self) -> dict:
        ec_df    = pd.DataFrame(self.equity_curve).set_index("date")
        returns  = ec_df["daily_return"]
        analytics = PerformanceAnalytics(returns)
        return {
            "equity_curve": ec_df,
            "returns":      returns,
            "trade_log":    pd.DataFrame(self.trade_log),
            "metrics":      analytics.all_metrics(),
            "analytics":    analytics,
        }


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

class WalkForwardValidator:
    """Rolling walk-forward analysis — no look-ahead bias."""

    def __init__(self, n_splits: int = 5, train_pct: float = 0.70):
        self.n_splits  = n_splits
        self.train_pct = train_pct

    def split(self, dates: pd.DatetimeIndex) -> list[tuple]:
        n         = len(dates)
        fold_size = n // (self.n_splits + 1)
        splits    = []
        for i in range(self.n_splits):
            train_end_idx  = fold_size * (i + 1) + int(fold_size * self.train_pct)
            test_start_idx = train_end_idx
            test_end_idx   = min(test_start_idx + fold_size, n - 1)
            if test_end_idx <= test_start_idx:
                continue
            splits.append((
                dates[fold_size * i],
                dates[train_end_idx - 1],
                dates[test_start_idx],
                dates[test_end_idx],
            ))
        return splits


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

class MonteCarloSimulator:
    """Bootstrap confidence intervals on strategy metrics."""

    def __init__(self, n_simulations: int = 1000, seed: int = 42):
        self.n_sims = n_simulations
        self.rng    = np.random.default_rng(seed)

    def bootstrap_metrics(self, returns: pd.Series, n_years: int = 1) -> dict:
        n_days  = n_years * 252
        metrics = []
        for _ in range(self.n_sims):
            sample = self.rng.choice(returns.values, size=n_days, replace=True)
            pa     = PerformanceAnalytics(pd.Series(sample))
            metrics.append({
                "sharpe":  pa.sharpe_ratio(),
                "ann_ret": pa.annualised_return(),
                "max_dd":  pa.max_drawdown(),
            })
        df      = pd.DataFrame(metrics)
        results = {}
        for col in df.columns:
            results[col] = {
                "mean": round(df[col].mean(), 4),
                "p5":   round(df[col].quantile(0.05), 4),
                "p25":  round(df[col].quantile(0.25), 4),
                "p50":  round(df[col].quantile(0.50), 4),
                "p75":  round(df[col].quantile(0.75), 4),
                "p95":  round(df[col].quantile(0.95), 4),
            }
        return results

    def simulate_paths(self, returns: pd.Series, n_days: int = 252,
                       initial_nav: float = 1.0) -> pd.DataFrame:
        paths = {}
        for i in range(min(self.n_sims, 500)):
            sample  = self.rng.choice(returns.values, size=n_days, replace=True)
            curve   = initial_nav * np.cumprod(1 + sample)
            paths[i] = curve
        return pd.DataFrame(paths)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    returns = pd.Series(np.random.randn(504) * 0.008 + 0.0003)
    pa = PerformanceAnalytics(returns)
    pa.print_summary()

    mc = MonteCarloSimulator(n_simulations=500)
    ci = mc.bootstrap_metrics(returns)
    print(f"\n[Monte Carlo] Sharpe 5th–95th pct: "
          f"[{ci['sharpe']['p5']}, {ci['sharpe']['p95']}]")