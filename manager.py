"""
Module 4: Risk Management
==========================
Implements pre-trade checks, real-time monitoring, and post-trade analysis.
  - VaR / CVaR (Historical Simulation + Parametric)
  - Maximum Drawdown tracking
  - Volatility regime detection
  - Position & concentration limits
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


# ── Risk Configuration ────────────────────────────────────────────────────────

@dataclass
class RiskLimits:
    max_position_pct: float = 0.10       # max single position (% of NAV)
    max_sector_pct: float = 0.30         # max sector concentration
    max_gross_exposure: float = 2.0      # max sum(|weights|)
    max_net_exposure: float = 0.50       # max |sum(weights)|
    max_var_95: float = 0.02             # max 1-day VaR at 95% confidence
    max_drawdown_pct: float = 0.15       # halt if drawdown > 15%
    min_cash_pct: float = 0.05           # minimum cash buffer
    max_turnover_daily: float = 0.30     # max daily portfolio turnover
    volatility_target: float = 0.15      # annualised target volatility


# ── VaR & CVaR ───────────────────────────────────────────────────────────────

def historical_var(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
) -> float:
    """
    Historical simulation VaR.
    returns: series of daily P&L or returns.
    """
    scaled = returns * np.sqrt(horizon)
    return float(-np.percentile(scaled.dropna(), (1 - confidence) * 100))


def parametric_var(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
) -> float:
    """Parametric (Gaussian) VaR assuming normal distribution."""
    from scipy.stats import norm
    mu  = returns.mean() * horizon
    sig = returns.std() * np.sqrt(horizon)
    z   = norm.ppf(1 - confidence)
    return float(-(mu + z * sig))


def conditional_var(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """CVaR / Expected Shortfall: mean of losses exceeding VaR."""
    cutoff = np.percentile(returns.dropna(), (1 - confidence) * 100)
    tail = returns[returns <= cutoff]
    return float(-tail.mean()) if len(tail) > 0 else 0.0


def portfolio_var(
    weights: pd.Series,
    covariance: pd.DataFrame,
    confidence: float = 0.95,
) -> float:
    """Parametric portfolio VaR from weights and covariance matrix."""
    from scipy.stats import norm
    w = weights.reindex(covariance.index, fill_value=0).values
    port_var = float(w @ covariance.values @ w) * 252
    port_vol_daily = np.sqrt(port_var / 252)
    z = norm.ppf(1 - confidence)
    return float(-z * port_vol_daily)


# ── Drawdown Tracker ──────────────────────────────────────────────────────────

class DrawdownTracker:
    """Real-time drawdown monitoring against rolling high-water mark."""

    def __init__(self):
        self.hwm   = 1.0          # high-water mark (normalised NAV)
        self.peak_date: Optional[pd.Timestamp] = None
        self.history: list[dict] = []

    def update(self, nav: float, date: pd.Timestamp) -> dict:
        if nav > self.hwm:
            self.hwm = nav
            self.peak_date = date

        drawdown = (nav - self.hwm) / self.hwm
        record = {
            "date": date,
            "nav":  nav,
            "hwm":  self.hwm,
            "drawdown": drawdown,
        }
        self.history.append(record)
        return record

    def current_drawdown(self) -> float:
        if not self.history:
            return 0.0
        return self.history[-1]["drawdown"]

    def max_drawdown(self) -> float:
        if not self.history:
            return 0.0
        return min(r["drawdown"] for r in self.history)

    def drawdown_duration(self) -> int:
        """Days since last high-water mark."""
        if not self.history or not self.peak_date:
            return 0
        last_date = self.history[-1]["date"]
        return (last_date - self.peak_date).days

    def to_series(self) -> pd.Series:
        if not self.history:
            return pd.Series(dtype=float)
        df = pd.DataFrame(self.history).set_index("date")
        return df["drawdown"]


# ── Pre-Trade Risk Checker ────────────────────────────────────────────────────

class PreTradeChecker:
    """
    Validates proposed portfolio weights against risk limits before execution.
    Returns a dict of {check_name: (passed: bool, value: float)}.
    """

    def __init__(self, limits: RiskLimits = None):
        self.limits = limits or RiskLimits()

    def check_all(
        self,
        proposed_weights: pd.Series,
        current_weights: pd.Series,
        returns_history: pd.DataFrame,
        sector_map: dict = None,
    ) -> dict:
        results = {}

        # 1. Max single position
        max_pos = proposed_weights.abs().max()
        results["max_position"] = (
            max_pos <= self.limits.max_position_pct,
            round(max_pos, 4),
        )

        # 2. Gross exposure
        gross = proposed_weights.abs().sum()
        results["gross_exposure"] = (
            gross <= self.limits.max_gross_exposure,
            round(gross, 4),
        )

        # 3. Net exposure
        net = abs(proposed_weights.sum())
        results["net_exposure"] = (
            net <= self.limits.max_net_exposure,
            round(net, 4),
        )

        # 4. Sector concentration
        if sector_map:
            sector_weights = {}
            for ticker, weight in proposed_weights.items():
                sector = sector_map.get(ticker, "unknown")
                sector_weights[sector] = sector_weights.get(sector, 0) + abs(weight)
            max_sector = max(sector_weights.values()) if sector_weights else 0
            results["sector_concentration"] = (
                max_sector <= self.limits.max_sector_pct,
                round(max_sector, 4),
            )

        # 5. Portfolio VaR
        if not returns_history.empty:
            port_ret = (returns_history * proposed_weights.reindex(
                returns_history.columns, fill_value=0)).sum(axis=1)
            var95 = historical_var(port_ret)
            results["var_95"] = (
                var95 <= self.limits.max_var_95,
                round(var95, 4),
            )

        # 6. Turnover
        all_t = proposed_weights.index.union(current_weights.index)
        n = proposed_weights.reindex(all_t, fill_value=0)
        c = current_weights.reindex(all_t, fill_value=0)
        turnover = (n - c).abs().sum() / 2
        results["turnover"] = (
            turnover <= self.limits.max_turnover_daily,
            round(turnover, 4),
        )

        return results

    def all_passed(self, check_results: dict) -> bool:
        return all(v[0] for v in check_results.values())

    def summary(self, check_results: dict) -> str:
        lines = ["Pre-Trade Risk Checks:"]
        for name, (passed, value) in check_results.items():
            status = "✓" if passed else "✗ BREACH"
            lines.append(f"  {status} {name}: {value}")
        return "\n".join(lines)


# ── Risk Monitor ──────────────────────────────────────────────────────────────

class RiskMonitor:
    """
    Aggregates all risk metrics for a running portfolio.
    Called after each trading day.
    """

    def __init__(self, limits: RiskLimits = None):
        self.limits    = limits or RiskLimits()
        self.dd_tracker = DrawdownTracker()
        self.metrics_history: list[dict] = []

    def update(
        self,
        date: pd.Timestamp,
        nav: float,
        daily_return: float,
        weights: pd.Series,
        returns_window: pd.Series,
    ) -> dict:
        # Drawdown
        dd_rec = self.dd_tracker.update(nav, date)

        # VaR / CVaR
        var95  = historical_var(returns_window, 0.95) if len(returns_window) > 20 else 0.0
        cvar95 = conditional_var(returns_window, 0.95) if len(returns_window) > 20 else 0.0

        # Realised vol (annualised)
        vol = returns_window.tail(21).std() * np.sqrt(252) if len(returns_window) > 5 else 0.0

        # Exposure
        gross = weights.abs().sum()
        net   = weights.sum()

        # Halt signal
        halt = dd_rec["drawdown"] < -self.limits.max_drawdown_pct

        metrics = {
            "date":        date,
            "nav":         nav,
            "daily_ret":   daily_return,
            "drawdown":    dd_rec["drawdown"],
            "max_dd":      self.dd_tracker.max_drawdown(),
            "var_95":      var95,
            "cvar_95":     cvar95,
            "realised_vol": vol,
            "gross_exp":   gross,
            "net_exp":     net,
            "halt":        halt,
        }
        self.metrics_history.append(metrics)
        return metrics

    def get_metrics_df(self) -> pd.DataFrame:
        if not self.metrics_history:
            return pd.DataFrame()
        return pd.DataFrame(self.metrics_history).set_index("date")

    def volatility_scaling(self, current_vol: float) -> float:
        """
        Target volatility scaling: scale exposure to hit vol target.
        Returns multiplier to apply to portfolio weights.
        """
        if current_vol < 1e-6:
            return 1.0
        return min(self.limits.volatility_target / current_vol, 2.0)

    def stress_test(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
        scenarios: dict = None,
    ) -> dict:
        """
        Apply historical or hypothetical stress scenarios.
        scenarios: {name: {ticker: shock_pct}}
        """
        if scenarios is None:
            scenarios = {
                "covid_crash_2020": {t: -0.35 for t in weights.index},
                "rate_shock_2022":  {t: -0.20 for t in weights.index},
                "flash_crash":      {t: -0.10 for t in weights.index},
                "tech_selloff":     {t: (-0.40 if "TECH" in t else -0.05) for t in weights.index},
            }

        results = {}
        for name, shocks in scenarios.items():
            shock_series = pd.Series(shocks).reindex(weights.index, fill_value=0)
            pnl = (weights * shock_series).sum()
            results[name] = round(pnl, 4)

        return results


if __name__ == "__main__":
    np.random.seed(42)
    returns = pd.Series(np.random.randn(252) * 0.01)
    print(f"[VaR 95%]  Historical: {historical_var(returns):.4f}")
    print(f"[CVaR 95%] {conditional_var(returns):.4f}")
    print(f"[VaR 95%]  Parametric: {parametric_var(returns):.4f}")

    tracker = DrawdownTracker()
    nav = 1.0
    for i, r in enumerate(returns):
        nav *= (1 + r)
        tracker.update(nav, pd.Timestamp("2023-01-01") + pd.Timedelta(days=i))

    print(f"\n[Max Drawdown] {tracker.max_drawdown():.4f}")
    print(f"[Current DD]   {tracker.current_drawdown():.4f}")
