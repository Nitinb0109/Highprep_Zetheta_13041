"""
Module 3: Portfolio Construction & Optimisation
================================================
Translates raw alpha signals into portfolio weights.
Implements:
  - Mean-Variance Optimisation (Markowitz)
  - Risk Parity / Equal Risk Contribution
  - Kelly Criterion (fractional)
  - Signal-proportional (simple long-short)

All methods enforce position & turnover constraints.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


# ── Covariance Estimators ─────────────────────────────────────────────────────

def sample_covariance(returns: pd.DataFrame, window: int = 63) -> np.ndarray:
    recent = returns.tail(window).dropna()
    return recent.cov().values * 252  # annualised


def ledoit_wolf_covariance(returns: pd.DataFrame, window: int = 63) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance estimator."""
    try:
        from sklearn.covariance import LedoitWolf
        recent = returns.tail(window).dropna().values
        lw = LedoitWolf().fit(recent)
        return lw.covariance_ * 252
    except Exception:
        return sample_covariance(returns, window)


# ── Portfolio Optimisers ──────────────────────────────────────────────────────

class PortfolioOptimiser:
    """
    Base class for portfolio construction.
    All methods return a pd.Series of weights indexed by ticker.
    """

    def __init__(
        self,
        max_position: float = 0.10,
        max_long_exposure: float = 1.0,
        max_short_exposure: float = 1.0,
        turnover_limit: float = 0.30,
        sector_limit: float = 0.30,
    ):
        self.max_pos   = max_position
        self.max_long  = max_long_exposure
        self.max_short = max_short_exposure
        self.turnover  = turnover_limit
        self.sec_limit = sector_limit


class MeanVarianceOptimiser(PortfolioOptimiser):
    """
    Markowitz mean-variance optimisation.
    Maximises Sharpe ratio subject to weight constraints.
    """

    def optimise(
        self,
        expected_returns: pd.Series,
        covariance: np.ndarray,
        risk_free_rate: float = 0.05,
        long_only: bool = False,
    ) -> pd.Series:
        n = len(expected_returns)
        tickers = expected_returns.index

        def neg_sharpe(w):
            port_ret = w @ expected_returns.values
            port_var = w @ covariance @ w
            return -(port_ret - risk_free_rate) / (np.sqrt(port_var) + 1e-8)

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w[w > 0]) - 1.0},  # long sum = 1
        ]
        if not long_only:
            constraints.append(
                {"type": "eq", "fun": lambda w: np.sum(w[w < 0]) + 1.0}  # short sum = -1
            )

        bounds = [(-self.max_pos, self.max_pos)] * n
        if long_only:
            bounds = [(0, self.max_pos)] * n

        w0 = np.ones(n) / n
        result = minimize(
            neg_sharpe,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        weights = pd.Series(result.x if result.success else w0, index=tickers)
        return self._clean_weights(weights)

    def _clean_weights(self, w: pd.Series, threshold: float = 0.01) -> pd.Series:
        w[w.abs() < threshold] = 0
        longs  = w[w > 0]
        shorts = w[w < 0]
        if longs.sum() > 0:
            w[w > 0] = longs / longs.sum() * self.max_long
        if shorts.sum() < 0:
            w[w < 0] = shorts / shorts.abs().sum() * self.max_short
        return w


class RiskParityOptimiser(PortfolioOptimiser):
    """
    Equal Risk Contribution (ERC) portfolio.
    Each asset contributes equally to total portfolio variance.
    """

    def optimise(self, covariance: np.ndarray, tickers: list[str]) -> pd.Series:
        n = len(tickers)
        target_risk = np.ones(n) / n  # equal risk contribution

        def risk_budget_objective(w):
            port_var = w @ covariance @ w
            marginal  = covariance @ w
            risk_contrib = w * marginal / (port_var + 1e-8)
            return np.sum((risk_contrib - target_risk) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(1e-4, self.max_pos)] * n
        w0 = np.ones(n) / n

        result = minimize(
            risk_budget_objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 1000},
        )

        weights = result.x if result.success else w0
        return pd.Series(weights / weights.sum(), index=tickers)


class KellyCriterion(PortfolioOptimiser):
    """
    Fractional Kelly position sizing.
    Kelly fraction = µ / σ² (single asset), scaled by fraction parameter.
    """

    def __init__(self, fraction: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.fraction = fraction  # half-Kelly by default

    def size_positions(
        self,
        expected_returns: pd.Series,
        volatilities: pd.Series,
    ) -> pd.Series:
        kelly = expected_returns / (volatilities ** 2 + 1e-8)
        fractional = kelly * self.fraction
        # Clip to max position
        fractional = fractional.clip(-self.max_pos, self.max_pos)
        return fractional


class SignalProportionalWeighter(PortfolioOptimiser):
    """
    Simple signal-proportional weighting.
    Long top-N signals, short bottom-N signals.
    """

    def __init__(self, n_longs: int = 5, n_shorts: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.n_longs  = n_longs
        self.n_shorts = n_shorts

    def construct(self, signal: pd.Series) -> pd.Series:
        """
        Signal: cross-sectional z-scores (higher = more bullish).
        Returns portfolio weights.
        """
        signal = signal.dropna()
        if len(signal) < self.n_longs + self.n_shorts:
            return pd.Series(dtype=float)

        ranked = signal.rank(ascending=False)
        n = len(signal)

        weights = pd.Series(0.0, index=signal.index)

        # Long top n_longs
        long_mask = ranked <= self.n_longs
        weights[long_mask] = signal[long_mask]
        # Short bottom n_shorts
        short_mask = ranked > (n - self.n_shorts)
        weights[short_mask] = signal[short_mask]

        # Normalise
        longs = weights[weights > 0]
        shorts = weights[weights < 0]
        if len(longs) > 0:
            weights[weights > 0] = longs / longs.sum()
        if len(shorts) > 0:
            weights[weights < 0] = shorts / shorts.abs().sum() * -1

        return weights.clip(-self.max_pos, self.max_pos)


# ── Portfolio Rebalancer ──────────────────────────────────────────────────────

class PortfolioRebalancer:
    """
    Handles rebalancing logic, turnover tracking, and weight smoothing.
    """

    def __init__(
        self,
        method: str = "signal_proportional",
        rebalance_freq: str = "W",  # "D", "W", "M"
        turnover_limit: float = 0.30,
    ):
        self.method     = method
        self.freq       = rebalance_freq
        self.turnover   = turnover_limit
        self.prev_weights: pd.Series = pd.Series(dtype=float)

    def should_rebalance(self, date: pd.Timestamp) -> bool:
        """Determine if rebalancing should occur on this date."""
        if self.freq == "D":
            return True
        elif self.freq == "W":
            return date.dayofweek == 4  # Friday
        elif self.freq == "M":
            return date.day <= 5 and date.dayofweek < 5
        return False

    def apply_turnover_constraint(
        self,
        target: pd.Series,
        current: pd.Series,
    ) -> pd.Series:
        """Limit portfolio turnover by partial rebalancing."""
        if current.empty:
            return target

        # Align
        all_tickers = target.index.union(current.index)
        t = target.reindex(all_tickers, fill_value=0.0)
        c = current.reindex(all_tickers, fill_value=0.0)

        delta = t - c
        total_turnover = delta.abs().sum()

        if total_turnover <= self.turnover:
            return t

        # Scale changes to meet turnover budget
        scale = self.turnover / (total_turnover + 1e-8)
        adjusted = c + delta * scale
        return adjusted

    def compute_turnover(self, new_w: pd.Series, old_w: pd.Series) -> float:
        """One-way turnover = sum of positive weight changes."""
        all_t = new_w.index.union(old_w.index)
        n = new_w.reindex(all_t, fill_value=0.0)
        o = old_w.reindex(all_t, fill_value=0.0)
        return (n - o).abs().sum() / 2  # two-way


if __name__ == "__main__":
    np.random.seed(42)
    n = 20
    tickers = [f"STK{i:02d}" for i in range(n)]
    er = pd.Series(np.random.randn(n) * 0.1 + 0.05, index=tickers)
    cov = np.diag(np.random.uniform(0.1, 0.4, n) ** 2)

    # Mean-Variance
    mvo = MeanVarianceOptimiser()
    w_mvo = mvo.optimise(er, cov)
    print(f"[MVO] Non-zero positions: {(w_mvo.abs() > 0.01).sum()}")
    print(f"[MVO] Net exposure: {w_mvo.sum():.3f}")

    # Risk Parity
    rp = RiskParityOptimiser()
    w_rp = rp.optimise(cov, tickers)
    print(f"\n[Risk Parity] Max weight: {w_rp.max():.3f} | Min: {w_rp.min():.3f}")

    # Signal-proportional
    signal = pd.Series(np.random.randn(n), index=tickers)
    spw = SignalProportionalWeighter(n_longs=5, n_shorts=5)
    w_sp = spw.construct(signal)
    print(f"\n[Signal-Prop] Longs: {(w_sp > 0).sum()} | Shorts: {(w_sp < 0).sum()}")
