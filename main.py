"""
QuantSys — Main Pipeline Orchestrator
=======================================
Integrates all modules into a cohesive end-to-end trading system.
Run from project root: python main.py
"""

import sys
import os
import argparse
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from features import DataIngester, FeatureEngineer
from signals import signal_fn
from optimiser import SignalProportionalWeighter, RiskParityOptimiser, ledoit_wolf_covariance
from manager import RiskMonitor, PreTradeChecker, RiskLimits
from engine import BacktestEngine, PerformanceAnalytics, MonteCarloSimulator, TransactionCostModel


# ── Strategy Factory ──────────────────────────────────────────────────────────

def build_momentum_strategy(panel_close: pd.DataFrame, panel_volume: pd.DataFrame):
    """
    Momentum long-short strategy.
    Buys top-quintile momentum stocks, shorts bottom-quintile.
    Adapts to market regime.
    """
    fs      = FactorSignals(panel_close)
    weighter = SignalProportionalWeighter(n_longs=5, n_shorts=5, max_position=0.12)
    regime  = RegimeDetector(n_regimes=3)

    # Fit regime on full history (in-sample — ok for regime labelling)
    mkt_ret = panel_close.mean(axis=1).pct_change().dropna()
    regime.fit(mkt_ret)
    regime_series = regime.predict(mkt_ret)

    def signal_fn(date, history):
        close_hist = history.reindex(panel_close.columns, axis=1)
        if len(close_hist) < 252:
            return None

        fs_local = FactorSignals(close_hist)
        mom      = fs_local.momentum_signal().iloc[-1]

        # Regime adjustment
        r = regime_series.get(date, 1)
        mult = regime.regime_multiplier(
            pd.Series([r], index=[date]), strategy="momentum"
        ).iloc[0]

        signal = mom * mult
        return weighter.construct(signal.dropna())

    return signal_fn


def build_risk_parity_strategy(panel_close: pd.DataFrame):
    """
    Long-only risk parity strategy. Equal risk contribution from each asset.
    """
    optimiser = RiskParityOptimiser(max_position=0.15)

    def signal_fn(date, history):
        if len(history) < 63:
            return None
        returns = history.pct_change().tail(63).dropna()
        if returns.shape[1] < 3:
            return None
        cov = ledoit_wolf_covariance(returns, window=63)
        tickers = list(returns.columns)
        return optimiser.optimise(cov, tickers)

    return signal_fn


# ── Full Pipeline Run ─────────────────────────────────────────────────────────

def run_pipeline(
    universe: str = "sp500_sample",
    strategy: str = "momentum",
    start: str = "2021-01-01",
    end: str = "2024-12-31",
    rebalance_freq: str = "W",
    initial_capital: float = 1_000_000,
    run_monte_carlo: bool = True,
    verbose: bool = True,
):
    print("\n" + "="*60)
    print("  QUANTSYS — PIPELINE START")
    print("="*60)

    # ── 1. Data Ingestion ────────────────────────────────────────
    print("\n[1/6] Loading market data...")
    ingester = DataIngester(universe)
    data     = ingester.fetch(start, end)
    if len(data) == 0:
        print("[ERROR] No data fetched. Check universe / internet connection.")
        return None

    panel_close  = ingester.get_panel("close").dropna(how="all", axis=1)
    panel_volume = ingester.get_panel("volume").dropna(how="all", axis=1)
    print(f"       Universe: {panel_close.shape[1]} tickers | "
          f"{panel_close.shape[0]} trading days")

    # ── 2. Feature Engineering (sample ticker) ───────────────────
    print("\n[2/6] Building features...")
    ticker_sample = list(data.keys())[0]
    fe = FeatureEngineer(data[ticker_sample])
    features = fe.build_all()
    print(f"       {ticker_sample}: {features.shape[1]} features built")

    # ── 3. Signal Generation / Strategy Selection ────────────────
    print(f"\n[3/6] Building strategy: {strategy}...")
    print(f"\n[3/6] Using COMPOSITE ALPHA SIGNAL (Sharpe Optimised)...")

    # ── 4. Backtest ───────────────────────────────────────────────
    print(f"\n[4/6] Running backtest ({start} → {end})...")
    tc_model = TransactionCostModel(
        commission_pct=0.0005,
        bid_ask_spread=0.0010,
        slippage_vol_mult=0.10,
    )
    engine = BacktestEngine(
        price_panel=panel_close,
        signal_fn=signal_fn,
        tc_model=tc_model,
        initial_capital=initial_capital,
        rebalance_freq=rebalance_freq,
    )
    results = engine.run(
    start=start,
    end=end,
    feature_data={
        "close": panel_close,
        "volume": panel_volume
    },
    warmup_days=252
)

    # ── 5. Risk Monitoring ────────────────────────────────────────
    print("\n[5/6] Risk metrics...")
    monitor = RiskMonitor(RiskLimits())
    ec = results["equity_curve"]
    rets = results["returns"]

    for i, (date, row) in enumerate(ec.iterrows()):
        monitor.update(
            date=date,
            nav=row["nav"],
            daily_return=row["daily_return"],
            weights=pd.Series(dtype=float),
            returns_window=rets.iloc[max(0, i-63):i],
        )

    risk_df = monitor.get_metrics_df()

    # ── 6. Performance Summary ────────────────────────────────────
    print("\n[6/6] Performance summary...")
    analytics = results["analytics"]
    analytics.print_summary()

    # Monte Carlo CI
    mc_results = None
    if run_monte_carlo:
        print("\n[MC]  Running Monte Carlo (500 simulations)...")
        mc = MonteCarloSimulator(n_simulations=500)
        mc_results = mc.bootstrap_metrics(rets)
        print(f"      Sharpe  90% CI: [{mc_results['sharpe']['p5']:+.3f}, {mc_results['sharpe']['p95']:+.3f}]")
        print(f"      Max DD  90% CI: [{mc_results['max_dd']['p5']:+.2%}, {mc_results['max_dd']['p95']:+.2%}]")

    # Save outputs
    save_outputs(results, risk_df, mc_results, strategy, universe)

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60 + "\n")

    return {
        "results":     results,
        "risk_df":     risk_df,
        "mc_results":  mc_results,
        "panel_close": panel_close,
    }


def save_outputs(results, risk_df, mc_results, strategy, universe):
    """Save all outputs to CSV / JSON."""
    os.makedirs("outputs", exist_ok=True)

    # Equity curve
    results["equity_curve"].to_csv(f"outputs/{strategy}_equity_curve.csv")

    # Trade log
    if not results["trade_log"].empty:
        results["trade_log"].to_csv(f"outputs/{strategy}_trade_log.csv", index=False)

    # Risk metrics
    if not risk_df.empty:
        risk_df.to_csv(f"outputs/{strategy}_risk_metrics.csv")

    # Performance metrics JSON
    metrics = results["metrics"].copy()
    if mc_results:
        metrics["monte_carlo"] = mc_results
    with open(f"outputs/{strategy}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\n  Outputs saved → outputs/{strategy}_*.csv / .json")


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuantSys Trading Pipeline")
    parser.add_argument("--strategy",  default="momentum",
                        choices=["momentum", "risk_parity"],
                        help="Strategy to run")
    parser.add_argument("--universe",  default="sp500_sample",
                        choices=["sp500_sample", "nifty50", "crypto"],
                        help="Instrument universe")
    parser.add_argument("--start",     default="2021-01-01",
                        help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end",       default="2024-12-31",
                        help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--freq",      default="W",
                        choices=["D", "W", "M"],
                        help="Rebalance frequency")
    parser.add_argument("--capital",   default=1_000_000, type=float,
                        help="Initial capital")
    parser.add_argument("--no-mc",     action="store_true",
                        help="Skip Monte Carlo simulation")
    args = parser.parse_args()

    run_pipeline(
        universe=args.universe,
        strategy=args.strategy,
        start=args.start,
        end=args.end,
        rebalance_freq=args.freq,
        initial_capital=args.capital,
        run_monte_carlo=not args.no_mc,
    )
