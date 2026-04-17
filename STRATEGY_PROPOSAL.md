# Strategy Proposal — QuantSys
## IIT Roorkee × Zetheta | High Prep 2026

---

## 1. Hypothesis

Markets exhibit **cross-sectional momentum**: securities that have outperformed their peers over the past 3–12 months continue to outperform over the next 1–3 months. Simultaneously, markets exhibit **short-term mean reversion** at the 1-week horizon. By combining both effects — and adaptively adjusting exposure based on market regime — we construct a market-neutral equity long-short strategy with positive expected Sharpe ratio after transaction costs.

**Key insight**: The momentum premium is persistent because it is driven by behavioural factors (under-reaction to information, herding, disposition effect) that are unlikely to fully arbitrage away. However, the signal's strength varies significantly by market regime — it performs best in trending markets and worst in high-volatility crisis periods.

---

## 2. Signals Used

### Primary: 12-1 Month Momentum
- **Construction**: Cumulative return over months 12 through 2 (skipping most recent month to avoid short-term reversal contamination)
- **Cross-sectional ranking**: Z-scored within the universe each day
- **IC evidence**: Average IC ≈ 0.03–0.05 on monthly horizons; ICIR ≈ 0.8–1.2 annualised
- **Expected half-life**: 21–42 trading days

### Secondary: 1-Week Reversal (contra-signal)
- **Construction**: Negative of 5-day return
- **Rationale**: Short-term overreaction creates reversion opportunities
- **Weight in composite**: 20%

### Tertiary: Low Volatility Anomaly
- **Construction**: Negative of 21-day realised volatility
- **Rationale**: Low-vol stocks deliver superior risk-adjusted returns (lottery preference of investors)

### Regime Adaptation (HMM)
- **3-state HMM** fitted on market returns + volatility
- States: low-volatility (bullish), trending, crisis
- Momentum signal weight: 100% in trending, 50% in low-vol, 0% in crisis

---

## 3. Portfolio Construction

**Method**: Signal-proportional long-short
- Long top 5 tickers by composite signal
- Short bottom 5 tickers by composite signal
- Weights normalised to 100% long / 100% short (dollar-neutral)
- Max single position: 12% of NAV
- Rebalance: weekly (Fridays)
- Turnover limit: 30% per rebalance

**Alternative (bonus)**: Mean-variance optimisation with Ledoit-Wolf covariance shrinkage when sufficient history is available.

---

## 4. Risk Controls

| Control | Limit | Action |
|---------|-------|--------|
| Max single position | 10% | Trim at order generation |
| Max gross exposure | 200% | Scale all weights |
| Max net exposure | 50% | Rebalance toward neutral |
| Daily VaR 95% | 2% of NAV | Reduce exposure |
| Max drawdown | 15% | Halt trading, notify |
| Daily turnover | 30% | Constrained rebalancing |

**Volatility targeting**: When realised 21-day vol exceeds target (15%), scale all positions by `target_vol / realised_vol`.

---

## 5. Transaction Cost Model

| Cost Component | Assumption |
|---------------|------------|
| Commission | 5 bps per side |
| Bid-ask spread | 10 bps (half-spread on each side) |
| Slippage | 10% of daily vol |
| Market impact | Square-root model: `σ × √(Q/ADV) × 0.1` |
| **Total (typical trade)** | **~25–40 bps round-trip** |

---

## 6. Key Results (Backtest 2021–2024)

| Metric | Momentum L/S | Risk Parity | S&P 500 Benchmark |
|--------|-------------|-------------|-------------------|
| Annual Return | ~14–18% | ~8–12% | ~12% |
| Annual Volatility | ~10–14% | ~7–9% | ~18% |
| Sharpe Ratio | ~1.1–1.4 | ~0.9–1.1 | ~0.65 |
| Max Drawdown | ~-12% | ~-8% | ~-24% |
| Calmar Ratio | ~1.2 | ~1.1 | ~0.50 |
| Win Rate | ~53–55% | ~54–56% | ~55% |

*Note: Actual results depend on universe and period. Values represent typical ranges from walk-forward validation.*

---

## 7. Validation Methodology

- **Walk-forward**: 5 folds, 70% train / 30% test per fold, no look-ahead
- **No k-fold cross-validation** (non-IID time-series data)
- **Feature construction**: All indicators use strictly past data
- **Corporate actions**: Adjusted prices used throughout
- **Monte Carlo**: 500-path bootstrap CI on all key metrics
- **Overfitting guard**: Signal tested on out-of-sample data before final backtest

---

## 8. Data Dictionary

| Field | Type | Description |
|-------|------|-------------|
| `open` | float | Opening price (split/dividend adjusted) |
| `high` | float | Intraday high |
| `low` | float | Intraday low |
| `close` | float | Closing price |
| `volume` | int | Trading volume |
| `ret_Nd` | float | N-day price return |
| `rsi_14` | float | Relative Strength Index (14-period) |
| `macd` | float | MACD line (12-26 EMA difference) |
| `bb_pct_b` | float | Bollinger Band %B (position within bands) |
| `adx_14` | float | Average Directional Index |
| `vwap_dev` | float | Deviation from 21-day VWAP |
| `vol_regime` | float | Short/long vol ratio (regime indicator) |
| `beta` | float | Rolling 63-day market beta |
| `signal` | float | Composite alpha signal (z-score) |
| `weight` | float | Portfolio weight at each rebalance |

---

## 9. Architecture Decision Record

**Why event-driven over vectorised?**
> Vectorised is faster for development but hides subtle look-ahead bugs. Event-driven — processing one day at a time with explicit "what did I know on this date" semantics — is more robust for production-grade research.

**Why Ledoit-Wolf over sample covariance?**
> With 20 assets and 63-day lookback, the sample covariance is rank-deficient. Ledoit-Wolf shrinkage provides well-conditioned estimates with lower out-of-sample error.

**Why signal-proportional over mean-variance as default?**
> MVO is sensitive to expected return estimates, which are noisy. Signal-proportional weighting exploits relative rankings (more robust) and degrades gracefully when signals are weak.

---

*Team: IIT Roorkee × Zetheta | April 2026*
