class BacktestEngine:
    """
    Event-driven backtesting engine.
    Iterates day by day, applying signals, rebalancing, and recording P&L.
    """

    def __init__(
        self,
        price_panel: pd.DataFrame,          # dates × tickers
        signal_fn: Callable,                 # f(date, history) → pd.Series weights
        tc_model: TransactionCostModel = None,
        initial_capital: float = 1_000_000,
        rebalance_freq: str = "W",
    ):
        self.prices    = price_panel
        self.signal_fn = signal_fn
        self.tc        = tc_model or TransactionCostModel()
        self.capital   = initial_capital
        self.freq      = rebalance_freq

        self.equity_curve: list[dict] = []
        self.trade_log:    list[dict] = []
        self.weights_history: dict[pd.Timestamp, pd.Series] = {}

    def _should_rebalance(self, date: pd.Timestamp) -> bool:
        if self.freq == "D":
            return True
        elif self.freq == "W":
            return date.dayofweek == 4
        elif self.freq == "M":
            return date.is_month_end
        return False

    def run(
    self,
    start: str,
    end: str,
    warmup_days: int = 252,
) -> dict:

    dates = self.prices.loc[start:end].index
    print(f"\n[BACKTEST] Running {start} → {end} | {len(dates)} trading days")

    nav = self.capital
    current_weights = pd.Series(dtype=float)
    prev_prices = None

    for i, date in enumerate(dates):
        prices_today = self.prices.loc[date]

        # ── DAILY P&L ───────────────────────────────────────────────
        if prev_prices is not None and not current_weights.empty:
            common = current_weights.index.intersection(prices_today.index)
            common = common.intersection(prev_prices.index)

            rets = (prices_today[common] / prev_prices[common] - 1).fillna(0)
            daily_pnl = (current_weights.reindex(common, fill_value=0) * rets).sum()

            nav *= (1 + daily_pnl)
        else:
            daily_pnl = 0.0

        # ── DRAWDOWN CONTROL (RISK MANAGEMENT) ─────────────────────
        if len(self.equity_curve) > 10:
            ec_df = pd.DataFrame(self.equity_curve)
            cum = ec_df["nav"]
            peak = cum.cummax()
            dd = (cum - peak) / peak

            if dd.min() < -0.2:
                current_weights = pd.Series(dtype=float)  # exit all

        # ── REBALANCING ────────────────────────────────────────────
        history = self.prices.loc[:date]

        if i >= warmup_days and self._should_rebalance(date):

            try:
                # 🔴 FIX 1: REMOVE LOOK-AHEAD BIAS
                signal_date = history.index[-2]
                new_weights = self.signal_fn(signal_date, history.iloc[:-1])

                if new_weights is not None and not new_weights.empty:

                    # 🔴 FIX 2: POSITION LIMIT
                    max_position = 0.1
                    new_weights = new_weights.clip(-max_position, max_position)

                    # normalize weights
                    if new_weights.abs().sum() > 0:
                        new_weights = new_weights / new_weights.abs().sum()

                    # ── TURNOVER ────────────────────────────────
                    all_t = new_weights.index.union(current_weights.index)
                    n_w = new_weights.reindex(all_t, fill_value=0)
                    c_w = current_weights.reindex(all_t, fill_value=0)

                    turnover = (n_w - c_w).abs().sum() / 2

                    # ── TRANSACTION COST ───────────────────────
                    vol_est = history.pct_change().tail(21).std().mean()

                    tc_cost = self.tc.total_cost(turnover, float(vol_est))

                    # 🔴 FIX 3: APPLY COST PROPERLY
                    cost_amount = nav * tc_cost
                    nav -= cost_amount

                    # ── TRADE LOG ──────────────────────────────
                    delta = n_w - c_w
                    for ticker, dw in delta.items():
                        if abs(dw) > 1e-4:
                            self.trade_log.append({
                                "date": date,
                                "ticker": ticker,
                                "weight_change": round(dw, 4),
                                "price": prices_today.get(ticker, np.nan),
                                "tc_cost": round(tc_cost * abs(dw), 6),
                            })

                    current_weights = new_weights.copy()
                    self.weights_history[date] = current_weights.copy()

            except Exception as e:
                pass

        # ── SAVE RESULTS ───────────────────────────────────────────
        self.equity_curve.append({
            "date": date,
            "nav": nav,
            "daily_return": daily_pnl,
            "turnover": turnover if 'turnover' in locals() else 0
        })

        prev_prices = prices_today.copy()

    print(f"[BACKTEST] Complete. Final NAV: ${nav:,.0f} | Trades: {len(self.trade_log)}")

    return self._compile_results()

    def _compile_results(self) -> dict:
        ec_df = pd.DataFrame(self.equity_curve).set_index("date")
        returns = ec_df["daily_return"]
        analytics = PerformanceAnalytics(returns)

        return {
            "equity_curve": ec_df,
            "returns":      returns,
            "trade_log":    pd.DataFrame(self.trade_log),
            "metrics":      analytics.all_metrics(),
            "analytics":    analytics,
        }


# ── Walk-Forward Validator ────────────────────────────────────────────────────

class WalkForwardValidator:
    """
    Implements rolling walk-forward analysis to assess out-of-sample performance.
    Prevents look-ahead bias and data snooping.
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_pct: float = 0.70,
    ):
        self.n_splits  = n_splits
        self.train_pct = train_pct

    def split(self, dates: pd.DatetimeIndex) -> list[tuple]:
        """
        Generate (train_start, train_end, test_start, test_end) tuples.
        Each fold moves forward in time.
        """
        n = len(dates)
        fold_size = n // (self.n_splits + 1)
        splits = []
        for i in range(self.n_splits):
            train_end_idx = fold_size * (i + 1) + int(fold_size * self.train_pct)
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