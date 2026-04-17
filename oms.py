"""
Module 6: Execution & Order Management
=======================================
Translates portfolio weight targets into executable orders.
Implements TWAP, VWAP, and Almgren-Chriss execution algorithms.
Tracks implementation shortfall.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


# ── Order Primitives ──────────────────────────────────────────────────────────

class OrderStatus(Enum):
    CREATED    = "created"
    VALIDATED  = "validated"
    SUBMITTED  = "submitted"
    PARTIAL    = "partial_fill"
    FILLED     = "filled"
    CANCELLED  = "cancelled"
    REJECTED   = "rejected"


class OrderSide(Enum):
    BUY  = "buy"
    SELL = "sell"


@dataclass
class Order:
    order_id:      str
    ticker:        str
    side:          OrderSide
    qty:           float           # shares or notional units
    decision_price: float          # price when decision was made
    status:        OrderStatus = OrderStatus.CREATED
    submitted_price: Optional[float] = None
    fill_price:    Optional[float] = None
    fill_qty:      float = 0.0
    timestamp:     Optional[pd.Timestamp] = None
    algo:          str = "market"
    notes:         str = ""

    @property
    def implementation_shortfall(self) -> float:
        """IS = fill_price - decision_price for buys (reversed for sells)."""
        if self.fill_price is None or self.decision_price is None:
            return np.nan
        if self.side == OrderSide.BUY:
            return (self.fill_price - self.decision_price) / self.decision_price
        else:
            return (self.decision_price - self.fill_price) / self.decision_price


# ── Order Management System ───────────────────────────────────────────────────

class OrderManagementSystem:
    """
    Full lifecycle management of orders:
    Created → Validated → Submitted → (Partial →) Filled / Cancelled / Rejected
    """

    def __init__(self, max_position_pct: float = 0.10, nav: float = 1_000_000):
        self.nav = nav
        self.max_pos = max_position_pct
        self.orders: dict[str, Order] = {}
        self._order_counter = 0

    def _new_id(self) -> str:
        self._order_counter += 1
        return f"ORD-{self._order_counter:06d}"

    def create_order(
        self,
        ticker: str,
        side: OrderSide,
        qty: float,
        decision_price: float,
        algo: str = "market",
    ) -> Order:
        order = Order(
            order_id=self._new_id(),
            ticker=ticker,
            side=side,
            qty=qty,
            decision_price=decision_price,
            timestamp=pd.Timestamp.now(),
            algo=algo,
        )
        self.orders[order.order_id] = order
        return order

    def validate(self, order: Order, current_positions: dict) -> bool:
        """Check order against risk limits before submission."""
        notional = order.qty * order.decision_price
        max_notional = self.nav * self.max_pos

        if notional > max_notional:
            order.status = OrderStatus.REJECTED
            order.notes = f"Exceeds max position: {notional:.0f} > {max_notional:.0f}"
            return False

        if order.qty <= 0:
            order.status = OrderStatus.REJECTED
            order.notes = "Invalid quantity"
            return False

        order.status = OrderStatus.VALIDATED
        return True

    def submit(self, order: Order, market_price: float):
        """Simulate order submission with bid-ask spread."""
        order.submitted_price = market_price
        order.status = OrderStatus.SUBMITTED

    def fill(self, order: Order, fill_price: float, fill_qty: float = None):
        """Record fill. Partial fill if fill_qty < order.qty."""
        order.fill_qty   = fill_qty or order.qty
        order.fill_price = fill_price
        order.status = OrderStatus.FILLED if order.fill_qty >= order.qty else OrderStatus.PARTIAL

    def cancel(self, order_id: str, reason: str = ""):
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            self.orders[order_id].notes  = reason

    def get_trade_log(self) -> pd.DataFrame:
        records = []
        for o in self.orders.values():
            records.append({
                "order_id": o.order_id,
                "ticker":   o.ticker,
                "side":     o.side.value,
                "qty":      o.qty,
                "decision_price": o.decision_price,
                "fill_price":     o.fill_price,
                "fill_qty":       o.fill_qty,
                "status":         o.status.value,
                "IS_bps":         round(o.implementation_shortfall * 10000, 2)
                                  if not np.isnan(o.implementation_shortfall) else np.nan,
                "algo":           o.algo,
                "timestamp":      o.timestamp,
            })
        return pd.DataFrame(records)


# ── Execution Algorithms ──────────────────────────────────────────────────────

class TWAPExecutor:
    """
    Time-Weighted Average Price execution.
    Splits large orders into equal-sized child orders over N intervals.
    """

    def __init__(self, n_slices: int = 10):
        self.n_slices = n_slices

    def schedule(
        self,
        order: Order,
        intraday_prices: pd.Series,
    ) -> list[dict]:
        """
        Generate child order schedule.
        intraday_prices: pd.Series with intraday timestamps as index.
        """
        slice_qty = order.qty / self.n_slices
        intervals = pd.date_range(
            start=intraday_prices.index[0],
            end=intraday_prices.index[-1],
            periods=self.n_slices,
        )

        schedule = []
        for t in intervals:
            # Find nearest available price
            idx = intraday_prices.index.get_indexer([t], method="nearest")[0]
            price = intraday_prices.iloc[idx]
            schedule.append({
                "time":       t,
                "qty":        round(slice_qty, 4),
                "limit_price": price,
            })

        return schedule

    def simulate_fill(
        self,
        order: Order,
        intraday_prices: pd.Series,
        volatility: float = 0.001,
    ) -> dict:
        """Simulate TWAP execution with random slippage per slice."""
        schedule = self.schedule(order, intraday_prices)
        fill_prices  = []
        fill_weights = []

        for child in schedule:
            slippage = np.random.randn() * volatility
            if order.side == OrderSide.BUY:
                actual_price = child["limit_price"] * (1 + abs(slippage))
            else:
                actual_price = child["limit_price"] * (1 - abs(slippage))

            fill_prices.append(actual_price)
            fill_weights.append(child["qty"])

        avg_fill = np.average(fill_prices, weights=fill_weights)
        is_bps   = ((avg_fill - order.decision_price) / order.decision_price) * 10000
        if order.side == OrderSide.SELL:
            is_bps = -is_bps

        return {
            "avg_fill_price": round(avg_fill, 4),
            "n_slices": len(schedule),
            "IS_bps":   round(is_bps, 2),
            "twap_price": round(np.mean([s["limit_price"] for s in schedule]), 4),
        }


class VWAPExecutor:
    """
    Volume-Weighted Average Price execution.
    Sizes child orders proportional to expected intraday volume profile.
    """

    # Typical intraday volume profile (normalised weights by half-hour)
    INTRADAY_VOL_PROFILE = np.array([
        0.06, 0.07, 0.06, 0.05, 0.05, 0.05, 0.05, 0.05,
        0.05, 0.05, 0.05, 0.06, 0.07, 0.08, 0.08, 0.07,
    ])
    INTRADAY_VOL_PROFILE = INTRADAY_VOL_PROFILE / INTRADAY_VOL_PROFILE.sum()

    def schedule(self, order: Order, total_qty: float = None) -> list[dict]:
        qty = total_qty or order.qty
        return [
            {"slice": i, "qty_pct": w, "qty": round(qty * w, 4)}
            for i, w in enumerate(self.INTRADAY_VOL_PROFILE)
        ]

    def simulate_fill(
        self,
        order: Order,
        daily_vwap: float,
        volatility: float = 0.001,
    ) -> dict:
        """
        Simulate VWAP fill. Good VWAP execution achieves close to daily VWAP.
        """
        tracking_error = np.random.randn() * volatility * 0.5
        fill_price = daily_vwap * (1 + tracking_error)
        is_bps = ((fill_price - order.decision_price) / order.decision_price) * 10000
        if order.side == OrderSide.SELL:
            is_bps = -is_bps

        return {
            "avg_fill_price": round(fill_price, 4),
            "daily_vwap": round(daily_vwap, 4),
            "vwap_slippage_bps": round(tracking_error * 10000, 2),
            "IS_bps": round(is_bps, 2),
        }


class AlmgrenChrissExecutor:
    """
    Almgren-Chriss optimal execution model.
    Balances market impact cost vs. timing risk.
    Finds optimal trading trajectory minimising expected cost + risk penalty.
    """

    def __init__(
        self,
        risk_aversion: float = 0.01,
        temp_impact_coeff: float = 0.1,
        perm_impact_coeff: float = 0.01,
    ):
        self.lam  = risk_aversion
        self.eta  = temp_impact_coeff   # temporary impact
        self.gamma = perm_impact_coeff  # permanent impact

    def optimal_trajectory(
        self,
        total_shares: float,
        n_periods: int = 10,
        sigma: float = 0.02,
        adv: float = 1_000_000,
    ) -> pd.Series:
        """
        Compute optimal selling schedule.
        Returns remaining inventory at each period.
        """
        kappa_sq = self.lam * sigma**2 / (self.eta + 1e-8)
        kappa    = np.sqrt(kappa_sq)

        j = np.arange(n_periods + 1)
        # Optimal holdings at each time step
        inventory = total_shares * np.sinh(kappa * (n_periods - j)) / (np.sinh(kappa * n_periods) + 1e-8)
        trades    = -np.diff(inventory)

        return pd.Series({
            "inventory": inventory.tolist(),
            "trade_schedule": trades.tolist(),
            "expected_cost_bps": round(
                (self.gamma * total_shares + self.eta * np.sum(trades**2 / adv)) * 10000, 2
            ),
        })


# ── Paper Trading Mode ────────────────────────────────────────────────────────

class PaperTrader:
    """
    Forward paper-trading simulation using real-time (delayed) prices.
    Useful for forward validation without real capital.
    """

    def __init__(self, initial_capital: float = 100_000):
        self.capital   = initial_capital
        self.positions: dict[str, float] = {}   # ticker → shares
        self.cash      = initial_capital
        self.trade_log: list[dict] = []
        self.oms       = OrderManagementSystem(nav=initial_capital)

    def execute_weights(
        self,
        target_weights: pd.Series,
        prices: pd.Series,
        date: pd.Timestamp,
    ) -> list[Order]:
        """Convert target portfolio weights to executed orders."""
        nav = self._compute_nav(prices)
        orders = []

        for ticker in target_weights.index.union(pd.Index(self.positions.keys())):
            target_w  = target_weights.get(ticker, 0.0)
            price     = prices.get(ticker)
            if price is None or np.isnan(price):
                continue

            target_shares = (target_w * nav) / price
            current_shares = self.positions.get(ticker, 0.0)
            delta_shares   = target_shares - current_shares

            if abs(delta_shares) < 0.1:
                continue

            side = OrderSide.BUY if delta_shares > 0 else OrderSide.SELL
            order = self.oms.create_order(
                ticker=ticker,
                side=side,
                qty=abs(delta_shares),
                decision_price=price,
                algo="market",
            )

            # Simulate fill with small spread
            spread_mult = 1.0005 if side == OrderSide.BUY else 0.9995
            fill_price  = price * spread_mult
            self.oms.fill(order, fill_price)

            # Update positions
            if side == OrderSide.BUY:
                self.positions[ticker] = current_shares + abs(delta_shares)
                self.cash -= fill_price * abs(delta_shares)
            else:
                self.positions[ticker] = current_shares - abs(delta_shares)
                self.cash += fill_price * abs(delta_shares)

            self.trade_log.append({
                "date": date, "ticker": ticker,
                "side": side.value, "qty": abs(delta_shares),
                "price": fill_price,
            })
            orders.append(order)

        return orders

    def _compute_nav(self, prices: pd.Series) -> float:
        pos_value = sum(
            self.positions.get(t, 0) * prices.get(t, 0)
            for t in self.positions
        )
        return self.cash + pos_value

    def get_nav(self, prices: pd.Series) -> float:
        return self._compute_nav(prices)


if __name__ == "__main__":
    oms = OrderManagementSystem(nav=1_000_000)
    order = oms.create_order("AAPL", OrderSide.BUY, 100, decision_price=180.0, algo="TWAP")
    valid = oms.validate(order, {})
    oms.submit(order, 180.5)
    oms.fill(order, 180.8)
    print(f"Order: {order.order_id} | IS: {order.implementation_shortfall*10000:.1f} bps")

    twap = TWAPExecutor(n_slices=5)
    intraday = pd.Series(
        np.linspace(180, 181, 50) + np.random.randn(50) * 0.1,
        index=pd.date_range("2024-01-01 09:30", periods=50, freq="5min"),
    )
    result = twap.simulate_fill(order, intraday)
    print(f"TWAP: avg fill = {result['avg_fill_price']:.2f} | IS = {result['IS_bps']:.1f} bps")
