#!/usr/bin/env python3
"""
alpaca_backtester.py

A single-file backtester for your Alpaca bot strategies, modeled after your Fidelity backtester,
but with full buy/sell execution, cash & buying power simulation, and biweekly contributions.

- Supports two buy modes: "momentum" and "rsi_reversal"
- Applies the same SPY 15m market gates as your live bots:
    * momentum: allow buys only if SPY MA60 > MA240
    * rsi_reversal: allow buys only if SPY MA60 < MA240
    * take-profit selling: allow sells only if SPY MA60 > MA240
- Adds $50 to buying power every 2 weeks starting at START_DATE
- Puts all sale proceeds back into buying power (compounding)
- One buy per asset per calendar day (like your Fidelity backtester)

ENV CONFIG (examples)
---------------------
TZ=America/Denver
START_DATE=2021-01-01
END_DATE=2025-10-22
BACKTEST_YEARS=5

# Data / universe
BAR_MINUTES=15
UNIVERSE=SPY,QQQ,IWM,DIA,XLK,XLF,XLV
ALPACA_DATA_FEED=iex     # or "sip" if you have access
APCA_API_KEY_ID=...      # required
APCA_API_SECRET_KEY=...  # required

# Strategy / sizing
MODE=momentum            # or "rsi_reversal"
TAKE_PROFIT_PCT=0.05
NOTIONAL_PCT=0.05        # fraction of *current cash* per buy
MIN_ORDER_DOLLARS=1.0
ONE_BUY_PER_ASSET_PER_DAY=1  # 1=true, 0=false

# Contributions / frictions
BIWEEKLY_CONTRIB=50.0
FEE_PCT=0.0000           # e.g., 0.0005 = 5 bps per side
SLIPPAGE_PCT=0.0000      # applied to fill price (worse by this pct)

USAGE
-----
python alpaca_backtester.py

Requires: alpaca-py, pandas, numpy
"""
from __future__ import annotations

import os
import math
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    from alpaca.data import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
except Exception as e:
    print("alpaca-py is required. pip install alpaca-py", file=sys.stderr)
    raise


# -----------------------
# Config via ENV
# -----------------------
TZ = os.getenv("TZ", "America/Denver")
START_DATE = os.getenv("START_DATE")
END_DATE   = os.getenv("END_DATE")
BACKTEST_YEARS = int(os.getenv("BACKTEST_YEARS", "5"))

BAR_MINUTES = int(os.getenv("BAR_MINUTES", "15"))
DATA_FEED   = os.getenv("ALPACA_DATA_FEED", "iex").strip() or None

UNIVERSE = [s.strip().upper() for s in os.getenv("UNIVERSE", "SPY,QQQ,IWM,DIA").split(",") if s.strip()]
if "SPY" not in UNIVERSE:
    UNIVERSE.append("SPY")

MODE = os.getenv("MODE", "momentum").strip().lower()

TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.05"))
NOTIONAL_PCT    = float(os.getenv("NOTIONAL_PCT", "0.05"))
MIN_ORDER_DOLLARS = float(os.getenv("MIN_ORDER_DOLLARS", "1.0"))
ONE_BUY_PER_ASSET_PER_DAY = os.getenv("ONE_BUY_PER_ASSET_PER_DAY", "1").lower() in ("1","true","yes")

BIWEEKLY_CONTRIB = float(os.getenv("BIWEEKLY_CONTRIB", "50.0"))
FEE_PCT = float(os.getenv("FEE_PCT", "0.0"))
SLIPPAGE_PCT = float(os.getenv("SLIPPAGE_PCT", "0.0"))

API_KEY = os.environ.get("APCA_API_KEY_ID")
API_SECRET = os.environ.get("APCA_API_SECRET_KEY")


# -----------------------
# Date helpers
# -----------------------
def _parse_dates() -> Tuple[pd.Timestamp, pd.Timestamp]:
    today_tz = pd.Timestamp.now(tz=TZ).normalize()
    end = pd.Timestamp(END_DATE, tz=TZ) if END_DATE else today_tz
    if START_DATE:
        start = pd.Timestamp(START_DATE, tz=TZ)
        print(f"[DATES] Using explicit dates: START_DATE={start}, END_DATE={end}")
    else:
        start = end - pd.DateOffset(years=BACKTEST_YEARS)
        print(f"[DATES] Using BACKTEST_YEARS={BACKTEST_YEARS}: start={start}, end={end}")
    return start, end

def _ensure_tz(index: pd.Index, target_tz: str) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(target_tz)


# -----------------------
# Data fetching
# -----------------------
def _fetch_bars(symbols: List[str], start: pd.Timestamp, end: pd.Timestamp, minutes: int) -> pd.DataFrame:
    if not (API_KEY and API_SECRET):
        raise RuntimeError("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY env vars")
    client = StockHistoricalDataClient(API_KEY, API_SECRET)
    req_kwargs = dict(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame(minutes, TimeFrameUnit.Minute),
        start=start.tz_convert("UTC").to_pydatetime(),
        end=end.tz_convert("UTC").to_pydatetime(),
        limit=None,
    )
    if DATA_FEED:
        req_kwargs["feed"] = DATA_FEED
    bars = client.get_stock_bars(StockBarsRequest(**req_kwargs))

    records = []
    for sym, sb in bars.data.items():
        for b in sb:
            records.append({"timestamp": b.timestamp, "symbol": sym, "close": float(b.close)})
    df = pd.DataFrame.from_records(records)
    pivot = df.pivot(index="timestamp", columns="symbol", values="close")
    pivot.index = _ensure_tz(pivot.index, TZ)
    return pivot.sort_index()


# -----------------------
# Indicators
# -----------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# -----------------------
# Signal builders
# -----------------------
def spy_gate_allow_buys(close: pd.DataFrame) -> pd.Series:
    spy = close["SPY"]
    ma60 = spy.rolling(60, min_periods=60).mean()
    ma240 = spy.rolling(240, min_periods=240).mean()
    return (ma60 > ma240) & (~ma60.isna()) & (~ma240.isna())

def spy_gate_allow_sells(close: pd.DataFrame) -> pd.Series:
    return spy_gate_allow_buys(close)

def spy_gate_allow_buys_reversal(close: pd.DataFrame) -> pd.Series:
    spy = close["SPY"]
    ma60 = spy.rolling(60, min_periods=60).mean()
    ma240 = spy.rolling(240, min_periods=240).mean()
    return (ma60 < ma240) & (~ma60.isna()) & (~ma240.isna())

def signals_momentum(close: pd.DataFrame) -> pd.DataFrame:
    sig = pd.DataFrame(False, index=close.index, columns=close.columns)
    for sym in close.columns:
        px = close[sym]
        ma60 = px.rolling(60, min_periods=60).mean()
        ma240 = px.rolling(240, min_periods=240).mean()
        r = rsi(px, 14)
        _, _, hist = macd(px)
        cond = (ma60 > ma240) & (px > ma60)
        cond &= r.between(55, 70) & (r > r.shift(1))
        cond &= (hist > 0) & (hist > hist.shift(1))
        sig[sym] = cond.fillna(False)
    return sig

def signals_rsi_reversal(close: pd.DataFrame) -> pd.DataFrame:
    sig = pd.DataFrame(False, index=close.index, columns=close.columns)
    for sym in close.columns:
        px = close[sym]
        ma60 = px.rolling(60, min_periods=60).mean()
        ma240 = px.rolling(240, min_periods=240).mean()
        r = rsi(px, 14)
        cond = (ma60 < ma240) & (r < 30)
        sig[sym] = cond.fillna(False)
    return sig


# -----------------------
# Portfolio structs
# -----------------------
@dataclass
class Position:
    qty: float = 0.0
    cost: float = 0.0

@dataclass
class Portfolio:
    cash: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    last_buy_date: Dict[str, Optional[pd.Timestamp.date]] = field(default_factory=dict)


# -----------------------
# Execution helpers
# -----------------------
def _apply_fills(price: float, side: str) -> float:
    if side == 'buy':
        return price * (1 + SLIPPAGE_PCT)
    else:
        return price * (1 - SLIPPAGE_PCT)

def _fee(notional: float) -> float:
    return abs(notional) * FEE_PCT


# -----------------------
# Backtest loop
# -----------------------
def run_backtest(close: pd.DataFrame) -> Tuple[pd.DataFrame, Portfolio]:
    pf = Portfolio(cash=0.0, positions={sym: Position() for sym in close.columns})
    equity_points: List[Tuple[pd.Timestamp, float]] = []

    if MODE == "momentum":
        sig = signals_momentum(close)
        gate = spy_gate_allow_buys(close)
    elif MODE == "rsi_reversal":
        sig = signals_rsi_reversal(close)
        gate = spy_gate_allow_buys_reversal(close)
    else:
        raise ValueError("MODE must be 'momentum' or 'rsi_reversal'")

    sell_gate = spy_gate_allow_sells(close)
    start_ts = close.index[0]
    next_contrib = start_ts.normalize()

    for ts, row in close.iterrows():
        while ts.normalize() >= next_contrib:
            pf.cash += BIWEEKLY_CONTRIB
            next_contrib += pd.Timedelta(days=14)

        if sell_gate.loc[ts]:
            for sym in close.columns:
                pos = pf.positions[sym]
                if pos.qty <= 0:
                    continue
                price = float(row[sym])
                avg_price = pos.cost / pos.qty if pos.qty > 0 else np.nan
                pnl_pct = (price - avg_price) / avg_price
                if pnl_pct >= TAKE_PROFIT_PCT:
                    fill = _apply_fills(price, 'sell')
                    notional = fill * pos.qty
                    fee = _fee(notional)
                    pf.cash += (notional - fee)
                    pf.positions[sym] = Position()
                    pf.last_buy_date.pop(sym, None)

        if gate.loc[ts]:
            notional = max(MIN_ORDER_DOLLARS, pf.cash * NOTIONAL_PCT)
            if notional >= MIN_ORDER_DOLLARS:
                current_date = ts.date()
                for sym in close.columns:
                    if ONE_BUY_PER_ASSET_PER_DAY and pf.last_buy_date.get(sym) == current_date:
                        continue
                    if not sig.loc[ts, sym]:
                        continue
                    price = float(row[sym])
                    fill = _apply_fills(price, 'buy')
                    qty = (notional - _fee(notional)) / fill
                    pf.cash -= (qty * fill + _fee(qty * fill))
                    p = pf.positions[sym]
                    p.qty += qty
                    p.cost += qty * fill
                    pf.positions[sym] = p
                    pf.last_buy_date[sym] = current_date

        mv = sum(
            pos.qty * float(row[sym])
            for sym, pos in pf.positions.items()
            if pos.qty > 0 and np.isfinite(row[sym])
        )
        equity_points.append((ts, pf.cash + mv))

    equity_df = pd.DataFrame(equity_points, columns=["timestamp", "equity"]).set_index("timestamp")
    return equity_df, pf


# -----------------------
# Reporting
# -----------------------
def summarize(close: pd.DataFrame, equity: pd.DataFrame, pf: Portfolio) -> None:
    start_dt, end_dt = close.index.min(), close.index.max()
    years = max((end_dt - start_dt).days / 365.25, 1e-9)

    total_contribs = BIWEEKLY_CONTRIB * math.ceil(((end_dt.normalize() - start_dt.normalize()).days + 1) / 14)
    end_cash = pf.cash
    end_px = close.ffill().iloc[-1]
    mv = sum(
        pos.qty * float(end_px.get(sym, np.nan))
        for sym, pos in pf.positions.items()
        if pos.qty > 0
    )
    final_value = end_cash + mv
    invested = total_contribs
    roi_pct = (final_value / invested - 1) * 100 if invested else float("nan")
    cagr = ((final_value / invested) ** (1 / years) - 1) * 100 if invested else float("nan")
    eq = equity["equity"]
    peak = eq.cummax().bfill()
    max_dd = float(((eq / peak - 1).min()) * 100)

    print("\n==================== ALPACA BOT BACKTEST ====================")
    print(f"Mode: {MODE} | Timeframe: {BAR_MINUTES}m | TZ: {TZ}")
    print(f"Window: {start_dt} → {end_dt}  ({years:.2f} years)")
    print(f"Universe: {UNIVERSE}")
    print("-------------------------------------------------------------")
    print(f"Total Contributions:             ${invested:,.2f}")
    print(f"Final Portfolio Value:           ${final_value:,.2f}")
    print(f"Ending Cash:                     ${end_cash:,.2f}")
    print(f"Net P&L:                         ${final_value - invested:,.2f}")
    print(f"ROI (Total Return):              {roi_pct:.2f}%")
    print(f"CAGR (Avg yearly return):        {cagr:.2f}%/yr")
    print(f"Max Drawdown:                    {max_dd:.2f}%")
    print("=============================================================\n")

    print("--- Positions snapshot (end) ---")
    for sym, pos in pf.positions.items():
        if pos.qty <= 0:
            continue
        last = float(end_px.get(sym, np.nan))
        avg = pos.cost / pos.qty if pos.qty > 0 else float("nan")
        mv_sym = last * pos.qty
        roi_sym = (last - avg) / avg * 100 if avg > 0 else float("nan")
        print(f"{sym:>6}: qty={pos.qty:.4f} avg=${avg:,.4f} last=${last:,.4f} MV=${mv_sym:,.2f} ROI={roi_sym:,.2f}%")

    print("\n--- Last 10 equity points ---")
    print(equity.tail(10).to_string())


# -----------------------
# Main
# -----------------------
def main():
    if not (API_KEY and API_SECRET):
        raise RuntimeError("APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set.")
    start, end = _parse_dates()
    close = _fetch_bars(UNIVERSE, start, end, BAR_MINUTES)
    if close.empty:
        raise RuntimeError("No bars returned. Check keys or date range.")
    close = close[[c for c in UNIVERSE if c in close.columns]].sort_index()
    equity, pf = run_backtest(close)
    summarize(close, equity, pf)


if __name__ == "__main__":
    main()
