#!/usr/bin/env python3
"""
alpaca_backtester.py

A single-file backtester for your Alpaca bot strategies, modeled after your Fidelity backtester,
with full buy/sell execution, cash & buying power simulation, and biweekly contributions.

- Modes:
    * MODE=momentum      → momentum-only buys (SPY uptrend gate)
    * MODE=rsi_reversal  → RSI-reversal-only buys (SPY downtrend gate)
    * MODE=both          → run BOTH strategies concurrently with a shared cash pool
- Sells: take-profit at TAKE_PROFIT_PCT (default 5%), allowed only when SPY 15m MA60 > MA240
- Contributions: adds $50 to buying power every 2 weeks starting at START_DATE
- All sale proceeds go back to buying power (compounding)
- One buy per asset per calendar day across BOTH strategies

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
MODE=both                # "momentum" | "rsi_reversal" | "both"
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
    UNIVERSE.append("SPY")  # ensure market gate symbol is present

MODE = os.getenv("MODE", "both").strip().lower()  # "momentum" | "rsi_reversal" | "both"

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
    """Resolve [start, end] like your Fidelity script."""
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
    try:
        bars = client.get_stock_bars(StockBarsRequest(**req_kwargs))
    except Exception as e:
        print(f"[WARN] fetch error: {e}", file=sys.stderr)
        raise

    # Convert to wide close-price frame
    records = []
    for sym, sb in bars.data.items():
        for b in sb:
            records.append({"timestamp": b.timestamp, "symbol": sym, "close": float(b.close)})
    if not records:
        return pd.DataFrame()
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
# Signal builders (mirror your live bots)
# -----------------------
def spy_gate_allow_buys(close: pd.DataFrame) -> pd.Series:
    """True where SPY MA60 > MA240 (momentum bot buy-gate)."""
    spy = close["SPY"]
    ma60 = spy.rolling(60, min_periods=60).mean()
    ma240 = spy.rolling(240, min_periods=240).mean()
    return (ma60 > ma240) & (~ma60.isna()) & (~ma240.isna())

def spy_gate_allow_sells(close: pd.DataFrame) -> pd.Series:
    """True where SPY MA60 > MA240 (take-profit seller gate)."""
    return spy_gate_allow_buys(close)

def spy_gate_allow_buys_reversal(close: pd.DataFrame) -> pd.Series:
    """True where SPY MA60 < MA240 (RSI reversal bot buy-gate)."""
    spy = close["SPY"]
    ma60 = spy.rolling(60, min_periods=60).mean()
    ma240 = spy.rolling(240, min_periods=240).mean()
    return (ma60 < ma240) & (~ma60.isna()) & (~ma240.isna())

def _safe_bool(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure boolean DataFrame with no NaNs."""
    return df.astype(bool).fillna(False)

def signals_momentum(close: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum rider (15m):
      1) MA60 > MA240
      2) Price > MA60
      3) RSI14 in [55,70] and rising
      4) MACD histogram > 0 and rising
    """
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
    """
    RSI reversal (15m):
      1) MA60 < MA240
      2) RSI14 < 30
    """
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
    cost: float = 0.0  # total dollars spent acquiring current qty

@dataclass
class Portfolio:
    cash: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    last_buy_date: Dict[str, Optional[pd.Timestamp.date]] = field(default_factory=dict)


# -----------------------
# Execution helpers
# -----------------------
def _apply_fills(price: float, side: str) -> float:
    """
    Returns fill price after slippage; fees are applied on notional.
    side: 'buy' or 'sell' (slippage always worsens your price).
    """
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
    # init portfolio
    pf = Portfolio(cash=0.0, positions={sym: Position() for sym in close.columns})
    equity_points: List[Tuple[pd.Timestamp, float]] = []

    # compute signals & gates according to MODE
    if MODE == "momentum":
        sig_momo = _safe_bool(signals_momentum(close))
        sig_rev  = None
        gate_momo = spy_gate_allow_buys(close)           # SPY uptrend
        gate_rev  = None
    elif MODE == "rsi_reversal":
        sig_momo = None
        sig_rev  = _safe_bool(signals_rsi_reversal(close))
        gate_momo = None
        gate_rev  = spy_gate_allow_buys_reversal(close)  # SPY downtrend
    elif MODE == "both":
        sig_momo = _safe_bool(signals_momentum(close))
        sig_rev  = _safe_bool(signals_rsi_reversal(close))
        gate_momo = spy_gate_allow_buys(close)           # uptrend gate
        gate_rev  = spy_gate_allow_buys_reversal(close)  # downtrend gate
    else:
        raise ValueError("MODE must be 'momentum', 'rsi_reversal', or 'both'")

    sell_gate = spy_gate_allow_sells(close)

    # biweekly contributions every 14 days starting at first index date
    start_ts = close.index[0]
    next_contrib = start_ts.normalize()

    for ts, row in close.iterrows():
        # Add contribution if due (can catch-up multiple periods if bars are sparse)
        while ts.normalize() >= next_contrib:
            pf.cash += BIWEEKLY_CONTRIB
            next_contrib += pd.Timedelta(days=14)

        # --------- Sells first (take-profit), only when sell gate is open ---------
        if sell_gate.loc[ts]:
            for sym in close.columns:
                pos = pf.positions[sym]
                if pos.qty <= 0:
                    continue
                price = float(row[sym])
                if not (np.isfinite(price) and price > 0):
                    continue
                avg_price = (pos.cost / pos.qty) if pos.qty > 0 else np.nan
                if not np.isfinite(avg_price) or avg_price <= 0:
                    continue
                pnl_pct = (price - avg_price) / avg_price
                if pnl_pct >= TAKE_PROFIT_PCT:
                    fill = _apply_fills(price, 'sell')
                    notional = fill * pos.qty
                    fee = _fee(notional)
                    pf.cash += (notional - fee)       # proceeds returned to buying power
                    pf.positions[sym] = Position()    # flat after TP sell
                    pf.last_buy_date.pop(sym, None)   # allow a same-day re-entry if rules allow

        # --------- Buys (momentum &/or reversal) ---------
        current_date = ts.date()

        def try_buys(sig_df: Optional[pd.DataFrame]):
            nonlocal pf
            if sig_df is None:
                return
            # sizing from *current* cash each time
            gross_budget = max(MIN_ORDER_DOLLARS, pf.cash * NOTIONAL_PCT)
            if gross_budget < MIN_ORDER_DOLLARS or gross_budget > pf.cash + 1e-9:
                return

            for sym in close.columns:
                # one-buy-per-asset-per-day across BOTH strategies
                if ONE_BUY_PER_ASSET_PER_DAY and pf.last_buy_date.get(sym) == current_date:
                    continue
                if not sig_df.loc[ts, sym]:
                    continue

                price = float(row[sym])
                if not (np.isfinite(price) and price > 0):
                    continue

                fill = _apply_fills(price, 'buy')
                fee  = _fee(gross_budget)
                qty  = (gross_budget - fee) / fill
                if qty <= 0:
                    continue

                # finalize cash & position (charge fee once on the gross budget)
                cash_delta = qty * fill + fee
                if cash_delta > pf.cash + 1e-9:
                    continue  # safety
                pf.cash -= cash_delta
                p = pf.positions[sym]
                p.qty  += qty
                p.cost += qty * fill
                pf.positions[sym] = p
                pf.last_buy_date[sym] = current_date

        # momentum buys only when uptrend gate is true
        if gate_momo is not None and gate_momo.loc[ts]:
            try_buys(sig_momo)

        # rsi-reversal buys only when downtrend gate is true
        if gate_rev is not None and gate_rev.loc[ts]:
            try_buys(sig_rev)

        # --------- Mark equity ---------
        mv = 0.0
        for sym, pos in pf.positions.items():
            if pos.qty <= 0:
                continue
            px = float(row[sym])
            if np.isfinite(px) and px > 0:
                mv += pos.qty * px
        equity_points.append((ts, pf.cash + mv))

    equity_df = pd.DataFrame(equity_points, columns=["timestamp", "equity"]).set_index("timestamp")
    return equity_df, pf


# -----------------------
# Reporting
# -----------------------
def summarize(close: pd.DataFrame, equity: pd.DataFrame, pf: Portfolio) -> None:
    start_dt, end_dt = close.index.min(), close.index.max()
    years = max((end_dt - start_dt).days / 365.25, 1e-9)

    # total contributions made based on window length (biweekly)
    total_contribs = BIWEEKLY_CONTRIB * math.ceil(((end_dt.normalize() - start_dt.normalize()).days + 1) / 14)

    end_cash = pf.cash
    end_px = close.ffill().iloc[-1]
    mv = 0.0
    for sym, pos in pf.positions.items():
        if pos.qty > 0:
            last = float(end_px.get(sym, np.nan))
            if np.isfinite(last) and last > 0:
                mv += pos.qty * last
    final_value = end_cash + mv

    invested = total_contribs
    if invested > 0:
        total_return = final_value / invested
        roi_pct = (total_return - 1.0) * 100.0
        cagr = (total_return ** (1 / years) - 1) * 100.0
    else:
        roi_pct, cagr = float("nan"), float("nan")

    # drawdown
    if not equity.empty:
        eq = equity["equity"].replace(0, np.nan)
        peak = eq.cummax().bfill().fillna(1.0)
        drawdown = (eq / peak - 1.0)
        max_dd = float(drawdown.min() * 100.0)
    else:
        max_dd = 0.0

    # Print
    print("\n==================== ALPACA BOT BACKTEST ====================")
    print(f"Mode: {MODE} | Timeframe: {BAR_MINUTES}m | TZ: {TZ}")
    print(f"Window: {start_dt} → {end_dt}  ({years:.2f} years)")
    print(f"Universe: {UNIVERSE}")
    print("-------------------------------------------------------------")
    print(f"Total Contributions (biweekly ${BIWEEKLY_CONTRIB:.2f}): ${invested:,.2f}")
    print(f"Final Portfolio Value:                          ${final_value:,.2f}")
    print(f"Ending Cash:                                    ${end_cash:,.2f}")
    print(f"Net P&L:                                        ${final_value - invested:,.2f}")
    print(f"ROI (Total Return):                             {roi_pct:.2f}%")
    print(f"CAGR (Avg yearly return):                       {cagr:.2f}%/yr")
    print(f"Max Drawdown:                                   {max_dd:.2f}%")
    print("=============================================================\n")

    # Per-asset snapshot
    print("--- Positions snapshot (end) ---")
    for sym, pos in pf.positions.items():
        if pos.qty <= 0:
            continue
        last = float(end_px.get(sym, np.nan))
        avg = (pos.cost / pos.qty) if pos.qty > 0 else float("nan")
        mv_sym = last * pos.qty if np.isfinite(last) else 0.0
        roi_sym = ((last - avg) / avg * 100.0) if (np.isfinite(last) and avg > 0) else float("nan")
        print(f"{sym:>6}: qty={pos.qty:.4f} avg=${avg:,.4f} last=${last:,.4f} MV=${mv_sym:,.2f} ROI={roi_sym:,.2f}%")

    # Equity tail
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
        raise RuntimeError("No bars returned. Check keys, feed, symbols, or date range.")

    # Keep only requested universe (columns that actually have data)
    keep = [c for c in UNIVERSE if c in close.columns]
    if not keep:
        raise RuntimeError("Universe symbols have no data in returned frame.")
    close = close[keep].sort_index()

    # Build & run
    equity, pf = run_backtest(close)
    summarize(close, equity, pf)


if __name__ == "__main__":
    main()
