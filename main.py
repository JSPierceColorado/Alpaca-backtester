#!/usr/bin/env python3
"""
main.py — Alpaca bots backtester (MODE=momentum | rsi_reversal | both)

- Data fetch: per-symbol pagination (limit=10000, next_page_token) with 1m fallback → resample to 15m
- Buys: NO once-per-day restriction
- Contributions: +$50 every 14 days starting at START_DATE
- Sells: take-profit at TAKE_PROFIT_PCT (default 5%), allowed only when SPY 15m MA60 > MA240
- Compounding: all sale proceeds return to cash and can be reused
- Counters printed in summary: buys_momentum, buys_reversal (dip), sells_count
- Diagnostics (DEBUG=1): fetch pages, bar counts, and signal/gate coverage

ENV (examples)
--------------
TZ=America/Denver
START_DATE=2024-10-22
END_DATE=2025-10-22
BACKTEST_YEARS=1

BAR_MINUTES=15
UNIVERSE=AAPL,MSFT,AMZN,GOOGL,META,NVDA,TSLA,JPM,V,JNJ,PG,XOM,CVX,HD,MA,UNH,KO,PEP,AVGO,LLY,SPY
ALPACA_DATA_FEED=iex      # or "sip" if your account has SIP
APCA_API_KEY_ID=...
APCA_API_SECRET_KEY=...

MODE=both                  # "momentum" | "rsi_reversal" | "both"
TAKE_PROFIT_PCT=0.05
NOTIONAL_PCT=0.05
MIN_ORDER_DOLLARS=1.0
BIWEEKLY_CONTRIB=50.0
FEE_PCT=0.0
SLIPPAGE_PCT=0.0
DEBUG=1                   # optional: prints fetch & signal diagnostics
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
BACKTEST_YEARS = int(os.getenv("BACKTEST_YEARS", "1"))

BAR_MINUTES = int(os.getenv("BAR_MINUTES", "15"))
DATA_FEED   = os.getenv("ALPACA_DATA_FEED", "iex").strip() or None
DEBUG       = os.getenv("DEBUG", "0").lower() in ("1","true","yes")

UNIVERSE = [s.strip().upper() for s in os.getenv("UNIVERSE", "SPY,QQQ,IWM,DIA").split(",") if s.strip()]
if "SPY" not in UNIVERSE:
    UNIVERSE.append("SPY")  # ensure market gate symbol is present

MODE = os.getenv("MODE", "both").strip().lower()  # "momentum" | "rsi_reversal" | "both"

TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.05"))
NOTIONAL_PCT    = float(os.getenv("NOTIONAL_PCT", "0.05"))
MIN_ORDER_DOLLARS = float(os.getenv("MIN_ORDER_DOLLARS", "1.0"))

BIWEEKLY_CONTRIB = float(os.getenv("BIWEEKLY_CONTRIB", "50.0"))
FEE_PCT = float(os.getenv("FEE_PCT", "0.0"))
SLIPPAGE_PCT = float(os.getenv("SLIPPAGE_PCT", "0.0"))

API_KEY = os.environ.get("APCA_API_KEY_ID")
API_SECRET = os.environ.get("APCA_API_SECRET_KEY")

# -----------------------
# Date helpers
# -----------------------
def _parse_dates() -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Resolve [start, end] and print like your Fidelity script."""
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
# Robust per-symbol data fetching (pagination + 1m fallback)
# -----------------------
def _fetch_symbol_bars(
    client: StockHistoricalDataClient,
    symbol: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    minutes: int,
    feed: Optional[str],
    max_pages: int = 1000,
) -> pd.Series:
    """Fetch bars for a single symbol with proper pagination."""
    tf = TimeFrame(minutes, TimeFrameUnit.Minute)
    page_token = None
    pages = 0
    rows = []

    while True:
        req_kwargs = dict(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start_utc.to_pydatetime(),
            end=end_utc.to_pydatetime(),
            limit=10000,
        )
        if feed:
            req_kwargs["feed"] = feed
        if page_token:
            req_kwargs["page_token"] = page_token

        resp = client.get_stock_bars(StockBarsRequest(**req_kwargs))

        got = 0
        sb = resp.data.get(symbol, [])
        for b in sb:
            rows.append((b.timestamp, float(b.close)))
            got += 1

        page_token = getattr(resp, "next_page_token", None)
        if DEBUG:
            print(f"[DEBUG] {symbol} {minutes}m page={pages} rows={got} next={bool(page_token)}")
        pages += 1
        if not page_token or pages >= max_pages:
            break

    if not rows:
        return pd.Series(dtype=float)

    ser = pd.Series(
        data=[v for _, v in rows],
        index=pd.DatetimeIndex([t for t, _ in rows]),
        name=symbol,
        dtype=float,
    )
    ser.index = _ensure_tz(ser.index, TZ)
    ser = ser[~ser.index.duplicated(keep="last")].sort_index()
    return ser


def _fetch_symbol_bars_with_fallback(
    client: StockHistoricalDataClient,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    minutes: int,
    feed: Optional[str],
) -> pd.Series:
    """
    Try native {minutes}-minute bars first; if too sparse (< 260 bars ~ < 10 trading days),
    refetch 1-minute and resample to {minutes} minutes (last price).
    """
    start_utc = start.tz_convert("UTC")
    end_utc = end.tz_convert("UTC")

    ser = _fetch_symbol_bars(client, symbol, start_utc, end_utc, minutes, feed)
    if ser.shape[0] >= 260 or minutes == 1:
        return ser

    if DEBUG:
        print(f"[DEBUG] {symbol}: only {ser.shape[0]} bars at {minutes}m — trying 1m fallback → resample")

    ser1 = _fetch_symbol_bars(client, symbol, start_utc, end_utc, 1, feed)
    if ser1.empty:
        return ser  # nothing better

    # resample to target minutes using last close
    ser1 = ser1.sort_index()
    ser_res = ser1.resample(f"{minutes}T").last().dropna()
    ser_res.name = symbol

    if DEBUG:
        print(f"[DEBUG] {symbol}: 1m fallback produced {ser_res.shape[0]} bars at {minutes}m")

    return ser_res


def _fetch_bars(symbols: List[str], start: pd.Timestamp, end: pd.Timestamp, minutes: int) -> pd.DataFrame:
    """Fetch each symbol independently with pagination and fallback to 1m→resample."""
    if not (API_KEY and API_SECRET):
        raise RuntimeError("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY env vars")

    client = StockHistoricalDataClient(API_KEY, API_SECRET)

    series_list = []
    for sym in symbols:
        try:
            s = _fetch_symbol_bars_with_fallback(client, sym, start, end, minutes, DATA_FEED)
            if s.empty:
                if DEBUG:
                    print(f"[DEBUG] {sym}: no bars returned")
                continue
            series_list.append(s)
        except Exception as e:
            print(f"[WARN] fetch failed for {sym}: {e}", file=sys.stderr)

    if not series_list:
        return pd.DataFrame()

    df = pd.concat(series_list, axis=1)
    # Deduplicate timestamps (paranoia) and sort
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df

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
# Signals & gates
# -----------------------
def spy_gate_allow_buys(close: pd.DataFrame) -> pd.Series:
    """True where SPY MA60 > MA240 (momentum buy-gate and sell-gate)."""
    spy = close["SPY"]
    ma60 = spy.rolling(60, min_periods=60).mean()
    ma240 = spy.rolling(240, min_periods=240).mean()
    return (ma60 > ma240) & (~ma60.isna()) & (~ma240.isna())

def spy_gate_allow_sells(close: pd.DataFrame) -> pd.Series:
    """Sells allowed only when SPY MA60 > MA240."""
    return spy_gate_allow_buys(close)

def spy_gate_allow_buys_reversal(close: pd.DataFrame) -> pd.Series:
    """True where SPY MA60 < MA240 (RSI-reversal buy-gate)."""
    spy = close["SPY"]
    ma60 = spy.rolling(60, min_periods=60).mean()
    ma240 = spy.rolling(240, min_periods=240).mean()
    return (ma60 < ma240) & (~ma60.isna()) & (~ma240.isna())

def _safe_bool(df: pd.DataFrame) -> pd.DataFrame:
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
    buys_momentum: int = 0
    buys_reversal: int = 0
    sells_count: int = 0

# -----------------------
# Execution helpers
# -----------------------
def _apply_fills(price: float, side: str) -> float:
    """Slippage worsens price; fees applied on notional."""
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

    # signals & gates based on MODE
    if MODE == "momentum":
        sig_momo = _safe_bool(signals_momentum(close)); sig_rev = None
        gate_momo = spy_gate_allow_buys(close);         gate_rev = None
    elif MODE == "rsi_reversal":
        sig_momo = None;                                sig_rev = _safe_bool(signals_rsi_reversal(close))
        gate_momo = None;                               gate_rev = spy_gate_allow_buys_reversal(close)
    elif MODE == "both":
        sig_momo = _safe_bool(signals_momentum(close))
        sig_rev  = _safe_bool(signals_rsi_reversal(close))
        gate_momo = spy_gate_allow_buys(close)
        gate_rev  = spy_gate_allow_buys_reversal(close)
    else:
        raise ValueError("MODE must be 'momentum', 'rsi_reversal', or 'both'")

    sell_gate = spy_gate_allow_sells(close)

    # biweekly contributions every 14 days from start
    start_ts = close.index[0]
    next_contrib = start_ts.normalize()

    for ts, row in close.iterrows():
        # contributions
        while ts.normalize() >= next_contrib:
            pf.cash += BIWEEKLY_CONTRIB
            next_contrib += pd.Timedelta(days=14)

        # --------- Sells (take-profit), only when sell gate is open ---------
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
                    pf.cash += (notional - fee)       # proceeds to buying power
                    pf.positions[sym] = Position()    # flat
                    pf.sells_count += 1

        # --------- Buys (momentum &/or reversal) ---------
        def try_buys(sig_df: Optional[pd.DataFrame], *, label: str):
            nonlocal pf
            if sig_df is None:
                return
            gross_budget = max(MIN_ORDER_DOLLARS, pf.cash * NOTIONAL_PCT)
            if gross_budget < MIN_ORDER_DOLLARS or gross_budget > pf.cash + 1e-9:
                return

            for sym in close.columns:
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

                cash_delta = qty * fill + fee
                if cash_delta > pf.cash + 1e-9:
                    continue

                pf.cash -= cash_delta
                p = pf.positions[sym]
                p.qty  += qty
                p.cost += qty * fill
                pf.positions[sym] = p

                if label == "momentum":
                    pf.buys_momentum += 1
                elif label == "reversal":
                    pf.buys_reversal += 1

        if gate_momo is not None and gate_momo.loc[ts]:
            try_buys(sig_momo, label="momentum")
        if gate_rev is not None and gate_rev.loc[ts]:
            try_buys(sig_rev, label="reversal")

        # mark equity
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

    # Diagnostics
    if DEBUG:
        counts = {c: int(close[c].notna().sum()) for c in close.columns}
        top = sorted(counts.items(), key=lambda x: -x[1])[:10]
        print(f"[DEBUG] bar counts (non-NaN) top10: {top}")
        missing = [c for c in UNIVERSE if c not in close.columns or close[c].dropna().empty]
        if missing:
            print(f"[DEBUG] symbols with no data: {missing}")

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

    if not equity.empty:
        eq = equity["equity"].replace(0, np.nan)
        peak = eq.cummax().bfill().fillna(1.0)
        drawdown = (eq / peak - 1.0)
        max_dd = float(drawdown.min() * 100.0)
    else:
        max_dd = 0.0

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

    print(f"Buys (momentum): {pf.buys_momentum} | Buys (dip/reversal): {pf.buys_reversal} | Sells (take-profit): {pf.sells_count}")

    print("\n--- Positions snapshot (end) ---")
    for sym, pos in pf.positions.items():
        if pos.qty <= 0:
            continue
        last = float(end_px.get(sym, np.nan))
        avg = (pos.cost / pos.qty) if pos.qty > 0 else float("nan")
        mv_sym = last * pos.qty if np.isfinite(last) else 0.0
        roi_sym = ((last - avg) / avg * 100.0) if (np.isfinite(last) and avg > 0) else float("nan")
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
        raise RuntimeError("No bars returned. Check keys, feed, symbols, date range, or permissions.")

    # Keep only symbols that actually have data
    have = [c for c in UNIVERSE if c in close.columns and close[c].notna().any()]
    if "SPY" not in have:
        raise RuntimeError("SPY data missing — cannot compute market gates. Try ALPACA_DATA_FEED=iex or enable SIP.")
    close = close[have].sort_index()

    # --- quick diagnostics (no logic changes) ---
    if DEBUG:
        gate_momo = spy_gate_allow_buys(close)
        gate_rev  = spy_gate_allow_buys_reversal(close)
        sig_momo  = _safe_bool(signals_momentum(close))
        sig_rev   = _safe_bool(signals_rsi_reversal(close))

        print("[DIAG] SPY gates: uptrend bars=", int(gate_momo.sum()),
              " downtrend bars=", int(gate_rev.sum()),
              " total bars=", close.shape[0])

        momo_counts = {c: int(sig_momo[c].sum()) for c in sig_momo.columns}
        rev_counts  = {c: int(sig_rev[c].sum())  for c in sig_rev.columns}
        print("[DIAG] top momentum signal counts:",
              sorted(momo_counts.items(), key=lambda x: -x[1])[:8])
        print("[DIAG] top reversal signal counts:",
              sorted(rev_counts.items(), key=lambda x: -x[1])[:8])

        allowed_momo = int((gate_momo & sig_momo.any(axis=1)).sum())
        allowed_rev  = int((gate_rev  & sig_rev.any(axis=1)).sum())
        print(f"[DIAG] bars with at least one momentum signal while buy-gate open: {allowed_momo}")
        print(f"[DIAG] bars with at least one reversal signal while buy-gate open: {allowed_rev}")

    equity, pf = run_backtest(close)
    summarize(close, equity, pf)

if __name__ == "__main__":
    main()
