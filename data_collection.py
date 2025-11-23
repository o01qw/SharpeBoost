"""
Download OHLCV via yfinance, build features, and save ONE LONG CSV:
rows = (Date, Ticker), columns = fields/features (Open, Close, return, momentum, …)
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf

# ========= CONFIG =========
tickers    = ["AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "GOOGL", "SPY",
    "QQQ", "NFLX", "AVGO", "ADBE", "CRM", "AMD", "COST", "PEP"]
start_date = "2015-01-01"
end_date   = "2025-01-01"
out_dir    = r"C:\Users\lazys\Desktop\sharpe_boost\data"
out_file   = "market_panel_long.csv"   # one file, tickers in rows
SAVE_PARQUET = True                    # also save a parquet (smaller, faster)
# Feature windows
MOM_LB      = 20
VOL_LB      = 20
DD_LB       = 252
VaR_LB      = 252
VaR_Q       = 0.05
Z_SPAN      = 252
MIN_PERIODS = 10
# =========================

os.makedirs(out_dir, exist_ok=True)

# ---------- helpers ----------
def _zscore(s: pd.Series, window=Z_SPAN, minp=MIN_PERIODS):
    m  = s.rolling(window, min_periods=minp).mean()
    sd = s.rolling(window, min_periods=minp).std(ddof=0).replace(0, np.nan)
    return (s - m) / sd

def _drawdown(close: pd.Series, window=DD_LB, minp=MIN_PERIODS):
    peak = close.rolling(window, min_periods=minp).max()
    return (close / peak) - 1.0

def _hist_var(returns: pd.Series, window=VaR_LB, q=VaR_Q, minp=MIN_PERIODS):
    return returns.rolling(window, min_periods=minp).quantile(q)

def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input columns: Open, High, Low, Close, Adj Close, Volume (yfinance style).
    Output adds: return, log_return, momentum, volatility, strev, drawdown, var,
                 return_z, volatility_z, momentum_z, volume_z.
    """
    out = df.copy()

    close   = df["Close"].astype(float)
    volume  = df["Volume"].astype(float)
    ret     = close.pct_change()
    log_ret = np.log(close).diff()
    momentum   = close.pct_change(MOM_LB)
    volatility = ret.rolling(VOL_LB, min_periods=MIN_PERIODS).std(ddof=0)
    strev      = -ret.shift(1)
    drawdown   = _drawdown(close, window=DD_LB, minp=MIN_PERIODS)
    var        = _hist_var(ret, window=VaR_LB, q=VaR_Q, minp=MIN_PERIODS)

    out["return"]       = ret
    out["log_return"]   = log_ret
    out["momentum"]     = momentum
    out["volatility"]   = volatility
    out["strev"]        = strev
    out["drawdown"]     = drawdown
    out["var"]          = var
    out["return_z"]     = _zscore(ret)
    out["volatility_z"] = _zscore(volatility)
    out["momentum_z"]   = _zscore(momentum)
    out["volume_z"]     = _zscore(volume.replace(0, np.nan).ffill())
    return out

# ---------- 1) download ----------
print("Downloading from yfinance…")
px = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)  # MultiIndex: (Field, Ticker)
if px.empty:
    raise RuntimeError("yfinance returned no data. Check tickers/dates.")
# Reorder to (Ticker, Field) for easier per-ticker ops
px = px.swaplevel(0, 1, axis=1).sort_index(axis=1)

# ---------- 2) compute features per ticker, combine ----------
frames = []
for tk in tickers:
    if tk not in px.columns.levels[0]:
        continue
    df_t = px[tk].copy().dropna(how="all")                     # OHLCV for this ticker
    feats = _compute_features(df_t)                            # add features
    cols_needed = [
        "return", "log_return",
        "momentum", "volatility", "strev",
        "drawdown", "var",
        "return_z", "volatility_z", "momentum_z", "volume_z",
    ]
    feats = feats.dropna(subset=cols_needed)
    feats["Ticker"] = tk
    frames.append(feats.reset_index().rename(columns={"index": "Date"}))  # ensure Date column

if not frames:
    raise RuntimeError("No per-ticker frames built.")

# Concatenate all tickers vertically -> rows = Date,Ticker
panel_long = pd.concat(frames, ignore_index=True)
panel_long.rename(columns={"Date": "Date"}, inplace=True)
panel_long = panel_long.sort_values(["Date", "Ticker"]).set_index(["Date", "Ticker"])

# Optional clean-up
panel_long = panel_long.replace([np.inf, -np.inf], np.nan)

# ---------- 3) save ONE CSV (tickers in rows) ----------
csv_path = os.path.join(out_dir, out_file)
panel_long.to_csv(csv_path, float_format="%.10f")
print(f"✅ Saved long panel CSV: {csv_path}")
print(f"   rows={len(panel_long):,}, cols={panel_long.shape[1]:,} (features incl. OHLCV)")

if SAVE_PARQUET:
    pq_path = os.path.join(out_dir, "market_panel_long.parquet")
    panel_long.to_parquet(pq_path)
    print(f"   Also saved Parquet: {pq_path}")
