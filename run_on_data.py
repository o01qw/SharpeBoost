# run_on_data.py
import os, numpy as np, pandas as pd
from trainer import SharpePruneTrainer
from trading import pnl_from_predictions
from metrics import sharpe_ema, sharpe_hac
from sklearn.ensemble import GradientBoostingRegressor

DATA_DIR   = r"C:\Users\lazys\Desktop\sharpeboost\data"
PANEL_PQ   = os.path.join(DATA_DIR, "market_panel_long.parquet")
PANEL_CSV  = os.path.join(DATA_DIR, "market_panel_long.csv")
OUTPUT_DIR = r"C:\Users\lazys\Desktop\sharpeboost\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FEATURES = ["log_return","momentum","volatility","strev",
            "drawdown","var","return_z","volatility_z","momentum_z","volume_z"]
TARGET = "return"

def load_panel():
    if os.path.exists(PANEL_PQ):
        df = pd.read_parquet(PANEL_PQ)
    elif os.path.exists(PANEL_CSV):
        df = pd.read_csv(PANEL_CSV, parse_dates=["Date"])
    else:
        raise FileNotFoundError("market_panel_long parquet/csv not found.")
    if not isinstance(df.index, pd.MultiIndex):
        df = df.set_index(["Date","Ticker"]).sort_index()
    return df

def build_arrays_from_long(df_long, features, target):
    """
    df_long: index = (Date, Ticker), columns include features + OHLCV.
    Returns:
      X: (T, N, F)   lagged features (by 1 day)
      y: (T, N)      target returns (same day)
      dates: DatetimeIndex
      tickers: list of tickers in the final aligned panel (length N)
    """
    # Start with all unique tickers present
    all_tickers = sorted(df_long.index.get_level_values("Ticker").unique().tolist())

    # Build wide mats per feature/target: Date x Ticker (columns ordered by all_tickers)
    wide = {}
    for f in features + [target]:
        if f not in df_long.columns:
            raise KeyError(f"Column '{f}' not found in panel.")
        w = (df_long[[f]]
             .reset_index()
             .pivot(index="Date", columns="Ticker", values=f)
             .reindex(columns=all_tickers)
             .sort_index())
        wide[f] = w

    # Lag features by 1 day
    for f in features:
        wide[f] = wide[f].shift(1)

    # Align on common dates across all frames
    common_idx = None
    for w in wide.values():
        common_idx = w.index if common_idx is None else common_idx.intersection(w.index)
    for k in wide:
        wide[k] = wide[k].loc[common_idx]

    # Concatenate with a feature key (MultiIndex columns: level0=feature, level1=ticker)
    stacked = pd.concat([wide[f] for f in features + [target]],
                        axis=1, keys=features + [target])

    # Optional diagnostics
    print("stacked columns levels:", stacked.columns.names)
    print("stacked shape before dropna:", stacked.shape)

    # Drop rows with any NaNs (startup windows)
    stacked = stacked.replace([np.inf, -np.inf], np.nan).dropna(how="any")

    # 1) Work on the target slice; deduplicate ticker columns (keep first)
    target_df = stacked[target]
    if target_df.columns.duplicated().any():
        dup_names = target_df.columns[target_df.columns.duplicated()].tolist()
        print("[WARN] Duplicate ticker columns in target:", dup_names)
        target_df = target_df.loc[:, ~target_df.columns.duplicated()]

    # 2) Choose tickers that have no NaNs in the target (you can relax this if needed)
    kept_tickers = [c for c in target_df.columns if target_df[c].notna().all()]
    if len(kept_tickers) == 0:
        raise ValueError("No tickers survived alignment; relax dropna or check data.")

    # 3) Reindex EVERY feature (and target) to exactly these tickers (and only once each)
    #    This guarantees each slice is shape (T, N) with the same N & order.
    cols_mi = pd.MultiIndex.from_product([features + [target], kept_tickers])
    # Drop any duplicate columns in the full stacked frame before reindexing
    stacked = stacked.loc[:, ~stacked.columns.duplicated()]
    # Now align to the exact feature×ticker grid (will raise if something is missing)
    stacked = stacked.loc[:, cols_mi]

    # --- end patch ---

    # Final shapes
    dates = stacked.index
    T = len(dates)
    N = len(kept_tickers)
    F = len(features)

    print(f"T={T}, N={N}, F={F}")
    # Sanity check: each feature slice must be (T, N)
    for f in features + [target]:
        sli = stacked[f]
        assert sli.shape == (T, N), f"{f} slice is {sli.shape}, expected {(T, N)}"

    # Build arrays
    X = np.zeros((T, N, F), dtype=np.float64)
    for j, f in enumerate(features):
        X[:, :, j] = stacked[f].to_numpy()  # shape (T, N)
    y = stacked[target].to_numpy()          # shape (T, N)

    return X, y, dates, kept_tickers


def flatten_for_xgb(X, y):
    T, N, F = X.shape
    return X.reshape(T * N, F), y.reshape(T * N)


def mean_cs_ic(preds_flat, y_flat, N, min_std=1e-12):
    """
    Cross-sectional IC averaged over time.
    Skips days where preds or returns have ~zero std or non-finite values.
    """
    T = preds_flat.size // N
    P = preds_flat.reshape(T, N)
    Y = y_flat.reshape(T, N)
    ics = []

    for t in range(T):
        x = P[t]
        y = Y[t]

        if (not np.all(np.isfinite(x))) or (not np.all(np.isfinite(y))):
            continue
        if x.std() < min_std or y.std() < min_std:
            continue

        ic = np.corrcoef(x, y)[0, 1]
        if np.isfinite(ic):
            ics.append(ic)

    return float(np.mean(ics)) if ics else np.nan

def run_rmse_baseline(X_tr_flat, y_tr_flat, X_test_flat, y_test_flat, ret_test):
    """
    Plain Gradient Boosting with MSE loss (sklearn).
    Trained on Train set, Evaluated on Test set.
    """
    # Use sklearn GradientBoostingRegressor
    model = GradientBoostingRegressor(
        loss='squared_error',
        learning_rate=0.05,
        n_estimators=200,
        subsample=0.8,
        max_depth=3,
        random_state=123
    )
    
    model.fit(X_tr_flat, y_tr_flat)
    preds_test_flat = model.predict(X_test_flat)
    
    pred_test = preds_test_flat.reshape(ret_test.shape)

    r_test, turnover, _ = pnl_from_predictions(pred_test, ret_test, tc_bps=5)
    S_ema = sharpe_ema(r_test, span=60)
    S_hac = sharpe_hac(r_test, lags=5)

    return {
        "Sharpe_rmse_ema_5bps": float(S_ema),
        "Sharpe_rmse_hac_5bps": float(S_hac),
        "Turnover_rmse": float(turnover.mean()),
    }

def run_one_split(X, y, dates, tickers,
                  train_start, train_end,
                  val_start, val_end,
                  test_start, test_end):
    """
    3-Way Split:
    1. Train: Fit trees (residuals)
    2. Val:   Accept/Reject trees (Sharpe optimization)
    3. Test:  Evaluate final model (Holdout)
    """
    # convert to Timestamps
    train_start = pd.Timestamp(train_start)
    train_end   = pd.Timestamp(train_end)
    val_start   = pd.Timestamp(val_start)
    val_end     = pd.Timestamp(val_end)
    test_start  = pd.Timestamp(test_start)
    test_end    = pd.Timestamp(test_end)

    # boolean masks
    train_mask = (dates >= train_start) & (dates <= train_end)
    val_mask   = (dates >= val_start) & (dates <= val_end)
    test_mask  = (dates >= test_start) & (dates <= test_end)

    if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("No data in one of the windows (Train/Val/Test).")

    X_tr, y_tr   = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    ret_val  = y_val.copy()
    ret_test = y_test.copy()

    # flatten
    X_tr_flat, y_tr_flat   = flatten_for_xgb(X_tr, y_tr)
    X_val_flat, y_val_flat = flatten_for_xgb(X_val, y_val)
    X_test_flat, y_test_flat = flatten_for_xgb(X_test, y_test)

    # --- Train SharpeBoost ---
    # Fits on Train, Selects on Val
    trainer = SharpePruneTrainer(
        n_estimators=200,
        sharpe_mode="ema",
        sharpe_span=60,
        tau=0.01,
        min_obs=60,
        tc_bps=5,
        prune_every=50,
        prune_tau=0.0,
    )

    trainer.fit_one_split(
        X_tr_flat, y_tr_flat,
        X_val_flat, y_val_flat,
        returns_val=ret_val,
    )
    
    n_trees = len(trainer.trees)
    print(f"SharpeBoost accepted {n_trees} trees (Optimized on Val).")
    
    # --- Evaluate on TEST (Holdout) ---
    preds_test_flat = trainer.predict(X_test_flat)
    pred_test = preds_test_flat.reshape(ret_test.shape)
    
    r_test, turnover, W = pnl_from_predictions(pred_test, ret_test, tc_bps=5)
    S_ema_5 = trainer._sharpe(r_test)

    # --- Metrics ---
    from sklearn.metrics import r2_score, mean_squared_error
    
    # Check R2 on Test
    r2_test  = r2_score(y_test_flat, preds_test_flat)
    mse_test = mean_squared_error(y_test_flat, preds_test_flat)
    
    N = X.shape[1]
    ic_test = mean_cs_ic(preds_test_flat, y_test_flat, N)

    print(f"Train: {train_start.date()}->{train_end.date()} | Val: {val_start.date()}->{val_end.date()} | Test: {test_start.date()}->{test_end.date()}")
    print(f"TEST Sharpe (5bps): {S_ema_5:.3f} | Turnover: {turnover.mean():.3f}")
    print(f"TEST R²: {r2_test:.4f} | IC: {ic_test:.4f}")

    # --- RMSE Baseline (Train on Train, Test on Test) ---
    # Note: Baseline doesn't use Val for selection, so we just train on Train.
    baseline = run_rmse_baseline(X_tr_flat, y_tr_flat, X_test_flat, y_test_flat, ret_test)
    print(f"Baseline RMSE Sharpe: {baseline['Sharpe_rmse_ema_5bps']:.3f}")

    out = {
        "test_start":   test_start.date(),
        "test_end":     test_end.date(),
        "universe_N":   N,

        # SharpeBoost metrics (Test)
        "trees_SB":            n_trees,
        "Sharpe_SB_ema_5bps":  float(S_ema_5),
        "Turnover_SB":         float(turnover.mean()),

        # baseline metrics (Test)
        "Sharpe_RMSE_ema_5bps": baseline["Sharpe_rmse_ema_5bps"],
        "Sharpe_RMSE_hac_5bps": baseline["Sharpe_rmse_hac_5bps"],
        "Turnover_RMSE":        baseline["Turnover_rmse"],

        "R2_test":   float(r2_test),
        "IC_test":   float(ic_test),
    }

    out["dSharpe_SB_minus_RMSE"] = (
        out["Sharpe_SB_ema_5bps"] - out["Sharpe_RMSE_ema_5bps"]
    )

    return out


def subset_universe(X, y, tickers, universe_tickers):
    """
    Select a subset of tickers (universe_tickers) out of the full `tickers` list.
    """
    idx = [i for i, tk in enumerate(tickers) if tk in universe_tickers]
    if not idx:
        raise ValueError("None of requested tickers found in tickers list.")

    X_sub = X[:, idx, :]
    y_sub = y[:, idx]
    tickers_sub = [tickers[i] for i in idx]
    return X_sub, y_sub, tickers_sub


if __name__ == "__main__":
    panel = load_panel()
    X, y, dates, tickers = build_arrays_from_long(panel, FEATURES, TARGET)

    print("Full ticker list:", tickers)

    # --- Define 3-Way Splits (Train, Val, Test) ---
    # Train: 3 years
    # Val:   1 year
    # Test:  1 year
    splits = [
        # Split 1
        ("2016-01-01", "2018-12-31",  # Train (3y)
         "2019-01-01", "2019-12-31",  # Val (1y)
         "2020-01-01", "2020-12-31"), # Test (1y)
         
        # Split 2
        ("2017-01-01", "2019-12-31",  # Train
         "2020-01-01", "2020-12-31",  # Val
         "2021-01-01", "2021-12-31"), # Test
         
        # Split 3
        ("2018-01-01", "2020-12-31",  # Train
         "2021-01-01", "2021-12-31",  # Val
         "2022-01-01", "2022-12-31"), # Test
    ]

    # --- define universes as subsets of tickers ---
    mega_candidates = ["AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN"]

    universes = {
        "all": tickers,
        "mega_only": [t for t in tickers if t in mega_candidates],
    }

    results = []

    for uname, u_list in universes.items():
        if not u_list:
            continue

        X_u, y_u, tickers_u = subset_universe(X, y, tickers, u_list)

        print("\n" + "#" * 72)
        print(f"UNIVERSE: {uname} (N={len(tickers_u)})")
        print("#" * 72)

        for (tr_s, tr_e, va_s, va_e, te_s, te_e) in splits:
            print("\n" + "=" * 72)
            print(f"Split: Test Year {te_s[:4]}")
            print("=" * 72)

            res = run_one_split(X_u, y_u, dates, tickers_u, tr_s, tr_e, va_s, va_e, te_s, te_e)
            res["universe"] = uname
            results.append(res)

    if results:
        print("\nSummary over universes and splits:")
        df_res = pd.DataFrame(results)
        print(df_res)
        print("\nAverage Sharpe by universe (TEST SET):")
        print(df_res.groupby("universe")[["Sharpe_SB_ema_5bps",
                                          "Sharpe_RMSE_ema_5bps",
                                          "dSharpe_SB_minus_RMSE"]].mean())
    
        csv_path = os.path.join(OUTPUT_DIR, "sharpeboost_results_3way.csv")
        df_res.to_csv(csv_path, index=False)
        print(f"\nSaved results to: {csv_path}")
