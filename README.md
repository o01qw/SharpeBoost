📈 SharpeBoost
Sharpe-Gated Boosting for Financial Prediction

A tree-ensemble training algorithm that maximises portfolio Sharpe ratio instead of RMSE.

🔍 Overview

SharpeBoost is an experimental machine-learning library that modifies the boosting process to optimise validation Sharpe ratio.
Instead of accepting every tree produced by a boosting algorithm, SharpeBoost:

trains one tree at a time

computes its marginal contribution to Sharpe (after transaction costs)

accepts the tree only if Sharpe improves

rejects it otherwise

and automatically stays flat in regimes where no tree improves Sharpe

This produces models that:

trade far less

avoid bad market regimes

often achieve higher risk-adjusted performance

produce interpretably conservative trading signals

SharpeBoost has been tested on multi-asset US equity panels (AAPL, MSFT, NVDA, TSLA, etc.) from 2016–2023.

🚀 Features

Sharpe-based gating for each new tree (EMA or HAC Sharpe)

Portfolio-aware training loop

Transaction-cost-adjusted P&L

Optional global pruning of low-impact trees

Automatic “stay in cash” behaviour when predictability breaks down

Compatible with XGBoost inputs (flat matrices, DMatrix style)

Utility functions for Sharpe, P&L, turnover

📦 Installation

Clone the repository:

git clone https://github.com/<your-username>/sharpeboost.git
cd sharpeboost
pip install -e .


Requires Python ≥ 3.10.

🧠 Quick Example
from sharpeboost import SharpeBoostRegressor
from sharpeboost.trading import pnl_from_predictions
from sharpeboost.metrics import sharpe_ema

# After you construct:
# X_train, y_train   (flat: T_train * N, F)
# X_val,   y_val     (flat)
# returns_val        (panel: T_val, N)

model = SharpeBoostRegressor(
    max_rounds=200,
    tau=0.01,         # minimum Sharpe improvement required
    min_obs=60,
    tc_bps=5,
    sharpe_mode="ema"
)

model.fit(X_train, y_train, X_val, y_val, returns_val)

preds = model.predict(X_val).reshape(returns_val.shape)

# trading P&L
r_val, turnover, W = pnl_from_predictions(preds, returns_val, tc_bps=5)

print("SharpeBoost Sharpe:", sharpe_ema(r_val, span=60))
print("Turnover:", turnover.mean())

📊 Example Results (US equities, 2016–2023)

Across 9 (universe × time-split) test windows:

SharpeBoost achieved higher Sharpe than RMSE-optimised XGBoost in most splits

It produced 2×–5× lower turnover

In difficult regimes, RMSE Sharpe was strongly negative
while SharpeBoost correctly produced 0 trades

In stable/predictable regimes, SharpeBoost improved Sharpe while still trading less

Example: (AAPL–TSLA universe, 2020–2021)

Model	Sharpe (EMA, 5bps)	Turnover
SharpeBoost	2.31	0.29
RMSE XGBoost	0.87	0.64
🧩 API Reference
SharpeBoostRegressor

The main estimator class.

model = SharpeBoostRegressor(
    max_rounds=200,
    tau=0.01,
    min_obs=60,
    tc_bps=5,
    sharpe_mode="ema",   # or "hac"
    sharpe_span=60,
    prune_every=50,
    prune_tau=0.0,
    xgb_params=None
)

.fit(X_train, y_train, X_val, y_val, returns_val)

Trains a Sharpe-gated model on your training and validation sets.

.predict(X)

Predict returns or scores for new samples.

📁 Project Structure
sharpeboost/
│
├── sharpeboost/
│   ├── __init__.py
│   ├── models.py          # SharpeBoostRegressor wrapper class
│   ├── trainer.py         # Sharpe-gated training logic
│   ├── metrics.py         # sharpe_ema, sharpe_hac, min_obs_ok
│   ├── trading.py         # pnl_from_predictions
│
├── examples/
│   ├── run_on_data.py     # US equity backtest
│   └── plots_results.py
│
├── data/                  # (ignored by pip) input data
├── output/                # experiment output
├── README.md
├── LICENSE
└── pyproject.toml

📝 License

SharpeBoost is released under the Apache License 2.0.

This permissive license allows commercial use, modification, distribution, and includes explicit patent protection for users of the library.

🤝 Contributing

Pull requests, feature requests, and discussions are welcome.
Please open an issue on the GitHub repo if you encounter any bugs.

📬 Contact

Ivan Levchenko
UNSW Sydney
Feel free to open an issue or reach out with questions.
