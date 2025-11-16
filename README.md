# SharpeBoost
### Sharpe-Gated Boosting for Financial Prediction

SharpeBoost is an experimental machine-learning library that modifies the boosting process to optimise validation Sharpe ratio instead of RMSE.  
Rather than accepting every tree produced by a boosting model, SharpeBoost:

- trains one tree at a time  
- computes its marginal contribution to portfolio Sharpe (after transaction costs)  
- accepts the tree only if Sharpe improves  
- rejects it otherwise  
- and stays flat (no trades) in regimes where predictability breaks down  

This yields models that trade less, avoid bad regimes, and often achieve higher risk-adjusted performance.

## Features

- Sharpe-based gating for each new tree (EMA or HAC)
- Portfolio-aware training loop with transaction costs
- Optional global pruning of low-impact trees
- Automatic stay-in-cash behavior in poor regimes
- Compatible with XGBoost-style input matrices
- Utility functions for Sharpe, P&L, turnover

## Installation

```bash
git clone https://github.com/o01qw/sharpeboost.git
cd sharpeboost
pip install -e .
```

Requires Python ≥ 3.10.

## Quick Example

```python
from sharpeboost import SharpeBoostRegressor
from sharpeboost.trading import pnl_from_predictions
from sharpeboost.metrics import sharpe_ema

model = SharpeBoostRegressor(
    max_rounds=200,
    tau=0.01,
    min_obs=60,
    tc_bps=5,
    sharpe_mode="ema"
)

model.fit(
    X_train, y_train,
    X_val, y_val,
    returns_val   # shape: (T_val, N)
)

preds = model.predict(X_val).reshape(returns_val.shape)

# P&L and Sharpe
r_val, turnover, W = pnl_from_predictions(preds, returns_val, tc_bps=5)
S = sharpe_ema(r_val, span=60)

print("SharpeBoost Sharpe:", S)
print("Turnover:", turnover.mean())
```

## Example Results (US Equities, 2016–2023)

Across 9 universe × time-split contexts:

- SharpeBoost achieved higher Sharpe than RMSE-XGBoost in most splits  
- Turnover reduced by 2×–5×  
- RMSE-XGBoost produced negative Sharpe in difficult regimes  
- SharpeBoost correctly produced 0 trades in those same regimes  
- In predictable regimes, SharpeBoost increased Sharpe while trading less  

Example (AAPL–TSLA universe, 2020–2021):

| Model | Sharpe (EMA, 5bps) | Turnover |
|-------|---------------------|----------|
| SharpeBoost | **2.31** | **0.29** |
| RMSE XGBoost | 0.87 | 0.64 |

## API Reference

### `SharpeBoostRegressor`

```python
model = SharpeBoostRegressor(
    max_rounds=200,
    tau=0.01,
    min_obs=60,
    tc_bps=5,
    sharpe_mode="ema",
    sharpe_span=60,
    prune_every=50,
    prune_tau=0.0,
    xgb_params=None
)
```

#### `.fit(X_train, y_train, X_val, y_val, returns_val)`  
Trains a Sharpe-gated model.

#### `.predict(X)`  
Predict returns or scores.

## Project Structure

```
sharpeboost/
│
├── sharpeboost/
│   ├── __init__.py
│   ├── models.py
│   ├── trainer.py
│   ├── metrics.py
│   ├── trading.py
│
├── examples/
│   ├── run_on_data.py
│   ├── plots_results.py
│
├── data/
├── output/
├── README.md
├── LICENSE
└── pyproject.toml
```

## Contributing

Pull requests, issues, and feature suggestions are welcome.

## Contact

**Ivan Levchenko** — ivan.levch@outlook.com
