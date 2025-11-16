# 📈 SharpeBoost
### *Sharpe-Gated Boosting for Financial Prediction*  
*A tree-ensemble training algorithm that maximises portfolio Sharpe ratio instead of RMSE.*

---

## 🔍 Overview

**SharpeBoost** is an experimental machine-learning library that modifies the boosting process to optimise **validation Sharpe ratio**.  
Instead of accepting every tree from a boosting model, SharpeBoost:

- trains **one tree at a time**
- computes **marginal Sharpe contribution** (after transaction costs)
- **accepts** the tree only if Sharpe improves
- **rejects** it otherwise  
- and **automatically stays flat** (no trades) in low-predictability regimes

This results in models that:

- trade **far less**
- avoid **bad market regimes**
- achieve **higher risk-adjusted performance**
- produce more **stable, conservative signals**

SharpeBoost has been tested on multi-asset US equity panels (AAPL, MSFT, NVDA, TSLA, etc.) from 2016–2023.

---

## 🚀 Features
- **Sharpe-based gating** for each new tree (EMA or HAC Sharpe)
- **Portfolio-aware training loop** with transaction costs
- **Optional pruning** of low-impact trees
- **Automatic “stay in cash” mode**
- **Compatible with XGBoost inputs** (flat matrices)
- **Utility functions**: Sharpe, P&L, turnover

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/sharpeboost.git
cd sharpeboost
pip install -e .
