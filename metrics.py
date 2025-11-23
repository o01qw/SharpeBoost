'''
How 'good' is scored. robust SHARPE calc.
'''
import numpy as np

def ema_mean_std(x, span=60, eps=1e-12):
    """EMA mean and std for stability on time-series."""
    alpha = 2.0 / (span + 1.0)
    mu = 0.0
    m2 = 0.0
    out_mu = []
    out_std = []
    for v in x:
        mu = alpha * v + (1 - alpha) * mu
        m2 = alpha * (v - mu) ** 2 + (1 - alpha) * m2
        out_mu.append(mu)
        out_std.append(np.sqrt(m2 + eps))
    return np.array(out_mu), np.array(out_std)

def sharpe_ema(returns, span=60, annualize=252):
    """EMA Sharpe: EMA(mean)/EMA(std) * sqrt(annualize)."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0: 
        return 0.0
    mu, sd = ema_mean_std(r, span=span)
    # last point represents current stabilized estimate
    S = (mu[-1] / (sd[-1] + 1e-12)) * np.sqrt(annualize)
    return float(S)

def newey_west_variance(x, lags=5):
    """HAC variance (Lo/Newey–West) of a zero-mean series."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return 0.0
    x = x - x.mean()
    T = len(x)
    gamma0 = np.dot(x, x) / T
    var = gamma0
    for k in range(1, min(lags, T-1) + 1):
        w = 1 - k / (lags + 1.0)
        cov = np.dot(x[k:], x[:-k]) / T
        var += 2 * w * cov
    return var

def sharpe_hac(returns, lags=5, annualize=252):
    """Lo's HAC Sharpe: mean / sqrt(HAC var) * sqrt(annualize)."""
    r = np.asarray(returns, dtype=float)
    if r.size < 3:
        return 0.0
    mu = r.mean()
    var = newey_west_variance(r, lags=lags)
    sd = np.sqrt(max(var, 1e-12))
    return float((mu / sd) * np.sqrt(annualize))

def min_obs_ok(n, min_obs=60):
    return n >= min_obs
