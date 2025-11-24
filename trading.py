'''
Convert model predictions → portfolio weights, apply transaction costs, and compute P&L
'''
import numpy as np

def zscore(x, axis=0, eps=1e-9):
    x = np.asarray(x, dtype=float)
    m = x.mean(axis=axis, keepdims=True)
    s = x.std(axis=axis, keepdims=True) + eps
    return (x - m) / s

def scores_to_weights(preds, top_k=None, cap=0.2, min_score=0.0):
    """
    Convert per-asset scores at a single time to long-only weights.
    - If top_k is set, take top_k assets only; else use all positive scores.
    - Cap per-asset weight at `cap` and renormalize.
    - min_score: ignore assets with score < min_score
    """
    s = np.asarray(preds, dtype=float)
    
    # Filter by min_score first
    s = np.where(s > min_score, s, 0.0)
    
    if top_k is not None and top_k < len(s):
        # Only consider indices that passed min_score
        valid_idx = np.where(s > 0)[0]
        if len(valid_idx) > top_k:
            # Sort only the valid ones
            # Get indices of top_k values among the valid ones
            # We can just sort all 's' descending, take top k. 
            # Zeros will be at the bottom (assuming s>=0).
            idx = np.argsort(-s)[:top_k]
            mask = np.zeros_like(s, dtype=bool)
            mask[idx] = True
            s = np.where(mask, s, 0.0)
            
    if s.sum() <= 0:
        return np.zeros_like(s)
        
    w = s / s.sum()
    if cap is not None:
        w = np.minimum(w, cap)
        if w.sum() > 0:
            w = w / w.sum()
    return w

def pnl_from_predictions(pred_matrix, ret_matrix, tc_bps=5, min_score=0.0):
    """
    pred_matrix: (T, N) predictions per time t, asset n
    ret_matrix:  (T, N) realized returns aligned to t (next-period or same if already shifted)
    tc_bps: round-trip cost in basis points applied to turnover per rebalance.
    min_score: minimum prediction value to take a position.
    Returns:
      r_port: (T,) portfolio return series after costs
      turnover: (T,) turnover series
      weights: (T, N) realized weights
    """
    P = np.asarray(pred_matrix, dtype=float)
    R = np.asarray(ret_matrix, dtype=float)
    T, N = P.shape
    W = np.zeros_like(P)
    prev_w = np.zeros(N)
    turnover = np.zeros(T)
    r_port = np.zeros(T)
    for t in range(T):
        w = scores_to_weights(P[t], top_k=None, cap=0.2, min_score=min_score)
        W[t] = w
        turnover[t] = np.abs(w - prev_w).sum()
        gross = (w * R[t]).sum()
        costs = turnover[t] * (tc_bps / 10000.0)
        r_port[t] = gross - costs
        prev_w = w
    return r_port, turnover, W
