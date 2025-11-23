'''Train a custom Gradient Boosting model (SharpeBoost) one tree at a time.
Accepts a tree only if it improves validation Sharpe after costs.
Optionally prunes previously accepted trees that stop helping.
'''
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from metrics import sharpe_ema, sharpe_hac, min_obs_ok
from trading import pnl_from_predictions

class SharpePruneTrainer:
    """
    Custom Gradient Boosting Trainer.
    Trains Decision Trees on residuals, but accepts them only if they improve
    validation Sharpe Ratio (net of trading costs).
    """
    def __init__(self,
                 learning_rate=0.05,
                 max_depth=3,
                 min_samples_leaf=20,
                 max_features=0.8,       # float: % of features to consider per split
                 subsample=0.8,          # float: % of samples to train each tree
                 n_estimators=500,       # max attempts
                 sharpe_mode="ema",      # "ema" or "hac"
                 sharpe_span=60,
                 sharpe_lags=5,
                 tau=0.01,               # min Sharpe improvement to accept a tree
                 min_obs=60,
                 tc_bps=5,
                 prune_every=25,         # global prune frequency (rounds); None to disable
                 prune_tau=0.0,          # drop tree if ΔSharpe < prune_tau
                 random_state=42):
        
        self.learning_rate = learning_rate
        self.tree_params = {
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "random_state": None # set per iteration
        }
        self.subsample = subsample
        self.n_estimators = n_estimators
        
        self.sharpe_mode = sharpe_mode
        self.sharpe_span = sharpe_span
        self.sharpe_lags = sharpe_lags
        self.tau = float(tau)
        self.min_obs = int(min_obs)
        self.tc_bps = float(tc_bps)
        self.prune_every = prune_every
        self.prune_tau = float(prune_tau)
        
        self.rng = np.random.default_rng(random_state)
        self.trees = []           # List of (DecisionTreeRegressor, seed)
        self.accepted_preds_val = [] # List of (T, N) arrays, one per tree * learning_rate

    def _sharpe(self, r):
        if self.sharpe_mode == "hac":
            return sharpe_hac(r, lags=self.sharpe_lags)
        return sharpe_ema(r, span=self.sharpe_span)

    def fit_one_split(self, X_train, y_train, X_val, y_val, returns_val):
        """
        X_train, y_train: Training data (flattened or not, but y_train must match X_train rows)
        X_val: Validation features
        y_val: Validation targets (for reference, though we optimize on returns_val)
        returns_val: (T, N) realized asset returns for validation P&L calc
        """
        # Ensure inputs are correct format
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        
        T_val, N_assets = returns_val.shape
        
        # Current ensemble predictions (start at 0)
        curr_pred_train = np.zeros_like(y_train)
        curr_pred_val = np.zeros((T_val, N_assets)) # Assuming X_val is ordered/shaped to produce this
        
        # If X_val is flattened (T*N, F), we need to reshape predictions
        # We assume X_val corresponds to returns_val flattened row-major or similar.
        # For safety in this custom loop, let's assume X_val produces (T*N) and we reshape.
        
        # Initial Sharpe
        # With 0 predictions, weights are 0, returns are 0, Sharpe is 0 (or undefined)
        curr_sharpe = 0.0
        
        print(f"Starting training. Max rounds: {self.n_estimators}, Tau: {self.tau}")

        for round_idx in range(1, self.n_estimators + 1):
            # 1. Compute Residuals
            residuals = y_train - curr_pred_train
            
            # 2. Subsample
            n_samples = X_train.shape[0]
            if self.subsample < 1.0:
                subset_size = int(n_samples * self.subsample)
                indices = self.rng.choice(n_samples, size=subset_size, replace=False)
                X_sub = X_train[indices]
                r_sub = residuals[indices]
            else:
                X_sub = X_train
                r_sub = residuals
            
            # 3. Train Candidate Tree
            seed = self.rng.integers(0, 100000)
            self.tree_params["random_state"] = seed
            tree = DecisionTreeRegressor(**self.tree_params)
            tree.fit(X_sub, r_sub)
            
            # 4. Predict on Validation
            # Raw tree prediction
            pred_val_raw = tree.predict(X_val) # (T*N, )
            pred_val_scaled = pred_val_raw * self.learning_rate
            
            # Reshape to (T, N)
            pred_val_scaled_2d = pred_val_scaled.reshape(T_val, N_assets)
            
            # 5. Check Sharpe Improvement
            cand_pred_val = curr_pred_val + pred_val_scaled_2d
            
            # Calculate P&L
            r_cand, _, _ = pnl_from_predictions(cand_pred_val, returns_val, tc_bps=self.tc_bps)
            
            if not min_obs_ok(len(r_cand), self.min_obs):
                # Warmup phase: accept blindly if not enough data? 
                # Or just skip Sharpe check. Let's assume we have data.
                # If not enough data, we can't trust Sharpe. 
                # For safety, let's just accept if we can't calc Sharpe? 
                # Or return 0. Let's stick to the logic: if we can't measure, we don't improve.
                # But usually min_obs is small (60).
                sharpe_cand = 0.0
            else:
                sharpe_cand = self._sharpe(r_cand)
            
            d_sharpe = sharpe_cand - curr_sharpe
            
            if d_sharpe >= self.tau:
                # ACCEPT
                self.trees.append(tree)
                self.accepted_preds_val.append(pred_val_scaled_2d)
                
                # Update current state
                curr_pred_val = cand_pred_val
                curr_sharpe = sharpe_cand
                
                # Update train predictions (needed for next residual)
                pred_train_raw = tree.predict(X_train)
                curr_pred_train += pred_train_raw * self.learning_rate
                
                print(f"Round {round_idx}: ACCEPTED. dS={d_sharpe:.4f}, S={curr_sharpe:.4f}, Trees={len(self.trees)}")
            else:
                # REJECT
                # print(f"Round {round_idx}: Rejected. dS={d_sharpe:.4f}")
                pass
            
            # 6. Periodic Prune
            if self.prune_every and (round_idx % self.prune_every == 0) and len(self.trees) > 0:
                curr_pred_val, curr_sharpe = self._global_prune(returns_val, curr_pred_val)
                # Note: We do NOT update curr_pred_train after pruning here for efficiency.
                # This means residuals might be slightly "off" relative to the actual ensemble,
                # but in boosting this is often acceptable (or we'd need to re-predict train for all kept trees).
                # For exactness, let's re-predict train.
                curr_pred_train = self.predict(X_train).ravel() # predict returns (N, 1) or similar, ensure shape matches y_train

        return self

    def _global_prune(self, returns_val, curr_pred_val):
        """
        Iteratively remove trees that contribute < prune_tau to Sharpe.
        Returns updated (curr_pred_val, curr_sharpe).
        """
        if not self.trees:
            return curr_pred_val, 0.0
            
        # Calculate baseline Sharpe
        r_all, _, _ = pnl_from_predictions(curr_pred_val, returns_val, tc_bps=self.tc_bps)
        S_all = self._sharpe(r_all)
        
        to_drop_indices = []
        
        # Check each tree's contribution
        # We can do this efficiently by subtracting the cached prediction
        for i, pred_chunk in enumerate(self.accepted_preds_val):
            # What if we remove tree i?
            pred_minus = curr_pred_val - pred_chunk
            r_minus, _, _ = pnl_from_predictions(pred_minus, returns_val, tc_bps=self.tc_bps)
            S_minus = self._sharpe(r_minus)
            
            contribution = S_all - S_minus
            
            if contribution < self.prune_tau:
                to_drop_indices.append(i)
        
        if to_drop_indices:
            print(f"Pruning {len(to_drop_indices)} trees...")
            # Remove from lists (in reverse order to keep indices valid? No, build new lists)
            new_trees = []
            new_preds = []
            
            drop_set = set(to_drop_indices)
            for i in range(len(self.trees)):
                if i not in drop_set:
                    new_trees.append(self.trees[i])
                    new_preds.append(self.accepted_preds_val[i])
            
            self.trees = new_trees
            self.accepted_preds_val = new_preds
            
            # Re-calculate current ensemble prediction
            if self.accepted_preds_val:
                curr_pred_val = np.sum(self.accepted_preds_val, axis=0)
            else:
                curr_pred_val = np.zeros_like(curr_pred_val)
                
            # Re-calculate Sharpe
            r_new, _, _ = pnl_from_predictions(curr_pred_val, returns_val, tc_bps=self.tc_bps)
            S_new = self._sharpe(r_new)
            print(f"Pruned. New Sharpe: {S_new:.4f} (was {S_all:.4f})")
            return curr_pred_val, S_new
            
        return curr_pred_val, S_all

    def predict(self, X):
        """
        Predict using the ensemble of accepted trees.
        """
        if not self.trees:
            return np.zeros(len(X))
            
        # Sum predictions of all trees
        # This could be slow for many trees; optimization possible (e.g. sklearn's predict uses optimized C)
        # Here we loop in Python.
        preds = np.zeros(len(X))
        for tree in self.trees:
            preds += tree.predict(X) * self.learning_rate
            
        return preds

# ----------------------------
# Minimal runnable example
# ----------------------------
if __name__ == "__main__":
    import pandas as pd

    rng = np.random.default_rng(123)
    T = 600   # time points
    N = 5     # assets
    F = 12    # features per asset

    # Build panel features and targets
    X = rng.normal(size=(T, N, F))
    beta = rng.normal(size=(F,))
    noise = 0.05 * rng.normal(size=(T, N)) # Increased noise
    y = (X @ beta) / F + noise 
    ret = y.copy()

    # Flatten to matrix form
    X_flat = X.reshape(T * N, F)
    y_flat = y.reshape(T * N)

    # Split
    cut = 400
    X_tr, y_tr = X_flat[:cut*N], y_flat[:cut*N]
    X_val, y_val = X_flat[cut*N:], y_flat[cut*N:]
    ret_val = ret[cut:]  # (T_val, N)

    trainer = SharpePruneTrainer(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        tau=0.001,             # Lower tau to encourage acceptance in random data
        min_obs=60,
        tc_bps=5,
        prune_every=20,
        prune_tau=-0.01        # Allow slight drop? Or 0.0.
    )

    trainer.fit_one_split(X_tr, y_tr, X_val, y_val, returns_val=ret_val)

    # Final Eval
    pred_val = trainer.predict(X_val).reshape(ret_val.shape)
    r_val, turnover, W = pnl_from_predictions(pred_val, ret_val, tc_bps=5)
    S_final = trainer._sharpe(r_val)
    print(f"Final Validation Sharpe: {S_final:.3f} | Trees: {len(trainer.trees)}")
