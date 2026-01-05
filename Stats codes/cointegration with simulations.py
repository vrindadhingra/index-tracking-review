#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vrindadhingra
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sant'Anna et al. (2017) cointegration using simulations

- Uses Engle–Granger two-step method:
  1) OLS regression of log(index) on log(asset prices)
  2) ADF test on residuals
- Randomly samples subsets of K assets and keeps the cointegrated ones.
- Among cointegrated subsets, selects the one with minimum SSR.
"""

import time
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# ==========================================================
# STEP 1: Load and preprocess price data
# ==========================================================


file_path = ".....enter file path to load asset price data...."
price_data = pd.read_excel(file_path)

# Drop Date column, keep index + asset prices as a NumPy array
price_matrix = price_data.iloc[:, 1:].values.astype(float)

# Take natural log of prices (as in Sant'Anna, Eq. (2))
log_price_matrix = np.log(price_matrix)

# Index and asset log-prices
ind_log = log_price_matrix[:, 0]     # log(It)
ast_log = log_price_matrix[:, 1:]    # log(p_{i,t})

# Dimensions
T, N = ast_log.shape   # T = time points, N = number of assets


# ============================================================
# STEP 2: Cointegration via simulations function definition 
# ============================================================


def coint_with_simulations(ast_log_window,
                                    ind_log_window,
                                    num_assets,
                                    num_iterations,
                                    adf_alpha=0.05,
                                    seed=42):
    """
    Sant'Anna et al. (2017) simulation-based cointegration portfolio.

    For a given in-sample window:
    1) Repeatedly sample num_assets assets at random.
    2) Run OLS regression: log(It) = beta0 + sum beta_i log(p_i,t) + eps_t
    3) Run ADF test on residuals.
       - If residuals are stationary at level adf_alpha, keep as candidate.
    4) Among all candidates, select the subset with minimum SSR.
    5) Normalize beta_i (i >= 1) so that sum_i beta_i = 1 (as in the paper).
    6) Return a full N-length weight vector (zeros for unselected assets).

    Parameters
    ----------
    ast_log_window : array, shape (T_window, N)
        Log prices of N assets in the in-sample window.
    ind_log_window : array, shape (T_window,)
        Log prices of the index in the in-sample window.
    num_assets : int
        K – number of assets per randomly sampled subset.
    num_iterations : int
        Number of random regressions to run (simulations).
    adf_alpha : float
        Significance level for ADF test on residuals (e.g., 0.05).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    full_weights : array, shape (N,)
        Portfolio weights, with nonzero entries only for the best subset.
    """

    rng = np.random.default_rng(seed)

    T_window, N_all = ast_log_window.shape
    y = ind_log_window.astype(float)          # log(It)
    X_all = ast_log_window.astype(float)      # log(p_{i,t})

    # Track best candidate
    best_ssr = np.inf
    best_weights_subset = None
    best_indices = None

    for _ in range(num_iterations):
        # 1) Randomly select K assets
        subset_idx = rng.choice(N_all, size=num_assets, replace=False)
        X_subset = X_all[:, subset_idx]                # shape (T_window, K)

        # 2) Build design matrix for OLS: [1, log p_1, ..., log p_K]
        X_design = np.column_stack([np.ones(T_window), X_subset])

        # OLS via least squares
        # coef[0] = beta0, coef[1:] = beta_i for selected assets
        coef, residuals_vec, rank, svals = np.linalg.lstsq(
            X_design, y, rcond=None
        )
        beta0 = coef[0]
        beta_subset = coef[1:]         # shape (K,)

        # 3) Residuals
        y_hat = X_design @ coef
        eps = y - y_hat

        # ADF test on residuals: check for stationarity
        adf_stat, p_value, _, _, _, _ = adfuller(eps, autolag="AIC")

        # If residuals are non-stationary, discard this subset
        if p_value >= adf_alpha:
            continue

        # 4) Compute SSR for this candidate
        ssr = np.sum(eps ** 2)

        # Keep the candidate with smallest SSR
        if ssr < best_ssr:
            # Normalization step as in Sant'Anna et al.
            beta_sum = np.sum(beta_subset)
            if np.isclose(beta_sum, 0.0):
                # If sum is essentially zero, skip normalization and candidate
                # (degenerate case, very unlikely but numerically possible)
                continue

            weights_subset = beta_subset / beta_sum

            best_ssr = ssr
            best_weights_subset = weights_subset
            best_indices = subset_idx

    # If no candidate subset satisfied cointegration, raise an error
    if best_weights_subset is None:
        raise ValueError("No cointegrated portfolio found in this window.")

    # 5) Build full-length weight vector
    full_weights = np.zeros(N_all)
    full_weights[best_indices] = best_weights_subset

    return full_weights


# ============================================================
# STEP 3: Rolling window setup
# ============================================================


in_sample = 21 * 24   # ~24 months of daily data (2-year in-sample)
out_sample = 21 * 3   # ~3 months reserved at end (not used to fit weights)
roll = 21 * 3         # step forward by 3 months each window

win_count = (T - in_sample - out_sample) // roll + 1
print(f"Total windows: {win_count}")

# Matrix of weights: one row per window, N columns (assets)
W = np.zeros((win_count, N))


# ==================================================================
# STEP 4: Rolling window run
# ==================================================================


start_time = time.time()

K = 45                 # number of assets (cardinality)
num_sims = 10000       # number of Monte Carlo regressions per window
alpha_adf = 0.05       # ADF significance level

for w in range(win_count):
    print(f"\n=== Window {w+1} / {win_count} ===")
    m1 = w * roll
    m2 = m1 + in_sample

    # In-sample log prices for this window
    y_win = ind_log[m1:m2]         # (in_sample,)
    X_win = ast_log[m1:m2, :]      # (in_sample, N)

    try:
        weights_win = coint_with_simulations(
            X_win,
            y_win,
            num_assets=K,
            num_iterations=num_sims,
            adf_alpha=alpha_adf,
            seed=42
        )
        W[w, :] = weights_win
   
    except ValueError as e:
        print(f"  Window {w+1}: {e}")
        W[w, :] = np.nan

elapsed = time.time() - start_time
print(f"\nTotal runtime: {elapsed:.2f} seconds")


# ============================================================
# STEP 5: Save weights to Excel
# ============================================================


output_file = "....enter file path to save output file.../W_factorIT.xlsx"
pd.DataFrame(W).to_excel(output_file, index=False)
print(f"Saved weights to: {output_file}")
