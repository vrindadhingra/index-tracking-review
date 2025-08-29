#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vrindadhingra
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


# ------------------------------
# Data Loading Function 
# ------------------------------

def load_price_data(file_path):
    """
    Loads stock price data from an Excel file and computes simple returns.
    Parameters:
      - file_path (str): Path to the Excel file containing price data.
    Returns:
      - ind_ret (numpy array): Returns of the index (benchmark).
      - ast_ret (numpy array): Returns of the constituent assets.
      - N (int): Number of assets.
      - T (int): Number of time periods.
    """
    # Load data
    price_data = pd.read_excel(file_path)
    # Convert to matrix (excluding the date column)
    price_matrix = price_data.iloc[:, 1:].values  # Exclude first column (Date)
    # Compute simple returns
    simple_returns = (price_matrix[1:, :] - price_matrix[:-1, :]) / price_matrix[:-1, :]
    # Define index and asset returns
    ind_ret = simple_returns[:, 0]       # Index returns
    ast_ret = simple_returns[:, 1:]        # Asset returns
    # Total number of assets and time periods
    N = ast_ret.shape[1]
    T = ast_ret.shape[0]
    print(f"Loaded data: {N} assets, {T} time periods")
    return ind_ret, ast_ret, N, T



# ------------------------------
# Simplex Projection Function (for x)
# ------------------------------
def simplex_projection(v):
    """
    Projects a vector v onto the simplex S = { x in R^n : x >= 0, sum(x)=1 }.
    Based on:
    Duchi, John, et al. "Efficient projections onto the l1-ball for learning in high dimensions."
    Proceedings of the 25th international conference on Machine learning. 2008.
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u > (cssv - 1) / np.arange(1, n+1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w


def project_to_C2(x, epsilon, u):
    """
    Projection for the (x, epsilon) block.
    Projects x onto the simplex (nonnegative, sum to one) and ensures epsilon >= 0.
    """
    x_proj = simplex_projection(x)
    epsilon_proj = max(epsilon, 0)
    # Optionally clip x_proj if u is finite, then re-project onto the simplex.
    x_proj = np.minimum(x_proj, u)
    x_proj = simplex_projection(x_proj)
    return x_proj, epsilon_proj

# ------------------------------
# Projection onto D (Sparsity Constraint)
# ------------------------------
def project_to_D(y, K, u):
    """
    Projection onto D: Enforces the sparsity constraint (||y||_0 ≤ K)
    and the bounds 0 <= y <= u.
    It selects the top-K elements (by absolute value) and zeros out the rest.
    """
    n = len(y)
    sorted_indices = np.argsort(np.abs(y))[::-1]
    y_sparse = np.zeros(n)
    top_K_indices = sorted_indices[:K]
    y_sparse[top_K_indices] = np.minimum(np.maximum(y[top_K_indices], 0), u[top_K_indices])
    return y_sparse

# ------------------------------
# Penalty PALM Algorithm for ν-SVR-IT
# ------------------------------
def penalty_palm_nu_svr_it(R, I, C1, C2, K, u, init_epsilon=0.005, 
                           rho_init=0.2, sigma=1.01, gamma=100, max_iter=100, tol=1e-6):
    """
    Implements the Penalty PALM algorithm for the ν-SVR-IT model.
    
    Objective:
      min_{x, ε}  0.5||x||^2 + C1 * Σ_t [ (R_t^T x - r_t - ε)_+^2 + (r_t - R_t^T x - ε)_+^2 ]
                 + C2 * ε
    Subject to:
      x in S = { x >= 0, sum(x)=1 }
      ε >= 0
      Cardinality: ||x||_0 ≤ K
    
    We split the variables by introducing an auxiliary variable y (which is projected onto D)
    and alternate updates.
    
    Returns:
      y       -- (n,) optimized portfolio weight vector (sparse, nonnegative, sum to one)
      ε       -- optimized tracking error tolerance.
    """
    T, n = R.shape
    # Initialize x randomly and project onto the simplex.
    x = np.random.rand(n)
    x, _ = project_to_C2(x, 0, u)
    # Initialize ε and auxiliary variable y.
    epsilon = init_epsilon
    y = np.copy(x)
    rho = rho_init

    for k in range(max_iter):
        # ---- Update (x, ε) block ----
        # Compute residuals:
        residuals1 = R @ x - I - epsilon
        residuals2 = I - R @ x - epsilon
        # Compute positive parts:
        loss1 = np.maximum(residuals1, 0)
        loss2 = np.maximum(residuals2, 0)
        
        # Compute gradients:
        grad_x = x + C1 * (R.T @ (2 * loss1 - 2 * loss2)) + rho * (x - y)
        grad_epsilon = C2 - 2 * C1 * np.sum(loss1 + loss2)
        
        # Proximal gradient step:
        x_temp = x - (1/gamma) * grad_x
        epsilon_temp = epsilon - (1/gamma) * grad_epsilon
        
        # Project (x, ε) onto C2:
        x_new, epsilon_new = project_to_C2(x_temp, epsilon_temp, u)
        
        # ---- Update y block (sparsity) ----
        grad_y = rho * (y - x_new)
        y_new = project_to_D(y - (1/gamma) * grad_y, K, u)
        
        # Convergence check:
        if np.linalg.norm(x_new - x) < tol and abs(epsilon_new - epsilon) < tol and np.linalg.norm(y_new - y) < tol:
            break
        
        x, epsilon, y = x_new, epsilon_new, y_new
        rho *= sigma
        
        if np.any(np.isnan(x)) or np.isnan(epsilon) or np.any(np.isnan(y)):
            x = np.full(n, np.nan)
            epsilon = np.nan
            break

    # Return y (which is sparse) and the optimized epsilon.
    return y, epsilon

# ------------------------------
# Cross-Validation for ν-SVR-IT
# ------------------------------


def cross_validate_nu(R, I, K, u, C1_grid, epsilon_grid, C2, folds=4, max_iter=100, tol=1e-6):
    """
    Performs cross-validation to select the best C1 and initial ε for the ν-SVR-IT model.
    C2 is assumed fixed.
    
    Returns:
      best_C1, best_epsilon: hyperparameters that minimize the average validation error.
    """
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    best_score = float("inf")
    best_C1, best_epsilon = None, None

    for C1 in C1_grid:
        for epsilon in epsilon_grid:
            fold_errors = []
            for train_idx, val_idx in kf.split(R):
                R_train, R_val = R[train_idx], R[val_idx]
                I_train, I_val = I[train_idx], I[val_idx]
                x_est, epsilon_est = penalty_palm_nu_svr_it(R_train, I_train, C1, C2, K, u,
                                                             init_epsilon=epsilon,
                                                             max_iter=max_iter, tol=tol)
                if np.any(np.isnan(x_est)) or np.isnan(epsilon_est):
                    fold_errors.append(np.inf)
                else:
                    val_error = np.mean(np.abs(R_val @ x_est - I_val))
                    fold_errors.append(val_error)
            avg_error = np.mean(fold_errors)
            if avg_error < best_score:
                best_score = avg_error
                best_C1 = C1
                best_epsilon = epsilon
    print(f"Optimal C1: {best_C1}, Optimal initial ε: {best_epsilon}")
    return best_C1, best_epsilon

# ------------------------------
# Rolling Window Implementation for ν-SVR-IT 
# ------------------------------
def run_rolling_window_nu(file_path):
    # Load data
    ind_ret, ast_ret, N, T = load_price_data(file_path)
    
    # Define rolling window parameters
    in_sample = 21 * 24  # 24 months of data
    out_sample = 21 * 3  # 3 months (for testing, not used in model fitting)
    roll = 21 * 3       # Rolling window step of 3 months
    win_count = (T - in_sample - out_sample) // roll + 1

    # Define hyperparameter grids for cross-validation
    C1_grid = [0.1, 1, 10, 50]
    epsilon_grid = [0.001, 0.005, 0.01, 0.05]
    # Set C2 (penalty on ε)
    C2 = 1  
    # Set sparsity constraint
    K = 45  
    # Define u as no upper bound
    u = np.full(N, np.inf)

    # Initialize results matrices
    W = np.zeros((win_count, N))
    Eps = np.zeros(win_count)

    start = time.time()
    for i in range(win_count):
        print(f"\n--- Window {i + 1} of {win_count} ---")
        m1 = roll * i
        m2 = in_sample + roll * i
        
        IR = ind_ret[m1:m2]
        AR = ast_ret[m1:m2, :]
        
        # Cross-validation for best C1 and initial ε in current window
        best_C1, best_epsilon = cross_validate_nu(AR, IR, K, u, C1_grid, epsilon_grid, C2, folds=4)
        
        try:
            # Run the ν-SVR-IT model with best hyperparameters
            final_weights, final_epsilon = penalty_palm_nu_svr_it(AR, IR, best_C1, C2, K, u,
                                                                  init_epsilon=best_epsilon,
                                                                  max_iter=100, tol=1e-6)
            W[i, :] = final_weights
            Eps[i] = final_epsilon
        except Exception as e:
            print(f"Skipping Window {i + 1}: {e}")
            W[i, :] = np.nan
            Eps[i] = np.nan

    end = time.time()
    print(f"Total time: {end - start:.2f} seconds")
    
    # Save results to Excel.
    df_W = pd.DataFrame(W, columns=[f"Asset_{j+1}" for j in range(N)])
    df_Eps = pd.DataFrame(Eps, columns=["Optimized_Epsilon"])
    
    output_file_W = ".....enter your file path here.../W_Nu-SVR-IT-cv.xlsx"
    output_file_Eps = ".....enter your file path here.../Eps_Nu-SVR-IT-cv.xlsx"
    
    
    df_W.to_excel(output_file_W, index=False)
    df_Eps.to_excel(output_file_Eps, index=False)
    
    print("Results saved to:")
    print(output_file_W)
    print(output_file_Eps)

# Call with your file path for ν-SVR-IT

file_path = "/....enter your file path to load price data..../"
run_rolling_window_nu(file_path)