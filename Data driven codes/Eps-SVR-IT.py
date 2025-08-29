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
# Simplex Projection Function
# ------------------------------


def simplex_projection(v):
    """
    Projects a vector v onto the simplex:
      S = { x in R^n : x_i >= 0, sum(x) = 1 }.
    Implementation based on:
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

def project_to_C1(x):
    """
    Projection onto C1: Ensures x is in the simplex,
    i.e., nonnegative and sums to one.
    """
    return simplex_projection(x)

# ------------------------------
# Projection onto D (Sparsity Constraint)
# ------------------------------


def project_to_D(y, K, u):
    """
    Projection onto D: Enforces the sparsity constraint (||y||_0 ≤ K)
    and the bounds 0 <= y <= u.
    It selects the top-K entries (by absolute value) and sets the rest to zero.
    Note: This projection also ensures nonnegativity (via np.maximum).
    """
    n = len(y)
    sorted_indices = np.argsort(np.abs(y))[::-1]  # descending order indices
    y_sparse = np.zeros(n)
    top_K_indices = sorted_indices[:K]
    # Clip the top K entries between 0 and the corresponding upper bounds.
    y_sparse[top_K_indices] = np.minimum(np.maximum(y[top_K_indices], 0), u[top_K_indices])
    return y_sparse

# --------------------------------------------
# Penalty PALM Algorithm for Sparse ε-SVR-IT
# --------------------------------------------


def penalty_palm_svr_it(R, I, C1, epsilon, K, u, rho_init=0.2, sigma=1.01, 
                        gamma_1=100, gamma_2=100, max_iter=100, tol=1e-6):
    """
    Implements the Penalty PALM algorithm for Sparse ε-SVR-IT.
    
    Arguments:
      R       -- (T, n) matrix of asset returns.
      I       -- (T,) vector of index returns.
      C1      -- float, penalty parameter for tracking deviations.
      epsilon -- float, tracking error tolerance.
      K       -- integer, maximum number of assets (cardinality constraint).
      u       -- (n,) vector of upper bounds for weights.
      rho_init-- initial penalty parameter.
      sigma   -- multiplicative factor to update the penalty parameter.
      gamma_1 -- proximal step size for x-update.
      gamma_2 -- proximal step size for y-update.
      max_iter-- maximum iterations.
      tol     -- convergence tolerance.
      
    Returns:
      x       -- (n,) optimized portfolio weight vector (which should be nonnegative, sum to one, and have at most K nonzero entries).
    """
    T, n = R.shape
    # Initialize x randomly and project onto the simplex (C1)
    x = np.random.rand(n)
    x = project_to_C1(x)
    # Initialize y as a copy of x (y will be sparsely projected)
    y = np.copy(x)
    rho = rho_init

    for k in range(max_iter):
        residuals = R @ x - I
        # Compute tracking error loss using the provided epsilon:
        loss = np.maximum(np.abs(residuals) - epsilon, 0) ** 2
        
        # Compute gradient for x update:
        # Includes gradient of 0.5*||x||^2 (which is x), the tracking error term, and the penalty term.
        gradient_x = x + C1 * (R.T @ (2 * loss * np.sign(residuals))) + rho * (x - y)
        # Update x with a proximal gradient step and then project onto C1.
        x_new = project_to_C1(x - (1 / gamma_1) * gradient_x)
        
        # Update y: using only the penalty term gradient.
        gradient_y = rho * (y - x_new)
        y_new = project_to_D(y - (1 / gamma_2) * gradient_y, K, u)
        
        # Convergence check:
        if np.linalg.norm(x_new - x) < tol and np.linalg.norm(y_new - y) < tol:
            break
        
        x, y = x_new, y_new
        rho *= sigma  # Increase penalty parameter gradually
        
        # Safeguard for numerical issues:
        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            x = np.full(n, np.nan)
            break

    # At convergence, ideally x ≈ y. Return y because it has been projected onto D (ensuring sparsity and nonnegativity).
    return y

# ------------------------------
# Cross-Validation Function
# ------------------------------


def cross_validate_palm(R, I, K, u, C1_grid, epsilon_grid, folds=4, max_iter=100, tol=1e-6):
    """
    Performs fourfold cross-validation to select the best C1 and epsilon for Sparse ε-SVR-IT.
    
    Returns:
      best_C1, best_epsilon -- the hyperparameters that minimize the average validation error.
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
                x_est = penalty_palm_svr_it(R_train, I_train, C1, epsilon, K, u,
                                            rho_init=0.2, sigma=1.01, gamma_1=100, gamma_2=100,
                                            max_iter=max_iter, tol=tol)
                
                # If algorithm diverged, assign infinite error.
                
                if np.any(np.isnan(x_est)):
                    fold_errors.append(np.inf)
                else:
                    val_error = np.mean(np.abs(R_val @ x_est - I_val))
                    fold_errors.append(val_error)
            avg_error = np.mean(fold_errors)
            if avg_error < best_score:
                best_score = avg_error
                best_C1 = C1
                best_epsilon = epsilon
    print(f"Optimal C1: {best_C1}, Optimal Epsilon: {best_epsilon}")
    return best_C1, best_epsilon

# ----------------------------------------------
# Rolling Window Implementation with Real Data
# ----------------------------------------------

def run_rolling_window(file_path):
    # Load data
    ind_ret, ast_ret, N, T = load_price_data(file_path)
    
    # Define rolling window parameters (adjust as needed)
    in_sample = 21 * 24  # In-sample period (24 months)
    out_sample = 21 * 3  # Out-of-sample period (3 months)
    roll = 21 * 3       # Rolling window step (3 months)
    win_count = (T - in_sample - out_sample) // roll + 1

    # Define hyperparameter grid for cross-validation.
    C1_grid = [0.1, 1, 10, 50]
    epsilon_grid = [0.001, 0.005, 0.01, 0.05]

    # Set sparsity constraint.
    K = 45  
    # Define u as no upper bound: vector of infinity.
    u = np.full(N, np.inf)

    # Initialize results matrix to store weights for each window.
    W1 = np.zeros((win_count, N))

    start = time.time()
    for i in range(win_count):
        print(f"\n--- Window {i + 1} of {win_count} ---")
        m1 = roll * i
        m2 = in_sample + roll * i
        
        IR = ind_ret[m1:m2]
        AR = ast_ret[m1:m2, :]
        
        # Perform cross-validation to select best C1 and epsilon for the current window.
        best_C1, best_epsilon = cross_validate_palm(AR, IR, K, u, C1_grid, epsilon_grid, folds=4)
        
        try:
            # Run the Penalty PALM algorithm using the best hyperparameters.
            final_weights = penalty_palm_svr_it(AR, IR, best_C1, best_epsilon, K, u, max_iter=100, tol=1e-6)
            W1[i, :] = final_weights
        except Exception as e:
            print(f"Skipping Window {i + 1}: {e}")
            W1[i, :] = np.nan  # Mark as failed if exception occurs.

    end = time.time()
    print(f"Total time: {end - start:.2f} seconds")
    
    # Save the results to Excel.
    df_w1 = pd.DataFrame(W1)
    output_file = "/....enter file path to save the output file/W_Eps-SVR-IT.xlsx"
    df_w1.to_excel(output_file, index=False)
    print("Results saved to:", output_file)

# Example call with your file path
file_path = "....enter file path to load price data...."
run_rolling_window(file_path)