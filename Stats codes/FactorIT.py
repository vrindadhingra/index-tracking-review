#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vrindadhingra
"""

import time
import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


def load_price_data(file_path):
    """
    Loads stock price data from an Excel file and computes both log prices and simple returns.
    Adjusts log prices to match simple returns by removing the first row.

    Parameters:
    - file_path (str): Path to the Excel file containing price data.

    Returns:
    - ind_log_price (numpy array): Adjusted log prices of the index (benchmark).
    - ind_ret (numpy array): Simple returns of the index.
    - ast_log_price (numpy array): Adjusted log prices of the constituent assets.
    - ast_ret (numpy array): Simple returns of the constituent assets.
    - N (int): Number of assets.
    - T (int): Number of time periods (after adjustment).
    """
    # Load data
    price_data = pd.read_excel(file_path)

    # Convert to matrix (excluding the date column)
    price_matrix = price_data.iloc[:, 1:].values  # Exclude first column (Date)

    # Compute log prices
    log_prices = np.log(price_matrix)

    # Compute simple returns
    simple_returns = (price_matrix[1:, :] - price_matrix[:-1, :]) / price_matrix[:-1, :]

    # Remove the first row from log prices to align with returns
    log_prices_adjusted = log_prices[1:, :]

    # Define index and asset log prices
    ind_log_price = log_prices_adjusted[:, 0]  # Adjusted index log prices
    ast_log_price = log_prices_adjusted[:, 1:]  # Adjusted asset log prices

    # Define index and asset returns
    ind_ret = simple_returns[:, 0]  # Index simple returns
    ast_ret = simple_returns[:, 1:]  # Asset simple returns

    # Total number of assets and scenarios
    N = ast_ret.shape[1]  # Number of assets
    T = ast_ret.shape[0]  # Number of time periods (after adjustment)

    print(f"Loaded data: {N} assets, {T} time periods (after alignment)")
    return ind_log_price, ind_ret, ast_log_price, ast_ret, N, T


def select_assets_factorIT(log_prices, num_factors, num_assets_to_select):
    """
    Perform PCA, OLS regression, and select a fixed number of assets that best replicate the factors.

    Args:
        log_prices (np.array): Matrix of log prices (T × N).
        num_factors (int): Number of principal components (factors) to extract.
        num_assets_to_select (int): Fixed number of assets to select.

    Returns:
        selected_assets (list): Indices of selected stocks.
    """
    T, N = log_prices.shape  # Time periods (T) and number of assets (N)

    # Step 1: Perform PCA on log prices
    pca = PCA(n_components=num_factors)
    factor_matrix = pca.fit_transform(log_prices)  # (T × num_factors)

    # Step 2: Run OLS regression for each asset against the extracted factors
    asset_r2_scores = []
    
    for i in range(N):
        asset_prices = log_prices[:, i].reshape(-1, 1)  # Convert to column vector
        
        # Fit OLS regression (log_prices = factors × loadings)
        model = LinearRegression()
        model.fit(factor_matrix, asset_prices)
        
        # Compute R² value
        R2 = model.score(factor_matrix, asset_prices)
        
        # Store asset index and R² score
        asset_r2_scores.append((i, R2))
    
    # Step 3: Select the top `num_assets_to_select` assets by R² value
    asset_r2_scores.sort(key=lambda x: x[1], reverse=True)  # Sort in descending order
    selected_assets = [x[0] for x in asset_r2_scores[:num_assets_to_select]]  # Select top assets

    return selected_assets

def factorIT_portfolio(log_prices, simple_returns, index_simple_returns, num_factors, num_assets_to_select):
    """
    Compute optimal portfolio weights using selected assets (optimized on simple returns).

    Args:
        log_prices (np.array): Matrix of log prices (T × N).
        simple_returns (np.array): Matrix of simple returns (T-1 × N).
        index_simple_returns (np.array): Vector of index simple returns (T-1,).
        num_factors (int): Number of principal components (factors) to extract.
        num_assets_to_select (int): Fixed number of assets to select.

    Returns:
        weights (np.array): Portfolio weights for all assets (zeros for unselected assets).
    """
    # Step 1: Select assets using PCA and OLS on log prices
    selected_assets = select_assets_factorIT(log_prices[:-1], num_factors, num_assets_to_select)  

    # Step 2: Get simple returns for selected assets
    R_S = simple_returns[:, selected_assets]  # (T-1 × num_assets_to_select)
    R_I = index_simple_returns  # (T-1,)

    # Step 3: Define Quadratic Programming Problem
    w = cp.Variable(num_assets_to_select)  # Portfolio weights for selected assets

    # Objective function: Minimize squared tracking error ||R_S w - R_I||^2
    objective = cp.Minimize(cp.sum_squares(R_S @ w - R_I))

    # Constraints:
    constraints = [
        cp.sum(w) == 1,  # Sum of weights must be 1
        w >= 0  # No short-selling (non-negative weights)
    ]

    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Step 4: Assign weights to all assets (zero for unselected assets)
    full_weights = np.zeros(log_prices.shape[1])
    full_weights[selected_assets] = w.value  # Assign weights to selected assets

    return full_weights


file_path = ".....enter file path to load asset price data...."
ind_log_price, ind_ret, ast_log_price, ast_ret, N, T = load_price_data(file_path)


# Rolling Window Setup
in_sample = 21 * 24  # In-sample period (24 months)
out_sample = 21 * 3  # Out-of-sample period (3 months)
roll = 21 * 3  # Rolling window size (3 months)
win_count = (T - in_sample - out_sample) // roll + 1

# Initialize results matrix
W = np.zeros((win_count, N))

start = time.time()

# Perform factor-based portfolio selection and optimization for each rolling window
for i in range(win_count):
    print(f"Window {i + 1} of {win_count}")
    m1 = roll * i
    m2 = in_sample + roll * i

    IP=ind_log_price[m1:m2]
    AP=ast_log_price[m1:m2, :]
    
    IR = ind_ret[m1:m2]
    AR = ast_ret[m1:m2, :]

    try:
        # Perform cointegration and get optimal portfolio weights
        W[i, :]=factorIT_portfolio(AP, AR, IR, num_factors=5, num_assets_to_select=45)
    except ValueError as e:
        print(f"Skipping Window {i + 1}: {e}")
        W[i, :] = np.nan  # Assign NaN to indicate this window was skipped

end = time.time()
total_time = end - start
print(f"Total time: {total_time} seconds")


# Save the results to Excel
df_w = pd.DataFrame(W)
output_file = "....enter file path to save output file.../W_factorIT.xlsx"
df_w.to_excel(output_file, index=False)