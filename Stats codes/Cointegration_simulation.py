#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vrindadhingra
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy.optimize import nnls
import time

# Load Data
file_path = ".......enter file path to load price data...."
price_data = pd.read_excel(file_path)

# Convert to matrix and remove the dates column
price_matrix = price_data.iloc[:, 1:].values  # Exclude the first column (Date)

# Calculate simple returns
price_row = price_matrix.shape[0] - 1
price_col = price_matrix.shape[1]
simple_returns = (price_matrix[1:, :] - price_matrix[:-1, :]) / price_matrix[:-1, :]

# Define index and asset returns
ind_ret = simple_returns[:, 0]
ast_ret = simple_returns[:, 1:]

# Total number of assets and scenarios
N = ast_ret.shape[1]  # Number of assets
T = ast_ret.shape[0]  # Number of scenarios

# Function: Cointegration with Simulations
def coint_with_simulations(ast_ret, ind_ret, num_assets, num_iterations, seed=42):
    """
    Perform simulations for cointegration-based portfolio construction.
    
    Parameters:
    - ast_ret: Matrix of asset returns.
    - ind_ret: Vector of index returns.
    - num_iterations: Number of simulations.
    - num_assets: Number of assets to select randomly in each simulation.
    
    Returns:
    - full_portfolio_weights: Optimal portfolio weights across all assets.
    """
    
    min_ssr = float('inf')
    optimal_weights = None

    best_selected_assets=None
    
    for _ in range(num_iterations):
        # Randomly select K assets
        selected_assets = np.random.choice(ast_ret.shape[1], num_assets, replace=False)
        selected_asset_data = ast_ret[:, selected_assets]

        # Perform Non-Negative Least Squares Regression
        nnls_result = nnls(selected_asset_data, ind_ret)
        residuals = ind_ret - selected_asset_data @ nnls_result[0]

        # Perform ADF test on residuals
        adf_result = adfuller(residuals, maxlag=1, autolag='AIC')
        p_value = adf_result[1]
        
        # If residuals are stationary, check for least SSR
        if p_value < 0.05:
            ssr = np.sum(residuals**2)
            if ssr < min_ssr:
                min_ssr = ssr
                optimal_weights = nnls_result[0] / np.sum(nnls_result[0])  # Normalize weights to sum to 1
                best_selected_assets=selected_assets
                
    if optimal_weights is None:
        raise ValueError("No cointegrated portfolios found after simulations.")

    # Return full portfolio weights, including zeros for unselected assets
    full_portfolio_weights = np.zeros(ast_ret.shape[1])
    full_portfolio_weights[best_selected_assets] = optimal_weights
    return full_portfolio_weights

# Rolling Window Setup
in_sample = 21 * 24  # In-sample period (24 months)
out_sample = 21 * 3  # Out-of-sample period (3 months)
roll = 21 * 3  # Rolling window size (3 months)
win_count = (T - in_sample - out_sample) // roll + 1

# Initialize results matrix
W = np.zeros((win_count, N))

start = time.time()

# Perform Cointegration Tests for Each Rolling Window
for i in range(win_count):
    print(f"Window {i + 1} of {win_count}")
    m1 = roll * i
    m2 = in_sample + roll * i

    IR = ind_ret[m1:m2]
    AR = ast_ret[m1:m2, :]

    try:
        # Perform cointegration and get optimal portfolio weights
        W[i, :] = coint_with_simulations(AR, IR, num_assets=45, num_iterations=10000)
    except ValueError as e:
        print(f"Skipping Window {i + 1}: {e}")
        W[i, :] = np.nan  # Assign NaN to indicate this window was skipped

end = time.time()
total_time = end - start
print(f"Total time: {total_time} seconds")


# Save the results to Excel
df_w = pd.DataFrame(W)
output_file = "....enter file path to save output file..../W_Coint_sim.xlsx"
df_w.to_excel(output_file, index=False)