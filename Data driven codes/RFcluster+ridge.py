#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vrindadhingra
"""

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# Load Data
file_path = "(.......mention your file path here...."
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


#RF Cluster Function
def rfcluster(X, y, num_trees=100, max_features=None, top_n_assets=50,
              train_size=0.9, random_state=42):
    """
    Perform Random Forest feature ranking (classification-based) on asset returns 
    to identify the most important assets relative to index movements.

    Parameters
    ----------
    X : numpy.ndarray of shape (T, N)
        Asset returns matrix with T time points and N assets.
    y : numpy.ndarray of shape (T,)
        Index returns vector. This will be converted into binary labels 
        ("Up" = 1 if return > 0, "Down" = 0 otherwise).
    num_trees : int, default=100
        Number of trees in the Random Forest.
    max_features : int or float, default=None
        Number of features considered at each split. If None, defaults to sqrt(N).
    top_n_assets : int, default=50
        Number of top-ranked assets to return.
    train_size : float, default=0.9
        Proportion of the windowed data used for training. For example, with train_size=0.9,
        90% of the observations are used for fitting the Random Forest and the remaining
        10% are used as a validation slice for computing permutation importance.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    top_assets_indices : list of int
        Indices of the top-ranked assets based on permutation importance.
    feature_importance_df : pandas.DataFrame
        DataFrame containing ranked feature importance scores for all assets.
    rf_model : sklearn.ensemble.RandomForestClassifier
        The trained Random Forest model.
    """

    
    # Convert y into binary labels: "Up" (1) if return > 0, "Down" (0) otherwise
    y_binary = (y > 0).astype(int)

    #Time-based train-test split
    split_point = int(train_size * len(X))
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y_binary[:split_point], y_binary[split_point:]

    #print(f"Training size: {X_train.shape}, Testing size: {X_test.shape}")

    # Set default max_features if not provided
    if max_features is None:
        max_features = int(np.sqrt(X.shape[1]))  # Default is sqrt(N)

    # Train Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=num_trees, max_features=max_features, random_state=random_state, oob_score=True)
    rf.fit(X_train, y_train)

    # Compute Test Accuracy
    y_pred = rf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Test Accuracy: {test_accuracy:.4f}")

    # Compute Feature Importance using Mean Decrease Accuracy (MDA)
    
    baseline_accuracy = accuracy_score(y_test, rf.predict(X_test))  # Original accuracy

    mda_scores = []
    for i in range(X.shape[1]):  # Loop through each feature (asset)
        X_permuted = X_test.copy()
        np.random.shuffle(X_permuted[:, i])  # Shuffle feature i

        # Compute accuracy after shuffling
        permuted_accuracy = accuracy_score(y_test, rf.predict(X_permuted))

        # Compute MDA score
        mda_score = baseline_accuracy - permuted_accuracy
        mda_scores.append(mda_score)

    # Create DataFrame to store feature importance scores
    feature_importance_df = pd.DataFrame({
        'Asset': np.arange(X.shape[1]),  # Asset indices
        'MDA_Importance': mda_scores
    }).sort_values(by='MDA_Importance', ascending=False)

    # Select top-ranked assets based on MDA
    top_assets_indices = feature_importance_df['Asset'].iloc[:top_n_assets].tolist()
    
    return top_assets_indices, feature_importance_df, rf


#Ridge Regression function to calculate portfolio weights
def ridge_regression(X, y, num_trees=100, top_n_assets=50, alpha_grid=None, cv_folds=5, random_state=42):
    """
    Calls rfcluster to get top-ranked assets, then fits a Ridge Regression model to compute portfolio weights.
    
    Parameters:
    - X: numpy.ndarray, shape (T, N)
        Asset returns matrix (T time points × N assets).
    - y: numpy.ndarray, shape (T,)
        Index returns vector.
    - num_trees: int, default=100
        Number of trees in the Random Forest used in rfcluster.
    - top_n_assets: int, default=50
        Number of top-ranked assets to select.
    - alpha_grid: list or None, default=None
        List of alpha values for Ridge Regression tuning. If None, defaults to log-spaced values.
    - cv_folds: int, default=5
        Number of cross-validation folds for Ridge Regression.
    - random_state: int, default=42
        Random seed for reproducibility.

    Returns:
    - selected_assets: list
        Indices of the top-ranked assets.
    - ridge_model: trained Ridge model
        The fitted Ridge Regression model.
    - portfolio_weights: numpy.ndarray
        Optimal portfolio weights for the selected assets.
    """

    # Step 1: Call rfcluster to get the top-ranked assets
    selected_assets, importance_df, rf_model = rfcluster(
        X, y, num_trees=num_trees, top_n_assets=top_n_assets, random_state=random_state
    )

    # Step 2: Subset X using selected assets
    X_selected = X[:, selected_assets]  # Keep only the top-ranked assets

    # Step 3: Define Alpha Grid for Ridge Regression (Regularization Strength)**
    if alpha_grid is None:
        alpha_grid = np.logspace(-4, 2, 50)  # Range from 10^-4 to 10^2

    # Step 4: Ridge Regression with Cross-Validation
    ridge = Ridge()
    ridge_cv = GridSearchCV(ridge, param_grid={'alpha': alpha_grid}, cv=cv_folds, scoring='neg_mean_squared_error')
    ridge_cv.fit(X_selected, y)

    # Step 5: Get optimal weights for selected assets
    best_ridge = ridge_cv.best_estimator_
    selected_weights = best_ridge.coef_

    # Normalize the selected weights
    selected_weights = np.abs(selected_weights)
    selected_weights /= np.sum(selected_weights)

    # Expand to full N assets:
    full_portfolio_weights = np.zeros(X.shape[1])  # Initialize all weights as zero
    full_portfolio_weights[selected_assets] = selected_weights  # Assign weights only to selected assets

    return full_portfolio_weights


# Rolling Window Setup
in_sample = 21 * 24  # In-sample period (24 months)
out_sample = 21 * 3  # Out-of-sample period (3 months)
roll = 21 * 3  # Rolling window size (3 months)
win_count = (T - in_sample - out_sample) // roll + 1

# Initialize results matrix
W = np.zeros((win_count, N))

start = time.time()

# Performing RF clustering +ridge regression for each rolling window
for i in range(win_count):
    print(f"Window {i + 1} of {win_count}")
    m1 = roll * i
    m2 = in_sample + roll * i

    IR = ind_ret[m1:m2]
    AR = ast_ret[m1:m2, :]

    try:
        # Perform cointegration and get optimal portfolio weights
        W[i, :]=ridge_regression(AR, IR, num_trees=100, top_n_assets=45, alpha_grid=None, cv_folds=5, random_state=42)
    except ValueError as e:
        print(f"Skipping Window {i + 1}: {e}")
        W[i, :] = np.nan  # Assign NaN to indicate this window was skipped

end = time.time()
total_time = end - start
print(f"Total time: {total_time} seconds")


# Saving the tracking portfolios obtained for each window into an Excel file
df_w = pd.DataFrame(W)
output_file = "(....mention path name here with the file name to want to save...)"
df_w.to_excel(output_file, index=False)