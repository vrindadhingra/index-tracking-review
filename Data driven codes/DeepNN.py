#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vrindadhingra
"""

import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


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
    ind_ret = simple_returns[:, 0]  # Index returns
    ast_ret = simple_returns[:, 1:]  # Asset returns

    # Total number of assets and scenarios
    N = ast_ret.shape[1]  # Number of assets
    T = ast_ret.shape[0]  # Number of time periods

    print(f"Loaded data: {N} assets, {T} time periods")
    return ind_ret, ast_ret, N, T


def deep_autoencoder(ast_train, h):
    """
    Trains a deep autoencoder to select top h assets based on reconstruction error.

    Parameters:
    - ast_train (numpy array): Training asset returns (T x N)
    - h (int): Number of assets to select for tracking

    Returns:
    - selected_assets (list): Indices of the selected assets
    """
    T_train, N = ast_train.shape  # Number of time periods (T_train) and assets (N)
    
    # Define autoencoder structure
    input_layer = Input(shape=(N,))
    encoded = Dense(64, activation="relu")(input_layer)
    encoded = Dense(32, activation="relu")(encoded)
    bottleneck = Dense(16, activation="relu")(encoded)  # Latent space
    decoded = Dense(32, activation="relu")(bottleneck)
    decoded = Dense(64, activation="relu")(decoded)
    output_layer = Dense(N, activation="linear")(decoded)

    # Create autoencoder model
    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer="adam", loss="mse")

    # Train autoencoder
    autoencoder.fit(ast_train, ast_train, epochs=100, batch_size=32, verbose=0)

    # Compute reconstruction errors
    reconstructed = autoencoder.predict(ast_train)
    d_i = np.mean(np.square(ast_train - reconstructed), axis=0)  # MSE per asset

    # Select top h assets based on smallest reconstruction error
    selected_assets = np.argsort(d_i)[:h]  # Indices of top h assets

    return selected_assets


def deep_NN(ast_train, ind_train, h):
    """
    Selects assets using deep_autoencoder, trains a deep NN, and computes portfolio weights 
    using gradient-based sensitivity analysis.

    Parameters:
    - ast_train: Training asset returns (T x N)
    - ind_train: Training index returns (T x 1)
    - h (int): Number of assets to include in the tracking portfolio

    Returns:
    - final_portfolio_weights (numpy array): Portfolio weights (N x 1)
    """
    T_train, N = ast_train.shape  # Number of time periods (T_train) and assets (N)

    # Step 1: Select h assets using deep autoencoder
    selected_assets = deep_autoencoder(ast_train, h)

    # Extract selected asset returns for training
    selected_ast_train = ast_train[:, selected_assets]

    # Step 2: Train a deep neural network on training data
    input_layer = Input(shape=(h,))
    hidden = Dense(64, activation="relu")(input_layer)
    hidden = Dense(32, activation="relu")(hidden)
    output_layer = Dense(1, activation="linear")(hidden)

    # Define model
    model = Model(input_layer, output_layer)
    model.compile(optimizer=Adam(learning_rate=0.01), loss="mse")

    # Train on training set only
    model.fit(selected_ast_train, ind_train, epochs=100, batch_size=16, verbose=0)

    # Compute portfolio weights using gradient-based sensitivity analysis
    with tf.GradientTape() as tape:
        inputs = tf.convert_to_tensor(selected_ast_train, dtype=tf.float32)
        tape.watch(inputs)
        predictions = model(inputs)
    gradients = tape.gradient(predictions, inputs)

    # Compute absolute sensitivity scores
    sensitivity_scores = np.mean(np.abs(gradients.numpy()), axis=0)  # Average over time

    # Normalize to obtain final portfolio weights
    portfolio_weights = sensitivity_scores / np.sum(sensitivity_scores)

    # Create full portfolio vector (all non-selected assets get zero weight)
    final_portfolio_weights = np.zeros(N)
    final_portfolio_weights[selected_assets] = portfolio_weights

    return final_portfolio_weights


file_path = "(.....enter file path here....)"
ind_ret, ast_ret, N, T = load_price_data(file_path)


# Rolling Window Setup
in_sample = 21 * 24  # In-sample period (24 months)
out_sample = 21 * 3  # Out-of-sample period (3 months)
roll = 21 * 3  # Rolling window size (3 months)
win_count = (T - in_sample - out_sample) // roll + 1

# Initialize results matrix
W = np.zeros((win_count, N))

start = time.time()

# Perform deep NN-based portfolio optimization
for i in range(win_count):
    print(f"Window {i + 1} of {win_count}")
    m1 = roll * i
    m2 = in_sample + roll * i

    IR = ind_ret[m1:m2]
    AR = ast_ret[m1:m2, :]

    try:
        # Perform cointegration and get optimal portfolio weights
        W[i, :]=deep_NN(AR, IR, h=45)
    except ValueError as e:
        print(f"Skipping Window {i + 1}: {e}")
        W[i, :] = np.nan  # Assign NaN to indicate this window was skipped

end = time.time()
total_time = end - start
print(f"Total time: {total_time} seconds")


# Save the results to Excel
df_w = pd.DataFrame(W)
output_file = ".......enter file path for saving output here....."
df_w.to_excel(output_file, index=False)