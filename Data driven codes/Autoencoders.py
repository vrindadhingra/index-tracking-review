"""
@author: vrindadhingra
"""

import pandas as pd
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Layer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l1
from tensorflow.keras.losses import mse
import tensorflow.keras.backend as K
from cvxopt import matrix, solvers


# Load Data
file_path = "......enter file path to load price data....."
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


def pick_mixed_by_loss(recon_loss, k, alpha_communal=0.7):
    order = np.argsort(recon_loss)              # low -> high
    k_low  = int(np.floor(k * alpha_communal))  # communal block
    k_high = k - k_low                           # idio block
    return np.r_[ order[:k_low], order[-k_high:] ]


def single_hidden_layer_autoencoder(asset_returns, k):
    """
    Train a single-hidden-layer autoencoder and rank assets based on reconstruction loss.
    """
    # Standardize data
    scaler = StandardScaler()
    normalized_returns = scaler.fit_transform(asset_returns)

    # Build autoencoder
    input_dim = normalized_returns.shape[1]
    input_layer = Input(shape=(input_dim,))
    latent = Dense(16, activation='relu')(input_layer)
    output_layer = Dense(input_dim, activation='linear')(latent)
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Train autoencoder
    autoencoder.fit(normalized_returns, normalized_returns, epochs=50, batch_size=32, verbose=0)

    # Reconstruction loss per asset
    reconstructed = autoencoder.predict(normalized_returns)
    reconstruction_loss = np.mean((normalized_returns - reconstructed) ** 2, axis=0)
    
    # Rank assets
    #ranked_assets = np.argsort(reconstruction_loss)[:k]
    #return ranked_assets
    
    ranked_assets = pick_mixed_by_loss(reconstruction_loss, k, alpha_communal=0.7)
    return ranked_assets


def sparse_autoencoder(asset_returns, k, sparsity_penalty=1e-4):
    """
    Train a sparse autoencoder and rank assets based on reconstruction loss.
    """
    # Standardize data
    scaler = StandardScaler()
    normalized_returns = scaler.fit_transform(asset_returns)

    # Build autoencoder
    input_dim = normalized_returns.shape[1]
    input_layer = Input(shape=(input_dim,))
    latent = Dense(16, activation='relu', activity_regularizer=l1(sparsity_penalty))(input_layer)
    output_layer = Dense(input_dim, activation='linear')(latent)
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Train autoencoder
    autoencoder.fit(normalized_returns, normalized_returns, epochs=50, batch_size=32, verbose=0)

    # Reconstruction loss per asset
    reconstructed = autoencoder.predict(normalized_returns)
    reconstruction_loss = np.mean((normalized_returns - reconstructed) ** 2, axis=0)
    
    # Rank assets
    ranked_assets = pick_mixed_by_loss(reconstruction_loss, k, alpha_communal=0.7)
    return ranked_assets


def contractive_autoencoder(asset_returns, k, lambda_penalty=1e-4):
    """
    Train a contractive autoencoder and rank assets based on reconstruction loss.

    Args:
        asset_returns (np.array): Matrix of asset returns (training set).
        k (int): Number of top-ranked assets to select.
        lambda_penalty (float): Penalty factor for the contractive loss.

    Returns:
        ranked_assets (list): Indices of the top k ranked assets.
    """
    # Standardize data
    scaler = StandardScaler()
    normalized_returns = scaler.fit_transform(asset_returns)

    # Build autoencoder
    input_dim = normalized_returns.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(16, activation='relu', name='encoder')(input_layer)  # Encoder layer
    output_layer = Dense(input_dim, activation='linear', name='decoder')(encoder)  # Decoder layer
    autoencoder = Model(inputs=input_layer, outputs=output_layer)

    # Custom loss with contractive penalty
    def contractive_loss(y_true, y_pred):
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))  # Mean squared error
        # Compute the Jacobian norm for the encoder layer
        encoder_weights = autoencoder.get_layer('encoder').kernel
        jacobian_norm = tf.reduce_sum(tf.square(encoder_weights))
        return mse_loss + lambda_penalty * jacobian_norm

    # Compile autoencoder
    autoencoder.compile(optimizer='adam', loss=contractive_loss)

    # Train autoencoder
    autoencoder.fit(normalized_returns, normalized_returns, epochs=50, batch_size=32, verbose=0)

    # Reconstruction loss per asset
    reconstructed = autoencoder.predict(normalized_returns)
    reconstruction_loss = np.mean((normalized_returns - reconstructed) ** 2, axis=0)
    
    # Rank assets
    ranked_assets = pick_mixed_by_loss(reconstruction_loss, k, alpha_communal=0.7)
    return ranked_assets


def stacked_autoencoder(asset_returns, k):
    """
    Train a stacked autoencoder and rank assets based on reconstruction loss.
    """
    # Standardize data
    scaler = StandardScaler()
    normalized_returns = scaler.fit_transform(asset_returns)

    # Build autoencoder
    input_dim = normalized_returns.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(64, activation='relu')(input_layer)
    encoder = Dense(32, activation='relu')(encoder)
    latent = Dense(16, activation='relu')(encoder)
    decoder = Dense(32, activation='relu')(latent)
    decoder = Dense(64, activation='relu')(decoder)
    output_layer = Dense(input_dim, activation='linear')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Train autoencoder
    autoencoder.fit(normalized_returns, normalized_returns, epochs=50, batch_size=32, verbose=0)

    # Reconstruction loss per asset
    reconstructed = autoencoder.predict(normalized_returns)
    reconstruction_loss = np.mean((normalized_returns - reconstructed) ** 2, axis=0)
    
    # Rank assets
    ranked_assets = pick_mixed_by_loss(reconstruction_loss, k, alpha_communal=0.7)
    return ranked_assets


def denoising_autoencoder(asset_returns, k, noise_factor=0.1):
    """
    Train a denoising autoencoder and rank assets based on reconstruction loss.
    """
    # Standardize data
    scaler = StandardScaler()
    normalized_returns = scaler.fit_transform(asset_returns)

    # Add noise to data
    noisy_data = normalized_returns + noise_factor * np.random.normal(size=normalized_returns.shape)

    # Build autoencoder
    input_dim = normalized_returns.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(16, activation='relu')(input_layer)
    output_layer = Dense(input_dim, activation='linear')(encoder)
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Train autoencoder
    autoencoder.fit(noisy_data, normalized_returns, epochs=50, batch_size=32, verbose=0)

    # Reconstruction loss per asset
    reconstructed = autoencoder.predict(normalized_returns)
    reconstruction_loss = np.mean((normalized_returns - reconstructed) ** 2, axis=0)
    
    # Rank assets
    ranked_assets = pick_mixed_by_loss(reconstruction_loss, k, alpha_communal=0.7)
    return ranked_assets


def variational_autoencoder(asset_returns, k, latent_dim=16, lambda_kl=1e-4, epochs=50, batch_size=32):
    """
    Train a Variational Autoencoder (VAE) and rank assets based on reconstruction loss.
    """

    # Standardize data
    scaler = StandardScaler()
    normalized_returns = scaler.fit_transform(asset_returns)
    input_dim = normalized_returns.shape[1]

    # Custom sampling layer
    class Sampling(Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            epsilon = tf.random.normal(shape=tf.shape(z_mean))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    # Custom KL regularization layer
    class KLRegularizer(Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            kl = -0.5 * tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
            )
            self.add_loss(lambda_kl * tf.reduce_mean(kl))
            return z_mean  # Pass-through

    # Build encoder
    inputs = Input(shape=(input_dim,))
    h = Dense(32, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(h)
    z_log_var = Dense(latent_dim, name='z_log_var')(h)

    # Register KL loss
    z_mean = KLRegularizer()([z_mean, z_log_var])

    # Sampling
    z = Sampling()([z_mean, z_log_var])

    # Decoder
    decoder_hidden = Dense(32, activation='relu')(z)
    decoder_output = Dense(input_dim, activation='linear')(decoder_hidden)

    # VAE model
    vae = Model(inputs, decoder_output)
    vae.compile(optimizer='adam', loss='mse')

    # Train
    vae.fit(normalized_returns, normalized_returns,
            epochs=epochs, batch_size=batch_size, verbose=0)

    # Reconstruction loss per asset
    reconstructed = vae.predict(normalized_returns)
    reconstruction_loss = np.mean((normalized_returns - reconstructed) ** 2, axis=0)

    # Select top-k assets
    ranked_assets = pick_mixed_by_loss(reconstruction_loss, k, alpha_communal=0.7)
    return ranked_assets



def portfolio_selection(index_returns, asset_returns, ranked_assets, k, reg_lambda=1e-2):
    """
    Select portfolio weights for all assets, with weights for unselected assets set to zero.

    Args:
        index_returns (np.array): Vector of index returns (shape: T,).
        asset_returns (np.array): Matrix of asset returns (shape: T, N).
        ranked_assets (list): Indices of the top-ranked assets to include in the portfolio.
        k (int): Number of assets to include in the portfolio.
        reg_lambda (float): Regularization parameter for weights.

    Returns:
        all_weights (np.array): Portfolio weights for all assets (zeros for unselected assets).
    """
    # Extract returns of selected assets
    selected_returns = asset_returns[:, ranked_assets[:k]]  # Shape: (T, k)

    # Define the quadratic programming problem
    T, N = selected_returns.shape

    # Objective: Minimize tracking error ||index_returns - selected_returns @ weights||^2 + reg_lambda * ||weights||^2
    P = 2 * (selected_returns.T @ selected_returns + reg_lambda * np.eye(N))
    q = -2 * selected_returns.T @ index_returns

    # Constraints: weights >= 0 and sum(weights) = 1
    G = -np.eye(N)  # Nonnegativity constraint
    h = np.zeros(N)
    A = np.ones((1, N))  # Sum of weights constraint
    b = np.array([1.0])

    # Convert to cvxopt format
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)

    # Solve the quadratic programming problem
    solvers.options['show_progress'] = False
    solution = solvers.qp(P, q, G, h, A, b)

    # Extract the weights for selected assets
    selected_weights = np.array(solution['x']).flatten()

    # Create a vector of weights for all assets (zeros for unselected assets)
    all_weights = np.zeros(asset_returns.shape[1])
    all_weights[ranked_assets[:k]] = selected_weights

    return all_weights

# rolling window set up
train_len = 504
test_len = 63
roll = 63
win_count = (T - train_len-test_len) // roll + 1

W1= np.zeros((win_count, N))
W2= np.zeros((win_count, N))
W3= np.zeros((win_count, N))
W4= np.zeros((win_count, N))
W5= np.zeros((win_count, N))
W6= np.zeros((win_count, N))

# Dictionary to store timingsfor each autoencoder
timings = {}


# Rolling window for each autoencoder separately to note time taken 


# --- Single Hidden Layer ---
start = time.time()
for i in range(win_count):
    m1 = roll * i
    m2 = train_len + roll * i
    IR = ind_ret[m1:m2]
    AR = ast_ret[m1:m2, :]
    k = 45
    W1[i] = portfolio_selection(IR, AR, single_hidden_layer_autoencoder(AR, k), k)
timings["Single_Hidden_Layer"] = time.time() - start


# --- Sparse AE ---
start = time.time()
for i in range(win_count):
    m1 = roll * i
    m2 = train_len + roll * i
    IR = ind_ret[m1:m2]
    AR = ast_ret[m1:m2, :]
    k = 45
    W2[i] = portfolio_selection(IR, AR, sparse_autoencoder(AR, k), k)
timings["Sparse_AE"] = time.time() - start



# --- Contractive AE ---
start = time.time()
for i in range(win_count):
    m1 = roll * i
    m2 = train_len + roll * i
    IR = ind_ret[m1:m2]
    AR = ast_ret[m1:m2, :]
    k = 45
    W3[i] = portfolio_selection(IR, AR, contractive_autoencoder(AR, k), k)
timings["Contractive_AE"] = time.time() - start



# --- Stacked AE ---
start = time.time()
for i in range(win_count):
    m1 = roll * i
    m2 = train_len + roll * i
    IR = ind_ret[m1:m2]
    AR = ast_ret[m1:m2, :]
    k = 45
    W4[i] = portfolio_selection(IR, AR, stacked_autoencoder(AR, k), k)
timings["Stacked_AE"] = time.time() - start



# --- Denoising AE ---
start = time.time()
for i in range(win_count):
    m1 = roll * i
    m2 = train_len + roll * i
    IR = ind_ret[m1:m2]
    AR = ast_ret[m1:m2, :]
    k = 45
    W5[i] = portfolio_selection(IR, AR, denoising_autoencoder(AR, k), k)
timings["Denoising_AE"] = time.time() - start



# --- Variational AE ---
start = time.time()
for i in range(win_count):
    m1 = roll * i
    m2 = train_len + roll * i
    IR = ind_ret[m1:m2]
    AR = ast_ret[m1:m2, :]
    k = 45
    W6[i] = portfolio_selection(IR, AR, variational_autoencoder(AR, k), k)
timings["Variational_AE"] = time.time() - start



# Print timings
print("\n--- Timing Summary ---")
for name, t in timings.items():
    print(f"{name}: {t:.2f} seconds")


# Create DataFrames for each weight matrix
df_W1 = pd.DataFrame(W1, columns=[f'Asset_{i+1}' for i in range(N)])
df_W2 = pd.DataFrame(W2, columns=[f'Asset_{i+1}' for i in range(N)])
df_W3 = pd.DataFrame(W3, columns=[f'Asset_{i+1}' for i in range(N)])
df_W4 = pd.DataFrame(W4, columns=[f'Asset_{i+1}' for i in range(N)])
df_W5 = pd.DataFrame(W5, columns=[f'Asset_{i+1}' for i in range(N)])
df_W6 = pd.DataFrame(W6, columns=[f'Asset_{i+1}' for i in range(N)])

# Add rolling window index
df_W1.index = [f'Rolling_Window_{i+1}' for i in range(win_count)]
df_W2.index = [f'Rolling_Window_{i+1}' for i in range(win_count)]
df_W3.index = [f'Rolling_Window_{i+1}' for i in range(win_count)]
df_W4.index = [f'Rolling_Window_{i+1}' for i in range(win_count)]
df_W5.index = [f'Rolling_Window_{i+1}' for i in range(win_count)]
df_W6.index = [f'Rolling_Window_{i+1}' for i in range(win_count)]


# Write matrices to Excel
output_file = ".....enter path to store the weights file..../portfolio_weights_AE.xlsx"

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df_W1.to_excel(writer, sheet_name='Single_Hidden_Layer')
    df_W2.to_excel(writer, sheet_name='Sparse_Autoencoder')
    df_W3.to_excel(writer, sheet_name='Contractive_Autoencoder')
    df_W4.to_excel(writer, sheet_name='Stacked_Autoencoder')
    df_W5.to_excel(writer, sheet_name='Denoising_Autoencoder')
    df_W6.to_excel(writer, sheet_name='Variational_Autoencoder')

print(f"Weights saved to {output_file}")

