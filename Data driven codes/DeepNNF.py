"""
@author: vrindadhingra
"""


import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout, Activation
from tensorflow.keras.models import Model

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
# Deep NNF Model Construction
# ------------------------------
def create_deep_NNF_model(num_assets, hidden_size=64, dropout_rate=0.5):
    """
    Creates the Deep NNF model. Instead of using the actual input,
    it uses a fixed noise vector to produce portfolio weights.
    
    Args:
        num_assets (int): Number of assets (output dimension).
        hidden_size (int): Number of neurons in hidden layers.
        dropout_rate (float): Dropout rate.
        
    Returns:
        model (tf.keras.Model): Keras model that outputs portfolio weights.
    """
    # Generate a fixed noise vector (constant across training)
    fixed_noise = tf.constant(np.random.randn(num_assets), dtype=tf.float32)
    
    # Dummy input (we use its batch dimension to tile the fixed noise)
    input_dummy = Input(shape=(num_assets,))
    # Lambda layer to ignore the dummy input and return fixed noise for each sample
    x = Lambda(lambda x: tf.tile(tf.reshape(fixed_noise, (1, num_assets)), [tf.shape(x)[0], 1]))(input_dummy)
    
    # Deep network: six fully connected layers with dropout and ReLU activations
    x = Dense(hidden_size, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(hidden_size, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(hidden_size, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(hidden_size, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(hidden_size, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(num_assets)(x)
    output = Activation('softmax')(x)  # Softmax ensures outputs sum to 1
    model = Model(inputs=input_dummy, outputs=output)
    return model

# --------------------------------------
# Custom Training Function for Deep NNF
# --------------------------------------
def train_deep_NNF_model(ast_train, ind_train, num_assets, hidden_size=64, dropout_rate=0.5, 
                           epochs=100, batch_size=16):
    """
    Trains the Deep NNF model using a custom training loop.
    
    Args:
        ast_train (np.array): Training asset returns (shape: T x num_assets).
        ind_train (np.array): Training index returns (shape: T,).
        num_assets (int): Number of assets (input/output dimension).
        hidden_size (int): Hidden layer size.
        dropout_rate (float): Dropout probability.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        
    Returns:
        model: Trained Deep NNF model.
    """
    model = create_deep_NNF_model(num_assets, hidden_size, dropout_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    # Create a TensorFlow dataset from training data.
    # Note: We use ast_train and ind_train to compute the loss.
    dataset = tf.data.Dataset.from_tensor_slices((ast_train.astype(np.float32), ind_train.astype(np.float32)))
    dataset = dataset.batch(batch_size)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for asset_batch, index_batch in dataset:
            current_batch = asset_batch.shape[0]
            # Create dummy input: shape = (batch_size, num_assets)
            dummy_input = tf.zeros((current_batch, num_assets))
            with tf.GradientTape() as tape:
                # Get portfolio weights from the model
                weights = model(dummy_input, training=True)  # shape: (batch_size, num_assets)
                # Compute portfolio return: dot product of weights and asset returns (for each sample)
                portfolio_return = tf.reduce_sum(weights * asset_batch, axis=1)
                # Loss: mean squared error between portfolio return and index return
                loss = tf.reduce_mean(tf.square(portfolio_return - index_batch))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss += loss.numpy()
        #print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    return model


# ----------------------------------
# Deep NNF for Partial Replication
# ----------------------------------
def deep_NNF_partial_replication(ast_train, ind_train, h, hidden_size=64, dropout_rate=0.5, epochs=100, batch_size=16):
    """
    Performs partial replication using the Deep NNF approach:
      1. Train the Deep NNF on the full set of assets.
      2. Select the top h assets based on the model's predicted weights.
      3. Train a new Deep NNF on only these h assets.
      4. Output the final portfolio weights in the full asset space.
    
    Args:
        ast_train (np.array): Training asset returns (T x N).
        ind_train (np.array): Training index returns (T,).
        h (int): Number of assets to include in the final portfolio.
        hidden_size, dropout_rate, epochs, batch_size: Training hyperparameters.
    
    Returns:
        final_portfolio_weights (np.array): Final portfolio weights (length = N).
    """
    T_train, N = ast_train.shape
    
    # Step 1: Train Deep NNF on full asset set.
    print("Training Deep NNF on full asset set...")
    model_full = train_deep_NNF_model(ast_train, ind_train, num_assets=N,
                                      hidden_size=hidden_size, dropout_rate=dropout_rate,
                                      epochs=epochs, batch_size=batch_size)
    # Use a dummy input (batch size 1) to get predicted weights.
    dummy_full = tf.zeros((1, N))
    full_weights = model_full(dummy_full, training=False).numpy()[0]
    
    # Step 2: Select top h assets (largest weights).
    selected_assets = np.argsort(full_weights)[-h:]
    print("Selected asset indices:", selected_assets)
    
    # Step 3: Train Deep NNF on the selected h assets.
    ast_train_subset = ast_train[:, selected_assets]
    print("Training Deep NNF on selected asset subset...")
    model_subset = train_deep_NNF_model(ast_train_subset, ind_train, num_assets=h,
                                        hidden_size=hidden_size, dropout_rate=dropout_rate,
                                        epochs=epochs, batch_size=batch_size)
    dummy_subset = tf.zeros((1, h))
    subset_weights = model_subset(dummy_subset, training=False).numpy()[0]
    
    # Step 4: Create the final portfolio vector (full dimension), with zeros for unselected assets.
    final_portfolio_weights = np.zeros(N)
    final_portfolio_weights[selected_assets] = subset_weights
    return final_portfolio_weights

# ------------------------------
# Rolling Window Implementation
# ------------------------------
file_path = "/.....enter file path to load price data.../sp500_daily.xlsx"
ind_ret, ast_ret, N, T = load_price_data(file_path)

# Define rolling window parameters (adjust as needed)
in_sample = 21 * 24  # In-sample period (24 months)
out_sample = 21 * 3  # Out-of-sample period (3 months)
roll = 21 * 3      # Rolling window size (3 months)
win_count = (T - in_sample - out_sample) // roll + 1


# Initialize results matrix
W = np.zeros((win_count, N))

start = time.time()
for i in range(win_count):
    print(f"\n--- Window {i + 1} of {win_count} ---")
    m1 = roll * i
    m2 = in_sample + roll * i
    IR = ind_ret[m1:m2]
    AR = ast_ret[m1:m2, :]
    
    try:
        # Perform Deep NNF partial replication with h selected assets (e.g., h=45)
        final_weights = deep_NNF_partial_replication(AR, IR, h=45, 
                                                     hidden_size
                                                     =64, dropout_rate=0.5,
                                                     epochs=100, batch_size=16)
        W[i, :] = final_weights
    except Exception as e:
        print(f"Skipping Window {i + 1}: {e}")
        W[i, :] = np.nan  # Mark window as failed

end = time.time()
print(f"Total time: {end - start:.2f} seconds")

# Save the results to Excel
df_w = pd.DataFrame(W)
output_file = ".....enter path to save your output file.../W_deepNNF.xlsx"
df_w.to_excel(output_file, index=False)