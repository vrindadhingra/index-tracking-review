"""
@author: vrindadhingra

"""

import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum

# Step 1: Load and Preprocess Data
file_path = "......enter file path to load price data"
price_data = pd.read_excel(file_path)

# Remove the date column and calculate log prices
price_matrix = price_data.iloc[:, 1:].values
price_matrix = np.array(price_matrix, dtype=float)
log_prices = np.log(price_matrix)

# Define benchmark and asset log prices
log_benchmark = log_prices[:, 0]  # Benchmark (Index)
log_assets = log_prices[:, 1:]  # Assets

# Get dimensions
T, N = log_assets.shape
K = 45  # Cardinality constraint (maximum number of assets in the portfolio)

# Rolling Window Setup
in_sample = 21 * 24  # In-sample period
out_sample = 21 * 3  # Out-of-sample period
roll = 21 * 3  # Rolling window size

# First Window Indices
start_in = roll+1
end_in = start_in + in_sample

# Extract in-sample data for the first window
in_sample_benchmark = log_benchmark[start_in:end_in]
in_sample_assets = log_assets[start_in:end_in, :]

# Step 2: Create Gurobi Model
model = Model("Cointegrated_Portfolio")

# Variables
beta = model.addVars(N, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="beta")  # Portfolio weights
mu = model.addVars(N, vtype=GRB.BINARY, name="mu")  # Binary inclusion variables

# Objective Function: Minimize squared residuals
residuals = [
    in_sample_benchmark[t] - quicksum(beta[i] * in_sample_assets[t, i] for i in range(N))
    for t in range(in_sample)
]
model.setObjective(quicksum(residuals[t] * residuals[t] for t in range(in_sample)), GRB.MINIMIZE)

# Constraints
model.addConstr(quicksum(mu[i] for i in range(N)) == K, name="cardinality")  # Cardinality
for i in range(N):
    model.addConstr(beta[i] <= mu[i], name=f"linking_upper_{i}")  # Upper linking
    model.addConstr(beta[i] >= 0.01 * mu[i], name=f"linking_lower_{i}")  # Lower linking (minimum weight)
model.addConstr(quicksum(beta[i] for i in range(N)) == 1, name="normalization")  # Normalization

# Step 3: Solve the Model
model.optimize()

# Step 4: Extract Results
if model.status == GRB.OPTIMAL:
    portfolio_weights = [beta[i].X for i in range(N)]
    selected_assets = [i for i in range(N) if mu[i].X > 0.5]
    print("Optimal Portfolio Weights:", portfolio_weights)
    print("Selected Assets:", selected_assets)
else:
    print("Optimization was not successful.")
    
#--------------ROLLING WINDOW------------------#  

import pandas as pd
import numpy as np
import time
from gurobipy import Model, GRB, quicksum

# Step 1: Load and Preprocess Data
file_path = "/Users/vrindadhingra/Desktop/NSUT Project/Index Tracking/R Codes/sp500_daily.xlsx"
price_data = pd.read_excel(file_path)

# Remove the date column and calculate log prices
price_matrix = price_data.iloc[:, 1:].values
price_matrix = np.array(price_matrix, dtype=float)
log_prices = np.log(price_matrix)

# Define benchmark and asset log prices
log_benchmark = log_prices[:, 0]  # Benchmark (Index)
log_assets = log_prices[:, 1:]  # Assets

# Rolling Window Parameters
in_sample = 21 * 24  # In-sample period
out_sample = 21 * 3  # Out-of-sample period
roll = 21 * 3  # Rolling window size
T, N = log_assets.shape  # Dimensions of the dataset
K = 45  # Cardinality constraint (maximum number of assets in the portfolio)
win_count = (T - in_sample - out_sample) // roll + 1  # Number of rolling windows

# Function: Solve Cointegration Problem Using Gurobi
def solve_cointegration(in_sample_benchmark, in_sample_assets, K):
    """
    Solve the convex MINLP problem for cointegration-based index tracking.
    
    Parameters:
    - in_sample_benchmark: Array of benchmark returns for the in-sample period.
    - in_sample_assets: Matrix of asset returns for the in-sample period.
    - K: Cardinality constraint (maximum number of assets in the portfolio).
    
    Returns:
    - portfolio_weights: Optimal portfolio weights for all assets.
    - selected_assets: Indices of the selected assets in the portfolio.
    """
    T, N = in_sample_assets.shape

    # Create a Gurobi model
    model = Model("Cointegrated_Portfolio")
    
    # --- Gurobi Parameters ---
    model.setParam("TimeLimit", 1800)   # max 30 minutes per window
    model.setParam("MIPGap", 0.001)   # stop if within 0.1% of optimal


    # Variables
    beta = model.addVars(N, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="beta")  # Portfolio weights
    mu = model.addVars(N, vtype=GRB.BINARY, name="mu")  # Binary inclusion variables

    # Objective Function: Minimize squared residuals
    residuals = [
    in_sample_benchmark[t] - (quicksum(beta[i] * in_sample_assets[t, i] for i in range(N)))
    for t in range(T)
    ]
    model.setObjective(quicksum(residuals[t] * residuals[t] for t in range(T)), GRB.MINIMIZE)

    # Constraints
    model.addConstr(quicksum(mu[i] for i in range(N)) == K, name="cardinality")  # Cardinality
    for i in range(N):
        model.addConstr(beta[i] <= mu[i], name=f"linking_upper_{i}")  # Upper linking
        model.addConstr(beta[i] >= 0.01 * mu[i], name=f"linking_lower_{i}")  # Lower linking (minimum weight)
    model.addConstr(quicksum(beta[i] for i in range(N)) == 1, name="normalization")  # Normalization

    # Solve the Model
    model.optimize()

    # Extract Results
    if model.status == GRB.OPTIMAL:
        portfolio_weights = np.array([beta[i].X for i in range(N)])
        selected_assets = [i for i in range(N) if mu[i].X > 0.5]
        return portfolio_weights, selected_assets
    else:
        raise ValueError("Optimization was not successful.")

# Step 2: Rolling Window Simulation
W = np.zeros((win_count, N))  # Matrix to store portfolio weights for each window

start=time.time()

for i in range(win_count):
    print(f"Processing Window {i + 1} of {win_count}...")

    # Define start and end indices for the in-sample period
    start_in = i * roll
    end_in = start_in + in_sample

    # Extract in-sample data for the current window
    in_sample_benchmark = log_benchmark[start_in:end_in]
    in_sample_assets = log_assets[start_in:end_in, :]

    try:
        # Solve the cointegration problem for the current window
        weights, selected_assets = solve_cointegration(in_sample_benchmark, in_sample_assets, K)
        W[i, :] = weights  # Save weights for the current window
    except ValueError as e:
        print(f"Skipping Window {i + 1}: {e}")
        W[i, :] = np.nan  # Assign NaN if optimization fails

end=time.time()
total_time=end-start
print(f"Total time: {total_time} seconds")

print(weights)

# Save Results
output_file = "......enter file path to save output weights file.../W_Coint_cvx.xlsx"
df_w = pd.DataFrame(W)
df_w.to_excel(output_file, index=False)

print("Rolling window simulation completed. Results saved.")