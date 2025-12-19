#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vrindadhingra
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
from gurobipy import Model, GRB, quicksum

# ------------------------------
# Load and preprocess data
# ------------------------------

file_path = ".......enter file path to load price data......"
price_data = pd.read_excel(file_path)

# Remove date column and take log prices
price_matrix = price_data.iloc[:, 1:].values.astype(float)
log_prices  = np.log(price_matrix)

# Benchmark (index) and assets in logs
log_benchmark = log_prices[:, 0]     # shape (T_all,)
log_assets    = log_prices[:, 1:]     # shape (T_all, N)

T_all, N = log_assets.shape
K = 45  # max number of assets allowed (cardinality)

# Rolling window setup (24m in-sample, 3m step, 3m oos kept out)
in_sample = 21 * 24
out_sample = 21 * 3
roll = 21 * 3
win_count = (T_all - in_sample - out_sample) // roll + 1


# --------------------------------------------------
# Solver: cointegration regression with cardinality
# --------------------------------------------------

from gurobipy import Model, GRB, quicksum
import numpy as np

def solve_cointegration(in_sample_benchmark, in_sample_assets, K,
                        timelimit=1800, mipgap=1e-3, output_flag=0,
                        add_ridge=False, lam=1e-4):
    """
    Minimize sum_t (y_t - (beta0 + X_t·beta))^2
    s.t. sum_i beta_i = 1, beta_i >= 0, sum_i mu_i <= K, beta_i <= mu_i, mu_i ∈ {0,1}.
    Uses centered (demeaned) X, y to improve numerical stability.
    If not OPTIMAL but a feasible incumbent exists, returns that solution.
    """

    # --- Center the data (improves numerics for level regression with intercept) ---
    y = in_sample_benchmark.astype(float)
    X = in_sample_assets.astype(float)
    y_center = y - y.mean()
    X_center = X - X.mean(axis=0)

    T, N = X_center.shape

    m = Model("Cointegrated_Portfolio")
    
    # ---- Gurobi parameters ----
    m.Params.TimeLimit   = timelimit
    m.Params.MIPGap      = mipgap
    m.Params.OutputFlag  = output_flag
    m.Params.MIPFocus    = 1          # find feasible solutions faster
    m.Params.NumericFocus= 2          # be more robust to numerics
    m.Params.FeasibilityTol = 1e-6
    m.Params.IntFeasTol     = 1e-5
    m.Params.Heuristics     = 0.2

    # ---- Variables ----
    beta0 = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="beta0")
    beta  = m.addVars(N, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="beta")
    mu    = m.addVars(N, vtype=GRB.BINARY, name="mu")

    # ---- Objective ----
    resid = [
        y_center[t] - (beta0 + quicksum(beta[i]*X_center[t, i] for i in range(N)))
        for t in range(T)
    ]
    obj = quicksum(resid[t]*resid[t] for t in range(T))
    if add_ridge and lam > 0.0:
        obj += lam * quicksum(beta[i]*beta[i] for i in range(N))
    m.setObjective(obj, GRB.MINIMIZE)

    # ---- Constraints ----
    m.addConstr(quicksum(mu[i] for i in range(N)) <= K, name="cardinality")
    for i in range(N):
        m.addConstr(beta[i] <= mu[i], name=f"link_{i}")
    m.addConstr(quicksum(beta[i] for i in range(N)) == 1.0, name="budget")

    # ---- Optimize ----
    m.optimize()

    # Helpful logging
    status = m.Status
    # Accept incumbent if available (even if not OPTIMAL)
    if m.SolCount > 0:
        weights  = np.array([beta[i].X for i in range(N)])
        selected = [i for i in range(N) if mu[i].X > 0.5]
        obj_val  = m.ObjVal if status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) else None
        return weights, selected, obj_val

    # If we got here, no feasible solution was found
    raise ValueError(f"Optimization was not successful (status {status}) and no incumbent found.")
    
# ------------------------------
# Rolling window run
# ------------------------------


W = np.zeros((win_count, N))
timings = []

start = time.time()
for k in range(win_count):
    print(f"Processing window {k+1} / {win_count} ...")
    m1 = k * roll
    m2 = m1 + in_sample

    y_in = log_benchmark[m1:m2]     # (in_sample,)
    X_in = log_assets[m1:m2, :]     # (in_sample, N)

    t0 = time.time()
    try:
        weights, selected, obj = solve_cointegration(
            y_in, X_in, K,
            timelimit=1800, mipgap=1e-3, output_flag=0,  # flip to 1 for Gurobi logs
            add_ridge=False, lam=1e-4                    # set add_ridge=True if you want mild dispersion
        )
        W[k, :] = weights
        print(f"  selected: {len(selected)} names; obj={obj:.4e}")
    except ValueError as e:
        print(f"  window {k+1} skipped: {e}")
        W[k, :] = np.nan
    timings.append(time.time() - t0)

elapsed = time.time() - start
print(f"\nDone. Total time: {elapsed:.2f}s  |  avg per window: {np.nanmean(timings):.2f}s")


# ------------------------------
# Save results
# ------------------------------

out_path = "/........enter path to save the output weights file...."

pd.DataFrame(W).to_excel(out_path, index=False)
print(f"Saved weights to: {out_path}")