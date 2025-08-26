# Load required libraries
library(glmnet)
library(nnls)
library(readxl)
library(writexl)

# Function 1: Nonnegative-Lasso (NNL)

nonnegative_lasso <- function(X, y, alpha){
  # Perform Nonnegative-Lasso (NNL) regression with positive weights
  #
  # Parameters:
  #   X: Design matrix (n_samples x n_features), e.g., asset returns.
  #   y: Response vector (n_samples), e.g., index returns.
  #   alpha: Regularization strength (controls sparsity).
  #
  # Returns:
  #   A list containing:
  #     - selected_features: Indices of selected features (nonzero coefficients).
  #     - beta: Nonnegative regression coefficients (n_features).
  
  # Fit the Lasso model with nonnegative constraints
  fit <- glmnet(
    X, y,
    alpha = 0.01,          # L1 regularization (Lasso)
    lambda = alpha,     # Regularization parameter
    lower.limits = 0    # Nonnegative constraint
  )
  
  # Extract coefficients (exclude the intercept)
  beta <- as.vector(coef(fit, s = "lambda.min")[-1])
  
  # Identify selected features (nonzero coefficients)
  selected_features <- which(beta > 0)
  
  return(list(selected_features = selected_features, beta = beta))
}

# Function 2: Nonnegative Least Squares (NNLS)
nonnegative_least_squares <- function(X_0, y) 
{
  
  fit <- nnls(X_0, y)
  
  # Extract coefficients
  beta <- coef(fit)
  return(beta)
}

# Function 3: Normalize Weights
normalize_weights <- function(beta) 
{
  return(beta / sum(beta))
}

#Function 4: Tune lambda 
tune_lambda <- function(X, y, desired_k) {
  lambda_max <- 100  # Start with a large lambda (empty selection)
  lambda_min <- 0  # Start with small lambda (full selection)
  tol <- 1e-2  # Convergence tolerance
  while (lambda_max - lambda_min > tol) {
    lambda_mid <- (lambda_max + lambda_min) / 2
    selected_features <- nonnegative_lasso(X, y, lambda_mid)$selected_features
    if (length(selected_features) > desired_k) {
      lambda_min <- lambda_mid  # Too many assets, increase sparsity
    } else {
      lambda_max <- lambda_mid  # Too few assets, decrease sparsity
    }
  }
  return(lambda_mid)
}

# Function 5: Full Two-Step Workflow with optimal tuning parameter calculation 

generate_portfolio <- function(X, y, desired_card) 
{
  # Generate a normalized index-tracking portfolio using NNL and NNLS
  #
  # Parameters:
  #   X: Design matrix (asset returns, n_samples x n_features).
  #   y: Response vector (index returns, n_samples).
  #   alpha: Regularization strength for NNL.
  #
  # Returns:
  #   A list containing:
  #     - selected_features: Indices of selected features.
  #     - portfolio_weights: Full portfolio weights (n_features).
  
  # Step 1: Nonnegative-Lasso for variable selection
  alpha<-tune_lambda(X, y, desired_card)
  print(alpha)
  result <- nonnegative_lasso(X, y, alpha)
  selected_features <- result$selected_features
  
  # If no features are selected, return zeros
  if (length(selected_features) == 0) {
    cat("No features selected. Try reducing alpha.\n")
    return(list(selected_features = NULL, portfolio_weights = rep(0, ncol(X))))
  }
  
  # Step 2: Nonnegative Least Squares for refinement
  X_selected <- X[, selected_features] # Subset of selected features
  beta_nnls <- nonnegative_least_squares(X_selected, y)
  
  # Normalize weights
  beta_nnls_normalized <- normalize_weights(beta_nnls)
  
  # Create a full portfolio with zeros for unselected assets
  full_portfolio <- rep(0, ncol(X))
  full_portfolio[selected_features] <- beta_nnls_normalized
  
  return(full_portfolio)
}
