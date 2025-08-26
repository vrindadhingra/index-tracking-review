# Load required libraries
library(glmnet)
library(nnls)
library(readxl)
library(writexl)

#Function 1: Optimal parameter tuning for lambda

tune_lambda_elastic_net <- function(X, y, K, lambda2_grid = c(1e-4, 1e-3, 1e-2, 0.05, 0.1)) {
  best_lambda1 <- NULL
  best_lambda2 <- NULL
  best_error <- Inf
  
  for (lambda2 in lambda2_grid) {
    lambda1_max <- 10  # Reduce max lambda1 (was 1e5 before)
    lambda1_min <- 1e-6  # Smallest lambda1
    tol <- 1e-3  
    
    selected_features <- c()
    while ((lambda1_max - lambda1_min) > tol) {
      lambda1_mid <- (lambda1_max + lambda1_min) / 2
      result <- nonnegative_elastic_net(X, y, lambda1_mid, lambda2)
      selected_features <- result$selected_features
      
      if (length(selected_features) > K) {
        lambda1_min <- lambda1_mid  
      } else {
        lambda1_max <- lambda1_mid  
      }
    }
    
    # Ensure at least K features are selected
    lambda1_opt <- (lambda1_max + lambda1_min) / 2
    result <- nonnegative_elastic_net(X, y, lambda1_opt, lambda2)
    selected_features <- result$selected_features
    
    if (length(selected_features) < K) {
      warning(paste("Could not select exactly", K, "features. Selected:", length(selected_features)))
      next
    }
    
    # Compute tracking error
    X_selected <- X[, selected_features]
    beta_nnls <- nonnegative_least_squares(X_selected, y)
    y_pred <- X_selected %*% beta_nnls
    tracking_error <- sqrt(mean((y - y_pred)^2))
    
    if (tracking_error < best_error) {
      best_lambda1 <- lambda1_opt
      best_lambda2 <- lambda2
      best_error <- tracking_error
    }
  }
  
  # If no valid values found, return the best found values
  if (is.null(best_lambda1) || is.null(best_lambda2)) {
    warning("No valid lambda1, lambda2 found. Trying alternative tuning...")
    best_lambda1 <- 0.001  # Reduce lambda1 further
    best_lambda2 <- 0.001  # Reduce lambda2 further
  }
  
  return(list(lambda1 = best_lambda1, lambda2 = best_lambda2))
}

#<=K assets selected 
tune_lambda_elastic_net <- function(X, y, K, lambda2_grid = c(1e-4, 1e-3, 1e-2, 0.05, 0.1)) {
  best_lambda1 <- NULL
  best_lambda2 <- NULL
  best_error <- Inf
  
  for (lambda2 in lambda2_grid) {
    lambda1_max <- 10  # Prevent excessive regularization
    lambda1_min <- 1e-6  
    tol <- 1e-3  
    
    best_lambda1_for_lambda2 <- NULL
    best_num_features <- Inf  # Track the best (≤ K) asset count
    
    while ((lambda1_max - lambda1_min) > tol) {
      lambda1_mid <- (lambda1_max + lambda1_min) / 2
      result <- nonnegative_elastic_net(X, y, lambda1_mid, lambda2)
      selected_features <- result$selected_features
      num_selected <- length(selected_features)
      
      if (num_selected > K) {
        lambda1_min <- lambda1_mid  # Increase sparsity (reduce selected assets)
      } else {
        lambda1_max <- lambda1_mid  # Decrease sparsity (increase selected assets)
        best_lambda1_for_lambda2 <- lambda1_mid
        best_num_features <- num_selected  # Track best valid count
      }
    }
    
    # Use the best lambda1 that gives ≤ K assets
    lambda1_opt <- best_lambda1_for_lambda2
    
    if (!is.null(lambda1_opt)) {
      result <- nonnegative_elastic_net(X, y, lambda1_opt, lambda2)
      selected_features <- result$selected_features
      
      if (length(selected_features) <= K) {
        # Compute tracking error
        X_selected <- X[, selected_features]
        beta_nnls <- nonnegative_least_squares(X_selected, y)
        y_pred <- X_selected %*% beta_nnls
        tracking_error <- sqrt(mean((y - y_pred)^2))
        
        # Update the best parameters
        if (tracking_error < best_error) {
          best_lambda1 <- lambda1_opt
          best_lambda2 <- lambda2
          best_error <- tracking_error
        }
      }
    }
  }
  
  if (is.null(best_lambda1) || is.null(best_lambda2)) {
    warning("No valid lambda1, lambda2 found. Assigning safe defaults.")
    best_lambda1 <- 0.001  
    best_lambda2 <- 0.001  
  }
  
  return(list(lambda1 = best_lambda1, lambda2 = best_lambda2))
}


#Function 2: Non-negative elastic net
nonnegative_elastic_net <- function(X, y, lambda1, lambda2) {
  alpha_value <- lambda1 / (lambda1 + lambda2)  # Mixing parameter for Elastic Net
  lambda_total <- sqrt(lambda1^2 + lambda2^2)  # Regularization strength
  
  fit <- glmnet(
    X, y,
    alpha = alpha_value,
    lambda = lambda_total,
    lower.limits = 0  # Enforce nonnegativity
  )
  
  beta <- as.vector(coef(fit, s = "lambda.min")[-1])
  selected_features <- which(beta > 0)
  
  return(list(selected_features = selected_features, beta = beta))
}


#Function 3: Non-negative least squares regression (NNLS)
nonnegative_least_squares <- function(X_0, y) {
  fit <- nnls(X_0, y)
  beta <- coef(fit)
  return(beta)
}


#Function 4: Calculate portfolio weights

generate_optimal_portfolio <- function(X, y, K) {
  # Step 1: Tune optimal lambda values
  lambda_values <- tune_lambda_elastic_net(X, y, K)
  lambda1 <- lambda_values$lambda1
  lambda2 <- lambda_values$lambda2
  
  # Step 2: Perform NNEN with optimal lambdas
  result <- nonnegative_elastic_net(X, y, lambda1, lambda2)
  selected_features <- result$selected_features
  
  if (length(selected_features) == 0) {
    cat("No features selected. Try adjusting lambda.\n")
    return(rep(0, ncol(X)))
  }
  
  # Step 3: Compute portfolio weights using NNLS
  X_selected <- X[, selected_features]
  beta_nnls <- nonnegative_least_squares(X_selected, y)
  
  # Step 4: Normalize weights to sum to 1
  beta_nnls_normalized <- beta_nnls / sum(beta_nnls)
  
  # Step 5: Create full portfolio vector (zero weights for non-selected assets)
  full_portfolio <- rep(0, ncol(X))
  full_portfolio[selected_features] <- beta_nnls_normalized
  
  return(full_portfolio)
}
