tev=function(indata,indexin, k)
{
  params <- list()
  params$TimeLimit<-1800
  params$OutputFlag<-0
  model <- list()
  
  n=dim(indata)[2]  # Total number of assets
  q=nrow(indata)     # In-sample data points 
  
  # Total number of variables: [1:n] (x = w - b), [n+1:2n] (z binary variables)
  m = 2*n  
  
  # Constraint Matrices
  A = c(rep(1, n), rep(0, n))  # sum(x) = 0 constraint
  B = c(rep(0,n), rep(1,n))
  I = diag(n)
  Z = matrix(0, nrow=n, ncol=n)  
  
  # Constructing the augmented constraint matrix
  Amat = rbind(
    A,             # constraint 1
    cbind(I, Z),   # constraint 2
    cbind(I, Z),   # constraint 3
    B,             # constraint 4
    cbind(I, -I)   # constraint 5
  )
  
  # Compute index weights using NNLS
  ols = nnls(indata, indexin)
  coeff = ols$x
  
  ind_weights = coeff
  s = sum(ind_weights)
  ind_weights = (1/s) * ind_weights  # Normalize weights
  
  # b = original index weight vector (from NNLS or known index)
  b_sparse <- ind_weights
  #top_k <- order(b_sparse, decreasing = TRUE)[1:k]
  #b_sparse[-top_k] <- 0
  #b_sparse <- b_sparse / sum(b_sparse)  # re-normalize to sum to 1

  b0 = -1 * b_sparse
  b1 = 1 + b0
  
  # Quadratic term in objective (covariance matrix)
  Qmat = matrix(0, m, m)
  Qmat[1:n, 1:n] = cov(indata)  # Only x has quadratic terms
  
  # Updating Model
  model$Q     <- Qmat
  model$obj   <- c(rep(0, n), rep(0, n))  # No linear term
  model$A     <- Amat
  model$rhs   <- c(0, b1, b0, k, b0)  # Adjusted RHS
  model$sense <- c('=', rep('<', n), rep('>', n), '<', rep('<', n))  # Constraint types
  model$modelsense <- 'min'
  model$vtype <- c(rep('C', n), rep('B', n))  # First n variables are continuous, last n are binary
  model$lb <- c(
    rep(-Inf, n),   # x_i ∈ [–∞, …)
    rep(0,     n)   # z_i ∈ [0, …)  (binary will then be [0,1])
  )
  model$ub <- c(
    rep( Inf, n),   # x_i ∈ (…, +∞]
    rep(1,     n)   # z_i ∈ [0,1]
  )
  
  print(Amat)
  print(model$rhs)
  print(Qmat)
  
  # Solve the model
  result_tev2 <- gurobi(model, params)
  print(result_tev2$status)
  
  print(result_tev2$x[1:n])
  
  # Extract optimized portfolio weights
  p_weights = c(result_tev2$x[1:n]) + b_sparse[1:n]  # Convert x back to w
  
  return(p_weights)
}