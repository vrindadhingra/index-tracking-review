mse=function(indata,indexin, k)
{
  params <- list()
  params$TimeLimit<-1800
  params$OutputFlag<-0
  model <-list()
  indata=indata
  
  n=dim(indata)[2] #total number of assets
  q=nrow(indata) #in-sample data points 
  
  #total number of variables: [1:n] (portfolio weights), [n+1:2n] cardinality variables (binary)
  m=2*n #total number of variables

  #constraint matrices
  
  A=c(rep(1,n), rep(0,n))
  B=c(rep(0,n), rep(1,n))
  C=diag(n)
  D=cbind(C,-C)
  Amat=rbind(A,B,D)
  
  #Quadratic objective: x^TQx+c^Tx+d (obj->c, objcon->d)
  
  #Quadratic coefficients (Q)
  Qmat=matrix(0, m, m)
  Smat=matrix(0, n, n)
  R=indata
  I=indexin
  
  for(j in 1:q)
  {
    Smat=Smat+(R[j,]%*%t(R[j,]))
  }
  Qmat[c(1:n), c(1:n)]=(1/q)*Smat
  
  #linear coefficients of objective
  Cmat=I*R
  w_vec=-2*(1/q)*colSums(Cmat)
  
  #constant coefficients of the objective for the portfolio weights
  cons_vec=(1/q)*(sum(I^2))
  
  #Model parameters
  model$Q=Qmat
  model$obj=c(w_vec, rep(0,n))
  model$con=c(cons_vec, rep(0,n))
  model$A     <- Amat
  model$rhs   <- c(1,k,rep(0,n))
  model$sense <- c('=',"<", rep('<',n))
  model$modelsense <- 'min'
  model$vtype <- c(rep('C',n), rep('B',n))
  result_mse<-gurobi(model,params)
  
  return(result_mse$x[1:n])
  
}
