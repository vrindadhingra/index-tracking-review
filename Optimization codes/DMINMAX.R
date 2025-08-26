dmin_max<-function(indata,indexin, k)
{
  params <- list()
  params$TimeLimit <- 1800
  params$OutputFlag<-0
  model <- list()
  indata=indata
  
  n=dim(indata)[2]
  q=nrow(indata)
  
  #total number of variables: [1:n] portfolio weights (w), [n+1:2n] cardinality binary variables (z), [2n+1] maximum deviation variable (a)
  
  A=c(rep(1,n), rep(0,n), 0) #sum(w)==1
  B=c(rep(0,n), rep(1,n), 0) #sum(z)<=k
  I=diag(n)
  Z=matrix(0,n,1)
  C=cbind(I,-I, Z) #w<=z
  E=cbind(indata, matrix(0,q,n), matrix(1,q,1))
  
  Amat=rbind(A,B,C,E)
  
  model$A     <- Amat
  model$obj   <- c(rep(0,n), rep(0,n), 1)
  model$rhs   <- c(1, k, rep(0,n),indexin)
  model$sense <- c("=", "<", rep('<',n), rep('>',q))
  model$modelsense <- 'min'
  model$vtype <- c(rep('C',n), rep('B',n), 'C')
  
  result_dminmax<- gurobi(model,params)
  
  return(result_dminmax$x[1:n])
  
}