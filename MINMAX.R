min_max=function(indata,indexin, k)
{
  params <- list()
  params$TimeLimit <- 1800
  params$OutputFlag<-0
  model <- list()
  
  n=dim(indata)[2]
  q=nrow(indata)
  
  
  #total number of variables: [1:n] portfolio weights (w), [n+1:2n] cardinality (binary) variables (z), [2n+1] maximum deviation variable (a)
  m=2*n+1
  
  A=c(rep(1,n), rep(0,n), 0) #sum(w)==1
  B=c(rep(0,n), rep(1,n), 0) #sum(z)<=k
  I=diag(n)
  Z=matrix(0,n,1)
  C=cbind(I,-I, Z) #w<=z
  D=cbind(indata, matrix(0,q,n), matrix(-1,q,1)) 
  E=cbind(indata, matrix(0,q,n), matrix(1,q,1))
  
  Amat=rbind(A,B,C,D,E)
  
  model$A     <- Amat
  model$obj   <- c(rep(0,n), rep(0,n), 1)
  model$rhs   <- c(1, k, rep(0,n), indexin, indexin)
  model$sense <- c("=", "<", rep('<',n), rep('<',q), rep('>',q))
  model$modelsense <- 'min'
  model$vtype <- c(rep('C',n), rep('B',n), 'C')
  
  result_minmax<- gurobi(model,params)
  
  return(result_minmax$x[1:n])
  
}
