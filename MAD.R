mad<-function(indata,indexin, k)
{
  params <- list()
  params$TimeLimit<-1800
  params$OutputFlag<-0
  model <-list()
  indata=indata
  
  n=dim(indata)[2] #total number of assets
  q=nrow(indata) #in-sample data points 
  
  #Variables: [1:n] (portfolio weights:w), [n+1:2n] card (binary) variables(z), 
  #           [2n+1, 2n+q]: a+ (+ve deviations), [2n+q+1: 2n+2q]: a- (-ve deviations)
  
  m=2*n+2*q
  
  #constraints matrices (LHS)
  
  A=c(rep(1,n), rep(0,n), rep(0,q), rep(0,q)) #sum(w)==1
  B=c(rep(0,n), rep(1,n), rep(0,q), rep(0,q)) #sum(z)<=k
  I=diag(n)
  Z=matrix(0,n,q)
  C=cbind(I,-I, Z, Z) #w<=z
  D=cbind(indata, matrix(0,q,n), diag(q), diag(-1,q,q)) #MAD constraint 
  
  Amat=rbind(A,B,C,D)
  
  model$A     <- Amat
  model$obj   <- c(rep(0,n),rep(0,n), rep(1,q),rep(1,q))
  model$rhs   <- c(1, k, rep(0,n), indexin)
  model$sense <- c("=", "<", rep('<',n), rep('=',q))
  model$modelsense <- 'min'
  model$vtype<-c(rep('C',n), rep('B',n), rep('C', q), rep('C', q))
  result_mad<-gurobi(model,params)
  
  return(result_mad$x[1:n])
  
}
