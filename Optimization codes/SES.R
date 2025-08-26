ses=function(indata, indexin, k)
{
  params <- list()
  params$Presolve<-0
  params$TimeLimit<-1800
  params$OutputFlag<-1
  model <-list()
  indata=indata
  
  n=dim(indata)[2] #total number of assets
  q=nrow(indata) #in-sample data points 
  
  #total number of variables: [1:n] (portfolio weights), [n+1:2n] cardinality variables (binary)
  m=2*n
  
  #constraint matrices
  A=c(rep(1,n), rep(0,n))
  B=c(rep(0,n), rep(1,n))
  C=diag(n)
  D=cbind(C,-C)
  Amat=rbind(A,B,D)
  
  #objective function 
  I=sum(indexin)
  R=colSums(indata)
  Qmat=matrix(0,m,m)
  Qmat[c(1:n), c(1:n)]=2*R%*%t(R)
  
  obj_wvec=(-2)*I*R
  
  obj_cons=I*I
  
  model$Q=Qmat
  model$obj=c(obj_wvec, rep(0,n))
  model$objcon=obj_cons
  model$A=Amat
  model$rhs=c(1,k,rep(0,n))
  model$sense=c('=',"=", rep('<',n))
  model$modelsense='min'
  model$vtype=c(rep('C',n), rep('B',n))
  result_ses=gurobi(model,params)
  
  result_ses$status
  return(result_ses$x[1:n])
  
}
