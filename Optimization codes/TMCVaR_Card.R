tmcvar_card=function(indata, indexin, alpha, delta, K)
{
  params <- list()
  params$TimeLimit<-1800
  params$OutputFlag<-0
  model <-list()
  
  delta=delta
  alpha=sort(alpha, decreasing=TRUE)
  m=length(alpha)
  n=dim(indata)[2] #total number of assets
  q=nrow(indata) #in-sample data points 
  
  #variable vector x=(w, beta_u, beta_l, u^U, u^L, z), w: n*1, beta_u & beta_l:m*1, u^U & u^L: (q*m), z:n*1
  
  v=n+2*m+2*q*m+n #total number of variables
  
  lambda=c(rep(0,m))
  alpha0=1
  #calculation of weights (lambda's)
  
  lambda[1]=((1-alpha[1])*((1-alpha[2])-(1-alpha0)))/((1-alpha[m])^2)
  s=m-1
  
  for(k in 2:s)
  {
    lambda[k]=((1-alpha[k])*((1-alpha[k+1])-(1-alpha[k-1])))/((1-alpha[m])^2)
  }
  
  lambda[m]=((1-alpha[m])*(alpha[m-1]-alpha[m]))/(1-alpha[m])^2
  
  l=c(rep(0,m))
  for(k in 1:m)
  {
    l[k]=lambda[k]/(1-alpha[k])
  }
  
  Ucoeff=delta*l
  Dcoeff=(1-delta)*l
  beta_u=delta*lambda
  beta_l=(1-delta)*lambda
  
  #objective function
  
  obj_fun=c(rep(0,n), beta_u, beta_l, rep(Ucoeff, q), rep(Dcoeff, q), rep(0,n))
  
  #print(obj_fun)
  
  #constraints coefficient matrix
  
  A1=matrix(c(rep(1,n), rep(0, 2*m+2*m*q+n)), nrow=1) #sum(w)
  #print(ncol(A1))
  #print(A1)
  
  #MCVaR constraint-upper
  O=matrix(0, q, n)
  W=c(rep(0,n))
  for(k in 1:m)
  {
    W=rbind(W, indata)
  }
  W=W[-1,]
  I1=diag(m)
  B1=c(rep(0,m))
  for(k in 1:m)
  {
    for(j in 1:q)
    {
      B1=rbind(B1,I1[k,])
    }
  }
  B1=B1[-1,]
  B2=matrix(0,q*m, m)
  I2=diag(q*m)
  O2=matrix(0, q*m, q*m)
  
  #MCVaR constraint-upper
  A2=cbind(W, B1, B2, I2, O2, matrix(0, q*m, n))
  #print(ncol(A2))
  #print(A2)
  
  #MCVaR constraint-lower
  A3=cbind(-W, B2, B1, O2, I2, matrix(0, q*m, n))
  #print(ncol(A3))
  #print(A3)
  
  A4=cbind(diag(n), matrix(0, n, 2*m+2*q*m), -diag(n))  # w_i - z_i <= 0
  #print(A4)
  #print(ncol(A4))
  
  A5=matrix(c(rep(0, n + 2*m + 2*q*m), rep(1, n)), nrow=1)  # sum(z)
  #print(A5)
  #print(ncol(A5))
  
  Amat=rbind(A1, A2, A3, A4, A5)
  
  rhs_vec = matrix(c(1, rep(indexin, m), rep(-1*indexin, m), rep(0, n), K), ncol = 1)
  
  sense_vec = c('=', rep('>', q*m), rep('>', q*m), rep('<', n), '<')
  
  
  model$obj   <-obj_fun
  model$A     <- Amat
  model$rhs   <- rhs_vec
  model$sense <- sense_vec
  model$modelsense <- 'min'
  model$vtype <- c(rep('C',n), rep('C', 2*m+2*m*q), rep('B',n))
  
  result_tmcvar<-gurobi(model,params)
  print(result_tmcvar$status)
  print(result_tmcvar$runtime)
  return(result_tmcvar$x[1:n])
  
}