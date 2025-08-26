lsq_reg<-function(indata,indexin, k)
{
  params <- list()
  params$TimeLimit<-1800
  params$OutputFlag<-0
  model1<-list()
  model2<-list()
  
  AR=indata
  IR=indexin
  
  n=dim(AR)[2] #total number of assets
  q=nrow(AR) #in-sample data points 
  
  #total number of variables=2n+2: 
  #[1:n] (portfolio weights), [n+1:2n] cardinality (binary) variables, [2n+1]: alpha_opt(d), [2n+2]: beta_opt(e)
  
  #Regression variables: Alpha and Beta calculation
  
  alpha=c(rep(0,n))
  beta=c(rep(0,n))
  
  for(j in 1:n)
  {
    ols=lm(AR[,j]~IR)
    alpha[j]=ols$coefficients[1]
    beta[j]=ols$coefficients[2]
  }
  
  
  #constraint matrices (LHS)
  A=c(rep(1,n), rep(0,n+2)) #const1: sum(w)==1
  B=c(rep(0,n), rep(1,n), rep(0,2)) #const2: sum(z)<=k
  I=diag(n)
  Z=matrix(0,n,2)
  C=cbind(I,-I, Z) # const3: w<=z
  D1=c(alpha, rep(0,n), -1, 0) 
  D2=c(alpha, rep(0,n), 1, 0)
  D3=c(alpha, rep(0,n), 0, 0)
  E1=c(beta, rep(0,n), 0, -1)
  E2=c(beta, rep(0,n), 0, 1)
  
  # Combine all constraints
  Amat1 = rbind(A, B, C, D1, D2, E1, E2)
  
  # Right-hand side and constraint sense
  model1$obj   <- c(rep(0,n), rep(0,n), 1, 0)  
  model1$A     <- Amat1
  model1$rhs   <- c(1, k, rep(0,n), 0, 0, 1, 1)
  model1$sense <- c('=', "<", rep('<', n), "<", ">", "<", ">")
  model1$modelsense <- 'min'
  model1$vtype <- c(rep('C', n), rep('B', n), rep('C', 2))
  result1<-gurobi(model1,params)
  print(result1$status)
  
  #Step2: Minimizing e=Beta-1
  
  w_opt1=result1$x[1:n]
  alpha_opt=alpha%*%w_opt1
  
  Amat2=rbind(A,B,C,E1,E2,D3)
  
  model2$obj   <- c(rep(0,n), rep(0,n), 0, 1)
  model2$A     <- Amat2
  model2$rhs   <- c(1,k,rep(0,n),1, 1, alpha_opt)
  model2$sense <- c('=',"<", rep('<',n), "<", ">", "=")
  model2$modelsense <- 'min'
  model2$vtype <- c(rep('C',n), rep('B',n), rep('C', 2))
  result2<-gurobi(model2,params)
  
  return(result2$x[1:n])
  
}