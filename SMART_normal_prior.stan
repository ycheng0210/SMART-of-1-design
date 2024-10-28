data {
  int<lower=0> J; //number of patients
  int<lower=0> resp1[J]; //stage1 response
  int<lower=0> resp2[J]; //stage2 response
  
  int<lower=0> trt1[J]; //stage1 trt
  int<lower=0> trt2[J]; //stage2 trt

  int<lower=0> link[J]; //linkage
 
 }

parameters {
  
  real <lower = 0 , upper = 1/1.5 > delta_mu [2];// trt effect
  real <lower = 0.9,upper = 1.1> beta1 [2]; //linkage
  real <lower = 0,upper = 1> beta0 [2]; //linkage
 }
 
transformed parameters{
   real <lower = 0,upper = 1> delta_s1[J];
   real <lower = 0,upper = 1> delta_s2[J];
   real <lower = 0> dtr_AB;
   real <lower = 0> dtr_BA;
   
  
  //  response rates for each stage
   for (j in 1:J){
       delta_s1[j] = delta_mu[trt1[j]];
       
      if(trt1[j]==trt2[j]){
        delta_s2[j] = delta_mu[trt2[j]] * beta1[link[j]];

      }else if(trt1[j]!=trt2[j]){
        delta_s2[j] = delta_mu[trt2[j]] * beta0[link[j]-2];
      }
       
   }
  
  // DTR rates
    dtr_AB = delta_mu[1] * (delta_mu[1] * beta1[1]) + (1 - delta_mu[1]) * (delta_mu[2] * beta0[1]);
    dtr_BA = delta_mu[2] * (delta_mu[2] * beta1[2]) + (1 - delta_mu[2]) * (delta_mu[1] * beta0[2]);
    
   
}
 
model {
  
  //Priors
      beta1[1] ~ normal(1,0.2)
      beta1[2] ~ normal(1,0.2); 
      beta0[1] ~ normal(0.7,0.2);
      beta0[2] ~ normal(0.7,0.2); 
      delta_mu[1] ~ normal(0.5,0.5);    
      delta_mu[2] ~ normal(0.5,0.5);
 
 
  //Likelihood
  {
   for (j in 1:J){
     
      resp1[j] ~ bernoulli(delta_s1[j]);
      resp2[j] ~ bernoulli(delta_s2[j]);
      
     }
  
  }
  
}
