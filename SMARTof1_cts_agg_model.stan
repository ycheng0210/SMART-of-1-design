data {
  int<lower=0> J; //number of patients
  int<lower=0> cycle;//number of cycle
  real resp1[cycle*J]; //stage1 response
  int<lower=0> stay[cycle*J]; //stay on the same trt, 0 or 1 
  real resp2[cycle*J]; //stage2 response
  
  int<lower=0> trt1[cycle*J]; //stage1 trt
  int<lower=0> trt2[cycle*J]; //stage2 trt
  int<lower=0> link[cycle*J]; //linkage indices
}


parameters {
  
  // population parameter
  row_vector[2] delta_mu; // population level trt effect
  row_vector<lower = 0>[4] betas_mu; // linkage
  real<lower = 0> delta_tau;       //  SD for trt effect 
  matrix[J,2] delta; // individual level trt effects
  real <lower = 0 > y_epsilon[J];// SD within patients


}

transformed parameters{
  
  real delta_s1[cycle*J]; // derived response rate with linkage
  real delta_s2[cycle*J]; // derived response rate with linkage
  
  for (j in 1:J){
    
    for (i in 1:cycle){
      
      delta_s1[(j-1)*cycle+i] = delta[j, trt1[(j-1)*cycle+i] ];
      delta_s2[(j-1)*cycle+i] = delta[j, trt2[(j-1)*cycle+i] ] * betas_mu[link[(j-1)*cycle+i]];
      
    }
  }
  
}

model {
  
  // Likelihood of stage 1 and 2 responses
  for (j in 1:J){
    for (i in 1:cycle){
      resp1[(j-1)*cycle+i] ~ normal(delta_s1[(j-1)*cycle+i],y_epsilon[j]);
      resp2[(j-1)*cycle+i] ~ normal(delta_s2[(j-1)*cycle+i],y_epsilon[j]);
    }
  }
  
  // Distribution of the individual-level effects 
  for(j in 1:J){
      delta[j,1] ~ normal(delta_mu[1],delta_tau);
      delta[j,2] ~ normal(delta_mu[2],delta_tau);
  }
  
  // Hyperpriors on population level distribution
  delta_mu[1] ~ normal(1, 10);  
  delta_mu[2] ~ normal(1, 10);  
  delta_tau ~ cauchy(0, 3); 
  
  betas_mu[1] ~ normal(1,5);
  betas_mu[2] ~ normal(1,5);
  betas_mu[3] ~ normal(0.5,5);
  betas_mu[4] ~ normal(0.5,5);

  y_epsilon ~ cauchy(0,3);


}
