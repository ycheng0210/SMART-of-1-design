// code from https://github.com/sshres/Bayesian-bandit-adaptive-n-of-1
data {
  int<lower=0> J; //number of patients
  int<lower=0> N; //number of total observations
  real y[N]; //outcome
  real t[N]; //treatment indicator
  int<lower=0> id[N]; //patient ID
}
parameters {
  vector[2] beta_mu; //population mean
  vector<lower=0>[2] beta_sigma; //population SD b/w patients
  vector[J] eta1; //individual variables for non-centered parameterization
  vector[J] eta2;
  real<lower=0> y_sig; //SD w/in patients
}
transformed parameters{
  vector[J] beta1; //individual coefficients
  vector[J] beta2;
  beta1 = beta_mu[1] + beta_sigma[1]*eta1;
  beta2 = beta_mu[2] + beta_sigma[2]*eta2;
}
model {
  //priors
  y_sig ~ cauchy(0,3); 
  beta_mu ~ normal(1, 10); 
  beta_sigma ~ student_t(1, 0, 10); 
  eta1 ~ normal(0,1);
  eta2 ~ normal(0,1);
  
  //likelihood
  {
    vector[N] xbeta;
    for(n in 1:N)
      xbeta[n] = beta1[id[n]] + beta2[id[n]].*t[n];
    y ~ normal(xbeta, y_sig);
  }
}
