data {
  int<lower=0> J; // number of patients
  int<lower=0> cycle; // number of cycles
  int<lower=0> resp1[cycle*J]; // stage 1 response
  int<lower=0> resp2[cycle*J]; // stage 2 response
  int<lower=0> trt1[cycle*J]; // stage 1 treatment
  int<lower=0> trt2[cycle*J]; // stage 2 treatment
  int<lower=0> link[cycle*J]; // linkage indices
}

parameters {
  // Population parameters
  row_vector[2] delta_log_odds_mu; // population-level effect (log-odds scale)
  vector<lower=0.5, upper=1.1>[4] betas_mu; // population-level linkage
  real<lower=0> delta_tau; // Population-level effect SD
  matrix[J, 2] delta_log_odds; // individual-level effects (log-odds scale)
}

transformed parameters {
  matrix<lower=0, upper=1>[J, 2] delta; // individual-level probabilities
  real<lower=0, upper=1> delta_s1[cycle*J]; // derived response rate stage 1
  real<lower=0, upper=1> delta_s2[cycle*J]; // derived response rate stage 2
  real<lower=0> dtr_AB; // DTR AB
  real<lower=0> dtr_BA; // DTR BA
  row_vector[2] delta_mu; // population-level probabilities

  // Transform individual log_odds to probabilities
  for (j in 1:J) {
    for (i in 1:2) {
      delta[j, i] = inv_logit(delta_log_odds[j, i]);
    }
  }

  // Transform population log-odds to probabilities
  for (i in 1:2) {
    delta_mu[i] = inv_logit(delta_log_odds_mu[i]);
  }

  // response rates for stages  
  for (j in 1:J) {
    for (i in 1:cycle) {
      delta_s1[(j-1)*cycle + i] = delta[j, trt1[(j-1)*cycle + i]];
      delta_s2[(j-1)*cycle + i] = delta[j, trt2[(j-1)*cycle + i]] * betas_mu[link[(j-1)*cycle + i]];
    }
  }

  // Population-level DTR
  dtr_AB = delta_mu[1] * (delta_mu[1] * betas_mu[1]) + (1 - delta_mu[1]) * (delta_mu[2] * betas_mu[3]);
  dtr_BA = delta_mu[2] * (delta_mu[2] * betas_mu[2]) + (1 - delta_mu[2]) * (delta_mu[1] * betas_mu[4]);
}

model {
  // Likelihood for stage 1 and 2 responses
  for (j in 1:J) {
    for (i in 1:cycle) {
      resp1[(j-1)*cycle + i] ~ bernoulli(delta_s1[(j-1)*cycle + i]);
      resp2[(j-1)*cycle + i] ~ bernoulli(delta_s2[(j-1)*cycle + i]);
    }
  }

  // Individual-level effects
  for (j in 1:J) {
    delta_log_odds[j, 1] ~ normal(delta_log_odds_mu[1], delta_tau);
    delta_log_odds[j, 2] ~ normal(delta_log_odds_mu[2], delta_tau);

  }

  // Population-level priors
  delta_log_odds_mu[1] ~ normal(-0.5, 10); // uninformative prior on pop. mean
  delta_log_odds_mu[2] ~ normal(0.5, 10); // uninformative prior on pop. mean
  delta_tau ~ cauchy(0, 3); // weakly informative prior on pop. variance
  
  for (i in 1:4) {
    
    betas_mu[i] ~ normal((1 - 0.3 * (i > 2)), 0.2 ); // shifted log-normal prior for positivity
  }
  
}
