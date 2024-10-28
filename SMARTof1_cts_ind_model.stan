data {
  int<lower=0> J; // number of patients
  int<lower=0> cycle; // number of cycles
  real resp1[cycle * J]; // stage 1 response
  int<lower=0> stay[cycle * J]; // stay on the same treatment, 0 or 1
  real resp2[cycle * J]; // stage 2 response
  int<lower=0> trt1[cycle * J]; // stage 1 treatment
  int<lower=0> trt2[cycle * J]; // stage 2 treatment
  int<lower=0> link[cycle * J]; // linkage
}

parameters {
  real<lower=0> delta[2, J]; // individual trt effects
  real beta1_raw[2, J]; // raw values for beta1, will transform using inv_logit
  real beta0_raw[2, J]; // raw values for beta0, will transform using inv_logit
  real<lower=0> y_epsilon[J]; // SD within patients
}

transformed parameters {
  real<lower=0.5, upper=1.5> beta1[2, J]; // transformed beta1 (constrained to [0.5, 1.5])
  real<lower=0, upper=1> beta0[2, J]; // transformed beta0 (constrained to [0, 1])
  real delta_s1[cycle * J]; // stage 1 response rate
  real delta_s2[cycle * J]; // stage 2 response rate

  // Apply the inv_logit transformation to beta1 and beta0
  for (j in 1:J) {
    for (i in 1:2) {
      beta1[i, j] = 0.5 + inv_logit(beta1_raw[i, j]); // beta1 constrained to [0.5, 1.5]
      beta0[i, j] = inv_logit(beta0_raw[i, j]); // beta0 constrained to [0, 1]
    }
  }

  // Calculate response rates for stage 1 and stage 2
  for (j in 1:J) {
    for (i in 1:cycle) {
      delta_s1[(j - 1) * cycle + i] = delta[trt1[(j - 1) * cycle + i], j];

      if (trt1[(j - 1) * cycle + i] == trt2[(j - 1) * cycle + i]) {
        delta_s2[(j - 1) * cycle + i] = delta[trt2[(j - 1) * cycle + i], j] * beta1[link[(j - 1) * cycle + i], j];
      } else {
        delta_s2[(j - 1) * cycle + i] = delta[trt2[(j - 1) * cycle + i], j] * beta0[link[(j - 1) * cycle + i] - 2, j];
      }
    }
  }
}

model {
  // Priors 
  for (j in 1:J) {
    for (i in 1:2) {
      delta[i, j] ~ normal(0, 5); 
      beta1_raw[i, j] ~ normal(0, 2); 
      beta0_raw[i, j] ~ normal(0, 2);
    }
    y_epsilon[j] ~ cauchy(0, 3); 
  }

  // Likelihood for stage 1 and 2 responses
  for (j in 1:J) {
    for (i in 1:cycle) {
      resp1[(j - 1) * cycle + i] ~ normal(delta_s1[(j - 1) * cycle + i], y_epsilon[j]);
      resp2[(j - 1) * cycle + i] ~ normal(delta_s2[(j - 1) * cycle + i], y_epsilon[j]);
    }
  }
}
