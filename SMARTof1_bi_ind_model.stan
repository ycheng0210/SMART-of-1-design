data {
  int<lower=0> J;            // number of patients
  int<lower=0> cycle;        // number of cycles
  int<lower=0> resp1[cycle * J];  // stage 1 response (bernoulli)
  int<lower=0> resp2[cycle * J];  // stage 2 response (bernoulli)
  int<lower=0> trt1[cycle * J];   // stage 1 treatment
  int<lower=0> trt2[cycle * J];   // stage 2 treatment
  int<lower=0> link[cycle * J];   // linkage
}

parameters {
  // Unconstrained real numbers that will be transformed
  real delta_raw[2, J];       // Raw delta
  real beta1_raw[2, J];       // Raw beta1 (unconstrained)
  real beta0_raw[2, J];       // Raw beta0 (unconstrained)
}

transformed parameters {
  real<lower=0, upper=1> delta[2, J];   // Delta between 0 and 1
  real<lower=0.5, upper=1.5> beta1[2, J];  // Beta1 between 0.5 and 1.5
  real<lower=0, upper=1> beta0[2, J];  // Beta0 between 0 and 1
  real delta_s1[cycle * J];             // Stage 1 response probabilities
  real delta_s2[cycle * J];             // Stage 2 response probabilities
  real dtr_AB[J];                       // Dynamic treatment regimen (A -> B)
  real dtr_BA[J];                       // Dynamic treatment regimen (B -> A)

  // Apply transformations
  for (j in 1:J) {
    for (i in 1:2) {
      // Use an inverse-logit to constrain delta to [0, 1]
      delta[i, j] = inv_logit(delta_raw[i, j]);

      // Transform beta1 to [0.5, 1.5]
      beta1[i, j] = 0.5 + inv_logit(beta1_raw[i, j]);  

      // Transform beta0 to [0, 1]
      beta0[i, j] = inv_logit(beta0_raw[i, j]);
    }

    // Calculate delta_s1 and delta_s2 for likelihood using inv_logit to ensure values stay in [0, 1]
    for (i in 1:cycle) {
      int index = (j - 1) * cycle + i;
      delta_s1[index] = delta[trt1[index], j];

      if (trt1[index] == trt2[index]) {
        // Apply inv_logit to ensure the product stays in [0, 1]
        delta_s2[index] = inv_logit(delta[trt2[index], j] * beta1[link[index], j]);
      } else {
        // Apply inv_logit to ensure the product stays in [0, 1]
        delta_s2[index] = inv_logit(delta[trt2[index], j] * beta0[link[index] - 2, j]);
      }
    }
  }

  // Dynamic treatment regimens for each patient
  for (j in 1:J) {
    dtr_AB[j] = delta[1, j] * (delta[1, j] * beta1[1, j]) + (1 - delta[1, j]) * (delta[2, j] * beta0[1, j]);
    dtr_BA[j] = delta[2, j] * (delta[2, j] * beta1[2, j]) + (1 - delta[2, j]) * (delta[1, j] * beta0[2, j]);
  }
}

model {
  // Priors for the raw, unconstrained parameters
  for (j in 1:J) {
    delta_raw[1, j] ~ normal(0, 2);   
    delta_raw[2, j] ~ normal(0, 2);   

    beta1_raw[1, j] ~ normal(0, 2);  
    beta1_raw[2, j] ~ normal(0, 2);

    beta0_raw[1, j] ~ normal(0, 2);  
    beta0_raw[2, j] ~ normal(0, 2);
  }

  // Likelihood
  for (j in 1:J) {
    for (i in 1:cycle) {
      int index = (j - 1) * cycle + i;
      resp1[index] ~ bernoulli(delta_s1[index]);
      resp2[index] ~ bernoulli(delta_s2[index]);
    }
  }
}
