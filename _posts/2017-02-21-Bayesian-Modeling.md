---
title: "Bayesian Modeling of Customer Behaviour"
author: "__Aditya__"
date:   "21-Feb-2017"
layout: post
category: R
output:
  md_document:
    preserve_yaml: true
---

Hello All, In this post I will demonstrate how to model a simple
customer behaviour using Bayesian statistics. First we will simulate
hypothetical data based on few distributions and see how modeling helps
us to simulate the future behaviour of the customers. We will also look

Introduction
------------

For this post, lets assume an imaginary museum which runs ona
subscription renewal model. Let's also assume that currently there are
5000 members and the musuem captures some customer specific data like
gender, interest in photography and sculptures (this informations is
primarily captured during the registration process) and some behaviour
data like digital content consumption and months since last visit to the
museum.

First, we will simulate this data assuming certain underlying
distributions and we will use the `rstan` package to indetify the
parameters of those distributions.

Congiguring the `rstan` for modeling the data.

    require(rstan)

    ## Loading required package: rstan

    ## Loading required package: ggplot2

    ## Loading required package: StanHeaders

    ## rstan (Version 2.14.1, packaged: 2016-12-28 14:55:41 UTC, GitRev: 5fa1e80eb817)

    ## For execution on a local, multicore CPU with excess RAM we recommend calling
    ## rstan_options(auto_write = TRUE)
    ## options(mc.cores = parallel::detectCores())

    rstan_options(auto_write = TRUE)
    options(mc.cores = parallel::detectCores()-2)

Simulating the Data
-------------------

We will simulate a data set with 5000 customers drawn from a population
of museum members. We will define various columns as below.

-   
    *X*<sub>1</sub>
     `gender` : Approximately 60% are females.

-   
    *X*<sub>2</sub>
     `fotography` : 50% expressed interest for fotography.

-   
    *X*<sub>3</sub>
     `sculpture` : 30% expressed interest for sculpture.

-   
    *X*<sub>4</sub>
     `digital_content` : The average number of digital contents consumed
    by the members is 0.5 and the sample follows a poisson distribution.

-   
    *X*<sub>5</sub>
     `months_since` : The average length since last visit is 5 months
    and the sample follows a poisson distribution.

<!-- -->

    N <- 5000
    # we assume 5000 members

    female <- rbinom(N,1,0.6) 
    # simulate 60% of female

    fotography <- rbinom(N,1,0.5) 
    # simulate 50% of customers who expressed interest for fotography

    sculpture <- rbinom(N,1,0.3) 
    # simulate 30% of customers who expressed interest for sculpture

    digital_content <- rpois(N,0.5) 
    # simulate digital content

    months_since <- rpois(N,5) 
    # simulate months since last visit, with a max of 12
    months_since[months_since>12] <- 12

Simulating renewal behavior and prepare the data
------------------------------------------------

The next step is to simulate the renewal behaviour. Let denote that `Y`
(`renewal`) equals 1 if the member decides to renew, 0 otherwise.

We assume the following probit model for Y given X. (This can be any
abstract relationship. We are assuming this for simplicity.)

*P*(*Y* = 1|*X*<sub>1</sub>, *X*<sub>2</sub>, *X*<sub>3</sub>, *X*<sub>4</sub>, *X*<sub>5</sub>)=*Φ*(*β*<sub>0</sub> + *β*<sub>1</sub>*X*<sub>1</sub> + *β*<sub>2</sub>*X*<sub>2</sub> + *β*<sub>3</sub>*X*<sub>3</sub> + *β*<sub>4</sub>*X*<sub>4</sub> + *β*<sub>5</sub>*X*<sub>5</sub>)
 where

*Φ*(⋅)

is the cumulative distribution function (CDF) of the standard normal
distribution.

We assume the below relationship between `renewal` and the `betas`.

    beta0 <- 0.6
    beta1 <- 0.9
    beta2 <- 0.6
    beta3 <- 10
    beta4 <- -0.01
    beta5 <- -0.2

Then we compute the probablities of renewal based on the standard normal
distribution.

    prob_simul <- pnorm(beta0 + beta1*female + beta2*fotography + beta3*sculpture + beta4*digital_content + beta5*months_since)

Finally the actual renewal status is determined by the probability we
computed above.

    renewal <- rbinom(N,1,prob_simul) 

Until this step we completed simulating our data. Usually this data is
already available and the actual modeling procedure starts from now.

Prepare data for Stan
---------------------

We will use the popular `rstan` package for Bayesian modeling our data.
[Stan](http://mc-stan.org/) is a popular programming language for
statistical modeling. Using Stan we can simulate the samples of
parameters from a complex distribution defined by the user. In this
case, our models is defined by `betas` hence the output of Stan
simulation will be the distribution of `betas`.

We first create an X matrix that combines all the input variables,
including a column corresponding to the intercept.

    X <- cbind(rep(1,N), # Intercept column
               female,
               fotography,
               sculpture,
               digital_content,
               months_since)

    # assign column names
    colnames(X) <- c("intercept",
                     "female",
                     "fotography",
                     "sculpture",
                     "digital_content",
                     "months_since")
    K<-dim(X)[2]

Code for Stan
-------------

Stan uses a specific modeling syntax. It requires the specification of
the types of data and parameters, in addition to model statements.

    probit <- '
    data{
      int<lower=0> N; # number of observations
      int<lower=0> K; # number of parameters

      int<lower=0,upper=1> y[N];
      vector[K] X[N];
    }
    parameters{
      vector[K] beta;
    }
    model{
      beta ~ cauchy(0,5);

    for(j in 1:N)
      y[j] ~ bernoulli(Phi_approx(dot_product(X[j],beta)));
    }
    '

Run STAN
--------

The initialization step of stan may take a little while but the running
time should be just a couple of minutes.

SUMMARY of results
------------------

In the summary of output, the posterior distribution for each model
coefficient is summaried using its percentiles.

    print(fit)

    ## Inference for Stan model: cf5de9320c057d6ba79f5297d8079d34.
    ## 2 chains, each with iter=4000; warmup=2000; thin=1; 
    ## post-warmup draws per chain=2000, total post-warmup draws=4000.
    ## 
    ##             mean se_mean    sd     2.5%      25%      50%      75%
    ## beta[1]     0.62    0.01  0.07     0.48     0.57     0.61     0.67
    ## beta[2]     0.92    0.00  0.05     0.82     0.89     0.92     0.95
    ## beta[3]     0.59    0.00  0.05     0.50     0.56     0.59     0.62
    ## beta[4]    25.29    4.18 51.25     3.81     5.81    10.12    21.85
    ## beta[5]     0.01    0.00  0.03    -0.05    -0.01     0.01     0.04
    ## beta[6]    -0.20    0.00  0.01    -0.23    -0.21    -0.20    -0.19
    ## lp__    -1873.67    0.18  2.32 -1879.11 -1874.95 -1873.25 -1871.94
    ##            97.5% n_eff Rhat
    ## beta[1]     0.75   122 1.00
    ## beta[2]     1.02   109 1.01
    ## beta[3]     0.68   176 1.02
    ## beta[4]   156.92   151 1.00
    ## beta[5]     0.08   242 1.00
    ## beta[6]    -0.18   107 1.01
    ## lp__    -1870.35   175 1.00
    ## 
    ## Samples were drawn using NUTS(diag_e) at Wed Mar 22 11:45:41 2017.
    ## For each parameter, n_eff is a crude measure of effective sample size,
    ## and Rhat is the potential scale reduction factor on split chains (at 
    ## convergence, Rhat=1).

    fitlist <- extract(fit)

### Convergence

Convergence plots show whether the draws from posterior distributions
are well mixed together.

    par(mfrow=c(2,3))
    plot(fitlist$beta[,1],type="l",xlab="",ylab="Intercept")
    plot(fitlist$beta[,2],type="l",xlab="",ylab="Gender Coefficient")
    plot(fitlist$beta[,3],type="l",xlab="",ylab="Fotography Coefficient")
    plot(fitlist$beta[,4],type="l",xlab="",ylab="Sculpture Coefficient")
    plot(fitlist$beta[,5],type="l",xlab="",ylab="Digital content Coefficient")
    plot(fitlist$beta[,6],type="l",xlab="",ylab="Months since login")

![](2017-02-21-Bayesian-Modeling_files/figure-markdown_strict/unnamed-chunk-10-1.png)

### Histograms of posterior distributions

Posterior distributions of model coefficients.

    par(mfrow=c(2,3))  
    par(bg="white", fg="black", 
        col.lab="black", col.axis="black", 
        col.main="black", lwd=1)
    #Gender
    param <- fitlist$beta[,2]
    min <- min(0,min(param))
    max <- max(0,max(param))
    hist(param,breaks=seq(min-0.05,max+0.05,0.05),main="Female",xlab="", ylab="",yaxt='n')
    axis(1, lab=T , lwd=2)
    abline(v=beta1, col=2)
    #PHOTO
    param <- fitlist$beta[,3]
    min <- min(0,min(param))
    max <- max(0,max(param))
    hist(param,breaks=seq(min-0.04,max+0.04,0.04),main="Photography",xlab="", ylab="",yaxt='n')
    axis(1, lab=T , lwd=2)
    abline(v=beta2, col=2)
    #SCULPTURE
    param <- fitlist$beta[,4]
    min <- min(0,min(param))
    max <- max(0,max(param))
    hist(param,breaks=seq(min-0.03,max+0.03,0.03),main="Sculpture",xlab="", ylab="",yaxt='n')
    axis(1, lab=T , lwd=2)
    abline(v=beta3, col=2)
    #ONLINE CONTENT
    param <- fitlist$beta[,5]
    min <- min(0,min(param))
    max <- max(0,max(param))
    hist(param,breaks=seq(min-0.02,max+0.02,0.02),main="Downloads",xlab="", ylab="",yaxt='n')
    axis(1, lab=T , lwd=2)
    abline(v=beta4, col=2)
    #VISIT
    param <- fitlist$beta[,6]
    min <- min(0,min(param))
    max <- max(0,max(param))
    hist(param,breaks=seq(min-0.01,max+0.01,0.01),main="Months since visit",xlab="", ylab="",yaxt='n')
    axis(1, lab=T , lwd=2)
    abline(v=beta5, col=2)

![](2017-02-21-Bayesian-Modeling_files/figure-markdown_strict/unnamed-chunk-11-1.png)

Next Steps
----------

The complexity in the model can be increased further. Specifically, we
can build a model which is either hierarchical or multivariate. I will
try to do that in the future posts.

Please leave a comment if you want to discuss further. I will be glad to
help.

Acknowledgements: This post is a replication of one of the exercises
from the Columbia Statistics
[course](https://github.com/tz33cu/ColumbiaX-Statistical-Thinking-for-Data-Science/blob/master/R%20Learning%20Activities/LearningActivity-BDA.Rmd)