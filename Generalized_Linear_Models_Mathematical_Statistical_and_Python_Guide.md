# Generalized Linear Models: A Mathematical, Statistical, and Python Guide

*From Linear Regression to Research-Level Modelling of Binary, Count, Rate, and Positive Outcomes*

Prepared as a self-study, NotebookLM, and AI-tutor source

> Source and version note. Mathematical claims are grounded in the foundational GLM literature. Library examples were checked against current official statsmodels and scikit-learn documentation available in July 2026. APIs may change, so reproducible projects should record exact package versions.


## Contents at a glance

- Part 0: probability, calculus, linear algebra, Python, and numerical prerequisites.

- Part I: exponential families, links, likelihood, inference, deviance, diagnostics, offsets, and prediction.

- Part II: Gaussian, binomial, Poisson, negative-binomial, Gamma, inverse-Gaussian, Tweedie, ordinal, and zero-modified models.

- Part III: a complete pure-Python IRLS implementation before any third-party library.

- Part IV: NumPy, SciPy, pandas, statsmodels, scikit-learn, and matplotlib workflows.

- Part V: GEE, mixed models, Bayesian GLMs, penalization, missing data, survey design, causal interpretation, and transportability.

- Part VI: learning sequence, exercises, NotebookLM prompts, paper-reading method, glossary, and annotated research papers.

- Appendices: full executable module, verification tests, and research project template.


## How to use this guide

This volume is the next topic after linear regression. It assumes that the reader has encountered fitted values, residuals, matrices, likelihood at an introductory level, and Python functions. It nevertheless rebuilds the required probability, calculus, and numerical foundations so that the material can be studied from zero.

The order is deliberate:

- Part 0 repairs prerequisite gaps in mathematics, statistics, and Python.

- Part I develops generalized linear models from first principles.

- Part II studies major response families and their interpretation.

- Part III implements GLMs from scratch using only the Python standard library.

- Part IV uses NumPy, pandas, SciPy, statsmodels, scikit-learn, and matplotlib.

- Part V moves from competent application to research practice.

- Part VI provides exercises, projects, NotebookLM prompts, a paper-reading method, an expanded glossary, and an annotated reading list.

Do not rush to library code. A research-level user should be able to explain why the likelihood has its form, derive the score for at least logistic and Poisson regression, connect Fisher scoring to weighted least squares, identify when a model is misspecified, and reproduce the main results with an independent implementation.


## What a generalized linear model is

A generalized linear model, or GLM, combines three components:

1. A probability distribution for the response, usually from the exponential family.
2. A linear predictor eta = X beta.
3. A link function g connecting the conditional mean mu to the linear predictor: g(mu) = eta.

This architecture includes ordinary Gaussian regression, logistic regression, Poisson regression, Gamma regression, inverse-Gaussian regression, and many related models. The word generalized refers to generalizing the response distribution and mean-variance relationship while retaining a linear predictor.


# Part 0. Prerequisites Rebuilt from Scratch


## 0.1 Outcomes are distributions, not just numbers

In ordinary regression it is easy to focus on the fitted mean and forget that a model describes a distribution. GLMs make the distribution explicit. A binary response can only be zero or one. A count response is non-negative and discrete. A cost or duration may be positive and strongly skewed. These constraints affect the possible means, variances, likelihoods, and predictions.

For each observation i, a GLM describes a conditional distribution of Y_i given predictors X_i. The key conditional quantities are:

```
mu_i = E[Y_i | X_i]
```

```
Var(Y_i | X_i) = phi V(mu_i) / w_i
```

Here V(mu) is a variance function, phi is a scale or dispersion parameter, and w_i may be a prior or frequency weight depending on the model.


## 0.2 Probability mass, density, and likelihood

For a discrete variable, a probability mass function gives P(Y=y). For a continuous variable, a density describes probability per unit of the outcome scale; probabilities arise by integrating the density over an interval.

A likelihood treats the observed data as fixed and the parameters as variable. If independent observations have densities or masses f(y_i | theta), the likelihood is

```
L(theta) = product_i f(y_i | theta)
```

The log-likelihood is

```
ell(theta) = sum_i log f(y_i | theta)
```

The logarithm is monotone, so maximizing L and ell gives the same parameter estimate. The sum is numerically and algebraically easier than a product of many small probabilities.


## 0.3 Expectation, variance, and the law of total variance

Expectation is a probability-weighted average. Variance measures average squared deviation from expectation. Conditional expectation and conditional variance depend on predictors.

The law of total expectation is

```
E[Y] = E[E[Y | X]]
```

The law of total variance is

```
Var(Y) = E[Var(Y | X)] + Var(E[Y | X])
```

The first term is average conditional noise. The second is variation in conditional means across predictor values. This decomposition helps explain why a population may be highly variable even if each conditional distribution is relatively concentrated.


## 0.4 Bernoulli, binomial, Poisson, Gamma, and Gaussian distributions

A Bernoulli variable takes values zero and one with probability p of one:

```
P(Y=y) = p^y (1-p)^(1-y)
```

Its mean is p and variance is p(1-p).

A binomial variable counts successes in m trials:

```
P(Y=y) = choose(m,y) p^y (1-p)^(m-y)
```

Its mean is mp and variance is mp(1-p).

A Poisson variable counts events with mean mu:

```
P(Y=y) = exp(-mu) mu^y / y!
```

Its variance also equals mu. This equality is a modelling restriction, not a universal law of counts.

A Gamma variable is positive and continuous. In the GLM parameterization its variance is often written

```
Var(Y | X) = phi mu^2
```

The Gaussian distribution has constant conditional variance phi under the simplest linear model.


## 0.5 Logs, odds, and exponentials

The odds corresponding to probability p are

```
odds = p / (1-p)
```

The logit is the log odds:

```
logit(p) = log[p/(1-p)]
```

Solving for p gives the logistic inverse link:

```
p = 1 / (1 + exp(-eta))
```

The log link is

```
log(mu) = eta
```

and therefore

```
mu = exp(eta)
```

The exponential guarantees a positive mean and turns additive coefficient changes on the eta scale into multiplicative changes on the outcome mean scale.


## 0.6 Derivatives required for GLMs

Core derivatives are:

```
d exp(x)/dx = exp(x)
```

```
d log(x)/dx = 1/x
```

```
d sigmoid(x)/dx = sigmoid(x)[1-sigmoid(x)]
```

The chain rule is essential because beta affects eta, eta affects mu, and mu affects the likelihood:

```
d ell / d beta = (d ell/d mu)(d mu/d eta)(d eta/d beta)
```

Since eta_i = x_i^T beta, the derivative of eta_i with respect to beta is x_i.


## 0.7 Gradient, Hessian, and curvature

For a parameter vector beta, the score is the gradient of the log-likelihood:

```
U(beta) = d ell(beta) / d beta
```

The Hessian is the matrix of second derivatives:

```
H(beta) = d^2 ell(beta) / d beta d beta^T
```

At a well-behaved maximum, the Hessian is negative definite. Newton-Raphson updates beta using observed curvature:

```
beta_new = beta_old - H(beta_old)^(-1) U(beta_old)
```

Fisher scoring replaces the negative Hessian with its expectation, the Fisher information I(beta):

```
beta_new = beta_old + I(beta_old)^(-1) U(beta_old)
```

For GLMs, Fisher scoring can be expressed as iteratively reweighted least squares.


## 0.8 Matrix dimensions

Let n be observations and p predictors. With an intercept, X is n by k where k = p+1. Beta is k by 1. Eta and mu are n by 1.

```
eta = X beta
```

The weighted normal equations use a diagonal n by n weight matrix W:

```
X^T W X beta_new = X^T W z
```

The matrix X^T W X is k by k. Checking dimensions catches many derivation and coding errors before numerical work begins.


## 0.9 Python foundations and defensive programming

Pure Python in this guide means the standard library only. Matrices are lists of row lists. Numerical functions validate dimensions, reject impossible responses, and raise informative exceptions.

A research implementation should separate:

- family definitions;

- matrix operations;

- fitting logic;

- diagnostics;

- prediction;

- evaluation;

- demonstrations and tests.

Never hide a convergence failure. A returned model should include a Boolean convergence flag, iteration count, and objective history.


## 0.10 Numerical stability

Probabilities very close to zero cause log(0). Large positive exponentials overflow. A stable sigmoid evaluates positive and negative inputs differently. Log-likelihood code clips probabilities only at machine-safe boundaries and documents the clipping.

IRLS can fail because of separation, rank deficiency, extreme weights, poor starting values, or a misspecified link. Production libraries use mature linear algebra, step control, and diagnostics. From-scratch code is for understanding and verification.


# Part I. Mathematical and Statistical Foundations


## 1. Why ordinary linear regression is not enough

A linear probability model can predict below zero or above one. A Gaussian model for counts can predict negative values and ignores the typical connection between count mean and variance. Constant-variance errors are often implausible for positive costs whose spread grows with their mean.

A GLM solves three related problems:

- it chooses a distribution whose support matches the outcome;

- it models how variance changes with the mean;

- it uses a link that maps allowable means to the unrestricted real line of the linear predictor.

The GLM is still a conditional model. It does not automatically solve confounding, selection, dependence, measurement error, or causal identification.


## 2. The three components of a GLM


### 2.1 Random component

The conditional distribution belongs to an exponential-family form. This determines the variance function and likelihood contribution.


### 2.2 Systematic component

Predictors enter through

```
eta_i = beta_0 + beta_1 x_i1 + ... + beta_p x_ip
```

Transformations, indicators, splines, and interactions can all appear as columns of X.


### 2.3 Link component

The link connects the mean to eta:

```
g(mu_i) = eta_i
```

The inverse link gives predictions:

```
mu_i = g^(-1)(eta_i)
```

A link is not merely a computational trick. It determines the scale on which predictors combine additively and coefficients have direct interpretation.


## 3. Exponential-family form

A one-parameter exponential-dispersion family can be written as

```
f(y | theta, phi) = exp{[y theta - b(theta)]/a(phi) + c(y,phi)}
```

The natural or canonical parameter is theta. The function b(theta) controls moments. Under regularity conditions:

```
E[Y] = b'(theta) = mu
```

```
Var(Y) = a(phi) b''(theta)
```

When a(phi)=phi/w, the variance is phi V(mu)/w.

This form is powerful because likelihood derivatives have a shared structure across Gaussian, binomial, Poisson, Gamma, and inverse-Gaussian models.


## 4. Deriving the mean and variance from the cumulant function

The density or mass integrates or sums to one. Differentiating this normalization identity with respect to theta yields the expectation of the score equal to zero. From that identity one obtains b'(theta)=E[Y]. Differentiating again gives b''(theta) proportional to Var(Y).

The derivation illustrates a general statistical principle: normalization constraints create moment identities. The cumulant function is not an arbitrary decoration; it encodes the distribution's mean and curvature.


## 5. Canonical links

A canonical link sets

```
eta = theta
```

Examples are:

- Gaussian: identity link;

- Bernoulli/binomial: logit link;

- Poisson: log link;

- Gamma: inverse link in the classical canonical parameterization;

- inverse Gaussian: inverse-squared link.

Canonical links often simplify score equations and make the sufficient statistic align with X^T y. They are not always the most interpretable or numerically convenient choice. Gamma regression often uses a log link to obtain multiplicative interpretations and positive means.


## 6. The score function for a GLM

For independent observations, the derivative of the log-likelihood contribution with respect to beta can be expressed as

```
U(beta) = X^T D V^(-1) (y - mu) / phi
```

Here D is diagonal with elements d mu_i / d eta_i and V is diagonal with variance-function values V(mu_i), adjusted for prior weights.

The score equals zero at a maximum-likelihood solution under regularity conditions. It resembles a weighted correlation between predictors and residuals. Unlike OLS, raw residuals need not be orthogonal to X; the orthogonality is weighted and transformed by D and V.


## 7. Fisher information

The expected information is

```
I(beta) = X^T W X / phi
```

where

```
w_i = [d mu_i/d eta_i]^2 / V(mu_i)
```

with any prior weights included. The approximate covariance of the maximum-likelihood estimator is

```
Var_hat(beta_hat) = phi_hat (X^T W X)^(-1)
```

This is the model-based covariance. Sandwich covariance can be used when the variance structure is misspecified but the mean model and independence conditions remain adequate.


## 8. IRLS derivation

At the current beta, define a working response

```
z_i = eta_i + (y_i - mu_i)/(d mu_i/d eta_i)
```

and working weight

```
w_i = [d mu_i/d eta_i]^2 / V(mu_i)
```

Then the Fisher-scoring update solves

```
beta_new = argmin_beta sum_i w_i (z_i - x_i^T beta)^2
```

Therefore a nonlinear maximum-likelihood problem becomes a sequence of weighted least-squares problems. The response z and weights W change after each iteration.

IRLS is an algorithm, not a new statistical model. Convergence of the algorithm does not prove model adequacy. Nonconvergence is a diagnostic signal that must be investigated.


## 9. Maximum likelihood and asymptotic inference

Under identification, regularity, and correct specification, maximum-likelihood estimators are consistent, asymptotically normal, and efficient within the model:

```
sqrt(n)(beta_hat-beta) -> Normal(0, I(beta)^(-1))
```

Finite samples may behave poorly under rare outcomes, separation, high leverage, weak information, or many predictors. Research reporting should include sample structure and event counts, not only the nominal n.


## 10. Wald, likelihood-ratio, and score tests

A Wald test uses the distance between an estimate and a null value divided by its standard error. For one coefficient:

```
z = (beta_hat_j - beta_j0)/SE(beta_hat_j)
```

A likelihood-ratio test compares maximized log-likelihoods:

```
LR = 2[ell_unrestricted - ell_restricted]
```

A score test evaluates the gradient at the restricted estimate without fitting the unrestricted model fully.

The three tests are asymptotically equivalent under regularity but can differ in finite samples. Wald tests are easiest to compute but can behave badly for strongly nonlinear parameterizations or estimates near boundaries.


## 11. Confidence intervals and transformation

A coefficient-scale Wald interval is

```
beta_hat_j +/- z_critical SE(beta_hat_j)
```

For a log-link coefficient, exponentiating the endpoints gives an interval for a multiplicative mean ratio. For a logit coefficient, exponentiation gives an odds-ratio interval.

Intervals for predicted means can be formed on the linear-predictor scale and transformed through the inverse link. This preserves allowable ranges and naturally creates asymmetric intervals on the outcome scale.


## 12. Deviance

The saturated model fits each observation as closely as the distribution permits. Deviance compares the fitted model with the saturated model:

```
D = 2[ell_saturated - ell_fitted]
```

Smaller deviance indicates closer fit relative to saturation. Residual deviance is not a universal goodness-of-fit test without attention to distribution, dispersion, sample size, sparsity, and asymptotic conditions.

For nested models, a deviance difference can support a likelihood-ratio test when both are maximum-likelihood fits to the same observations with compatible distributional assumptions.


## 13. Null deviance, pseudo-R-squared, AIC, and BIC

Null deviance comes from an intercept-only model. Comparing it with residual deviance indicates improvement over a constant conditional mean.

Many quantities are called pseudo-R-squared. They do not share the exact OLS variance-decomposition meaning. Always name the definition used.

AIC is

```
AIC = -2 ell(beta_hat) + 2k
```

BIC is

```
BIC = -2 ell(beta_hat) + k log(n)
```

They compare likelihood-based models fitted to the same response observations. They do not replace validation, calibration, or substantive judgement.


## 14. Dispersion and scale

For binomial and Poisson GLMs, the classical model fixes phi=1. For Gaussian, Gamma, and inverse-Gaussian models, phi is estimated.

Pearson dispersion can be estimated by

```
phi_hat = sum_i (y_i-mu_i)^2 / V(mu_i) divided by (n-k)
```

Overdispersion means observed conditional variation exceeds the nominal model variance. Underdispersion means it is smaller. Overdispersion can arise from omitted heterogeneity, dependence, excess zeros, measurement problems, or an inappropriate distribution.

Changing a standard error by estimating an overdispersion factor does not automatically repair a wrong likelihood or mean model.


## 15. Residuals in GLMs

Response residual:

```
r_i = y_i - mu_hat_i
```

Pearson residual:

```
r_Pi = (y_i-mu_hat_i)/sqrt[V(mu_hat_i)]
```

Deviance residual is the signed square root of the observation's deviance contribution.

Working residuals arise inside IRLS. Anscombe and randomized quantile residuals aim to create distributions more suitable for diagnostic plots.

Residual plots must respect discreteness and changing variance. A cloud of binary response residuals has limited visual resolution; binned residual plots, calibration plots, or simulation-based diagnostics may be more informative.


## 16. Leverage and influence

GLM leverage is computed from the final weighted design:

```
H_W = W^(1/2) X (X^T W X)^(-1) X^T W^(1/2)
```

The diagonal h_i measures leverage in the local weighted least-squares approximation. Influence combines leverage, residual size, and model curvature.

Case-deletion diagnostics are approximations. Observations with high influence should be checked for data quality, structural uniqueness, and sensitivity. Deletion requires substantive justification.


## 17. Offsets, exposure, and rates

An offset is a predictor with coefficient fixed at one. In a Poisson rate model with exposure t_i:

```
log(mu_i) = log(t_i) + x_i^T beta
```

Then

```
mu_i = t_i exp(x_i^T beta)
```

The model describes a rate per exposure unit while predicting counts for each observed exposure. Treating log exposure as an estimated ordinary predictor answers a different question and may create bias if the coefficient should be fixed by definition.


## 18. Frequency, prior, analytic, and sampling weights

Weights have distinct interpretations. Frequency weights represent replicated observations. Binomial trial totals determine the precision of proportions. Analytic or variance weights encode known relative precision. Sampling weights arise from inclusion probabilities.

A software argument named weights may implement only one of these meanings. Research users must read documentation and confirm how weights enter the likelihood, scale, and covariance.


## 19. Prediction and uncertainty

A GLM predicts the conditional mean mu, not necessarily an individual outcome. Individual predictive distributions require the selected response family and dispersion.

Prediction uncertainty has at least two layers:

- uncertainty in beta and therefore in the conditional mean;

- outcome variation conditional on the mean.

For discrete outcomes, prediction intervals are sets of integer values obtained from the predictive distribution. For binary outcomes, the individual future outcome remains zero or one even when the estimated probability is precise.


## 20. Model misspecification and sandwich covariance

If the conditional mean is correctly parameterized but the variance function is wrong, estimating equations may still target a meaningful mean-model parameter. A sandwich covariance can provide asymptotically valid uncertainty under independence and suitable regularity.

The sandwich does not correct the fitted mean, link, omitted confounding, dependence, or separation. It also may perform poorly with small samples or high leverage. Model-based and robust results should be compared and their assumptions stated.


# Part II. Major Families and Interpretation


## 21. Gaussian GLM

The Gaussian identity-link GLM is ordinary linear regression estimated by maximum likelihood. With constant variance, maximizing likelihood is equivalent to minimizing SSE.

Using the Gaussian family with a log link creates a positive conditional mean but still assumes Gaussian outcome noise around that mean, including possible negative observations. Distribution and link are separate choices.


## 22. Binary logistic regression

For Y in {0,1}:

```
log[p_i/(1-p_i)] = x_i^T beta
```

A one-unit increase in x_j multiplies the odds by exp(beta_j), holding other predictors fixed. It does not multiply probability by that amount.

The marginal effect on probability is

```
d p_i / d x_ij = beta_j p_i(1-p_i)
```

Therefore probability effects vary by observation. Report predicted probabilities, discrete changes, or average marginal effects when odds ratios are not substantively intuitive.


## 23. Grouped binomial models

If y_i successes occur in m_i trials, model the proportion y_i/m_i with trial count m_i or pass successes and failures in a supported format. Larger m_i gives more information.

Grouped-binomial data assume a common success probability within the group conditional on predictors. Unmodelled heterogeneity can produce overdispersion relative to binomial variance.


## 24. Separation in logistic regression

Complete separation occurs when a predictor combination perfectly distinguishes outcomes. The likelihood can improve as some coefficients diverge, so finite maximum-likelihood estimates do not exist.

Warning signs include huge coefficients, huge standard errors, nonconvergence, and predicted probabilities essentially zero or one. Remedies include better data, removing deterministic leakage, penalized likelihood, Bayesian priors, or Firth bias reduction. Deleting a predictor merely to force convergence may discard the scientific phenomenon causing separation.


## 25. Rare events and small samples

When events are scarce, nominal sample size overstates information. Models with many parameters per event are unstable. Cross-validation folds may contain few or no events.

Use parsimonious models, penalization, careful calibration assessment, and uncertainty methods designed for small samples. Threshold metrics such as accuracy are especially misleading under rare prevalence.


## 26. Poisson regression for counts

For a Poisson mean mu_i:

```
log(mu_i) = x_i^T beta
```

A one-unit increase in x_j multiplies the expected count by exp(beta_j). With an exposure offset, exp(beta_j) is an incidence-rate ratio.

The Poisson model assumes conditional mean equals conditional variance. This must be checked. A good mean fit can coexist with overdispersion that invalidates model-based standard errors.


## 27. Count offsets and exposure examples

Examples of exposure include person-time, kilometres driven, number of households observed, or days at risk. The exposure must represent proportional opportunity for events under the model.

If doubling exposure does not plausibly double expected count, a simple offset may be wrong. Consider nonlinear exposure effects, saturation, varying risk periods, or a different outcome definition.


## 28. Overdispersion and quasi-Poisson reasoning

If Var(Y|X)=phi mu with phi>1, coefficient estimates from the Poisson score may remain useful for the mean, while standard errors are multiplied approximately by sqrt(phi).

Quasi-likelihood specifies a mean-variance relationship without a full probability distribution. It supports estimating equations and robust uncertainty but does not provide an ordinary full likelihood for AIC comparison.

Overdispersion is a symptom. Investigate omitted predictors, clustering, excess zeros, contagion, and heterogeneous event rates rather than only inflating standard errors.


## 29. Negative binomial regression

A common negative-binomial model has

```
Var(Y|X) = mu + alpha mu^2
```

It can arise as a Poisson-Gamma mixture with unobserved rate heterogeneity. Compared with Poisson, it allows variance to grow faster than the mean.

Different software uses different parameterizations. Always report the variance formula and definition of alpha. The log link usually gives incidence-rate-ratio interpretations similar to Poisson.


## 30. Zero-inflated and hurdle models

A zero-inflated model combines a structural-zero process with a count process that can also generate zeros. A hurdle model first models zero versus positive and then models positive counts with a zero-truncated distribution.

These models answer different questions. Excess observed zeros do not automatically prove a distinct structural-zero class. Compare substantive mechanisms, residual patterns, predictive calibration, and simpler overdispersed alternatives.


## 31. Gamma regression

Gamma GLMs are useful for positive continuous outcomes whose variance grows roughly with the square of the mean, such as costs or durations.

With a log link:

```
log(mu_i) = x_i^T beta
```

and exp(beta_j) is a conditional mean ratio. Zero outcomes are not allowed by the Gamma distribution. If zeros are genuine, use a two-part model, Tweedie family, or another distribution that represents the mechanism.


## 32. Inverse-Gaussian regression

The inverse-Gaussian distribution is positive and more strongly right-skewed than many Gamma distributions. Its variance function is proportional to mu^3.

It can model waiting times and other positive outcomes with rapidly increasing variance. Choice between Gamma and inverse Gaussian should be based on data-generating considerations and diagnostic evidence, not only AIC.


## 33. Tweedie models

Tweedie variance functions have

```
Var(Y|X) = phi mu^p
```

The power p identifies important cases: p=0 Gaussian, p=1 Poisson, p=2 Gamma, and p=3 inverse Gaussian. For 1<p<2, the compound Poisson-Gamma distribution has a point mass at zero plus positive continuous values, useful for claim costs and similar semicontinuous outcomes.

Power estimation and interpretation require care. A Tweedie model can fit support well while hiding distinct processes that a two-part model would explain more clearly.


## 34. Multinomial and ordinal logistic regression

Multinomial logistic regression models unordered categories relative to a reference. Ordinal models exploit category ordering, commonly through proportional odds.

The proportional-odds assumption states that predictor effects are constant across cumulative category cut points on the log-odds scale. Test and diagnose it substantively and statistically. These models extend beyond the simplest scalar-response GLM architecture but are natural next steps.


## 35. Link-function alternatives

Binary models may use probit or complementary log-log links. Probit arises from a latent Gaussian threshold interpretation. Complementary log-log is asymmetric and connects naturally to discrete-time hazards.

Poisson and Gamma models can use identity or inverse links under constraints, but predictions must remain in the allowed mean domain. Select links based on mechanism, interpretation, fit, and stability.


# Part III. GLMs from Scratch in Pure Python


## 36. Design principles

The companion module glm_from_scratch.py uses only dataclasses, math, random, and typing. It implements a generic IRLS engine with family objects defining:

- inverse link;

- derivative dmu/deta;

- variance function;

- deviance contribution;

- log-likelihood contribution;

- response-domain validation;

- scale treatment and initial mean.

This design separates shared optimization logic from family-specific mathematics.


## 37. Stable logistic functions

A naive sigmoid exp(eta)/(1+exp(eta)) overflows for large eta. A stable implementation is:

```python
def stable_sigmoid(eta):
    if eta >= 0.0:
        z = math.exp(-eta)
        return 1.0 / (1.0 + z)
    z = math.exp(eta)
    return z / (1.0 + z)
```

The formulas are algebraically identical but numerically safer in different regions.


## 38. Weighted least squares core

IRLS repeatedly solves:

```
(X^T W X) beta = X^T W z
```

The pure implementation forms weighted cross-products directly without constructing a large diagonal W matrix.

```python
def weighted_crossproduct(X, weights):
    n = len(X)
    p = len(X[0])
    result = [[0.0 for _ in range(p)] for _ in range(p)]
    for i, row in enumerate(X):
        for a in range(p):
            for b in range(p):
                result[a][b] += weights[i] * row[a] * row[b]
    return result
```

Production software should use stable factorization rather than explicit matrix inversion.


## 39. Generic IRLS algorithm

The central loop is:

```python
for iteration in range(max_iterations):
    eta = X_beta_plus_offset(X, beta, offset)
    mu = [family.inverse_link(v) for v in eta]
    d = [family.dmu_deta(v) for v in eta]
    var = [family.variance(m) for m in mu]
    weights = [prior[i] * d[i] ** 2 / var[i] for i in range(n)]
    z = [eta[i] + (y[i] - mu[i]) / d[i] - offset[i]
         for i in range(n)]
    beta_new = weighted_least_squares(X, z, weights)
```

The full module adds domain checks, ridge stabilization, convergence criteria, diagnostics, covariance, and result objects.


## 40. Logistic regression from scratch

```python
from glm_from_scratch import fit_glm_irls, simulate_logistic

X, y = simulate_logistic(seed=42, n=300)
model = fit_glm_irls(X, y, family="binomial")

print(model.coefficients)
print(model.standard_errors)
print(model.converged, model.iterations)
print(model.predict_mean([[0.0, 1.0]]))
```

The implementation matches statsmodels estimates on well-conditioned test data to numerical tolerance.


## 41. Poisson rates with an offset

```python
import math
from glm_from_scratch import fit_glm_irls, simulate_poisson

X, y, exposure = simulate_poisson(seed=42, n=300)
offset = [math.log(value) for value in exposure]
model = fit_glm_irls(X, y, family="poisson", offset=offset)

print(model.coefficients)
print(model.deviance)
```

Predictions with future exposures require the corresponding log exposures as offsets.


## 42. Gamma regression from scratch

```
model = fit_glm_irls(X, positive_y, family="gamma")
print("mean ratios", [math.exp(value) for value in model.coefficients[1:]])
print("estimated scale", model.scale)
```

The companion implementation uses a log link for Gamma. It estimates scale from Pearson residuals and calculates a Gamma log-likelihood using shape=1/phi and scale=phi*mu.


## 43. Deviance and residual computation

The module stores response, Pearson, and deviance residuals. For Poisson:

```
d_i = 2[y_i log(y_i/mu_i) - (y_i-mu_i)]
```

with the y_i log term defined as zero when y_i=0. The deviance residual is sign(y_i-mu_i)sqrt(d_i).

Always handle mathematical limits explicitly rather than allowing log(0) errors.


## 44. Leverage and Cook-style influence

At convergence, construct weighted rows sqrt(w_i)x_i and compute

```
h_i = w_i x_i^T (X^T W X)^(-1) x_i
```

The companion Cook-style measure is a screening diagnostic based on Pearson residual and leverage. It is not a substitute for exact case deletion.


## 45. Sandwich covariance from scratch

Observation score contributions have the form

```
s_i = x_i (y_i-mu_i)(dmu_i/deta_i)/V(mu_i)
```

A sandwich estimator is

```
A^(-1) [sum_i s_i s_i^T] A^(-1)
```

where A is the sensitivity matrix. The module includes HC0, HC1, and leverage-adjusted HC3-style variants for educational comparison.


## 46. Cross-validation metrics by family

Use metrics aligned with the predictive distribution:

- binary: log loss, Brier score, calibration, ROC AUC, precision-recall measures;

- Poisson/count: Poisson deviance and count calibration;

- Gamma/positive: Gamma deviance, mean absolute error, and domain-relevant relative error;

- Gaussian: RMSE, MAE, and R-squared.

Cross-validation must preserve groups, time, or spatial structure where relevant.


## 47. Verification tests

A from-scratch implementation should be tested against:

- known analytic special cases;

- independent software such as statsmodels;

- simulated data with known parameters;

- invalid inputs and boundary cases;

- convergence and nonconvergence examples;

- invariants such as positive means and probabilities in [0,1].

Agreement with a library on one dataset is necessary but not sufficient. Shared mistakes, loose tolerances, or untested branches can remain.


# Part IV. Python Libraries for Research Workflows


## 48. Division of labour among libraries

NumPy provides arrays and linear algebra. SciPy provides probability distributions and optimization. pandas manages tabular data. statsmodels emphasizes statistical models, covariance estimators, hypothesis tests, and results objects. scikit-learn emphasizes predictive pipelines, regularization, tuning, and generalization. matplotlib provides plots.

As of mid-2026, statsmodels documents one-parameter exponential-family GLMs with families, links, variance functions, and result classes. Scikit-learn provides regularized PoissonRegressor, GammaRegressor, TweedieRegressor, and LogisticRegression, with regularization defaults that differ from classical unpenalized inference.


## 49. NumPy implementation of one IRLS step

```python
import numpy as np

eta = X @ beta + offset
mu = 1.0 / (1.0 + np.exp(-eta))
d = mu * (1.0 - mu)
variance = mu * (1.0 - mu)
weights = d**2 / variance
z = eta + (y - mu) / d - offset

sqrt_w = np.sqrt(weights)
Xw = X * sqrt_w[:, None]
zw = z * sqrt_w
beta_new, *_ = np.linalg.lstsq(Xw, zw, rcond=None)
```

Use clipping or stable special functions for extreme eta. np.linalg.lstsq is generally preferable to explicit inversion.


## 50. pandas data audit

```
required = ["outcome", "age", "programme", "exposure", "district"]
df = pd.read_csv("analysis.csv", usecols=required)

print(df.info())
print(df.describe(include="all"))
print(df.isna().sum())
print(df.duplicated().sum())
print(df["district"].value_counts(dropna=False))
```

Check support restrictions: binary coding, non-negative integer counts, positive Gamma outcomes, and positive exposure. Audit impossible combinations before fitting.


## 51. statsmodels logistic regression

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

model = smf.glm(
    "outcome ~ age + programme + age:programme + C(district)",
    data=df,
    family=sm.families.Binomial(),
).fit()

print(model.summary())
print(model.get_robustcov_results(cov_type="HC3").summary())
```

Depending on statsmodels result class and version, robust covariance can also be requested at fit time. Confirm the current official API rather than relying on copied code.


## 52. statsmodels Poisson with exposure

```
model = smf.glm(
    "events ~ remoteness + training + C(region)",
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df["person_years"]),
).fit()

print(np.exp(model.params))
```

Exponentiated slopes are incidence-rate ratios under the log-link model. The intercept depends on the exposure unit and reference categories.


## 53. statsmodels Gamma GLM

```
model = smf.glm(
    "cost ~ distance + beneficiaries + C(project_type)",
    data=df,
    family=sm.families.Gamma(link=sm.families.links.Log()),
).fit()
```

statsmodels' default Gamma link may differ from the log link. Specify the intended link explicitly and report it.


## 54. Predictions and intervals

```
new_data = pd.DataFrame({
    "age": [30, 50],
    "programme": [0, 1],
    "district": ["A", "B"],
})
frame = model.get_prediction(new_data).summary_frame(alpha=0.05)
print(frame)
```

Inspect whether intervals are on the mean scale or linear-predictor scale and whether observation-level predictive intervals are supported for the family.


## 55. scikit-learn logistic pipelines

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric = ["age", "income"]
categorical = ["district"]

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), numeric),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical),
])

pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=5000)),
])
```

Scikit-learn applies regularization by default. Its coefficients are not the same inferential object as unpenalized maximum-likelihood coefficients unless penalty is disabled or negligible under a supported configuration.


## 56. PoissonRegressor, GammaRegressor, and TweedieRegressor

```python
from sklearn.linear_model import PoissonRegressor, GammaRegressor, TweedieRegressor

poisson = PoissonRegressor(alpha=0.0, max_iter=2000)
gamma = GammaRegressor(alpha=0.0, max_iter=2000)
tweedie = TweedieRegressor(power=1.5, alpha=0.1, link="log", max_iter=2000)
```

These estimators emphasize prediction and regularization. They do not provide classical coefficient p-values. Exposure handling may require modelling a rate with sample weights or a transformed target depending on the estimator; verify the current API and ensure the loss corresponds to the scientific target.


## 57. Calibration for binary predictions

```python
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

probability = pipeline.predict_proba(X_test)[:, 1]
print("log loss", log_loss(y_test, probability))
print("Brier", brier_score_loss(y_test, probability))
print("ROC AUC", roc_auc_score(y_test, probability))

observed, predicted = calibration_curve(y_test, probability, n_bins=10)
```

Calibration bins are descriptive and sensitive to binning. Use smooth calibration methods, bootstrap uncertainty, or cross-validated predictions for serious assessment.


## 58. Diagnostics with plots

Useful plots include:

- deviance or Pearson residuals versus fitted means;

- residuals versus important predictors;

- leverage and influence index plots;

- observed versus predicted counts by risk group;

- calibration curves for probabilities;

- rootograms for counts;

- randomized quantile residual Q-Q plots;

- partial residual or component-plus-residual plots when supported.

Plots should include enough context to reveal sparse cells, exposure variation, and influential groups.


## 59. Nested and non-nested model comparison

Likelihood-ratio tests require nested models and compatible maximum-likelihood fits. AIC and BIC can compare non-nested likelihood models fitted to the same observations. Cross-validation compares predictive performance but estimates a different target.

Quasi-likelihood models do not possess an ordinary full likelihood, so ordinary AIC is not directly available. Specialized criteria such as QIC are used in related settings.


## 60. Regularized GLMs

Penalized estimation solves

```
minimize -ell(beta) + lambda penalty(beta)
```

L2 penalties stabilize correlated predictors. L1 penalties create sparsity. Elastic net combines them.

Penalty selection belongs inside cross-validation. Standardize predictors when penalty treats coefficients symmetrically. Classical p-values after data-driven selection are not valid without post-selection methods or a separate confirmatory sample.


# Part V. Research-Level Extensions and Judgement


## 61. Mean model versus full distribution

A GLM can be used as a full likelihood model or as an estimating equation for the mean. These are different claims. Full-likelihood inference and predictive distributions require the selected family to be credible. Robust mean-model inference can tolerate some variance misspecification but not an incorrect mean or dependence structure.

State explicitly whether the family is substantive, a working variance model, or a computational device.


## 62. Generalized estimating equations

GEE extends GLM mean models to correlated observations such as repeated measures or clustered data. It specifies a marginal mean, variance function, and working correlation.

Regression coefficients have population-average interpretations. Sandwich covariance can remain consistent when the working correlation is wrong, given enough independent clusters and correct mean structure. Small-cluster corrections may be necessary.


## 63. Generalized linear mixed models

GLMMs add random effects to the linear predictor, for example:

```
g(mu_ij) = x_ij^T beta + z_ij^T b_j
```

They model cluster-specific heterogeneity and induce within-cluster dependence. Logistic GLMM coefficients are conditional on random effects and differ from marginal GEE coefficients.

Estimation requires integrating over random effects, using Laplace approximation, adaptive quadrature, variational methods, or simulation. Convergence and singular random-effect fits require careful diagnosis.


## 64. Bayesian GLMs

Bayesian analysis combines likelihood with priors:

```
p(beta | y) proportional to p(y | beta) p(beta)
```

Weakly informative priors can stabilize separation and rare-event models. Posterior intervals have direct conditional probability interpretations given model and prior.

Research reporting should include prior predictive checks, posterior predictive checks, sensitivity to plausible priors, convergence diagnostics, and effective sample sizes for simulation-based inference.


## 65. Firth bias reduction and penalized likelihood

Firth's method modifies likelihood using a Jeffreys-prior penalty to reduce first-order bias. In logistic regression it often produces finite estimates under separation.

Firth estimates answer a penalized-likelihood problem and should not be described as ordinary maximum likelihood. Profile penalized-likelihood intervals are generally preferable to naive Wald intervals.


## 66. Robustness to outliers and contamination

Likelihood-based GLMs can be sensitive to influential outcomes or covariate patterns. Robust GLM methods bound influence or downweight observations, but they change the estimand and require tuning choices.

Before applying robust methods, distinguish data error, valid rare cases, distributional heavy tails, and structural subpopulations. A model that ignores a real subgroup is not repaired merely by downweighting it.


## 67. Missing data

Complete-case GLM estimates are valid only under conditions tied to the missingness mechanism and model. Multiple imputation should respect outcome type, nonlinear terms, interactions, and clustering.

For prediction, imputation must occur inside each training fold. For inference, uncertainty from imputation and analysis must be combined. Sensitivity analysis is required for plausible missing-not-at-random mechanisms.


## 68. Survey and case-control designs

Sampling changes interpretation. In case-control studies, logistic slopes can estimate odds ratios under appropriate sampling even though the intercept and prevalence are distorted. Probability calibration requires prevalence correction or population data.

Complex surveys require design weights, strata, and primary sampling units in variance estimation. A standard GLM with a weight column is not automatically design-correct.


## 69. Causal interpretation

A GLM coefficient is an adjusted association unless a design and assumptions identify a causal estimand. Nonlinearity creates additional complications: conditional odds ratios are non-collapsible, so adding a prognostic covariate can change an odds ratio even without confounding.

For causal work, define potential outcomes, treatment timing, adjustment set, positivity, consistency, and interference assumptions. Consider standardized risks, risk differences, or risk ratios rather than relying only on conditional odds ratios.


## 70. Marginal effects and standardization

For nonlinear models, average predictions under counterfactual predictor settings can be more interpretable than coefficients.

For a binary exposure A, the standardized risk difference is estimated by predicting each observation twice, once with A=1 and once with A=0, then averaging the differences:

```
ATE_hat = n^(-1) sum_i [mu_hat(A=1,X_i)-mu_hat(A=0,X_i)]
```

Uncertainty should account for model estimation, often through the delta method, bootstrap, or simulation from the coefficient covariance.


## 71. Distribution shift and transportability

A model validated on one population may fail when predictor distributions, outcome mechanisms, measurement systems, or decision policies change.

Assess calibration and performance by site, time, and subgroup. External validation is stronger than a random split within one dataset. Recalibration can correct intercept or slope drift but cannot repair missing predictors or changed causal mechanisms.


## 72. High-dimensional GLMs

When p is large relative to n, unpenalized maximum likelihood may be unstable or undefined. Penalization, screening, dimension reduction, and sparsity assumptions become central.

Prediction-oriented tuning does not automatically yield valid coefficient inference. Debiased estimators, selective inference, sample splitting, or prespecified low-dimensional targets are research alternatives.


## 73. Simulation studies for GLMs

A rigorous simulation varies factors such as:

- sample size and event rate;

- predictor correlation and leverage;

- effect sizes;

- dispersion and zero inflation;

- link misspecification;

- cluster count and intraclass correlation;

- missingness;

- penalty strength.

Report bias, empirical standard deviation, mean estimated standard error, interval coverage, power, calibration, predictive loss, convergence failures, and computation time. Monte Carlo error should accompany simulation summaries.


## 74. Reproducibility and transparent reporting

Preserve raw data provenance, analysis code, environment, seeds, model formulas, link and family, weight definitions, offset units, convergence output, exclusion rules, and all model-selection steps.

Report events and non-events for binary models, count distribution and zero fraction for counts, exposure distribution for rate models, and support including zeros for positive outcomes.


# Part VI. Learning, Paper Reading, Glossary, and Literature


## 75. Step-by-step learning sequence


### Stage 1: Distribution intuition

Explain why binary, count, and positive outcomes need different support and variance structures.


### Stage 2: Link and coefficient interpretation

Move fluently between eta, mu, probability, odds, expected count, and mean ratios.


### Stage 3: Likelihood and derivatives

Derive logistic and Poisson log-likelihoods, scores, and Hessian or expected information.


### Stage 4: IRLS

Derive working response and weights and connect each code line to the equation.


### Stage 5: Diagnostics

Use residuals, leverage, influence, dispersion, calibration, and validation.


### Stage 6: Libraries

Reproduce pure-Python results with statsmodels and build predictive pipelines with scikit-learn.


### Stage 7: Research extensions

Study GEE, GLMMs, penalized likelihood, Bayesian GLMs, missing data, and causal standardization.


### Stage 8: Independent research project

Write a question and estimand, justify family and link, implement a baseline from scratch, fit library models, diagnose, validate, conduct sensitivity analysis, and reproduce one published result.


This part turns the GLM chapter into four graduated tiers of practice, the same structure used in the Regression chapter: **Conceptual (C)** -> **Derivation (D)** -> **Pure-Python coding (P)** -> **Library (L)**. Work a topic through all four tiers before moving to the next topic, and do not attempt a derivation until its conceptual question is easy.

## 76. Conceptual exercises

**Distributions and link functions**

- **C1.** Why is a link needed if eta can take any real value but mu may be restricted?
- **C2.** Distinguish a response distribution from a link function.
- **C3.** Why does Poisson variance equal its mean under the model?
- **C4.** What does a canonical link simplify?
- **C5.** Explain why exp(beta) is an odds ratio in logistic regression but a mean ratio in Poisson regression.

**Interpretation and confounding**

- **C6.** Why can a logistic coefficient change when a prognostic covariate is added even without confounding?

**Counts, exposure, and overdispersion**

- **C7.** What does an offset do that an ordinary predictor does not?
- **C8.** Why is overdispersion a diagnostic rather than a final explanation?

**Clustered and hierarchical models**

- **C9.** Distinguish GEE population-average coefficients from GLMM cluster-specific coefficients.

**Model evaluation**

- **C10.** Why can a model discriminate well but be poorly calibrated?

<details markdown="1">
<summary>Answer key: Conceptual exercises (C1-C10)</summary>

**Marking rubric.** Full credit names the specific mechanism and connects it to the mathematics in Section 7, not just the vocabulary.

**C1.** The linear predictor `eta = X*beta` ranges over all real numbers, but the mean `mu` of many response families is restricted (probabilities in `[0,1]`, counts/rates in `[0, infinity)`). The link function `g` maps the restricted `mu` to the unrestricted `eta` (`g(mu) = eta`) so that an ordinary linear model in `eta` never has to produce an impossible value of `mu`.

**C2.** The response distribution (Bernoulli, Poisson, Gamma, ...) describes how the *outcome* varies around its mean for a fixed `mu` — it is a statement about randomness. The link function describes how the *mean itself* relates to the predictors — it is a statement about the systematic part of the model. A GLM needs both, chosen somewhat independently (see C4 for the canonical pairing).

**C3.** The Poisson distribution is entirely parameterized by its rate `lambda`, and its variance is derived to equal `lambda` as a direct consequence of the counting-process (Poisson process) assumptions that generate it — there is no separate variance parameter to estimate. This is also why real count data, which usually has additional unmodeled heterogeneity, tends to show variance *greater* than the mean (overdispersion, C8).

**C4.** The canonical link is the one for which the link function equals the exponential family's natural parameter, which makes the score equations and Fisher information (Section 7) simplify algebraically — specifically, the expected and observed information coincide, so Fisher scoring and Newton-Raphson become the same algorithm.

**C5.** In logistic regression the link is the log-odds; `exp(beta_j)` is the multiplicative change in *odds* per unit increase in `x_j`. In Poisson regression with a log link, `exp(beta_j)` is the multiplicative change in the *mean count* per unit increase in `x_j`. Same algebraic form (`exp` of a coefficient), different link, different quantity being multiplied.

**C6.** Unlike in linear regression, a logistic (or any nonlinear-link) coefficient reflects the effect on `eta`, and the mapping from `eta` to `mu` is nonlinear — adding a covariate that predicts the outcome well (even one uncorrelated with existing predictors) changes the scale on which the remaining coefficients are measured. This is a non-collapsibility effect, not confounding, and it is one of the more counterintuitive facts in the GLM chapter.

**C7.** An offset is a predictor forced to have a coefficient of exactly 1 (entered as `log(exposure)` under a log link) rather than estimated — it converts a model for a count into a model for a *rate* (count per unit exposure) without spending a degree of freedom or letting the data decide how strongly exposure should matter.

**C8.** Overdispersion (variance exceeding the mean-implied value) tells you the assumed variance function is wrong, but not *why* — it could be omitted covariates, clustering, excess zeros, or a genuinely different data-generating process. Diagnosing overdispersion is the start of the investigation (quasi-Poisson, negative binomial, zero-inflation), not a model in itself.

**C9.** A GEE coefficient answers "how does the population-average mean change with `x`," marginalizing over the random effects/cluster structure. A GLMM coefficient answers "how does one specific cluster's (or individual's) mean change with `x`, holding its random effect fixed." The two differ numerically for nonlinear links (non-collapsibility again, C6) even though both are valid, just answering different questions.

**C10.** Discrimination (e.g. ROC-AUC) asks whether the model ranks higher-risk cases above lower-risk cases correctly — it is invariant to any monotonic rescaling of the predicted probabilities. Calibration asks whether the predicted probability of 0.3 actually corresponds to an observed frequency of about 30% — a model can rank cases perfectly while being systematically over- or under-confident in its actual probability values.

</details>

## 77. Derivation exercises

**Likelihood and score functions**

- **D1.** Derive the Bernoulli log-likelihood.
- **D2.** Derive the logistic score X^T(y-p).
- **D4.** Derive the Poisson score under a log link.

**IRLS mechanics**

- **D3.** Derive the logistic expected information X^T W X.
- **D5.** Show that the Poisson IRLS weight equals mu.
- **D6.** Derive the working response for a log-link model.

**Deviance**

- **D7.** Derive the binomial deviance contribution.
- **D8.** Derive the Poisson deviance contribution using the saturated likelihood.

**Inference and marginal effects**

- **D9.** Show how coefficient covariance transforms to an odds-ratio interval.
- **D10.** Derive the delta-method variance for a predicted probability.
- **D11.** Derive an average marginal effect for a continuous predictor in logistic regression.
- **D12.** Derive a standardized risk difference from model predictions.

<details markdown="1">
<summary>Answer key: Derivation exercises (D1-D12)</summary>

**Marking rubric.** Full credit shows every step and connects the result to the corresponding IRLS or diagnostic formula in Section 7-9.

**D1.** For a single Bernoulli trial, `P(y) = mu^y * (1-mu)^(1-y)`. Taking the log: `l = y*log(mu) + (1-y)*log(1-mu)`. Summed over `n` independent observations: `L = sum(y_i*log(mu_i) + (1-y_i)*log(1-mu_i))`.

**D2.** With the logit link, `mu = sigmoid(eta)`, `eta = X*beta`. Differentiating `D1`'s log-likelihood with respect to `beta` and using `d(mu)/d(eta) = mu*(1-mu)`, the `mu*(1-mu)` terms cancel exactly against the same term appearing in `d(eta)/d(beta)`'s chain-rule denominator, leaving the clean score `dL/d(beta) = X^T(y - mu) = X^T(y-p)`.

**D3.** The Hessian of `D2`'s score is `d^2 L/d(beta) d(beta)^T = -X^T diag(mu_i*(1-mu_i)) X`. The *expected* information is the negative expectation of this (already deterministic here since it doesn't depend on `y`), giving `X^T W X` with `W = diag(mu_i*(1-mu_i))` — this is exactly the weight matrix used in IRLS.

**D4.** For Poisson with a log link, `mu = exp(eta)`. Log-likelihood per observation (dropping the `y!` normalizing constant, which doesn't depend on `beta`): `l_i = y_i*eta_i - exp(eta_i)`. Differentiating: `dl_i/d(beta) = (y_i - mu_i)*x_i`, giving score `X^T(y-mu)` — the same *form* as D2, which is the point of the unified GLM notation (Section 7's "why this unifies GLMs").

**D5.** The IRLS weight in general is `w_i = 1/(V(mu_i) * g'(mu_i)^2)`, where `V` is the variance function and `g'` the link derivative. For Poisson (`V(mu)=mu`) with a log link (`g(mu)=log(mu)`, so `g'(mu)=1/mu`): `w_i = 1/(mu_i * (1/mu_i)^2) = mu_i`. The weight equals the mean itself.

**D6.** The IRLS working response is `z_i = eta_i + (y_i - mu_i)*g'(mu_i)`. For the log link, `g'(mu_i) = 1/mu_i`, giving `z_i = eta_i + (y_i - mu_i)/mu_i` — this is exactly what a weighted-least-squares regression of `z` on `X` with weights `w_i` from D5 needs at each IRLS iteration.

**D7.** Binomial deviance contribution is `2 * [y*log(y/mu) + (1-y)*log((1-y)/(1-mu))]` (with the usual `0*log(0)=0` convention), derived as `2*(l_saturated - l_model)` where the saturated model sets `mu_i = y_i` exactly. Each term is non-negative and zero only when `mu_i = y_i`.

**D8.** For Poisson, the saturated log-likelihood replaces `mu_i` with `y_i`: `l_saturated_i = y_i*log(y_i) - y_i`. Subtracting the fitted model's contribution and doubling: `D_i = 2*[y_i*log(y_i/mu_i) - (y_i - mu_i)]`, with `y_i*log(y_i/mu_i)` taken to be 0 when `y_i=0`.

**D9.** If `Var(beta_hat_j) = v_j` (from the sandwich or model-based covariance), then `beta_hat_j +/- 1.96*sqrt(v_j)` is a Wald CI for the coefficient on the log-odds scale. Exponentiating both endpoints, `exp(beta_hat_j - 1.96*sqrt(v_j))` to `exp(beta_hat_j + 1.96*sqrt(v_j))`, gives a valid CI for the **odds ratio** because `exp` is monotonic (order-preserving), even though the interval is no longer symmetric around `exp(beta_hat_j)`.

**D10.** For `p = sigmoid(eta)`, the delta method gives `Var(p) ~= (dp/d(eta))^2 * Var(eta) = [p*(1-p)]^2 * x^T * Cov(beta_hat) * x`, using the same `d(mu)/d(eta) = mu(1-mu)` derivative from D2/D3.

**D11.** The marginal effect of `x_j` at a given point is `d(mu)/d(x_j) = mu*(1-mu)*beta_j` (chain rule through the logistic link). The *average* marginal effect averages this quantity, `mu_i*(1-mu_i)*beta_j`, over the observed sample (or over a predictor distribution), rather than evaluating it at a single reference point.

**D12.** A standardized risk difference compares average predicted risk under two counterfactual settings of one predictor (e.g. `x_j=1` vs `x_j=0`) while holding the *distribution* of all other predictors at its observed sample values: `RD = mean_i[mu(x_j=1, x_{-j,i})] - mean_i[mu(x_j=0, x_{-j,i})]`, i.e. predict twice for every observation and average the difference — this is what "standardization" means in the causal-inference sense used in Part V.

</details>

## 78. Pure-Python coding exercises

**Link and family extensions**

- **P1.** Add grouped binomial trials to the IRLS engine.
- **P2.** Implement a probit link using math.erf.
- **P3.** Add the complementary log-log link.
- **P4.** Implement a negative-binomial log-likelihood and optimizer.

**IRLS numerics and stability**

- **P5.** Add step halving when deviance increases.
- **P6.** Replace Gauss-Jordan with Householder QR.
- **P14.** Compare ridge-stabilized IRLS with unpenalized IRLS under collinearity.

**Inference and diagnostics**

- **P7.** Add cluster-robust sandwich covariance.
- **P8.** Implement score and likelihood-ratio tests.
- **P9.** Add parametric bootstrap intervals for predicted counts.
- **P10.** Add randomized quantile residuals.
- **P11.** Detect separation heuristically and return a warning.
- **P12.** Add calibration slope and intercept calculations.

**Model selection and validation**

- **P13.** Implement Poisson and Gamma deviance cross-validation.
- **P15.** Build automated tests against statsmodels on random seeds.

<details markdown="1">
<summary>Answer key: Pure-Python coding exercises (P1-P15)</summary>

**Marking rubric.** Full credit requires the implementation to match `statsmodels` on a shared synthetic example to numerical tolerance, plus one edge-case test. Hints and the core formula are given below; write the implementation yourself against `glm_from_scratch.py`'s existing style.

**P1.** Extend the binomial family so each observation carries `(successes, trials)` instead of a single 0/1; the deviance, weights, and working response formulas are unchanged except that `y` becomes `successes/trials` and each observation's contribution to the log-likelihood is scaled by `trials` (the binomial coefficient itself drops out of the score, same as in D1-D2).

**P2.** The probit link is `g(mu) = Phi^-1(mu)` where `Phi` is the standard normal CDF; `math.erf` gives you `Phi(z) = 0.5*(1+erf(z/sqrt(2)))`, and the inverse (`Phi^-1`) needs a numerical root-finder (bisection or Newton) since it has no closed form in terms of `erf`.

**P3.** The complementary log-log link is `g(mu) = log(-log(1-mu))`, with inverse `mu = 1 - exp(-exp(eta))`. Unlike the symmetric logit, cloglog is asymmetric — it approaches `mu=1` more slowly than `mu=0`, useful when events are rare.

**P4.** The negative binomial log-likelihood (mean `mu`, dispersion `alpha`) is `l_i = log(Gamma(y_i + 1/alpha)) - log(Gamma(1/alpha)) - log(y_i!) + (1/alpha)*log(1/(1+alpha*mu_i)) + y_i*log(alpha*mu_i/(1+alpha*mu_i))`. Optimize `alpha` by 1D search (e.g. golden section) around an IRLS fit of `beta` for fixed `alpha`, alternating the two until both stabilize.

**P5.** After each IRLS update, check whether deviance increased; if so, halve the step (`beta_new = beta_old + 0.5*(beta_proposed - beta_old)`) and re-check, repeating the halving until deviance decreases or a maximum number of halvings is reached — this is what prevents divergence when starting far from the optimum or with near-separated data.

**P6.** Replace your Gauss-Jordan solve of the weighted normal equations with a QR decomposition of `sqrt(W)*X` (and correspondingly transform the working response), solving the resulting triangular system by back-substitution — this avoids ever forming `X^T W X` explicitly, which is the main source of numerical instability under near-collinearity.

**P7.** Cluster-robust sandwich covariance: `V = (X^T W X)^-1 * [sum_g (X_g^T W_g e_g)(X_g^T W_g e_g)^T] * (X^T W X)^-1`, summing the inner term over clusters `g` rather than individual observations (contrast with the observation-level HC0 sandwich from the Regression chapter).

**P8.** The score test statistic is `S^T * I^-1 * S` evaluated at the restricted (null) MLE, requiring no refit of the full model. The likelihood-ratio statistic is `2*(l_full - l_restricted)`, requiring both models fit — implement both and confirm they agree approximately (they're asymptotically equivalent) on a case where the restriction is far from true.

**P9.** Refit the model (or resample residuals) many times under the fitted parameters to generate a distribution of predicted counts at a chosen `x`; the parametric bootstrap interval is the 2.5th/97.5th percentile of that simulated distribution — verify it roughly matches the delta-method interval (D10-style) when the sample size is large.

**P10.** A randomized quantile residual maps each observation's predicted CDF value to a standard normal quantile, adding continuity-correcting uniform randomization for discrete outcomes (Poisson, binomial) so the resulting residuals are exactly normally distributed under a correctly specified model — test by checking the residuals pass a normality test (e.g. Shapiro-Wilk) when the model is correctly specified and fail when it is not.

**P11.** A simple separation heuristic: if any fitted probability reaches within `1e-6` of 0 or 1 while the corresponding coefficient's magnitude keeps growing across IRLS iterations without converging, flag likely (quasi-)separation — cross-check against Firth's penalized-likelihood correction (Part V), which resolves the non-convergence directly.

**P12.** Regress the observed binary outcome on the model's predicted logit (a single-predictor logistic regression) — the resulting slope is the calibration slope (1.0 = perfectly calibrated) and the intercept is the calibration intercept (0.0 = perfectly calibrated). Verify on simulated data where you know the true model that both are close to their ideal values.

**P13.** For each family, cross-validated deviance sums the per-fold deviance contributions (D7/D8's formulas) computed with parameters fit on the other folds, rather than mean squared error — this is the natural GLM generalization of cross-validated RMSE and rewards the correct variance structure, not just point-prediction accuracy.

**P15.** Generate many random synthetic datasets (varying `n`, number of predictors, and true parameters), fit both your from-scratch IRLS and `statsmodels.GLM` on each, and assert the coefficients agree to a fixed tolerance — a single passing example is not evidence of correctness (§13.17 of the site's own code rules); many random seeds are.

</details>

## 79. Library exercises

**Cross-library comparison**

- **L1.** Fit the same logistic model in pure Python and statsmodels and compare every coefficient and fitted probability.

**Count and continuous families**

- **L2.** Fit Poisson, negative-binomial, and quasi-Poisson-style analyses to overdispersed counts.
- **L3.** Use an exposure offset and verify that changing exposure rescales predicted counts but not predicted rates.
- **L4.** Fit Gamma and Tweedie models to positive or semicontinuous costs.

**Standard errors and diagnostics**

- **L5.** Compare model-based and robust standard errors.
- **L6.** Produce binned residual and calibration plots.

**Validation and model selection**

- **L7.** Use nested cross-validation for penalized logistic regression.
- **L8.** Compare random and group cross-validation on clustered data.

**Clustered, hierarchical, and separation**

- **L9.** Fit a GEE and compare with an independence GLM.
- **L10.** Fit a random-intercept GLMM in a suitable library and compare conditional with marginal predictions.
- **L11.** Demonstrate complete separation and apply penalized or Bayesian correction.

**Causal standardization**

- **L12.** Use bootstrap standardization to estimate a risk difference.

<details markdown="1">
<summary>Answer key: Library exercises (L1-L12)</summary>

**Marking rubric.** Full credit requires running code plus one sentence connecting the library output back to the underlying GLM concept — a bare printout of numbers with no interpretation earns partial credit only.

**L1.** Fit with your `glm_from_scratch` IRLS solver, then `sm.GLM(y, X, family=sm.families.Binomial()).fit()`. Coefficients and fitted probabilities should agree to `1e-6` on well-behaved data; any larger gap points to a convergence or weighting bug in the from-scratch solver, not in statsmodels.

**L2.** Fit `sm.GLM(..., family=sm.families.Poisson())`, then `sm.GLM(..., family=sm.families.NegativeBinomial())` on the same overdispersed counts, then compare a quasi-Poisson fit (`sm.GLM(..., family=sm.families.Poisson()).fit(scale="X2")`, which inflates standard errors by the Pearson dispersion without changing the point estimates). Report how much wider the standard errors get moving from plain Poisson to the overdispersion-corrected versions.

**L3.** Add `offset=np.log(exposure)` to the model call and refit at a different exposure value for the same underlying rate; confirm that the predicted *count* scales proportionally with exposure while the predicted *rate* (`count/exposure`) stays constant — that separation is exactly what an offset is for (see C7).

**L4.** Fit `sm.GLM(..., family=sm.families.Gamma(link=sm.families.links.Log()))` for strictly positive costs, and `sm.GLM(..., family=sm.families.Tweedie(var_power=1.5))` for data with an exact zero mass plus a continuous positive part (e.g. insurance claims). Compare deviance residual plots between the two to see which handles the zero mass more sensibly.

**L5.** Fit the same model with `cov_type="nonrobust"` and `cov_type="HC0"` (or `cov_type="cluster"` if data are clustered); the coefficients are identical, only the standard errors move — report by how much, and note the direction (robust SEs are typically, but not always, larger).

**L6.** Use `sm.graphics.plot_ccpr` or manually bin residuals against fitted values and plot mean observed vs. mean predicted per bin (a calibration/reliability plot) — a well-calibrated model's points should track the 45-degree line.

**L7.** Wrap `LogisticRegression(penalty="l1")` (or `"l2"`) in `GridSearchCV` for the inner loop (choosing the penalty strength) and an outer `cross_val_score` loop for honest performance estimation — nesting matters here because using the same folds to both choose the penalty and report performance overstates accuracy (the same leakage issue as C14 in the Regression chapter).

**L8.** Compare `KFold` and `GroupKFold` (grouping by cluster) cross-validated deviance on the same clustered count or binary data; random CV will typically look better than group CV because it lets information about a cluster leak between train and validation folds.

**L9.** Fit with `statsmodels`' GEE (`sm.GEE(..., groups=cluster_id)`) and compare coefficients to an ordinary GLM fit ignoring clustering — for a correctly specified mean model the point estimates should be similar, but the GEE standard errors (which account for within-cluster correlation) will generally differ from the naive GLM's.

**L10.** Fit a random-intercept GLMM (e.g. via `statsmodels.genmod.bayes_mixed_glm` or `pymer4`/`lme4` if available) and compare its cluster-specific predictions (conditional on the estimated random effect) to the population-average predictions from L9's GEE — the two are estimating genuinely different quantities (C9), and the gap between them grows with the amount of between-cluster heterogeneity.

**L11.** Construct data with a predictor that perfectly (or near-perfectly) separates the outcome and confirm `statsmodels` either fails to converge or reports enormous standard errors; refit with Firth's penalized likelihood (`statsmodels`' `Logit.fit(method="firth")` or an equivalent implementation) and confirm it converges to finite, interpretable coefficients.

**L12.** Fit the model, predict `mu` for every observation under two counterfactual settings of the predictor of interest (holding everything else at its observed value), and bootstrap the whole standardization procedure (resample observations with replacement, refit, re-predict, recompute the risk difference) to get a confidence interval on the risk difference — the bootstrap accounts for uncertainty in both the coefficient estimates and the standardization step itself.

</details>

## 80. How to read a GLM research paper

Read in seven passes.


### Pass 1: Outcome and support

What values can the outcome take? Does the chosen family match binary, count, positive, semicontinuous, ordinal, or repeated data?


### Pass 2: Estimand and interpretation scale

Is the target a probability, odds ratio, rate ratio, mean ratio, marginal risk difference, cluster-specific effect, or prediction risk? Write it in words and symbols.


### Pass 3: Data-generating process

Study sampling, exposure time, clustering, missingness, case-control selection, and temporal ordering.


### Pass 4: Family, link, and linear predictor

Write the complete model including offset, random effects, weights, interactions, nonlinear terms, and dispersion.


### Pass 5: Estimation and uncertainty

Identify maximum likelihood, quasi-likelihood, GEE, penalization, Bayesian estimation, or another method. Check covariance, cluster count, priors, and convergence.


### Pass 6: Diagnostics and validation

Look for dispersion, separation, calibration, residuals, influence, support, external validation, and sensitivity to family and link.


### Pass 7: Reproduction and critique

Reconstruct one table or figure. Determine whether the numerical result supports the paper's verbal claim. Propose the single most informative robustness check or new data collection.

A paper note should record: question, estimand, population, outcome support, family, link, predictor structure, likelihood or estimating equation, identification assumptions, diagnostics, key result, limitation, and extension.


## 81. NotebookLM and AI-tutor prompts

- "Teach exponential-family notation one symbol at a time and quiz me on the role of theta, b(theta), phi, and V(mu)."

- "Derive logistic regression from the Bernoulli likelihood, stopping after every algebraic step."

- "Make me derive Poisson IRLS and connect each equation to glm_from_scratch.py."

- "Give me ten coefficient interpretations and make me identify whether each is an odds ratio, rate ratio, mean ratio, or marginal probability effect."

- "Create a dataset that causes separation and ask me to diagnose it before showing a remedy."

- "Act as a statistical reviewer and challenge my family, link, offset, dispersion, and covariance choices."

- "Give me a GLM paper and guide me through the seven-pass reading method without revealing the critique first."

- "Ask me to reproduce a statsmodels result using only pure Python and compare tolerances."

- "Create progressively harder exercises on deviance, likelihood-ratio tests, and prediction intervals."

- "Build a personalized review plan based only on concepts I answer incorrectly."


## 82. Expanded glossary


### Akaike information criterion

A likelihood-based criterion balancing fit and parameter count: -2 log likelihood plus 2k. It estimates relative expected out-of-sample information loss under assumptions. AIC values are meaningful only in comparisons fitted to the same response data with comparable likelihood definitions.


### Bernoulli distribution

A distribution for one binary trial with probability p of success. Its mean is p and variance p(1-p). Bernoulli logistic regression models p conditional on predictors.


### Binomial distribution

A distribution for the number of successes in a fixed number of conditionally exchangeable trials with common success probability. Grouped-binomial regression must account for the number of trials.


### Brier score

The mean squared difference between a binary outcome and predicted probability. It evaluates both calibration and refinement and is a proper scoring rule. Its scale depends on prevalence, so compare it with sensible reference models.


### Calibration

Agreement between predicted probabilities or means and observed outcomes. Calibration-in-the-large concerns systematic over- or underprediction; calibration slope concerns predictions being too extreme or too weak.


### Canonical link

The link that equates the linear predictor with the natural parameter of an exponential family. It often simplifies score equations but is not mandatory.


### Complementary log-log link

The binary link log[-log(1-p)]. It is asymmetric and has a connection to discrete-time proportional-hazards models.


### Conditional likelihood

A likelihood formed after conditioning on sufficient statistics or nuisance structures. It can eliminate nuisance parameters in matched or fixed-effect models but changes what information is used.


### Count exposure

The amount of opportunity for events, such as person-time or distance. In a log-link rate model, its logarithm enters as an offset with fixed coefficient one.


### Deviance

Twice the log-likelihood difference between a saturated model and the fitted model. It measures relative discrepancy within the chosen family and supplies residuals and nested likelihood-ratio comparisons.


### Deviance residual

The signed square root of an observation's deviance contribution. It places observations from different mean levels on a more comparable discrepancy scale than raw residuals.


### Dispersion parameter

A scale multiplying the variance function. It is fixed at one in classical Poisson and binomial GLMs and estimated in Gaussian, Gamma, and inverse-Gaussian GLMs.


### Distributional regression

Models in which multiple distribution parameters, not only the mean, depend on predictors. It extends GLMs when scale, shape, zero mass, or tail behavior changes systematically.


### Exponential family

A class of distributions expressible through a natural parameter, cumulant function, dispersion, and base measure. It creates shared formulas for moments, score, information, and IRLS.


### Fisher information

Expected negative Hessian of the log-likelihood, equivalently the variance of the score under regularity. Its inverse approximates estimator covariance.


### Fisher scoring

An iterative optimization method replacing the observed negative Hessian in Newton-Raphson with expected information. In GLMs it yields IRLS.


### Gamma distribution

A positive continuous distribution with flexible right skew. In Gamma GLMs the variance is proportional to mu squared, scaled by dispersion.


### Generalized estimating equation

An estimating-equation approach for correlated or longitudinal outcomes that specifies a marginal mean and working covariance. Robust covariance can protect against working-correlation misspecification with sufficient independent clusters.


### Generalized linear mixed model

A GLM with random effects in the linear predictor. It models conditional, cluster-specific relationships and requires integration over random effects.


### Hurdle model

A two-part model in which one process determines zero versus positive and a zero-truncated distribution models positive values. It differs from zero inflation because the positive-count component cannot generate zero.


### Incidence-rate ratio

The multiplicative change in an expected event rate for a one-unit predictor change under a log-link count model. It is exp(beta_j) when other predictors and exposure are held fixed.


### Inverse link

The function mapping linear predictor eta back to mean mu. Examples are sigmoid for logit and exp for log.


### Iteratively reweighted least squares

The algorithm that fits many GLMs by repeatedly constructing a working response and weights and solving weighted least squares until convergence.


### Link function

A monotone function connecting the conditional mean to the linear predictor. It defines the additive modelling scale and coefficient interpretation.


### Log likelihood

The logarithm of likelihood, turning independent products into sums. Its gradient is the score and its curvature determines information.


### Log link

The link log(mu)=eta. It guarantees positive means and gives multiplicative coefficient interpretations.


### Logistic function

The inverse of the logit, mapping the real line to probabilities between zero and one.


### Marginal effect

The change in predicted outcome mean from a small continuous change or specified discrete change in a predictor. In nonlinear models it depends on covariate values.


### Maximum likelihood

Estimation by choosing parameters that maximize probability or density of the observed data under the model. Its optimal properties rely on identification and regularity and are often asymptotic.


### Mean-variance relationship

The modelled function connecting conditional variance to conditional mean. It is constant for basic Gaussian, mu(1-mu) for Bernoulli, mu for Poisson, and mu squared for Gamma, up to dispersion.


### Negative binomial distribution

An overdispersed count distribution. A common GLM-like parameterization has variance mu+alpha mu squared. Multiple parameterizations exist.


### Newton-Raphson

An iterative optimizer using the gradient and observed Hessian. It can converge quickly near a solution but can be unstable with poor starting values or problematic curvature.


### Non-collapsibility

A property of measures such as odds ratios whereby conditional and marginal values can differ even without confounding. It complicates comparison of logistic coefficients across adjustment sets.


### Null deviance

Deviance of an intercept-only model. It provides a baseline but is not equivalent to total sum of squares.


### Offset

A linear-predictor term whose coefficient is fixed, usually at one. Log exposure is a common Poisson offset.


### Odds ratio

A ratio of odds. In logistic regression exp(beta_j) is the conditional odds ratio for a one-unit predictor change under the specified model.


### Overdispersion

Conditional variance greater than the nominal family variance. It may signal heterogeneity, clustering, excess zeros, or family misspecification.


### Pearson residual

The response residual divided by the square root of the variance function. Its squared sum is used in dispersion assessment.


### Poisson distribution

A count distribution with equal conditional mean and variance. Poisson regression usually uses a log link.


### Prior weight

A model weight entering likelihood or variance calculations. Its meaning is software- and context-specific and must not be confused with sampling weights.


### Probit link

The inverse-standard-normal CDF link for binary probability. It corresponds to a latent Gaussian threshold model and often yields predictions close to logit with rescaled coefficients.


### Proper scoring rule

A prediction score whose expected value is optimized by reporting the true predictive distribution. Log loss and Brier score are proper for probabilities.


### Quasi-likelihood

An estimating framework based on a specified mean-variance relationship rather than a full distribution. It supports coefficients and covariance but not an ordinary full likelihood for AIC.


### Randomized quantile residual

A residual obtained by mapping an observation through its fitted distribution and then through the inverse normal CDF, with randomization for discrete outcomes. Under a correct model it is approximately standard normal apart from parameter estimation.


### Rate

Expected count per exposure unit. A rate model requires a meaningful exposure denominator and proportionality assumptions.


### Response residual

Observed outcome minus fitted conditional mean. Its variance changes across observations in most GLMs.


### Saturated model

A model with enough parameters to reproduce each observation's fitted mean as closely as the family permits. It provides the deviance reference.


### Score function

Gradient of the log-likelihood with respect to parameters. At an interior maximum the score is zero.


### Separation

A logistic-regression condition in which predictor combinations perfectly or nearly classify outcomes, causing maximum-likelihood coefficients to diverge or become unstable.


### Sandwich covariance

A robust covariance estimator combining inverse sensitivity matrices around empirical score variability. It protects against specified model misspecification under asymptotic conditions.


### Standardized prediction

A marginal prediction formed by averaging conditional predictions over a chosen covariate distribution. It can translate nonlinear regression into population risks or mean outcomes.


### Tweedie distribution

An exponential-dispersion family with variance phi mu^p. Values between one and two give compound Poisson-Gamma outcomes with exact zeros and positive continuous values.


### Underdispersion

Conditional variance smaller than nominal model variance. It occurs in constrained or regular event processes and can also signal dependence or measurement design.


### Variance function

The family-specific function V(mu) determining how conditional variance changes with the mean, apart from dispersion and weights.


### Wald test

A test based on estimate minus null value relative to estimated standard error. It is convenient but can perform poorly near boundaries, under separation, or with strong nonlinearity.


### Working correlation

A model for within-cluster association in GEE. Coefficient consistency can survive its misspecification under conditions, but efficiency and finite-sample covariance can be affected.


### Working response

The pseudo-outcome z used in IRLS. It linearizes the mean relation locally and changes at every iteration.


### Zero-inflated model

A mixture of a structural-zero component and a count distribution that can itself generate zeros. Interpretation requires separating the two latent processes.


## 83. Suggested research papers with reading purpose

1. Nelder, J. A., and Wedderburn, R. W. M. (1972). Generalized Linear Models. Journal of the Royal Statistical Society Series A, 135(3), 370-384. The founding paper; focus on the unified architecture and IRLS logic.
2. Wedderburn, R. W. M. (1974). Quasi-likelihood Functions, Generalized Linear Models, and the Gauss-Newton Method. Biometrika, 61(3), 439-447. Read for mean-variance modelling without full likelihood.
3. Wedderburn, R. W. M. (1976). On the Existence and Uniqueness of the Maximum Likelihood Estimates for Certain Generalized Linear Models. Biometrika, 63(1), 27-32. Connects estimation to existence and boundary problems.
4. Pregibon, D. (1981). Logistic Regression Diagnostics. Annals of Statistics, 9(4), 705-724. Foundational influence and diagnostic development for logistic models.
5. Liang, K.-Y., and Zeger, S. L. (1986). Longitudinal Data Analysis Using Generalized Linear Models. Biometrika, 73(1), 13-22. The foundational GEE paper.
6. Firth, D. (1993). Bias Reduction of Maximum Likelihood Estimates. Biometrika, 80(1), 27-38. Read for penalized likelihood and separation-related bias reduction.
7. Dunn, P. K., and Smyth, G. K. (1996). Randomized Quantile Residuals. Journal of Computational and Graphical Statistics, 5(3), 236-244. A general diagnostic residual for non-Gaussian regression.
8. Lambert, D. (1992). Zero-Inflated Poisson Regression, with an Application to Defects in Manufacturing. Technometrics, 34(1), 1-14. Introduces a major excess-zero model and its latent interpretation.
9. McCullagh, P. (1980). Regression Models for Ordinal Data. Journal of the Royal Statistical Society Series B, 42(2), 109-142. Foundational proportional-odds modelling.
10. Breslow, N. E., and Clayton, D. G. (1993). Approximate Inference in Generalized Linear Mixed Models. Journal of the American Statistical Association, 88(421), 9-25. Important for GLMM approximation and limitations.
11. Zeger, S. L., Liang, K.-Y., and Albert, P. S. (1988). Models for Longitudinal Data: A Generalized Estimating Equation Approach. Biometrics, 44(4), 1049-1060. Extends population-average longitudinal modelling.
12. Lee, Y., and Nelder, J. A. (1996). Hierarchical Generalized Linear Models. Journal of the Royal Statistical Society Series B, 58(4), 619-678. A broad framework for random effects and dispersion.
13. King, G., and Zeng, L. (2001). Logistic Regression in Rare Events Data. Political Analysis, 9(2), 137-163. Read critically for rare-event bias, sampling, and correction.
14. Zou, H., and Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net. Journal of the Royal Statistical Society Series B, 67(2), 301-320. Applies directly to high-dimensional GLMs as well as linear models.
15. van der Laan, M. J., and Rubin, D. (2006). Targeted Maximum Likelihood Learning. International Journal of Biostatistics, 2(1). Connects flexible nuisance models to targeted causal parameters.
16. Zeileis, A., Kleiber, C., and Jackman, S. (2008). Regression Models for Count Data in R. Journal of Statistical Software, 27(8). A practical comparison of Poisson, negative-binomial, hurdle, and zero-inflated models.
17. Gelman, A., Jakulin, A., Pittau, M. G., and Su, Y.-S. (2008). A Weakly Informative Default Prior Distribution for Logistic and Other Regression Models. Annals of Applied Statistics, 2(4), 1360-1383. Read for Bayesian stabilization and scale-aware priors.
18. Austin, P. C., and Steyerberg, E. W. (2012). Interpreting the Concordance Statistic of a Logistic Regression Model: Relation to the Variance and Distribution of Prognostic Effects. Statistics in Medicine, 31(20), 2298-2308. Helps separate discrimination from effect size and calibration.
19. Gneiting, T., and Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. Journal of the American Statistical Association, 102(477), 359-378. Foundational for probabilistic prediction evaluation.
20. Brooks, M. E., Kristensen, K., van Benthem, K. J., et al. (2017). glmmTMB Balances Speed and Flexibility Among Packages for Zero-inflated Generalized Linear Mixed Modeling. The R Journal, 9(2), 378-400. Read for computational trade-offs in modern mixed count models.

For every paper, record the outcome support, estimand, family and link, likelihood or estimating equation, dependence assumptions, diagnostic strategy, computational method, and one result you can independently reproduce.


# Appendix A. Complete Pure-Python Module

The complete executable module follows. Save it as glm_from_scratch.py. It is verified against statsmodels for Gaussian, binomial-logit, Poisson-log, and Gamma-log examples within numerical tolerance.

```python
"""Educational generalized linear models using only Python's standard library.

Implements:
- matrix utilities and weighted least squares
- Gaussian identity, Bernoulli/binomial logit, Poisson log, and Gamma log GLMs
- iteratively reweighted least squares (IRLS/Fisher scoring)
- coefficient covariance, standard errors, Wald z statistics and confidence intervals
- deviance, Pearson and deviance residuals, leverage and Cook-style influence
- HC0/HC3 sandwich covariance
- train/test splitting, k-fold cross-validation, and basic metrics
- synthetic-data helpers and verification tests

This is intended for learning and small datasets, not production-scale analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

Vector = List[float]
Matrix = List[List[float]]


# ---------------------------------------------------------------------------
# Validation and matrix operations
# ---------------------------------------------------------------------------

def _float_vector(values: Iterable[float]) -> Vector:
    result = [float(v) for v in values]
    if not result:
        raise ValueError("expected at least one value")
    return result


def _validate_matrix(a: Matrix) -> Tuple[int, int]:
    if not a or not a[0]:
        raise ValueError("matrix must be non-empty")
    rows = len(a)
    cols = len(a[0])
    if any(len(row) != cols for row in a):
        raise ValueError("all matrix rows must have the same length")
    return rows, cols


def transpose(a: Matrix) -> Matrix:
    rows, cols = _validate_matrix(a)
    return [[float(a[i][j]) for i in range(rows)] for j in range(cols)]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return sum(float(x) * float(y) for x, y in zip(a, b))


def matvec(a: Matrix, x: Sequence[float]) -> Vector:
    _, cols = _validate_matrix(a)
    if cols != len(x):
        raise ValueError("incompatible matrix and vector dimensions")
    return [dot(row, x) for row in a]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    _, a_cols = _validate_matrix(a)
    b_rows, b_cols = _validate_matrix(b)
    if a_cols != b_rows:
        raise ValueError("incompatible matrix dimensions")
    bt = transpose(b)
    return [[dot(row, bt[j]) for j in range(b_cols)] for row in a]


def identity(n: int) -> Matrix:
    if n <= 0:
        raise ValueError("n must be positive")
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def add_intercept_column(x: Matrix) -> Matrix:
    _validate_matrix(x)
    return [[1.0] + [float(v) for v in row] for row in x]


def solve_linear_system(a: Matrix, b: Sequence[float], tolerance: float = 1e-12) -> Vector:
    """Solve A x = b using Gauss-Jordan elimination with partial pivoting."""
    n_rows, n_cols = _validate_matrix(a)
    if n_rows != n_cols:
        raise ValueError("A must be square")
    if len(b) != n_rows:
        raise ValueError("b has incompatible length")

    aug = [
        [float(a[i][j]) for j in range(n_cols)] + [float(b[i])]
        for i in range(n_rows)
    ]
    for col in range(n_cols):
        pivot_row = max(range(col, n_rows), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < tolerance:
            raise ValueError("matrix is singular or nearly singular")
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        pivot = aug[col][col]
        aug[col] = [value / pivot for value in aug[col]]
        for row in range(n_rows):
            if row == col:
                continue
            factor = aug[row][col]
            if factor != 0.0:
                aug[row] = [
                    aug[row][j] - factor * aug[col][j]
                    for j in range(n_cols + 1)
                ]
    return [aug[i][-1] for i in range(n_rows)]


def inverse(a: Matrix) -> Matrix:
    n_rows, n_cols = _validate_matrix(a)
    if n_rows != n_cols:
        raise ValueError("matrix must be square")
    eye = identity(n_rows)
    columns = [
        solve_linear_system(a, [eye[i][j] for i in range(n_rows)])
        for j in range(n_cols)
    ]
    return transpose(columns)


def weighted_crossproduct(x: Matrix, weights: Sequence[float]) -> Matrix:
    n, p = _validate_matrix(x)
    if len(weights) != n:
        raise ValueError("weights have incompatible length")
    result = [[0.0 for _ in range(p)] for _ in range(p)]
    for i, row in enumerate(x):
        w = float(weights[i])
        if w < 0.0 or not math.isfinite(w):
            raise ValueError("weights must be finite and non-negative")
        for a in range(p):
            for b in range(p):
                result[a][b] += w * row[a] * row[b]
    return result


def weighted_xty(x: Matrix, weights: Sequence[float], y: Sequence[float]) -> Vector:
    n, p = _validate_matrix(x)
    if len(weights) != n or len(y) != n:
        raise ValueError("weights or y have incompatible length")
    result = [0.0] * p
    for i, row in enumerate(x):
        wy = float(weights[i]) * float(y[i])
        for j in range(p):
            result[j] += row[j] * wy
    return result


def weighted_least_squares(x: Matrix, z: Sequence[float], weights: Sequence[float]) -> Vector:
    xtwx = weighted_crossproduct(x, weights)
    xtwz = weighted_xty(x, weights, z)
    return solve_linear_system(xtwx, xtwz)


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------

def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))


def normal_ppf(probability: float, tolerance: float = 1e-12) -> float:
    if not (0.0 < probability < 1.0):
        raise ValueError("probability must lie strictly between zero and one")
    if probability == 0.5:
        return 0.0
    if probability < 0.5:
        return -normal_ppf(1.0 - probability, tolerance=tolerance)
    low, high = 0.0, 1.0
    while normal_cdf(high) < probability:
        high *= 2.0
    for _ in range(120):
        mid = (low + high) / 2.0
        if normal_cdf(mid) < probability:
            low = mid
        else:
            high = mid
        if high - low < tolerance:
            break
    return (low + high) / 2.0


def stable_sigmoid(eta: float) -> float:
    if eta >= 0.0:
        z = math.exp(-eta)
        return 1.0 / (1.0 + z)
    z = math.exp(eta)
    return z / (1.0 + z)


def safe_log(value: float, floor: float = 1e-15) -> float:
    return math.log(max(float(value), floor))


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


@dataclass(frozen=True)
class GLMFamily:
    name: str
    link_name: str
    inverse_link: Callable[[float], float]
    dmu_deta: Callable[[float], float]
    variance: Callable[[float], float]
    deviance_component: Callable[[float, float], float]
    loglike_component: Callable[[float, float, float], float]
    valid_response: Callable[[float], bool]
    fixed_scale: Optional[float]
    initialize_mu: Callable[[float], float]


def gaussian_identity_family() -> GLMFamily:
    return GLMFamily(
        name="gaussian",
        link_name="identity",
        inverse_link=lambda eta: eta,
        dmu_deta=lambda eta: 1.0,
        variance=lambda mu: 1.0,
        deviance_component=lambda y, mu: (y - mu) ** 2,
        loglike_component=lambda y, mu, phi: -0.5 * (
            math.log(2.0 * math.pi * phi) + (y - mu) ** 2 / phi
        ),
        valid_response=lambda y: math.isfinite(y),
        fixed_scale=None,
        initialize_mu=lambda y: float(y),
    )


def binomial_logit_family() -> GLMFamily:
    def deviance(y: float, mu: float) -> float:
        mu = _clip(mu, 1e-15, 1.0 - 1e-15)
        first = 0.0 if y == 0.0 else y * math.log(y / mu)
        second = 0.0 if y == 1.0 else (1.0 - y) * math.log((1.0 - y) / (1.0 - mu))
        return 2.0 * (first + second)

    return GLMFamily(
        name="binomial",
        link_name="logit",
        inverse_link=stable_sigmoid,
        dmu_deta=lambda eta: max(stable_sigmoid(eta) * (1.0 - stable_sigmoid(eta)), 1e-15),
        variance=lambda mu: max(mu * (1.0 - mu), 1e-15),
        deviance_component=deviance,
        loglike_component=lambda y, mu, phi: y * safe_log(mu) + (1.0 - y) * safe_log(1.0 - mu),
        valid_response=lambda y: y in (0.0, 1.0),
        fixed_scale=1.0,
        initialize_mu=lambda y: (float(y) + 0.5) / 2.0,
    )


def poisson_log_family() -> GLMFamily:
    def inverse_link(eta: float) -> float:
        return math.exp(_clip(eta, -35.0, 35.0))

    def deviance(y: float, mu: float) -> float:
        mu = max(mu, 1e-15)
        if y == 0.0:
            return 2.0 * mu
        return 2.0 * (y * math.log(y / mu) - (y - mu))

    return GLMFamily(
        name="poisson",
        link_name="log",
        inverse_link=inverse_link,
        dmu_deta=lambda eta: inverse_link(eta),
        variance=lambda mu: max(mu, 1e-15),
        deviance_component=deviance,
        loglike_component=lambda y, mu, phi: y * safe_log(mu) - mu - math.lgamma(y + 1.0),
        valid_response=lambda y: y >= 0.0 and abs(y - round(y)) < 1e-9,
        fixed_scale=1.0,
        initialize_mu=lambda y: max(float(y), 0.1),
    )


def gamma_log_family() -> GLMFamily:
    def inverse_link(eta: float) -> float:
        return math.exp(_clip(eta, -35.0, 35.0))

    def deviance(y: float, mu: float) -> float:
        ratio = y / max(mu, 1e-15)
        return 2.0 * (ratio - 1.0 - math.log(ratio))

    def loglike(y: float, mu: float, phi: float) -> float:
        # Gamma parameterization: shape=1/phi, scale=phi*mu.
        phi = max(phi, 1e-15)
        shape = 1.0 / phi
        scale = phi * mu
        return (
            (shape - 1.0) * math.log(y)
            - y / scale
            - math.lgamma(shape)
            - shape * math.log(scale)
        )

    return GLMFamily(
        name="gamma",
        link_name="log",
        inverse_link=inverse_link,
        dmu_deta=lambda eta: inverse_link(eta),
        variance=lambda mu: max(mu * mu, 1e-15),
        deviance_component=deviance,
        loglike_component=loglike,
        valid_response=lambda y: y > 0.0 and math.isfinite(y),
        fixed_scale=None,
        initialize_mu=lambda y: max(float(y), 1e-6),
    )


def get_family(name: str) -> GLMFamily:
    key = name.lower().replace("-", "_")
    mapping = {
        "gaussian": gaussian_identity_family,
        "gaussian_identity": gaussian_identity_family,
        "binomial": binomial_logit_family,
        "binomial_logit": binomial_logit_family,
        "logistic": binomial_logit_family,
        "poisson": poisson_log_family,
        "poisson_log": poisson_log_family,
        "gamma": gamma_log_family,
        "gamma_log": gamma_log_family,
    }
    if key not in mapping:
        raise ValueError(f"unsupported family: {name}")
    return mapping[key]()


# ---------------------------------------------------------------------------
# GLM fitting and results
# ---------------------------------------------------------------------------

@dataclass
class GLMResult:
    family: GLMFamily
    coefficients: Vector
    covariance_matrix: Matrix
    standard_errors: Vector
    z_statistics: Vector
    p_values: Vector
    confidence_intervals: List[Tuple[float, float]]
    fitted_mean: Vector
    linear_predictor: Vector
    residuals_response: Vector
    residuals_pearson: Vector
    residuals_deviance: Vector
    leverage: Vector
    cooks_distance: Vector
    deviance: float
    null_deviance: float
    log_likelihood: float
    scale: float
    aic: float
    bic: float
    n_observations: int
    n_parameters: int
    degrees_of_freedom: int
    converged: bool
    iterations: int
    design_matrix: Matrix
    working_weights: Vector
    xtwx_inverse: Matrix
    includes_intercept: bool
    offset: Vector
    deviance_history: Vector

    def predict_linear(self, x_new: Matrix, offset: Optional[Sequence[float]] = None) -> Vector:
        design = add_intercept_column(x_new) if self.includes_intercept else x_new
        values = matvec(design, self.coefficients)
        if offset is None:
            offset_values = [0.0] * len(values)
        else:
            offset_values = [float(v) for v in offset]
            if len(offset_values) != len(values):
                raise ValueError("offset has incompatible length")
        return [a + b for a, b in zip(values, offset_values)]

    def predict_mean(self, x_new: Matrix, offset: Optional[Sequence[float]] = None) -> Vector:
        return [self.family.inverse_link(eta) for eta in self.predict_linear(x_new, offset=offset)]

    def coefficient_table(self) -> List[Dict[str, float]]:
        rows = []
        for j in range(self.n_parameters):
            rows.append(
                {
                    "index": float(j),
                    "coefficient": self.coefficients[j],
                    "standard_error": self.standard_errors[j],
                    "z": self.z_statistics[j],
                    "p_value": self.p_values[j],
                    "ci_lower": self.confidence_intervals[j][0],
                    "ci_upper": self.confidence_intervals[j][1],
                }
            )
        return rows


def _initial_eta(family: GLMFamily, y: Sequence[float]) -> Vector:
    result = []
    for value in y:
        mu = family.initialize_mu(float(value))
        if family.name == "gaussian":
            eta = mu
        elif family.name == "binomial":
            mu = _clip(mu, 1e-6, 1.0 - 1e-6)
            eta = math.log(mu / (1.0 - mu))
        else:
            eta = math.log(max(mu, 1e-12))
        result.append(eta)
    return result


def _deviance_residual(family: GLMFamily, y: float, mu: float) -> float:
    component = max(family.deviance_component(y, mu), 0.0)
    sign = 1.0 if y >= mu else -1.0
    return sign * math.sqrt(component)


def fit_glm_irls(
    x: Matrix,
    y: Sequence[float],
    family: str | GLMFamily = "binomial",
    add_intercept: bool = True,
    offset: Optional[Sequence[float]] = None,
    prior_weights: Optional[Sequence[float]] = None,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
    ridge: float = 0.0,
    confidence: float = 0.95,
) -> GLMResult:
    """Fit a GLM using Fisher scoring / IRLS.

    `ridge` adds a small L2 penalty to slope terms. It can stabilize educational
    examples but changes the estimator and classical covariance interpretation.
    """
    n, p_raw = _validate_matrix(x)
    ys = [float(v) for v in y]
    if len(ys) != n:
        raise ValueError("X and y have incompatible dimensions")
    fam = get_family(family) if isinstance(family, str) else family
    if any(not fam.valid_response(v) for v in ys):
        raise ValueError(f"response contains values invalid for {fam.name}")
    if max_iterations <= 0 or tolerance <= 0.0:
        raise ValueError("max_iterations and tolerance must be positive")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")

    design = add_intercept_column(x) if add_intercept else [[float(v) for v in row] for row in x]
    _, p = _validate_matrix(design)
    if n <= p:
        raise ValueError("more observations than parameters are required")

    offsets = [0.0] * n if offset is None else [float(v) for v in offset]
    if len(offsets) != n:
        raise ValueError("offset has incompatible length")
    prior = [1.0] * n if prior_weights is None else [float(v) for v in prior_weights]
    if len(prior) != n or any(v <= 0.0 for v in prior):
        raise ValueError("prior_weights must be positive and match the data")

    eta0 = _initial_eta(fam, ys)
    z0 = [eta0[i] - offsets[i] for i in range(n)]
    coefficients = weighted_least_squares(design, z0, prior)

    converged = False
    deviance_history: Vector = []
    working_weights = [1.0] * n
    for iteration in range(1, max_iterations + 1):
        linear = [value + offsets[i] for i, value in enumerate(matvec(design, coefficients))]
        mu = [fam.inverse_link(value) for value in linear]
        derivatives = [max(abs(fam.dmu_deta(value)), 1e-15) for value in linear]
        variances = [max(fam.variance(value), 1e-15) for value in mu]
        working_weights = [
            prior[i] * derivatives[i] ** 2 / variances[i]
            for i in range(n)
        ]
        z = [
            linear[i] + (ys[i] - mu[i]) / derivatives[i] - offsets[i]
            for i in range(n)
        ]

        xtwx = weighted_crossproduct(design, working_weights)
        if ridge > 0.0:
            start = 1 if add_intercept else 0
            for j in range(start, p):
                xtwx[j][j] += ridge
        xtwz = weighted_xty(design, working_weights, z)
        new_coefficients = solve_linear_system(xtwx, xtwz)

        new_linear = [value + offsets[i] for i, value in enumerate(matvec(design, new_coefficients))]
        new_mu = [fam.inverse_link(value) for value in new_linear]
        new_deviance = sum(fam.deviance_component(ys[i], new_mu[i]) * prior[i] for i in range(n))
        deviance_history.append(new_deviance)

        max_change = max(abs(a - b) for a, b in zip(new_coefficients, coefficients))
        coefficients = new_coefficients
        if max_change <= tolerance * (1.0 + max(abs(v) for v in coefficients)):
            converged = True
            break

    linear = [value + offsets[i] for i, value in enumerate(matvec(design, coefficients))]
    mu = [fam.inverse_link(value) for value in linear]
    derivatives = [max(abs(fam.dmu_deta(value)), 1e-15) for value in linear]
    variances = [max(fam.variance(value), 1e-15) for value in mu]
    working_weights = [prior[i] * derivatives[i] ** 2 / variances[i] for i in range(n)]
    xtwx = weighted_crossproduct(design, working_weights)
    if ridge > 0.0:
        start = 1 if add_intercept else 0
        for j in range(start, p):
            xtwx[j][j] += ridge
    xtwx_inverse = inverse(xtwx)

    df = n - p
    response_residuals = [ys[i] - mu[i] for i in range(n)]
    pearson = [response_residuals[i] / math.sqrt(variances[i]) for i in range(n)]
    deviance_residuals = [_deviance_residual(fam, ys[i], mu[i]) for i in range(n)]
    pearson_chi2 = sum(prior[i] * pearson[i] ** 2 for i in range(n))
    scale = fam.fixed_scale if fam.fixed_scale is not None else pearson_chi2 / df

    covariance = [[scale * value for value in row] for row in xtwx_inverse]
    standard_errors = [math.sqrt(max(covariance[j][j], 0.0)) for j in range(p)]
    z_statistics = [
        coefficients[j] / standard_errors[j] if standard_errors[j] > 0.0 else math.inf
        for j in range(p)
    ]
    p_values = [2.0 * (1.0 - normal_cdf(abs(value))) for value in z_statistics]
    critical = normal_ppf(0.5 + confidence / 2.0)
    intervals = [
        (coefficients[j] - critical * standard_errors[j], coefficients[j] + critical * standard_errors[j])
        for j in range(p)
    ]

    leverage = []
    cooks = []
    for i, row in enumerate(design):
        weighted_row = [math.sqrt(working_weights[i]) * value for value in row]
        h = dot(weighted_row, matvec(xtwx_inverse, weighted_row))
        h = _clip(h, 0.0, 1.0 - 1e-12)
        leverage.append(h)
        cooks.append(
            (pearson[i] ** 2 / max(p * scale, 1e-15)) * h / max((1.0 - h) ** 2, 1e-15)
        )

    deviance = sum(prior[i] * fam.deviance_component(ys[i], mu[i]) for i in range(n))
    weighted_mean = sum(prior[i] * ys[i] for i in range(n)) / sum(prior)
    null_mu = [weighted_mean] * n
    if fam.name == "binomial":
        null_mu = [_clip(weighted_mean, 1e-15, 1.0 - 1e-15)] * n
    elif fam.name in {"poisson", "gamma"}:
        null_mu = [max(weighted_mean, 1e-15)] * n
    null_deviance = sum(prior[i] * fam.deviance_component(ys[i], null_mu[i]) for i in range(n))
    log_likelihood = sum(
        prior[i] * fam.loglike_component(ys[i], mu[i], scale)
        for i in range(n)
    )
    aic = -2.0 * log_likelihood + 2.0 * p
    bic = -2.0 * log_likelihood + p * math.log(n)

    return GLMResult(
        family=fam,
        coefficients=coefficients,
        covariance_matrix=covariance,
        standard_errors=standard_errors,
        z_statistics=z_statistics,
        p_values=p_values,
        confidence_intervals=intervals,
        fitted_mean=mu,
        linear_predictor=linear,
        residuals_response=response_residuals,
        residuals_pearson=pearson,
        residuals_deviance=deviance_residuals,
        leverage=leverage,
        cooks_distance=cooks,
        deviance=deviance,
        null_deviance=null_deviance,
        log_likelihood=log_likelihood,
        scale=scale,
        aic=aic,
        bic=bic,
        n_observations=n,
        n_parameters=p,
        degrees_of_freedom=df,
        converged=converged,
        iterations=iteration,
        design_matrix=design,
        working_weights=working_weights,
        xtwx_inverse=xtwx_inverse,
        includes_intercept=add_intercept,
        offset=offsets,
        deviance_history=deviance_history,
    )


# ---------------------------------------------------------------------------
# Sandwich covariance and metrics
# ---------------------------------------------------------------------------

def sandwich_covariance(model: GLMResult, kind: str = "HC0") -> Matrix:
    """Observation-level sandwich covariance based on GLM score contributions."""
    key = kind.upper()
    if key not in {"HC0", "HC1", "HC3"}:
        raise ValueError("kind must be HC0, HC1, or HC3")
    x = model.design_matrix
    p = model.n_parameters
    n = model.n_observations
    meat = [[0.0 for _ in range(p)] for _ in range(p)]

    for i, row in enumerate(x):
        mu = model.fitted_mean[i]
        eta = model.linear_predictor[i]
        derivative = max(abs(model.family.dmu_deta(eta)), 1e-15)
        variance = max(model.family.variance(mu), 1e-15)
        scalar_score = model.residuals_response[i] * derivative / variance
        if key == "HC3":
            scalar_score /= max(1.0 - model.leverage[i], 1e-15)
        for a in range(p):
            for b in range(p):
                meat[a][b] += scalar_score ** 2 * row[a] * row[b]

    covariance = matmul(matmul(model.xtwx_inverse, meat), model.xtwx_inverse)
    if key == "HC1":
        factor = n / (n - p)
        covariance = [[factor * value for value in row] for row in covariance]
    return covariance


def mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("inputs must be non-empty and have equal length")
    return sum(abs(float(a) - float(b)) for a, b in zip(y_true, y_pred)) / len(y_true)


def root_mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("inputs must be non-empty and have equal length")
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true))


def binary_log_loss(y_true: Sequence[float], probabilities: Sequence[float]) -> float:
    if len(y_true) != len(probabilities) or not y_true:
        raise ValueError("inputs must be non-empty and have equal length")
    total = 0.0
    for y, p in zip(y_true, probabilities):
        p = _clip(float(p), 1e-15, 1.0 - 1e-15)
        total -= float(y) * math.log(p) + (1.0 - float(y)) * math.log(1.0 - p)
    return total / len(y_true)


def poisson_deviance(y_true: Sequence[float], mu: Sequence[float]) -> float:
    family = poisson_log_family()
    return sum(family.deviance_component(float(y), float(m)) for y, m in zip(y_true, mu))


def train_test_split(
    x: Matrix,
    y: Sequence[float],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[Matrix, Matrix, Vector, Vector]:
    n, _ = _validate_matrix(x)
    if len(y) != n or not (0.0 < test_fraction < 1.0):
        raise ValueError("invalid dimensions or test_fraction")
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    n_test = max(1, int(round(n * test_fraction)))
    test = set(indices[:n_test])
    x_train, x_test, y_train, y_test = [], [], [], []
    for i in range(n):
        target_x, target_y = (x_test, y_test) if i in test else (x_train, y_train)
        target_x.append([float(v) for v in x[i]])
        target_y.append(float(y[i]))
    return x_train, x_test, y_train, y_test


def cross_validate_glm(
    x: Matrix,
    y: Sequence[float],
    family: str,
    k: int = 5,
    seed: int = 42,
) -> Dict[str, object]:
    n, _ = _validate_matrix(x)
    if len(y) != n or not (2 <= k <= n):
        raise ValueError("invalid dimensions or k")
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    folds = [[] for _ in range(k)]
    for position, index in enumerate(indices):
        folds[position % k].append(index)

    scores = []
    for validation in folds:
        validation_set = set(validation)
        x_train = [x[i] for i in range(n) if i not in validation_set]
        y_train = [float(y[i]) for i in range(n) if i not in validation_set]
        x_valid = [x[i] for i in validation]
        y_valid = [float(y[i]) for i in validation]
        model = fit_glm_irls(x_train, y_train, family=family)
        prediction = model.predict_mean(x_valid)
        if family.lower().startswith("binomial") or family.lower() == "logistic":
            score = binary_log_loss(y_valid, prediction)
            metric = "log_loss"
        elif family.lower().startswith("poisson"):
            score = poisson_deviance(y_valid, prediction) / len(y_valid)
            metric = "mean_poisson_deviance"
        else:
            score = root_mean_squared_error(y_valid, prediction)
            metric = "rmse"
        scores.append(score)
    return {"metric": metric, "fold_scores": scores, "mean_score": sum(scores) / len(scores)}


# ---------------------------------------------------------------------------
# Synthetic data and demonstration
# ---------------------------------------------------------------------------

def simulate_logistic(seed: int = 42, n: int = 300) -> Tuple[Matrix, List[int]]:
    rng = random.Random(seed)
    x, y = [], []
    for _ in range(n):
        age = rng.uniform(-2.0, 2.0)
        programme = 1.0 if rng.random() < 0.4 else 0.0
        eta = -0.4 + 1.1 * age + 0.8 * programme
        probability = stable_sigmoid(eta)
        x.append([age, programme])
        y.append(1 if rng.random() < probability else 0)
    return x, y


def simulate_poisson(seed: int = 42, n: int = 300) -> Tuple[Matrix, List[int], Vector]:
    rng = random.Random(seed)
    x, y, exposure = [], [], []
    for _ in range(n):
        remoteness = rng.uniform(0.0, 2.0)
        trained = 1.0 if rng.random() < 0.5 else 0.0
        exp_time = rng.uniform(0.5, 3.0)
        eta_rate = 0.3 + 0.5 * remoteness - 0.4 * trained
        mean_count = exp_time * math.exp(eta_rate)
        # Knuth algorithm for educational Poisson sampling.
        limit = math.exp(-mean_count)
        product = 1.0
        count = 0
        while product > limit:
            count += 1
            product *= rng.random()
        x.append([remoteness, trained])
        y.append(count - 1)
        exposure.append(exp_time)
    return x, y, exposure


def demo() -> None:
    x, y = simulate_logistic()
    model = fit_glm_irls(x, y, family="binomial")
    print("Logistic coefficients:", [round(v, 4) for v in model.coefficients])
    print("Converged:", model.converged, "iterations:", model.iterations)

    xp, yp, exposure = simulate_poisson()
    poisson = fit_glm_irls(
        xp,
        yp,
        family="poisson",
        offset=[math.log(value) for value in exposure],
    )
    print("Poisson rate coefficients:", [round(v, 4) for v in poisson.coefficients])
    print("Poisson deviance:", round(poisson.deviance, 3))


if __name__ == "__main__":
    demo()
```


# Appendix B. Minimal Verification Tests

The verification suite checks analytic behavior, probability bounds, convergence, and coefficient recovery. Expand it whenever a family or diagnostic is added.

```python
import math
import random

from glm_from_scratch import (
    fit_glm_irls,
    simulate_logistic,
    simulate_poisson,
    stable_sigmoid,
    normal_cdf,
    normal_ppf,
)


def close(a, b, tol=1e-7):
    return abs(a - b) <= tol * (1.0 + max(abs(a), abs(b)))


# Stable sigmoid and normal inverse checks.
assert 0.0 <= stable_sigmoid(-1000.0) < 1e-100
assert 1.0 - stable_sigmoid(1000.0) < 1e-12
for p in [0.001, 0.025, 0.5, 0.975, 0.999]:
    assert close(normal_cdf(normal_ppf(p)), p, tol=1e-9)

# Gaussian identity recovers an exact plane.
X = [[1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0], [5.0, 0.0], [6.0, 1.0]]
y = [3.0 + 2.0 * a - 4.0 * b for a, b in X]
gaussian = fit_glm_irls(X, y, family="gaussian")
assert gaussian.converged
assert all(close(a, b) for a, b in zip(gaussian.coefficients, [3.0, 2.0, -4.0]))

# Logistic probabilities are valid and the simulation fit converges.
Xb, yb = simulate_logistic(seed=7, n=400)
logit = fit_glm_irls(Xb, yb, family="binomial")
assert logit.converged
assert all(0.0 <= p <= 1.0 for p in logit.fitted_mean)
assert len(logit.coefficients) == 3

# Poisson model with exposure produces positive predictions.
Xp, yp, exposure = simulate_poisson(seed=8, n=400)
poisson = fit_glm_irls(
    Xp,
    yp,
    family="poisson",
    offset=[math.log(v) for v in exposure],
)
assert poisson.converged
assert all(mu > 0.0 for mu in poisson.fitted_mean)
assert poisson.deviance >= 0.0

# Gamma model on generated positive data.
rng = random.Random(11)
Xg, yg = [], []
for _ in range(500):
    x1 = rng.uniform(-1.5, 1.5)
    x2 = rng.uniform(-1.0, 1.0)
    mu = math.exp(0.4 + 0.3 * x1 - 0.2 * x2)
    # Exponential is Gamma with shape 1, scale mu.
    y_value = rng.expovariate(1.0 / mu)
    Xg.append([x1, x2])
    yg.append(y_value)
gamma = fit_glm_irls(Xg, yg, family="gamma")
assert gamma.converged
assert all(mu > 0.0 for mu in gamma.fitted_mean)
assert gamma.scale > 0.0

print("All pure-Python GLM verification tests passed.")
```


# Appendix C. Research Project Template


## Question

State the outcome, unit, target population, predictor or intervention, and intended use.


## Estimand

Write the target in words and mathematical notation. Specify whether it is conditional or marginal.


## Data generation

Describe sampling, timing, exposure, repeated observations, missingness, and measurement.


## Model

State family, link, linear predictor, offset, weights, dispersion, and dependence assumptions.


## Estimation

State algorithm, software, convergence criteria, covariance estimator, and penalty or prior.


## Diagnostics

Include support checks, residuals, dispersion, influence, calibration, and sensitivity.


## Validation

Use a split matching deployment structure and report family-appropriate metrics.


## Results

Report coefficients on both link and outcome scales, uncertainty, and meaningful predictions.


## Limitations

Separate design, measurement, model, computation, external validity, and causal limitations.


## Reproducibility

Provide code, environment, seeds, data provenance, and an execution guide.


# Final Perspective

A generalized linear model is not simply linear regression with a different button. It is a disciplined agreement among outcome support, conditional mean, variance function, link, likelihood or estimating equation, computation, diagnostics, and interpretation.

Research-level competence requires four abilities at once: derive the model, implement its core, diagnose its failure modes, and connect its numerical output to a precisely defined scientific estimand. The next topics after this volume are multilevel models, survival analysis, panel and longitudinal models, causal inference, Bayesian modelling, and flexible nonlinear statistical learning.
