# Regression: A Mathematical, Statistical, and Python Guide

*From Zero to Research-Level Modelling — Expanded Edition*

Prepared as a self-study, NotebookLM, and AI-tutor source

> Expanded edition note. This revision preserves the original regression text and adds prerequisite mathematics and Python, research-grade covariance and resampling methods, missing-data and survey considerations, a paper-reading framework, an expanded glossary, and an annotated research literature path.


## How to use this guide

**Purpose.** This is a self-contained source for learning regression through mathematics, intuition, pure Python, and modern Python libraries. It is designed to be read directly or uploaded as a source to an AI study tool. It does not assume prior knowledge of calculus beyond basic derivatives, although the matrix chapters become easier if you know elementary linear algebra.

**Meaning of “pure Python.”** In Part II, “pure Python” means no third-party numerical, statistical, or machine-learning packages. Only the Python standard library is used. The point is not to compete with NumPy or statsmodels. The point is to expose every important calculation.

**Scope.** The guide covers:

- simple and multiple linear regression;
- least squares derivations and matrix geometry;
- assumptions, standard errors, confidence intervals, hypothesis tests, and joint tests;
- prediction and prediction intervals;
- residual diagnostics, leverage, influence, multicollinearity, heteroskedasticity, and dependence;
- categorical variables, interactions, transformations, and polynomial terms;
- train/test evaluation and cross-validation;
- ridge, lasso, and elastic-net regularization;
- logistic regression as the most important extension to a non-continuous outcome;
- implementations from scratch and then with NumPy, pandas, statsmodels, scikit-learn, and matplotlib;
- exercises, teaching prompts, and a glossary.

Regression is not one algorithm. It is a family of ways to describe how an outcome varies with one or more predictors. The mathematical core is a conditional relationship; the computational core is optimization; the statistical core is uncertainty; and the practical core is disciplined judgement.


### Contents at a glance

- Part 0, Sections 0.1-0.12: prerequisite algebra, probability, calculus, linear algebra, Python, numerical stability, and simulation.
- Part I, Sections 1-14: mathematical foundations, assumptions, inference, diagnostics, regularization, logistic regression, and causal cautions.
- Part II, Sections 15-24: pure-Python implementation using only the standard library.
- Part III, Sections 25-33: NumPy, pandas, statsmodels, scikit-learn, plots, workflows, and common mistakes.
- Part IV, Sections 34-41: exercises, learning sequence, AI-tutor prompts, glossary, and further study.

Part V, Sections 42-55: research-grade regression practice, paper reading, expanded glossary, and research papers.

Appendices A-D: executable code, verification tests, and research-extension exercises.


# Part 0. Foundations for a Zero-to-Research Pathway


## 0.1 What this expanded edition adds

The original guide already provides a strong account of ordinary least squares, model assumptions, inference, diagnostics, regularization, logistic regression, and implementation. This expanded edition adds the prerequisites and research practices that are usually left implicit. It is designed for a learner who may be starting with elementary mathematics, introductory statistics, and basic Python, but who intends to reach the level at which research articles can be read, reproduced, criticized, and extended.

Use Part 0 before Part I when any notation or programming construct feels unfamiliar. Return to it whenever a later derivation becomes opaque. Research-level competence does not mean memorizing every formula. It means being able to reconstruct the logic, state the assumptions, implement the method, verify the implementation, and explain what evidence the fitted model does and does not provide.


## 0.2 A map of the knowledge you are building

Regression draws on six connected forms of knowledge:

- substantive reasoning: what is being measured, why it matters, and how the data arose;

- algebra: manipulating equations and understanding functions;

- probability and statistics: distinguishing a population process from a realized sample;

- linear algebra: expressing many equations compactly and understanding projection geometry;

- optimization and numerical analysis: finding estimates reliably on a finite-precision computer;

- scientific workflow: making decisions reproducible, testable, and open to criticism.

A learner often becomes stuck because one of these layers is missing, not because regression itself is mysterious. When difficulty arises, identify the missing layer explicitly.


## 0.3 Essential algebra

A variable is a symbol that can take a value. A parameter is an unknown but fixed feature of a population model. An estimator is a rule that turns sample data into an estimate of that parameter. A random variable represents an uncertain numerical outcome before observation.

The distributive law is central:

```
a(b + c) = ab + ac
```

The rule for moving a term across an equality is not a special operation; it is shorthand for applying the same operation to both sides. If

```
y = a + bx
```

then subtracting a and dividing by b, provided b is not zero, gives

```
x = (y - a) / b
```

Summation notation compresses repeated addition:

```
sum_{i=1}^n x_i = x_1 + x_2 + ... + x_n
```

Linearity of summation allows constants to be moved outside:

```
sum_i (a + b x_i) = n a + b sum_i x_i
```

This identity is used repeatedly in deriving normal equations and expectations.


## 0.4 Functions, logs, exponentials, and transformations

A function maps an input to an output. In y = f(x), x is the argument and y is the function value. A linear function has constant slope. A nonlinear function may still be used in a model that is linear in its unknown coefficients.

The exponential function exp(x) is always positive. The natural logarithm log(x) is its inverse:

```
log(exp(x)) = x, and exp(log(x)) = x for x > 0
```

Important log rules are:

```
log(ab) = log(a) + log(b)
```

```
log(a/b) = log(a) - log(b)
```

```
log(a^c) = c log(a)
```

These rules turn products in likelihoods into sums in log-likelihoods. They also explain why multiplicative relationships often become additive on a log scale.


## 0.5 Derivatives and optimization

A derivative measures local rate of change. If f(x) = x^2, then f'(x) = 2x. If f(x) = exp(x), then f'(x) = exp(x). If f(x) = log(x), then f'(x) = 1/x.

The chain rule differentiates a composition:

```
d f(g(x)) / dx = f'(g(x)) g'(x)
```

Optimization uses derivatives to locate candidate minima or maxima. A differentiable interior minimum usually satisfies f'(x) = 0. The second derivative indicates local curvature: f''(x) > 0 suggests a local minimum, while f''(x) < 0 suggests a local maximum.

For several parameters, the gradient is the vector of first derivatives and the Hessian is the matrix of second derivatives. Ordinary least squares has a quadratic objective with a constant Hessian 2 X^T X. Positive definiteness of X^T X under full column rank gives a unique minimum.


## 0.6 Probability foundations

A probability lies between zero and one. A probability distribution assigns probabilities to possible outcomes. The expected value is a probability-weighted average over repeated realizations:

```
E[X] = sum_x x P(X=x) for a discrete variable
```

Variance measures spread around the expectation:

```
Var(X) = E[(X - E[X])^2] = E[X^2] - E[X]^2
```

Covariance measures joint movement:

```
Cov(X,Y) = E[(X-E[X])(Y-E[Y])]
```

Independence is stronger than zero covariance. Independent variables have zero covariance when moments exist, but zero covariance does not generally imply independence.

Conditional expectation is the central object in regression. E[Y | X=x] is not a prediction for a metaphysically identical individual; it is an average over the conditional distribution of Y among cases represented by x.


## 0.7 Samples, estimands, estimators, and estimates

An estimand is the precise population quantity the analysis seeks. Examples include a conditional mean difference, a prediction risk, an average treatment effect, or a slope in a specified projection. An estimator is the procedure used to target it. An estimate is the realized numerical answer.

These distinctions prevent a common mistake: treating a coefficient as if its meaning came solely from software output. Meaning comes from the estimand, the model, the data-generating process, and the design.

Sampling variation means that an estimator changes across hypothetical repeated samples. Bias, variance, and mean squared error refer to this sampling distribution, not merely to the residuals of one fitted model.


## 0.8 Linear algebra without fear

A vector is an ordered list of numbers. A matrix is a rectangular array. Matrix multiplication is defined so that each output element is a dot product of a row and a column. Dimensions must agree:

```
(n x k)(k x p) = (n x p)
```

The transpose X^T exchanges rows and columns. The dot product a^T b measures aligned magnitude. Orthogonality means a^T b = 0.

The rank of X is the number of linearly independent column directions. Full column rank is required for a unique ordinary least-squares coefficient vector. The condition number describes sensitivity: a large value indicates that small perturbations in data or arithmetic may cause large changes in coefficients.

An eigenvector is a direction whose orientation is preserved by a matrix transformation. A singular value measures how strongly a matrix stretches an orthogonal direction. Singular-value decomposition is valuable because it reveals rank, condition, and low-dimensional structure even when X^T X is poorly behaved.


## 0.9 Python foundations required by this guide

A Python list stores an ordered collection. A list of lists can represent a matrix. A function packages a reusable calculation. A class packages state and behavior. Exceptions signal invalid inputs rather than silently producing misleading output.

A minimal research-quality function should:

- state what inputs it expects;

- validate dimensions and types where failure would mislead;

- document the returned quantities;

- separate numerical computation from printing;

- include tests for known answers and edge cases;

- avoid hidden dependence on global variables;

- expose random seeds when randomness is used.

Example:

```python
def mean(values):
    values = [float(v) for v in values]
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)
```

The conversion to float makes the intended numerical type explicit. The validation prevents a division-by-zero error whose message would be less informative.


## 0.10 Floating-point arithmetic and numerical error

Computers represent most real numbers approximately. Therefore expressions that are mathematically equal may differ by a tiny amount in code. Never test floating-point results using exact equality unless the result is guaranteed to be exactly representable.

```python
def is_close(a, b, rel_tol=1e-9, abs_tol=1e-12):
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))
```

Numerical stability asks whether an algorithm controls amplification of rounding error. The normal-equation formula is mathematically correct but often less stable than QR or SVD because forming X^T X squares the condition number.


## 0.11 Simulation as a bridge between theory and research

Simulation is an experiment in which the data-generating process is known. It can verify theoretical claims, expose finite-sample behavior, and test code.

A simulation study should specify:

- the estimand and true parameter values;

- the data-generating process;

- sample sizes and design conditions;

- the number of replications;

- evaluation measures such as bias, RMSE, coverage, and convergence failure;

- random seeds and software versions;

- uncertainty in the simulation summaries themselves.

A single simulated dataset is an illustration. A simulation study requires repeated datasets and a systematic design.


## 0.12 A prerequisite mastery checklist

Before moving beyond simple regression, the reader should be able to:

- rearrange a linear equation and use summation notation;

- explain expectation, variance, covariance, and conditional expectation;

- differentiate a quadratic and apply the chain rule;

- multiply small matrices by hand and check dimensions;

- explain why X^T X can be singular;

- write and test a Python function with input validation;

- distinguish a population disturbance from a sample residual;

- distinguish an estimand, estimator, estimate, and standard error.


# Part I. Mathematical Foundations


## 1. What regression is trying to do

Suppose we observe an outcome Y and predictors X. Examples include:

- crop yield as a function of rainfall, fertilizer, and soil quality;
- household income as a function of education and experience;
- programme cost as a function of remoteness and number of beneficiaries;
- electricity demand as a function of temperature and time;
- the probability of loan default as a function of borrower characteristics.

The central object is often the **conditional mean**:

E[Y | X = x]

This means the average value of Y among cases having predictor value x. A regression model proposes a mathematical form for this conditional relationship.

For linear regression with one predictor:

E[Y | X = x] = beta_0 + beta_1 x

For several predictors:

E[Y | X_1, ..., X_p] = beta_0 + beta_1 X_1 + ... + beta_p X_p

A fitted model replaces unknown population parameters beta_j with estimates b_j or beta_hat_j:

y_hat = b_0 + b_1 x_1 + ... + b_p x_p


### 1.1 Four different goals

A regression analysis can pursue four distinct goals. Confusing them causes many errors.

- **Description.** Summarize patterns in the observed data.
- **Prediction.** Estimate outcomes for new cases.
- **Explanation.** Quantify conditional associations and explore mechanisms.
- **Causal inference.** Estimate what would happen under an intervention.

The same regression equation can be used for all four, but the evidence required is not the same. Good predictive performance does not establish causality. A causal coefficient may come from a model that is not optimized for prediction. A descriptive association may be useful even when neither predictive nor causal claims are warranted.


### 1.2 Regression does not automatically mean a straight line

“Linear regression” means linear in the unknown coefficients, not necessarily linear in the raw predictor. These are all linear regression models in their parameters:

- Y = beta_0 + beta_1 X + error
- Y = beta_0 + beta_1 X + beta_2 X^2 + error
- log(Y) = beta_0 + beta_1 X + error
- Y = beta_0 + beta_1 log(X) + error
- Y = beta_0 + beta_1 X + beta_2 Z + beta_3 XZ + error

They can be fitted by ordinary least squares because the coefficients enter additively.


### 1.3 Population, sample, signal, and noise

A useful conceptual model is:

observed outcome = systematic component + unexplained component

or

Y_i = f(X_i) + epsilon_i

For a linear model:

Y_i = beta_0 + beta_1 X_i1 + ... + beta_p X_ip + epsilon_i

Here:

- i indexes observations;
- beta_0 is the intercept;
- beta_j is a population slope;
- epsilon_i is the disturbance: everything affecting Y_i that the model does not explicitly include.

A fitted residual is not the same as the unobservable population disturbance:

e_i = y_i - y_hat_i

The disturbance exists in the population model. The residual is computed after fitting the sample model.


## 2. Notation and data geometry

Let there be n observations and p predictors.

- y is an n x 1 vector of outcomes.
- X is an n x (p+1) design matrix when an intercept column is included.
- beta is a (p+1) x 1 vector of population coefficients.
- epsilon is an n x 1 vector of disturbances.

The model is:

y = X beta + epsilon

A typical design matrix is:

```
X = [1  x_11  x_12  ...  x_1p]
    [1  x_21  x_22  ...  x_2p]
    [                 ...       ]
    [1  x_n1  x_n2  ...  x_np]
```

The first column of ones represents the intercept.


### 2.1 Vectors as points and columns as directions

Each column of X is a vector in n-dimensional observation space. All linear combinations of the columns form the **column space of X**. The fitted vector is:

y_hat = X beta_hat

Therefore y_hat must lie in the column space of X. Least squares finds the point in that column space closest to the observed vector y.

The residual vector is:

e = y - y_hat

At the least-squares solution, e is perpendicular to every column of X:

X^T e = 0

This orthogonality is one of the most important facts in regression. It explains the normal equations, why residuals sum to zero when an intercept is included, and why fitted values are uncorrelated with residuals in the sample.


## 3. Simple linear regression

The simple population model is:

Y_i = beta_0 + beta_1 X_i + epsilon_i

The fitted line is:

y_hat_i = b_0 + b_1 x_i

The residual is:

e_i = y_i - b_0 - b_1 x_i

Ordinary least squares chooses b_0 and b_1 to minimize the sum of squared residuals:

SSE(b_0, b_1) = sum_i (y_i - b_0 - b_1 x_i)^2

Why squares?

- positive and negative errors cannot cancel;
- large errors are penalized more strongly;
- the objective is differentiable;
- under normally distributed errors, least squares is also maximum likelihood;
- the geometry leads to a unique projection when the design has full rank.

### 3.1 Deriving the slope and intercept

Differentiate the SSE with respect to the two coefficients and set the derivatives equal to zero.

With respect to b_0:

dSSE/db_0 = -2 sum_i (y_i - b_0 - b_1 x_i) = 0

Therefore:

sum_i e_i = 0

and:

b_0 = y_bar - b_1 x_bar

With respect to b_1:

dSSE/db_1 = -2 sum_i x_i(y_i - b_0 - b_1 x_i) = 0

Substitute the intercept expression and simplify:

b_1 = sum_i (x_i - x_bar)(y_i - y_bar) / sum_i (x_i - x_bar)^2

The intercept is:

b_0 = y_bar - b_1 x_bar

Using sample covariance and variance:

b_1 = Cov(X,Y) / Var(X)

provided both use the same denominator. The slope is also related to correlation:

b_1 = r_xy (s_y / s_x)

This shows that correlation is scale-free, whereas the regression slope depends on measurement units.


### 3.2 Interpreting the coefficients

- b_0 is the predicted outcome when X = 0.
- b_1 is the predicted change in Y for a one-unit increase in X.

The intercept may have no substantive meaning when zero is outside the data range. It is still algebraically important because it anchors the fitted line and allows residuals to sum to zero.


### 3.3 Worked numerical example

Suppose:

```
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]
```

Then:

```
x_bar = 3
y_bar = 4
sum((x-x_bar)(y-y_bar)) = 6
sum((x-x_bar)^2) = 10
b_1 = 6/10 = 0.6
b_0 = 4 - 0.6(3) = 2.2
```

The fitted equation is:

y_hat = 2.2 + 0.6x

At x = 4, the prediction is 4.6; the observed value is 4; the residual is -0.6.


### 3.4 What least squares guarantees in the sample

When the model includes an intercept:

- residuals sum to zero: sum e_i = 0;
- the fitted line passes through (x_bar, y_bar);
- residuals are orthogonal to x: sum x_i e_i = 0;
- residuals are orthogonal to fitted values: sum y_hat_i e_i = 0;
- the sample mean of fitted values equals the sample mean of observed outcomes.

These are algebraic properties. They do not prove that the model is correct.


## 4. Multiple linear regression

The multiple model is:

Y_i = beta_0 + beta_1 X_i1 + ... + beta_p X_ip + epsilon_i

A coefficient has a **partial** interpretation. For example, in:

income = beta_0 + beta_1 education + beta_2 experience + epsilon

beta_1 is the expected difference in income associated with one additional unit of education **holding experience constant**.

“Holding constant” is mathematical, not magical. It means the coefficient uses variation in education not linearly explained by the other included predictors. It does not guarantee that all relevant differences have been controlled.


### 4.1 The least-squares objective in matrix form

The residual vector is:

e = y - Xb

The sum of squared errors is:

SSE(b) = e^T e = (y - Xb)^T(y - Xb)

Expand:

SSE(b) = y^T y - 2b^T X^T y + b^T X^T X b

Differentiate with respect to b:

dSSE/db = -2X^T y + 2X^T Xb

Set equal to zero:

X^T X b = X^T y

These are the **normal equations**. If X^T X is invertible:

beta_hat = (X^T X)^(-1) X^T y

In serious numerical computing, one usually avoids explicitly forming the inverse. QR decomposition or singular-value decomposition is more stable. The inverse formula remains central for theory.


### 4.2 Projection matrix and residual-maker matrix

Substitute the OLS estimator into the fitted values:

y_hat = X(X^T X)^(-1)X^T y = Hy

where:

H = X(X^T X)^(-1)X^T

H is the **hat matrix** because it puts the “hat” on y. It is symmetric and idempotent:

H^T = H

H^2 = H

The residuals are:

e = y - Hy = (I-H)y = My

where M = I-H is the residual-maker matrix.


### 4.3 Frisch-Waugh-Lovell theorem

Suppose you want the coefficient on X_1 while controlling for X_2, ..., X_p.

- Regress X_1 on the other predictors and retain residuals r_X.
- Regress Y on the other predictors and retain residuals r_Y.
- Regress r_Y on r_X without an intercept.

The resulting slope is exactly the same as the coefficient on X_1 in the full multiple regression.

This theorem gives a precise meaning to “holding other variables constant”: compare the portions of X_1 and Y not explained by the controls.


### 4.4 Perfect and near multicollinearity

**Perfect multicollinearity** occurs when one column of X is an exact linear combination of others. Then X^T X is singular and the coefficients are not uniquely identified.

Examples:

- including an intercept and all categories of a categorical variable;
- including both total expenditure and components that sum exactly to total expenditure;
- including age, birth year, and survey year when the identity is exact.

**Near multicollinearity** means predictors are highly, but not perfectly, related. OLS still exists, but coefficient estimates can become unstable and standard errors large. Prediction may remain good even when individual coefficient interpretation becomes weak.


## 5. Assumptions: what each one is needed for

Regression assumptions are often taught as a single list. A better approach is to ask what conclusion each assumption supports.


### 5.1 Linearity in parameters and correct functional form

The conditional mean is represented correctly by the chosen terms:

E[Y | X] = X beta

This does not require a straight-line relationship in raw variables. It requires that transformations, interactions, and nonlinear terms be adequate.

If the functional form is wrong, residuals often show systematic patterns and predictions can be biased.


### 5.2 Independent or appropriately modelled observations

Classical formulas assume observations are independent across i, or at least that dependence is handled correctly. Dependence arises with:

- repeated measurements of the same person;
- students within schools;
- households within villages;
- observations over time;
- spatially neighbouring units.

Ignoring dependence usually makes standard errors too small. Remedies include cluster-robust standard errors, mixed models, generalized estimating equations, time-series models, or spatial models.


### 5.3 No perfect multicollinearity

The columns of X must be linearly independent. This is necessary for unique coefficient estimates.


### 5.4 Zero conditional mean

The critical exogeneity assumption is:

E[epsilon | X] = 0

It says that, after conditioning on the included predictors, the average omitted disturbance is zero for every predictor combination.

Violations include:

- omitted confounders correlated with included predictors;
- reverse causality;
- simultaneous determination;
- systematic measurement error in predictors;
- sample selection tied to unobserved determinants of the outcome.

If zero conditional mean fails, OLS coefficients may be biased and inconsistent. More data does not repair structural bias.


### 5.5 Homoskedasticity

Classical OLS inference assumes:

Var(epsilon_i | X) = sigma^2

for all observations. If variance changes with predictors, errors are heteroskedastic. OLS coefficients can still be unbiased under exogeneity, but classical standard errors are wrong. Heteroskedasticity-robust standard errors provide a common remedy for inference.


### 5.6 Normality

Exact small-sample t and F inference traditionally assumes:

epsilon | X ~ Normal(0, sigma^2 I)

Normality is not required for OLS coefficients to exist or for the Gauss-Markov theorem. In large samples, approximate inference often follows from asymptotic theory. Normality matters most for exact small-sample inference and prediction intervals.


### 5.7 Gauss-Markov theorem

Under linearity, exogeneity, no perfect multicollinearity, and homoskedastic uncorrelated errors, OLS is **BLUE**: the Best Linear Unbiased Estimator.

- **Linear**: the estimator is linear in y.
- **Unbiased**: E[beta_hat | X] = beta.
- **Best**: among linear unbiased estimators, OLS has the smallest variance.

“Best” does not mean best prediction under every loss function, best causal design, or best model when assumptions fail.


## 6. Sampling uncertainty and statistical inference

A fitted coefficient is a random quantity because another sample would produce another estimate.

Under the classical model:

Var(beta_hat | X) = sigma^2 (X^T X)^(-1)

Because sigma^2 is unknown, estimate it using:

s^2 = SSE / (n-k)

where k is the number of estimated parameters, including the intercept.

The estimated covariance matrix is:

Var_hat(beta_hat) = s^2 (X^T X)^(-1)

The standard error of coefficient j is the square root of the corresponding diagonal element.


### 6.1 t statistic

To test:

H_0: beta_j = beta_j0

compute:

t = (beta_hat_j - beta_j0) / SE(beta_hat_j)

Under the classical normal model, this follows a Student t distribution with n-k degrees of freedom under the null.

A two-sided p-value is:

p = 2 P(T_df >= |t_observed|)


### 6.2 Confidence interval

A (1-alpha) confidence interval is:

beta_hat_j +/- t_(1-alpha/2, df) SE(beta_hat_j)

A 95% confidence interval is not a 95% probability statement about a fixed parameter after the data have been observed. Its frequentist meaning is that the procedure covers the true parameter in 95% of repeated samples under the model assumptions.


### 6.3 Statistical significance is not substantive importance

A tiny effect can be statistically significant with a large sample. A large estimated effect can be imprecise with a small sample. Always report:

- effect size and units;
- uncertainty interval;
- sample size;
- practical implications;
- sensitivity to modelling choices.

### 6.4 Joint F tests

To test several restrictions simultaneously, such as:

H_0: beta_2 = beta_3 = beta_4 = 0

compare a restricted and unrestricted model:

F = [(SSE_R - SSE_U)/q] / [SSE_U/(n-k_U)]

where:

- q is the number of restrictions;
- SSE_R is the restricted-model SSE;
- SSE_U is the unrestricted-model SSE;
- k_U is the number of unrestricted parameters.

The F test asks whether the variables improve fit jointly, not whether every individual coefficient is significant.


### 6.5 Robust standard errors

When heteroskedasticity is suspected, use a sandwich covariance estimator. In conceptual form:

Var_robust(beta_hat) = (X^T X)^(-1) X^T Omega_hat X (X^T X)^(-1)

where Omega_hat contains squared residual-based variance estimates. HC0, HC1, HC2, and HC3 differ in finite-sample corrections. HC3 is often a reasonable default in modest samples because it adjusts more strongly for leverage.

Robust standard errors change estimated uncertainty, not the OLS coefficients themselves.


## 7. Prediction and uncertainty

For a new predictor row x_0, the predicted conditional mean is:

y_hat_0 = x_0^T beta_hat

There are two different intervals.


### 7.1 Confidence interval for the mean response

This estimates the average outcome at x_0:

SE_mean = s sqrt[x_0^T (X^T X)^(-1) x_0]

The interval is:

y_hat_0 +/- t_critical SE_mean


### 7.2 Prediction interval for a new individual outcome

A new outcome contains both coefficient uncertainty and new observation noise:

SE_prediction = s sqrt[1 + x_0^T (X^T X)^(-1) x_0]

The prediction interval is wider:

y_hat_0 +/- t_critical SE_prediction


### 7.3 Extrapolation

Regression can calculate a prediction far outside the observed predictor range, but the formula has no knowledge of whether the relationship continues. Extrapolation is a substantive assumption and should be clearly marked.


## 8. Measuring model fit


### 8.1 Sum-of-squares decomposition

With an intercept:

SST = SSR + SSE

where:

- SST = sum(y_i - y_bar)^2, total variation;
- SSR = sum(y_hat_i - y_bar)^2, explained variation;
- SSE = sum(y_i - y_hat_i)^2, unexplained variation.

### 8.2 R-squared

R^2 = 1 - SSE/SST

It is the fraction of sample variation in Y explained by the fitted values under the sum-of-squares decomposition.

Important limitations:

- adding predictors cannot lower training R^2;
- high R^2 does not imply causality;
- low R^2 does not mean a coefficient is unimportant;
- high R^2 does not validate functional form or assumptions;
- comparisons across very different outcomes can be misleading.

### 8.3 Adjusted R-squared

Adjusted R^2 = 1 - (1-R^2)(n-1)/(n-k)

It penalizes additional parameters. It can decrease when a weak predictor is added, but it is still an in-sample criterion.


### 8.4 Error metrics

For prediction:

MAE = mean(|y_i - y_hat_i|)

MSE = mean((y_i - y_hat_i)^2)

RMSE = sqrt(MSE)

- MAE is easier to interpret and less sensitive to extreme errors.
- RMSE penalizes large errors more heavily and is in outcome units.
- MSE is convenient mathematically but is in squared units.

Percentage errors can behave badly near zero and should not be used automatically.


### 8.5 Information criteria

AIC and BIC compare likelihood-based models fitted to the same outcome data:

AIC = -2 log-likelihood + 2k

BIC = -2 log-likelihood + k log(n)

Lower values are preferred within a valid comparison. BIC penalizes complexity more strongly as sample size grows. Neither substitutes for out-of-sample validation or substantive judgement.


## 9. Residual diagnostics

A regression should be interrogated, not merely fitted.


### 9.1 Residuals versus fitted values

Plot residuals against fitted values.

Look for:

- curvature: omitted nonlinear structure;
- funnel shape: heteroskedasticity;
- clusters: group structure or missing categorical variables;
- isolated points: outliers or data errors.

A healthy plot is not proof of correctness, but visible structure is evidence against the current specification.


### 9.2 Q-Q plot

A quantile-quantile plot compares residual quantiles with normal quantiles. Strong tail departures matter for small-sample t inference and prediction intervals. Mild deviations are common and not automatically fatal.


### 9.3 Leverage

The diagonal elements of the hat matrix are leverages:

h_ii = x_i^T (X^T X)^(-1) x_i

Leverage measures how unusual an observation’s predictor combination is. The average leverage is:

mean(h_ii) = k/n

A common screening threshold is 2k/n or 3k/n, but thresholds are not universal decision rules.


### 9.4 Standardized and studentized residuals

An internally standardized residual is approximately:

r_i = e_i / [s sqrt(1-h_ii)]

Large absolute values indicate outcomes unusual relative to the fitted model. Externally studentized residuals re-estimate the variance leaving observation i out and are often preferable for formal outlier assessment.


### 9.5 Cook’s distance

Cook’s distance combines residual size and leverage:

D_i = [e_i^2/(k s^2)] [h_ii/(1-h_ii)^2]

Large values indicate observations that materially affect the fitted coefficient vector. Values above 4/n are often inspected, but context matters more than a mechanical cutoff.

Do not delete influential observations merely because they are influential. Check data quality, investigate the case, assess sensitivity, and report justified decisions.


### 9.6 Multicollinearity and VIF

For predictor j, regress it on all other predictors and obtain R_j^2:

VIF_j = 1/(1-R_j^2)

A VIF of 1 means no linear relationship with the other predictors. Values of 5 or 10 are often treated as warning levels, but interpretation depends on purpose. Interactions and polynomial terms naturally create collinearity; centering can improve numerical interpretation.


### 9.7 Heteroskedasticity

Signs include a residual funnel and outcome variability rising with fitted values. Responses include:

- robust standard errors for inference;
- transformations such as log(Y) when substantively appropriate;
- weighted least squares when variance structure is defensible;
- a model whose distribution matches the outcome.

### 9.8 Autocorrelation and clustering

Plot residuals over time or within groups. A Durbin-Watson statistic is a limited diagnostic for first-order serial correlation in ordered residuals, but time-series structure should usually be modelled explicitly. Clustered data require group-aware uncertainty estimates and often multilevel modelling.


## 10. Transformations, categories, interactions, and nonlinear terms


### 10.1 Centering and scaling

Center a predictor:

X_centered = X - mean(X)

Then the intercept represents the expected outcome at the average predictor value. Centering is especially useful with interactions and polynomial terms.

Standardize:

Z = (X - mean(X))/SD(X)

A standardized coefficient is expressed per standard-deviation change. Standardization is essential for distance-based methods and usually important for ridge, lasso, and gradient descent. It is not automatically required for ordinary least squares.


### 10.2 Log transformations

Four common specifications:

**Level-level:** Y = beta_0 + beta_1 X

- One-unit increase in X is associated with beta_1 units in Y.

**Log-level:** log(Y) = beta_0 + beta_1 X

- One-unit increase in X is associated approximately with 100 beta_1% change in Y for small beta_1.
- Exact percentage: 100[exp(beta_1)-1]%.

**Level-log:** Y = beta_0 + beta_1 log(X)

- A 1% increase in X is associated approximately with beta_1/100 units in Y.

**Log-log:** log(Y) = beta_0 + beta_1 log(X)

- beta_1 is an elasticity: a 1% increase in X is associated with approximately beta_1% change in Y.

Logs require positive values unless a carefully justified alternative transformation is used. Adding an arbitrary constant can distort interpretation.


### 10.3 Categorical variables

A categorical variable with g categories is represented by g-1 indicator variables when an intercept is included. One category is the reference.

For region with categories Central, North, South:

Y = beta_0 + beta_1 North + beta_2 South + error

- beta_0 is the mean for Central.
- beta_1 is the North-Central difference.
- beta_2 is the South-Central difference.

Changing the reference category changes coefficient presentation, not fitted values.


### 10.4 Interactions

Consider:

Y = beta_0 + beta_1 X + beta_2 Z + beta_3 XZ + error

The effect of X depends on Z:

dE[Y|X,Z]/dX = beta_1 + beta_3 Z

The coefficient beta_1 is the effect of X when Z=0. Therefore centering Z often makes the main effect more meaningful.

When including an interaction, normally include the corresponding main effects. Omitting them imposes strong constraints, such as forcing groups to have the same intercept or setting an effect to zero at the reference point.


### 10.5 Polynomial regression

A quadratic model is:

Y = beta_0 + beta_1 X + beta_2 X^2 + error

The marginal slope is:

dE[Y|X]/dX = beta_1 + 2 beta_2 X

Interpret the curve, turning point, and predictions rather than treating beta_1 as a constant effect. The turning point is:

X* = -beta_1/(2 beta_2)

only if it lies in a meaningful data range.


### 10.6 Splines

High-degree global polynomials can oscillate. Splines fit smooth piecewise polynomials joined at knots. They provide flexible nonlinear relationships while retaining controlled smoothness. Modern predictive workflows often prefer splines, generalized additive models, or tree-based models to high-degree polynomials.


## 11. Generalization, validation, and leakage

Training fit describes how well the model matches data it has already seen. Prediction requires performance on new data.


### 11.1 Train/test split

Divide the data into:

- training set: fit the model;
- test set: estimate final out-of-sample performance.

Do not repeatedly tune decisions on the test set. Once used for tuning, it becomes part of the training process.


### 11.2 Cross-validation

In k-fold cross-validation:

- split data into k folds;
- fit on k-1 folds;
- validate on the remaining fold;
- repeat until each fold has served as validation;
- average the metric.

Cross-validation estimates performance under repeated data splits and is useful for choosing model complexity and regularization strength.


### 11.3 Leakage

Leakage occurs when information unavailable at prediction time enters model training or preprocessing.

Examples:

- standardizing using the entire dataset before splitting;
- imputing missing values using test-set information;
- using an outcome-derived variable as a predictor;
- random splitting of time-series data so future information predicts the past;
- having the same person in both training and test data.

Use pipelines so transformations are fitted only on training folds.


### 11.4 Bias-variance trade-off

Expected prediction error can be conceptualized as:

irreducible noise + squared bias + variance

- overly simple models have high bias;
- overly flexible models have high variance;
- regularization and cross-validation help choose a useful balance.

### 11.5 Data splitting must respect the data-generating structure

- Time data: use forward or rolling validation.
- Grouped data: split by group when predicting new groups.
- Spatial data: use spatial blocks when nearby observations are similar.
- Rare outcomes: stratify classification splits where appropriate.

## 12. Regularization

Ordinary least squares minimizes:

SSE(beta)

Regularization adds a penalty for coefficient size.


### 12.1 Ridge regression

Ridge minimizes:

SSE(beta) + lambda sum_{j=1}^p beta_j^2

The intercept is usually not penalized. The closed-form estimator is:

beta_hat_ridge = (X^T X + lambda P)^(-1)X^T y

where P has zero for the intercept and ones for slopes.

Ridge:

- shrinks coefficients toward zero;
- handles correlated predictors well;
- reduces variance at the cost of bias;
- usually retains all predictors;
- requires scaling when predictors have different units.

### 12.2 Lasso regression

Lasso minimizes:

(1/(2n))SSE(beta) + lambda sum_{j=1}^p |beta_j|

The absolute-value penalty can set coefficients exactly to zero, performing variable selection. With strongly correlated predictors, lasso may select one somewhat arbitrarily.


### 12.3 Elastic net

Elastic net combines the penalties:

SSE + lambda [alpha sum |beta_j| + (1-alpha) sum beta_j^2]

It can retain groups of correlated predictors better than pure lasso while still producing sparsity.


### 12.4 Choosing the penalty

Do not choose lambda from training error. Use cross-validation. Larger penalties increase shrinkage. The best penalty depends on the objective, metric, and validation design.

Regularized coefficients have different inferential properties from classical OLS coefficients. Standard OLS p-values should not simply be attached after data-driven lasso selection.


## 13. Logistic regression

Despite its name, logistic regression is primarily used for binary classification.

Let Y be 0 or 1. The model describes the probability:

p(X) = P(Y=1 | X)

A linear model for probability can predict below 0 or above 1. Logistic regression instead models log-odds:

log[p/(1-p)] = beta_0 + beta_1 X_1 + ... + beta_p X_p

Solving for probability:

p = 1 / [1 + exp(-(beta_0 + beta_1X_1 + ... + beta_pX_p))]


### 13.1 Coefficient interpretation

A one-unit increase in X_j, holding other variables constant, changes log-odds by beta_j and multiplies odds by:

exp(beta_j)

If exp(beta_j)=1.5, the odds are 50% higher, not necessarily the probability. Probability changes depend on the starting probability and all predictors.


### 13.2 Likelihood and loss

Logistic regression is not fitted by ordinary least squares. For observation i, the Bernoulli likelihood is:

p_i^(y_i) (1-p_i)^(1-y_i)

The negative average log-likelihood, or log loss, is:

-[1/n] sum [y_i log(p_i) + (1-y_i)log(1-p_i)]

Optimization methods such as Newton-Raphson, iteratively reweighted least squares, or gradient descent minimize this loss.


### 13.3 Classification thresholds

A probability model becomes a classifier only after choosing a threshold, often 0.5. The correct threshold depends on error costs and prevalence.

Metrics include:

- sensitivity/recall: TP/(TP+FN);
- specificity: TN/(TN+FP);
- precision: TP/(TP+FP);
- F1: harmonic mean of precision and recall;
- ROC AUC: ranking performance across thresholds;
- log loss: probability calibration and confidence;
- Brier score: mean squared probability error.

Accuracy alone can be misleading for rare outcomes.


## 14. Regression and causal claims

A regression coefficient is generally an adjusted association. It becomes a causal effect only under a credible identification strategy.


### 14.1 Omitted-variable bias in a simple case

Suppose the true model is:

Y = beta_0 + beta_1 X + beta_2 Z + epsilon

but Z is omitted. The expected simple-regression slope is:

E[b_1_tilde] = beta_1 + beta_2 Cov(X,Z)/Var(X)

Bias requires both:

- Z affects Y (beta_2 != 0);
- Z is correlated with X.

The sign of bias depends on both signs.


### 14.2 Bad controls

Not every available variable should be controlled. Variables caused by the treatment can block part of the causal effect or induce collider bias. Control selection should follow a causal model, temporal logic, domain knowledge, and research design—not an automatic p-value procedure.


### 14.3 Stronger designs

Causal regression often relies on:

- randomized experiments;
- natural experiments;
- instrumental variables;
- regression discontinuity;
- difference-in-differences;
- panel fixed effects;
- matching or weighting under explicit assumptions.

Regression is then a component of a design rather than the source of identification by itself.


# Part II. Regression from Scratch in Pure Python


## 15. Why implement regression yourself?

A from-scratch implementation teaches what packages do on your behalf:

- compute means and cross-products;
- solve a least-squares optimization problem;
- estimate residual variance;
- invert or factor a matrix;
- calculate standard errors and t statistics;
- produce predictions and diagnostics;
- separate training from evaluation.

The implementation in this guide uses only:

```python
from dataclasses import dataclass
import math
import random
```

The standard library is allowed; NumPy, pandas, scipy, statsmodels, and scikit-learn are not used in Part II.


### 15.1 Numerical caution

The textbook formula (X^T X)^(-1)X^T y is educational but can be numerically unstable because forming X^T X squares the condition number. A QR-based solver is better. The companion implementation therefore offers:

- normal equations for transparency;
- modified Gram-Schmidt QR for the primary OLS fit;
- partial pivoting in linear-system solution.

Production software uses highly optimized and more robust linear algebra, often QR or singular-value decomposition.


## 16. Simple linear regression from scratch

The algorithm is direct:

```python
def simple_ols(x, y):
    n = len(x)
    x_bar = sum(x) / n
    y_bar = sum(y) / n

    sxx = sum((xi - x_bar) ** 2 for xi in x)
    sxy = sum((xi - x_bar) * (yi - y_bar)
              for xi, yi in zip(x, y))

    slope = sxy / sxx
    intercept = y_bar - slope * x_bar

    fitted = [intercept + slope * xi for xi in x]
    residuals = [yi - yhat for yi, yhat in zip(y, fitted)]
    sse = sum(e ** 2 for e in residuals)
    sst = sum((yi - y_bar) ** 2 for yi in y)
    r_squared = 1 - sse / sst

    return intercept, slope, fitted, residuals, r_squared
```

Example:

```
x = [1, 2, 3, 4, 5, 6]
y = [3.1, 4.9, 7.2, 9.1, 11.2, 12.8]

intercept, slope, fitted, residuals, r2 = simple_ols(x, y)
print(intercept, slope, r2)
```

Expected values are approximately:

```
intercept = 1.12
slope = 1.98
R-squared = 0.9981
```


### 16.1 Computing the slope by covariance and variance

```python
def sample_variance(values):
    n = len(values)
    m = sum(values) / n
    return sum((v - m) ** 2 for v in values) / (n - 1)


def sample_covariance(x, y):
    n = len(x)
    x_bar = sum(x) / n
    y_bar = sum(y) / n
    return sum((a - x_bar) * (b - y_bar)
               for a, b in zip(x, y)) / (n - 1)

slope = sample_covariance(x, y) / sample_variance(x)
```

The (n-1) denominators cancel. This is a useful cross-check.


## 17. Matrix operations without NumPy

A matrix is represented as a list of equal-length row lists.

```
X = [
    [1.0, 10.0],
    [1.0, 12.0],
    [1.0, 15.0],
]
```


### 17.1 Transpose

```python
def transpose(A):
    rows = len(A)
    cols = len(A[0])
    return [[A[i][j] for i in range(rows)] for j in range(cols)]
```


### 17.2 Dot product and matrix multiplication

```python
def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def matmul(A, B):
    B_T = transpose(B)
    return [[dot(row, col) for col in B_T] for row in A]


def matvec(A, x):
    return [dot(row, x) for row in A]
```


### 17.3 Solving a linear system

To solve Ax=b, Gauss-Jordan elimination with partial pivoting:

- augment A with b;
- choose the largest available pivot by absolute value;
- swap rows;
- scale the pivot row;
- eliminate the pivot column from every other row;
- read the solution from the final column.

Partial pivoting reduces numerical damage when a candidate pivot is tiny.


### 17.4 QR decomposition

For a design matrix X, QR decomposition writes:

X = QR

where columns of Q are orthonormal and R is upper triangular. Least squares becomes:

min ||y - QR beta||^2

Multiply by Q^T:

R beta_hat = Q^T y

Then solve by back substitution. This avoids explicitly calculating (X^T X)^(-1) for the coefficient fit.

The supplied module implements modified Gram-Schmidt QR. It is clearer than Householder reflections, although production libraries usually prefer Householder QR or SVD for better stability.


## 18. Multiple OLS from scratch

Using the companion module:

```python
from regression_from_scratch import fit_ols

X = [
    [2.0, 10.0],
    [3.0, 12.0],
    [5.0, 13.0],
    [7.0, 16.0],
    [9.0, 18.0],
    [11.0, 21.0],
]
y = [20.0, 24.0, 31.0, 39.0, 46.0, 55.0]

model = fit_ols(X, y, add_intercept=True, method="qr")

print(model.coefficients)
print(model.standard_errors)
print(model.t_statistics)
print(model.p_values)
print(model.confidence_intervals)
print(model.r_squared)
print(model.adjusted_r_squared)
```

The fit_ols procedure performs these steps:

- Add the intercept column.
- Estimate coefficients by QR.
- Compute fitted values X beta_hat.
- Compute residuals y - y_hat.
- Calculate SSE, SST, R^2, and adjusted R^2.
- Estimate s^2 = SSE/(n-k).
- Compute (X^T X)^(-1) for the covariance matrix.
- Obtain standard errors, t statistics, p-values, and intervals.

### 18.1 Student t distribution from scratch

Python’s standard library provides math.gamma and math.lgamma but not a Student t cumulative distribution function. The companion implementation computes it through the regularized incomplete beta function and a continued fraction. It obtains t quantiles by bisection.

For degrees of freedom v and t > 0:

F_v(t) = 1 - 0.5 I_[v/(v+t^2)](v/2, 1/2)

where I_x(a,b) is the regularized incomplete beta function.

This is included so that the pure-Python inference is genuinely t-based rather than using a normal approximation.


### 18.2 Confidence and prediction intervals

```
mean_intervals = model.prediction_intervals(
    [[6.0, 15.0]],
    confidence=0.95,
    include_observation_noise=False,
)

new_observation_intervals = model.prediction_intervals(
    [[6.0, 15.0]],
    confidence=0.95,
    include_observation_noise=True,
)
```

The first is for the conditional mean. The second is for a new individual outcome and is wider.


## 19. Diagnostics from scratch

```python
from regression_from_scratch import ols_diagnostics

diagnostics = ols_diagnostics(model)
for row in diagnostics:
    print(row)
```

Each row contains:

- raw residual;
- leverage h_ii;
- standardized residual;
- Cook’s distance.

A practical screening process is:

- Sort observations by Cook’s distance.
- Inspect data quality and substantive circumstances.
- Refit the model without highly influential cases only as a sensitivity check, not as an automatic deletion rule.
- Compare coefficients, predictions, and conclusions.
- Report any consequential dependence on a small number of cases.

### 19.1 A pure-Python residual check

```python
residual_mean = sum(model.residuals) / len(model.residuals)
print(residual_mean)  # approximately zero with an intercept

orthogonality = []
for j in range(model.n_parameters):
    value = sum(
        model.design_matrix[i][j] * model.residuals[i]
        for i in range(model.n_observations)
    )
    orthogonality.append(value)
print(orthogonality)  # approximately all zeros
```

Floating-point calculations produce tiny numerical deviations rather than exact zeros.


## 20. Gradient descent for linear regression

Closed-form or factorization-based OLS is ideal for ordinary linear regression of moderate size. Gradient descent becomes useful when connecting regression to machine learning and very large optimization problems.

For mean squared error:

J(beta) = (1/n) sum_i (x_i^T beta - y_i)^2

The gradient for coefficient j is:

dJ/dbeta_j = (2/n) sum_i (y_hat_i - y_i)x_ij

The update is:

beta_j <- beta_j - learning_rate * gradient_j

```python
from regression_from_scratch import fit_linear_gradient_descent

model_gd = fit_linear_gradient_descent(
    X,
    y,
    learning_rate=0.05,
    epochs=5000,
    standardize=True,
)
print(model_gd.coefficients)
```


### 20.1 Why scaling matters

If one feature ranges from 0 to 1 and another from 0 to 1,000,000, the loss surface is elongated. A single learning rate may overshoot one direction and crawl in another. Standardization makes the geometry more balanced.


### 20.2 Learning-rate symptoms

- Loss rises or becomes inf: learning rate too large.
- Loss falls extremely slowly: learning rate too small or features poorly scaled.
- Loss oscillates: learning rate often too large.
- Training stops far from OLS: too few epochs, bad scaling, or implementation error.

Plot or inspect loss_history to diagnose optimization.


## 21. Train/test splitting and cross-validation from scratch

```python
from regression_from_scratch import (
    train_test_split,
    fit_ols,
    root_mean_squared_error,
    r_squared_score,
    cross_validate_ols,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_fraction=0.25, seed=42
)

model = fit_ols(X_train, y_train)
predictions = model.predict(X_test)

print(root_mean_squared_error(y_test, predictions))
print(r_squared_score(y_test, predictions))

cv_results = cross_validate_ols(X, y, k=5, seed=42)
print(cv_results)
```

Caution: fold-level R^2 can be unstable or negative when validation folds are tiny or outcomes vary little. RMSE is often more interpretable in such cases.


## 22. Ridge and lasso from scratch


### 22.1 Ridge

```python
from regression_from_scratch import fit_ridge

ridge = fit_ridge(X, y, alpha=10.0, standardize=True)
print(ridge.coefficients)
```

The implementation adds alpha to the diagonal of X^T X for slope terms but not the intercept.


### 22.2 Lasso coordinate descent

Coordinate descent updates one coefficient at a time. For standardized feature j, compute the partial residual and apply the soft-threshold operator:

S(z, gamma) = sign(z) max(|z|-gamma, 0)

```python
from regression_from_scratch import fit_lasso

lasso = fit_lasso(X, y, alpha=0.1)
print(lasso.coefficients)
```

The implementation uses the objective:

(1/(2n)) SSE + alpha sum |beta_j|

Because lasso solutions depend strongly on scaling, the implementation standardizes predictors internally.


### 22.3 What the from-scratch implementation does not replace

A production regularization library provides:

- efficient paths over many penalty values;
- convergence diagnostics;
- sparse matrix support;
- cross-validated tuning;
- alternative solvers;
- careful handling of sample weights and edge cases.

The pure implementation is for understanding, verification, and small educational problems.


## 23. Logistic regression from scratch

```python
from regression_from_scratch import fit_logistic_gradient_descent

X_class = [
    [20.0, 0.0],
    [25.0, 0.0],
    [30.0, 1.0],
    [35.0, 0.0],
    [40.0, 1.0],
    [50.0, 1.0],
    [60.0, 1.0],
]
y_class = [0, 0, 0, 0, 1, 1, 1]

logit = fit_logistic_gradient_descent(
    X_class,
    y_class,
    learning_rate=0.1,
    epochs=10000,
    l2=0.1,
    standardize=True,
)

print(logit.predict_proba([[45.0, 1.0]]))
print(logit.predict([[45.0, 1.0]], threshold=0.5))
```

The gradient is:

(1/n) X^T(p-y)

with an optional L2 penalty on slopes. A numerically stable sigmoid uses different formulas for positive and negative scores to avoid overflow.


## 24. A complete from-scratch workflow

```python
from regression_from_scratch import (
    train_test_split,
    fit_ols,
    ols_diagnostics,
    root_mean_squared_error,
    mean_absolute_error,
    r_squared_score,
)

# 1. Organize rows as predictors and outcomes.
X = [[...], [...], ...]
y = [...]

# 2. Preserve a test set.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_fraction=0.2, seed=2026
)

# 3. Fit on training data only.
model = fit_ols(X_train, y_train, add_intercept=True, method="qr")

# 4. Examine coefficients and uncertainty.
for j, (coef, se, p, interval) in enumerate(zip(
    model.coefficients,
    model.standard_errors,
    model.p_values,
    model.confidence_intervals,
)):
    print(j, coef, se, p, interval)

# 5. Diagnose training fit.
for row in sorted(
    ols_diagnostics(model),
    key=lambda item: item["cooks_distance"],
    reverse=True,
)[:10]:
    print(row)

# 6. Evaluate untouched test data.
pred = model.predict(X_test)
print("MAE", mean_absolute_error(y_test, pred))
print("RMSE", root_mean_squared_error(y_test, pred))
print("R2", r_squared_score(y_test, pred))

# 7. Refit final model on all data only after model choices are fixed.
final_model = fit_ols(X, y)
```

This sequence separates estimation, diagnosis, validation, and final fitting.


# Part III. Regression with Python Libraries


## 25. Which library should do what?

- **NumPy:** numerical arrays and linear algebra; ideal for transparent matrix calculations.
- **pandas:** data tables, cleaning, joins, missing values, categories, and feature construction.
- **statsmodels:** statistical modelling, coefficient tables, hypothesis tests, confidence intervals, robust covariance, influence diagnostics, and formula syntax.
- **scikit-learn:** predictive pipelines, preprocessing, train/test splits, cross-validation, tuning, and regularization.
- **matplotlib:** diagnostic and explanatory plots.

A common mistake is to force one package to do every task. Use statsmodels when inference is central; use scikit-learn when generalization and pipelines are central; use both when appropriate.


## 26. NumPy: OLS through linear algebra

```python
import numpy as np

X = np.array([
    [2.0, 10.0],
    [3.0, 12.0],
    [5.0, 13.0],
    [7.0, 16.0],
    [9.0, 18.0],
    [11.0, 21.0],
])
y = np.array([20.0, 24.0, 31.0, 39.0, 46.0, 55.0])

X_design = np.column_stack([np.ones(len(X)), X])
beta_hat, residual_sums, rank, singular_values = np.linalg.lstsq(
    X_design, y, rcond=None
)

predictions = X_design @ beta_hat
residuals = y - predictions
```

Prefer np.linalg.lstsq to:

```
beta_hat = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
```

The latter explicitly forms an inverse and is less stable. If solving the normal equations for demonstration, at least use:

```
beta_hat = np.linalg.solve(X_design.T @ X_design, X_design.T @ y)
```


### 26.1 Manual standard errors in NumPy

```
n, k = X_design.shape
sse = residuals @ residuals
sigma2_hat = sse / (n - k)
xtx_inverse = np.linalg.inv(X_design.T @ X_design)
cov_beta = sigma2_hat * xtx_inverse
standard_errors = np.sqrt(np.diag(cov_beta))
t_statistics = beta_hat / standard_errors
```

Use statsmodels for p-values and mature covariance options rather than rebuilding an incomplete inference layer in ordinary work.


### 26.2 Condition number

```
condition_number = np.linalg.cond(X_design)
print(condition_number)
```

A large condition number warns of scaling problems or near linear dependence. There is no universal cutoff independent of units and problem structure.


## 27. pandas: preparing regression data

```python
import pandas as pd

required = ["income", "experience", "education", "training", "region"]
df = pd.read_csv("data.csv", usecols=required)

print(df.info())
print(df.describe(include="all"))
print(df.isna().sum())
print(df.duplicated().sum())
```


### 27.1 Explicit feature construction

```
df = df.assign(
    experience_centered=lambda d: d["experience"] - d["experience"].mean(),
    log_income=lambda d: np.log(d["income"]),
    experience_training=lambda d: d["experience"] * d["training"],
)
```


### 27.2 Missing data

Do not call dropna() reflexively. Ask:

- how much is missing?
- is missingness related to the outcome or predictors?
- is absence itself meaningful?
- should imputation be fitted inside cross-validation?

For prediction, imputation belongs inside a scikit-learn pipeline. For inference, missing-data assumptions and sensitivity may require more careful methods such as multiple imputation.


### 27.3 Category checks

```
df["region"] = df["region"].astype("category")
print(df["region"].value_counts(dropna=False))
```

Check spelling, case, and sparse categories before modelling.


## 28. statsmodels for inference


### 28.1 Formula interface

```python
import statsmodels.formula.api as smf

model = smf.ols(
    "income ~ experience + education + training + C(region)",
    data=df,
).fit()

print(model.summary())
```

Formula syntax automatically creates an intercept and encodes categories.

Useful patterns:

```
# No intercept
smf.ols("y ~ x1 + x2 - 1", data=df)

# Interaction and main effects
smf.ols("y ~ x1 * x2", data=df)

# Interaction only, generally a strong constraint
smf.ols("y ~ x1:x2", data=df)

# Polynomial term: I() means interpret arithmetically
smf.ols("y ~ x + I(x ** 2)", data=df)

# Log transformation
smf.ols("np.log(y) ~ x1 + np.log(x2)", data=df)
```


### 28.2 Robust standard errors

```
robust = model.get_robustcov_results(cov_type="HC3")
print(robust.summary())
```

Cluster-robust example:

```
clustered = model.get_robustcov_results(
    cov_type="cluster",
    groups=df["village_id"],
)
```

The number of clusters matters. With few clusters, ordinary cluster-robust asymptotics can be unreliable.


### 28.3 Joint tests

```
print(model.f_test("training = 0, experience:training = 0"))
```

or compare nested models:

```
restricted = smf.ols("income ~ experience + education", data=df).fit()
unrestricted = smf.ols(
    "income ~ experience + education + training + C(region)",
    data=df,
).fit()

print(unrestricted.compare_f_test(restricted))
```


### 28.4 Predictions and intervals

```
new_data = pd.DataFrame({
    "experience": [5, 15],
    "education": [14, 18],
    "training": [0, 1],
    "region": ["Central", "North"],
})

prediction = model.get_prediction(new_data).summary_frame(alpha=0.05)
print(prediction)
```

Typical columns include:

- mean: point prediction;
- mean_ci_lower, mean_ci_upper: interval for mean response;
- obs_ci_lower, obs_ci_upper: interval for a new observation.

### 28.5 Influence and diagnostic tests

```
influence = model.get_influence()
diagnostic_table = influence.summary_frame()
print(diagnostic_table.nlargest(10, "cooks_d"))
```

Breusch-Pagan test:

```python
from statsmodels.stats.diagnostic import het_breuschpagan

lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(
    model.resid,
    model.model.exog,
)
```

A diagnostic p-value is evidence, not a substitute for plots and substantive assessment.


### 28.6 VIF

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_matrix = model.model.exog
vif = [
    variance_inflation_factor(X_matrix, j)
    for j in range(X_matrix.shape[1])
]
```

Do not interpret the intercept VIF as an ordinary predictor VIF.


## 29. scikit-learn for predictive workflows


### 29.1 Basic numerical regression

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X = df[["experience", "education", "training"]]
y = df["income"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("MAE", mean_absolute_error(y_test, pred))
print("RMSE", mean_squared_error(y_test, pred) ** 0.5)
print("R2", r2_score(y_test, pred))
```

Scikit-learn emphasizes prediction and does not supply a classical OLS coefficient table with p-values.


### 29.2 Mixed numerical and categorical data

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_features = ["experience", "education", "training"]
categorical_features = ["region"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
])

preprocess = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", LinearRegression()),
])

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
```

The pipeline prevents leakage by fitting imputation, scaling, and category encoding only on the training data within each fit.


### 29.3 Cross-validation

```python
from sklearn.model_selection import KFold, cross_validate

cv = KFold(n_splits=5, shuffle=True, random_state=42)
results = cross_validate(
    pipeline,
    X,
    y,
    cv=cv,
    scoring={
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
    },
    return_train_score=True,
)

print(results["test_r2"].mean())
print(-results["test_rmse"].mean())
print(-results["test_mae"].mean())
```

Negative loss scores are a scikit-learn convention: higher is always better. Reverse the sign for interpretation.


### 29.4 Ridge, lasso, and elastic net

```python
import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

ridge_pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", RidgeCV(alphas=np.logspace(-4, 4, 100), cv=5)),
])

lasso_pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", LassoCV(
        alphas=np.logspace(-4, 1, 100),
        cv=5,
        max_iter=20000,
    )),
])

elastic_pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9, 1.0],
        alphas=np.logspace(-4, 1, 100),
        cv=5,
        max_iter=20000,
    )),
])
```

For a rigorous final evaluation, tune within inner cross-validation and evaluate in an outer loop or untouched test set.


### 29.5 Polynomial features

```python
from sklearn.preprocessing import PolynomialFeatures

polynomial_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scale", StandardScaler()),
    ("ridge", RidgeCV(alphas=np.logspace(-4, 4, 100), cv=5)),
])
```

Polynomial expansion can create many features quickly. Regularization and validation become important.


## 30. Diagnostic plots with matplotlib

```python
import matplotlib.pyplot as plt
import statsmodels.api as sm

fitted = model.fittedvalues
residuals = model.resid

plt.figure()
plt.scatter(fitted, residuals)
plt.axhline(0)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals versus fitted")
plt.show()

plt.figure()
sm.qqplot(residuals, line="45", fit=True)
plt.title("Normal Q-Q plot")
plt.show()
```

Scale-location plot:

```
standardized = model.get_influence().resid_studentized_internal
plt.figure()
plt.scatter(fitted, np.sqrt(np.abs(standardized)))
plt.xlabel("Fitted values")
plt.ylabel("Square root of absolute standardized residual")
plt.title("Scale-location plot")
plt.show()
```

Influence plot:

```
sm.graphics.influence_plot(model, criterion="cooks")
plt.show()
```

For large datasets, plotting every point may obscure structure. Use transparency, sampling, hexbin plots, or binned residual summaries.


## 31. Logistic regression with libraries


### 31.1 statsmodels for coefficient inference

```
logit_model = smf.logit(
    "purchased ~ age + income + prior_contact",
    data=df_class,
).fit()

print(logit_model.summary())
print(np.exp(logit_model.params))  # odds ratios
```

Confidence intervals for odds ratios:

```
odds_ratio_ci = np.exp(logit_model.conf_int())
```


### 31.2 scikit-learn for prediction

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

classifier = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000)),
])

classifier.fit(X_train, y_train)
probability = classifier.predict_proba(X_test)[:, 1]
prediction = (probability >= 0.5).astype(int)

print(roc_auc_score(y_test, probability))
print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))
```

Choose thresholds using costs, decision capacity, precision-recall trade-offs, and validation data. Do not optimize and report on the same test set.


## 32. End-to-end regression checklist


### Step 1: State the question

Write one sentence specifying:

- outcome;
- unit of observation;
- target population;
- prediction or inference goal;
- time horizon if relevant.

### Step 2: Understand how data were generated

Document sampling, measurement, exclusions, repeated observations, clusters, and timing. A regression cannot repair a fundamentally unsuitable design.


### Step 3: Audit data

Check types, units, plausible ranges, duplicates, missingness, coding, outliers, and target leakage.


### Step 4: Explore without overfitting

Use distributions and plots. Identify transformations and interactions from theory and visible structure, while remembering that repeated exploration consumes evidence.


### Step 5: Define a baseline

For prediction, compare against a simple baseline such as the training mean. For inference, define a parsimonious specification grounded in the question.


### Step 6: Split or structure validation

Choose random, grouped, temporal, or spatial validation before extensive tuning.


### Step 7: Build preprocessing into a pipeline

Imputation, scaling, encoding, and feature generation should be fitted inside the training process.


### Step 8: Fit and diagnose

Examine residual structure, influential observations, collinearity, and dependence. For inference, use an appropriate covariance estimator.


### Step 9: Evaluate with more than one metric

Report error in outcome units and a relative fit metric. For classification, report threshold-dependent and probability metrics.


### Step 10: Test sensitivity

Try plausible specifications, robust standard errors, transformations, alternative validation seeds, and influence checks. Explain which conclusions are stable.


### Step 11: Interpret within the data range

Translate coefficients and predictions into meaningful units. Avoid causal language without a design that supports it.


### Step 12: Preserve reproducibility

Record data version, code, random seeds, package versions, formula, exclusions, and all model-selection decisions.


## 33. Common mistakes and corrections


| Mistake | Why it is wrong | Better practice |
| --- | --- | --- |
| “The coefficient is significant, so the effect is important.” | Significance depends on uncertainty and sample size. | Report magnitude, units, interval, and practical relevance. |
| “R-squared is high, so the model is correct.” | R-squared does not test assumptions or causality. | Diagnose residuals and validate on new data. |
| “The p-value is the probability the null is true.” | A frequentist p-value conditions on the null. | State what data extremity means under the null. |
| “Controlling for more variables always improves causal estimates.” | Bad controls can induce bias. | Use causal logic and temporal ordering. |
| “A non-significant coefficient means no effect.” | It may mean low precision or weak design. | Examine interval width and power. |
| “Delete every outlier.” | Unusual observations may be valid and informative. | Verify, investigate, and conduct sensitivity analysis. |
| “Scale the full dataset, then split.” | Test information leaks into training. | Fit scaling inside a pipeline on training folds. |
| “Use the test set until the model looks good.” | The test set becomes training information. | Keep a final untouched test set or use nested CV. |
| “Prediction interval and confidence interval are the same.” | New outcomes include irreducible noise. | Distinguish mean-response and observation intervals. |
| “Correlation and slope are interchangeable.” | Slope depends on units. | Use correlation for standardized association and slope for unit change. |
| “Lasso p-values can be read like ordinary OLS p-values.” | Selection changes the sampling process. | Use post-selection methods or treat lasso primarily as predictive. |
| “A 0.5 classification threshold is always correct.” | Costs and prevalence differ. | Choose threshold from decision objectives and validation. |


# Part IV. Exercises and Teaching Sequence

This part turns Parts I-III into four graduated tiers of practice. Each tier tests a different kind of mastery, and each is a prerequisite for the one after it:

- **Tier 1, Conceptual (C):** can you explain the idea in plain language, without hiding behind a formula? If you cannot answer a conceptual question, do not move on to its derivation.
- **Tier 2, Derivation (D):** can you reconstruct the mathematics from first principles on paper? Derivation is what makes a formula yours instead of something you looked up.
- **Tier 3, Pure-Python coding (P):** can you make the derivation run, using only the standard library, so that every intermediate quantity is visible and checkable?
- **Tier 4, Library (L):** can you reproduce the same result with NumPy, pandas, statsmodels, and scikit-learn, and explain any difference between the from-scratch and library answers?

Exercises inside each tier are grouped by topic so you can work through a topic's full C -> D -> P -> L stack before moving to the next topic, instead of clearing an entire tier before starting the next one. Either path works; the first is recommended for a first pass.

## 34. Conceptual questions

**Foundations, simple and multiple regression**

- **C1.** Explain the difference between epsilon_i and e_i.
- **C2.** Why does an intercept make the residuals sum to zero?
- **C3.** What exactly does “holding other predictors constant” mean algebraically?

**Assumptions and inference**

- **C4.** Which assumption is most directly threatened by omitted confounding?
- **C5.** Can OLS coefficients remain unbiased under heteroskedasticity? What becomes invalid?

**Prediction and fit**

- **C6.** Why is a prediction interval wider than a confidence interval for the mean?
- **C7.** Give an example where R^2 is low but a coefficient is substantively important.
- **C15.** What is extrapolation, and why is it dangerous even when the algebra is correct?

**Diagnostics and multicollinearity**

- **C9.** Why can near multicollinearity damage coefficient interpretation without badly damaging prediction?

**Regularization**

- **C10.** Why does ridge require attention to feature scale?
- **C11.** Why can lasso choose one variable among a correlated group?

**Logistic regression**

- **C12.** Explain odds versus probability in logistic regression.
- **C13.** Why can accuracy be misleading for rare outcomes?

**Validation and causal claims**

- **C8.** Give an example where predictive accuracy is high but causal interpretation is invalid.
- **C14.** Why should transformations be fitted inside cross-validation?

<details markdown="1">
<summary>Answer key: Conceptual questions (C1-C15)</summary>

**Marking rubric.** A full-credit answer names the specific mechanism (not just the vocabulary word) and gives or references a concrete case. Partial credit for identifying the right concept without the mechanism.

**C1.** `epsilon_i` is the population disturbance in the true, unobservable data-generating process; `e_i = y_i - y_hat_i` is the observed residual from the *fitted* model. They differ because `beta_hat != beta`: `e_i` is your best available estimate of `epsilon_i`, not the thing itself.

**C2.** The first-order condition for the intercept in the OLS minimization is `d(SSE)/d(b0) = -2 * sum(e_i) = 0`, i.e. `sum(e_i) = 0`. This is a direct algebraic consequence of including an intercept term (see D1) — it is not true in general for a no-intercept model.

**C3.** It is a partial-derivative statement: the coefficient on `x_j` is `dE[y]/dx_j` with every other regressor fixed. Equivalently, by the Frisch-Waugh-Lovell theorem, it is the slope you would get from regressing the part of `y` left over after removing the other regressors' effects on the part of `x_j` left over after doing the same.

**C4.** Zero conditional mean, `E[epsilon | X] = 0`. An omitted confounder gets absorbed into `epsilon`; if it correlates with an included regressor, that regressor is now correlated with the error term, and zero conditional mean fails.

**C5.** Yes — unbiasedness only requires `E[epsilon | X] = 0`, which does not depend on the error variance being constant. What breaks is the *classical* standard-error formula `sigma^2 (X^T X)^-1`, which assumes homoskedasticity; use robust (HC0-HC3) standard errors instead so hypothesis tests and intervals stay valid.

**C6.** The confidence interval for the mean response only reflects uncertainty in estimating `E[y|x]` (parameter uncertainty). A prediction interval for one new observation must *also* include that observation's own irreducible noise term `epsilon`, which adds `sigma^2` to the variance — hence it is always wider, even as `n -> infinity`.

**C7.** A well-run randomized trial with a strong, precisely estimated treatment effect but enormous individual-level outcome variability (health, behavioral, or economic outcomes are full of unexplained individual variation). `R^2` stays low because most *variance* is unexplained, but the coefficient itself is large, precise, and substantively decisive.

**C15.** Extrapolation is predicting `y` at `x` values outside the range the data actually covered. The fitted line is only *validated* — by residual checks, by the data itself — within the observed range; nothing in the algebra guarantees the same relationship continues to hold outside it, even though the equation is happy to produce a number there.

**C9.** Multicollinearity inflates `Var(beta_hat_j)` (formally, via the VIF) because it becomes hard to tell which correlated predictor "deserves credit" for a shared effect — that destabilizes *individual* coefficients. But the fitted values `X * beta_hat` (the combined prediction) stay well-identified, because the *joint* effect of the correlated block is still pinned down by the data even when its split among them is not.

**C10.** The ridge penalty `lambda * sum(beta_j^2)` penalizes every coefficient in the raw units of its feature. A feature on a large numeric scale needs a small coefficient to have the same effect on `y`, so it gets penalized less than an equally-important feature on a small scale gets penalized more. Standardizing all features first makes the penalty treat them comparably.

**C11.** The L1 penalty's constraint region has corners exactly on the coordinate axes. When two predictors are highly correlated, the loss surface's optimum tends to land on one of those corners — lasso arbitrarily zeroes one and keeps the other near the combined coefficient, rather than splitting the weight evenly the way ridge's smooth, rotation-invariant penalty does.

**C12.** Probability `p` lives in `[0, 1]`. Odds `= p / (1 - p)` lives in `[0, infinity)` and expresses "chance of success relative to chance of failure." Logistic regression models the *log-odds* (logit) as linear in `x`; consequently `exp(beta_j)` is the multiplicative change in the **odds**, not in the probability, per one-unit change in `x_j`.

**C13.** If the positive class is 2% of the data, a classifier that always predicts "negative" scores 98% accuracy while being useless. Accuracy cannot distinguish that degenerate classifier from a genuinely good one; use precision, recall, F1, or ROC/PR-AUC, which are sensitive to how the model handles the minority class.

**C8.** Ice-cream sales predict drowning deaths very well (high R^2) because both are driven by hot weather — but reducing ice-cream sales will not reduce drownings. High predictive accuracy only shows association is strong enough to extrapolate *within the same data-generating regime*; it says nothing about what happens if you intervene on the predictor.

**C14.** If a transformation's parameters (a Box-Cox `lambda`, a scaler's mean/variance, a selected polynomial degree) are fit on the *entire* dataset before splitting, information from the validation fold has already leaked into the transformation. Refitting the transform only on each training fold, inside the cross-validation loop, gives an honest estimate of out-of-sample performance.

</details>

## 35. Derivation exercises

**Simple regression from first principles**

- **D1.** Starting from SSE = sum(y_i-b_0-b_1x_i)^2, derive the two normal equations.
- **D2.** Use the first normal equation to show b_0 = y_bar-b_1x_bar.
- **D3.** Derive b_1 = S_xy/S_xx.
- **D4.** Prove that the fitted simple-regression line passes through (x_bar,y_bar).

**Multiple regression in matrix form**

- **D5.** Show that X^T e = 0 in multiple OLS.
- **D6.** Starting from beta_hat = beta + (X^T X)^(-1)X^T epsilon, derive unbiasedness under E[epsilon|X]=0.
- **D7.** Derive Var(beta_hat|X)=sigma^2(X^T X)^(-1) under spherical errors.
- **D8.** Show that H^2=H and H^T=H.
- **D9.** Show that the average leverage is k/n using trace(H)=k.

**Extensions: causal, logistic, regularized**

- **D10.** Derive the omitted-variable-bias formula for one included and one omitted predictor.
- **D11.** Derive the logistic-regression gradient X^T(p-y).
- **D12.** Derive the ridge normal equations.

<details markdown="1">
<summary>Answer key: Derivation exercises (D1-D12)</summary>

**Marking rubric.** Full credit requires every algebraic step shown (no "it follows that" skips) and a one-sentence statement of which assumption, if any, the result depends on.

**D1.** Differentiate `SSE = sum((y_i - b0 - b1*x_i)^2)` with respect to each parameter and set both to zero:
`d(SSE)/d(b0) = -2 * sum(y_i - b0 - b1*x_i) = 0` -> `sum(y_i) = n*b0 + b1*sum(x_i)` (normal equation 1)
`d(SSE)/d(b1) = -2 * sum(x_i*(y_i - b0 - b1*x_i)) = 0` -> `sum(x_i*y_i) = b0*sum(x_i) + b1*sum(x_i^2)` (normal equation 2)

**D2.** Divide normal equation 1 by `n`: `y_bar = b0 + b1*x_bar`, so `b0 = y_bar - b1*x_bar`. No further assumptions needed — this is pure algebra from D1.

**D3.** Substitute `b0 = y_bar - b1*x_bar` into normal equation 2 and collect terms in `b1`:
`sum(x_i*y_i) - n*x_bar*y_bar = b1*(sum(x_i^2) - n*x_bar^2)`.
The left side equals `S_xy = sum((x_i-x_bar)(y_i-y_bar))` and the right side's bracket equals `S_xx = sum((x_i-x_bar)^2)` (standard centering identities), giving `b1 = S_xy / S_xx`.

**D4.** At `x = x_bar`: `y_hat = b0 + b1*x_bar = (y_bar - b1*x_bar) + b1*x_bar = y_bar`. The line always passes through `(x_bar, y_bar)` — a direct consequence of D2, true for every OLS fit with an intercept.

**D5.** OLS minimizes `||y - X*beta||^2`; setting the gradient to zero gives `-2*X^T*(y - X*beta_hat) = 0`, i.e. `X^T*(y - X*beta_hat) = X^T*e = 0` directly. This is the matrix-form generalization of D1 — every column of `X` (including the intercept column, if present) is orthogonal to the residual vector.

**D6.** `beta_hat = (X^T X)^-1 X^T y = (X^T X)^-1 X^T (X*beta + epsilon) = beta + (X^T X)^-1 X^T epsilon`. Taking expectation conditional on `X`: `E[beta_hat | X] = beta + (X^T X)^-1 X^T E[epsilon | X] = beta`, using `E[epsilon | X] = 0`. Unbiasedness depends on exactly this assumption — nothing else.

**D7.** From D6, `beta_hat - beta = (X^T X)^-1 X^T epsilon`. Then `Var(beta_hat | X) = (X^T X)^-1 X^T Var(epsilon | X) X (X^T X)^-1`. Under spherical errors, `Var(epsilon | X) = sigma^2 * I`, so this collapses to `sigma^2 * (X^T X)^-1 X^T X (X^T X)^-1 = sigma^2 * (X^T X)^-1`. This step is exactly where homoskedasticity and no-autocorrelation enter — replace `sigma^2*I` with a general `Omega` and you get the sandwich/GLS formula instead.

**D8.** `H = X(X^T X)^-1 X^T`. `H^2 = X(X^T X)^-1 X^T X (X^T X)^-1 X^T = X(X^T X)^-1 X^T = H`, since `X^T X (X^T X)^-1 = I`. `H^T = (X(X^T X)^-1 X^T)^T = X ((X^T X)^-1)^T X^T = X(X^T X)^-1 X^T = H`, since `(X^T X)^-1` is symmetric (the inverse of a symmetric matrix is symmetric).

**D9.** Leverage `h_i` is the `i`-th diagonal entry of `H`. `trace(H) = sum(h_i)`. Since `H` is an orthogonal projection onto the `k`-dimensional column space of `X`, `trace(H) = rank(X) = k`. Therefore the average leverage is `sum(h_i)/n = k/n`.

**D10.** True model: `y = b0 + b1*x + b2*z + u`. Regressing `y` on `x` alone gives `b1_short = Cov(x,y)/Var(x)`. Substitute the true model for `y`: `Cov(x,y) = b1*Var(x) + b2*Cov(x,z)`. Dividing by `Var(x)`: `b1_short = b1 + b2 * [Cov(x,z)/Var(x)] = b1 + b2*delta`, where `delta` is the coefficient from regressing `z` on `x`. The bias term `b2*delta` is zero only if the omitted variable has no effect (`b2=0`) or is uncorrelated with `x` (`delta=0`).

**D11.** Log-likelihood for one observation: `l_i = y_i*log(p_i) + (1-y_i)*log(1-p_i)`, with `p_i = sigmoid(x_i^T beta)`. Using `dp_i/d(beta) = p_i*(1-p_i)*x_i` and the chain rule, `dl_i/d(beta) = (y_i - p_i)*x_i`. Summing over observations and writing in matrix form gives the gradient `X^T(y - p)` — note the sign convention matches maximizing log-likelihood (equivalently minimizing the negative log-likelihood gives `X^T(p-y)`).

**D12.** Ridge objective: `||y - X*beta||^2 + lambda*||beta||^2`. Gradient: `-2*X^T(y - X*beta) + 2*lambda*beta = 0` -> `X^T y = X^T X beta + lambda*beta = (X^T X + lambda*I)*beta`, so `beta_hat_ridge = (X^T X + lambda*I)^-1 X^T y`. Note `(X^T X + lambda*I)` is invertible even when `X^T X` is singular (perfect multicollinearity) for any `lambda > 0` — this is the formal reason ridge stabilizes the near-collinear case.

</details>

## 36. Pure-Python coding exercises

**Statistics and diagnostics extensions**

- **P1.** Add median and interquartile range functions to the module.
- **P3.** Implement leave-one-out cross-validation.
- **P4.** Add externally studentized residuals.
- **P5.** Implement a partial F test comparing two nested OLS models.
- **P6.** Add HC0 and HC3 robust covariance estimators.
- **P15.** Compare normal-equation and QR results as predictors become nearly collinear.

**Weighted and feature-engineered models**

- **P2.** Write weighted least squares from scratch.
- **P7.** Write a function to one-hot encode string categories without pandas.

**Regularization and optimization**

- **P8.** Add elastic-net coordinate descent.
- **P9.** Implement early stopping with a validation set for gradient descent.

**Logistic regression and classification metrics**

- **P10.** Add probability calibration metrics to logistic regression.
- **P11.** Create a confusion-matrix function and compute precision, recall, specificity, and F1.

**Simulation studies**

- **P12.** Simulate omitted-variable bias and show how it changes with Cov(X,Z).
- **P13.** Simulate heteroskedastic errors and compare classical with robust standard errors.
- **P14.** Simulate multicollinearity and examine coefficient instability across repeated samples.

<details markdown="1">
<summary>Answer key: Pure-Python coding exercises (P1-P15)</summary>

**Marking rubric.** These are implementation tasks: full credit requires the function to run, to match a library (NumPy/statsmodels) reference on a known example within `1e-6`, and to include at least one edge-case test (e.g. `n=2`, a repeated value, a zero-variance column). A hint plus the core formula is given below rather than full code — write the implementation yourself and check it against Appendix A's style.

**P1.** Sort the values; the median is the middle value (`n` odd) or the mean of the two middle values (`n` even). IQR = `Q3 - Q1`, where `Q1`/`Q3` are the 25th/75th percentiles of the sorted list — use linear interpolation between the two nearest ranks to match NumPy's default `percentile` behavior, and test against `numpy.percentile`.

**P2.** WLS minimizes `sum(w_i * (y_i - x_i^T beta)^2)`. The closed-form solution is `beta_hat = (X^T W X)^-1 X^T W y` where `W = diag(w_1,...,w_n)` — reuse your existing Gaussian-elimination solver on `X^T W X` and `X^T W y` rather than writing a new one.

**P3.** LOOCV refits the model `n` times, each time leaving out observation `i`, and averages the squared error on the held-out point. For OLS there is a shortcut that avoids `n` refits: `LOOCV_i = e_i / (1 - h_i)`, using the residual and leverage from a *single* fit — implement the brute-force version first, then verify it matches the shortcut.

**P4.** An externally studentized residual excludes observation `i` when estimating the error variance: `t_i = e_i / (s_(i) * sqrt(1 - h_i))`, where `s_(i)^2` is the residual variance estimated *without* observation `i`. Use the same leave-one-out variance shortcut as P3 rather than refitting for each `i`.

**P5.** The partial F test compares a restricted (fewer predictors) and unrestricted (full) model: `F = [(SSE_restricted - SSE_full)/q] / [SSE_full/(n-k-1)]`, where `q` is the number of restrictions. Fit both models with your existing OLS routine, take the two `SSE` values, and compare `F` against `scipy.stats.f` or a printed F-table for the same degrees of freedom.

**P6.** HC0: `V_HC0 = (X^T X)^-1 X^T diag(e_i^2) X (X^T X)^-1`. HC3 divides each `e_i^2` by `(1-h_i)^2` before building the diagonal matrix (a finite-sample correction that up-weights high-leverage points' contribution). Verify both against `statsmodels`' `cov_type="HC0"` / `"HC3"` on the same data.

**P7.** For each category level, create a binary column that is 1 where the original column equals that level and 0 otherwise; drop one level's column (the reference category) to avoid the dummy-variable trap (perfect multicollinearity with the intercept). Test on a column with 3+ categories and confirm your design matrix has full column rank.

**P8.** Elastic-net coordinate descent updates one coefficient at a time holding the others fixed, using the soft-thresholding operator `S(z, lambda*alpha) = sign(z)*max(|z|-lambda*alpha, 0)` combined with the ridge-style shrinkage `1/(1+lambda*(1-alpha))` from the L2 term. Cycle through all coefficients repeatedly until the largest coefficient change is below a tolerance.

**P9.** Track validation loss after every epoch; if it fails to improve (beyond a small tolerance) for a set "patience" number of consecutive epochs, stop and restore the parameters from the best epoch seen so far (not the final ones) — the second part (restoring, not just stopping) is the detail most implementations miss.

**P10.** A calibration slope/intercept is obtained by regressing the true outcome on the model's predicted logit (`glm(y ~ logit_p, family=binomial)`); a slope near 1 and intercept near 0 indicates good calibration. Bin predictions into deciles and compare mean predicted probability to observed frequency per bin as a visual/numeric check (a reliability diagram).

**P11.** Build the 2x2 confusion matrix by counting true/false positives/negatives at your chosen threshold, then `precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`, `specificity = TN/(TN+FP)`, `F1 = 2*precision*recall/(precision+recall)`. Test on a case with zero predicted positives to confirm you handle the resulting division by zero gracefully.

**P12.** Generate `x`, then `z = rho*x + sqrt(1-rho^2)*noise` (controls `Cov(x,z)` via `rho`), then `y = b0 + b1*x + b2*z + u`. Regress `y` on `x` alone across several values of `rho` and confirm the bias matches the D10 formula `b2 * Cov(x,z)/Var(x)` to within simulation noise.

**P13.** Generate errors whose variance depends on `x` (e.g. `u_i ~ Normal(0, x_i^2)`), fit OLS, and compare the classical standard errors to HC0/HC3 (P6) over many repeated samples — the classical SEs should be systematically wrong (too narrow or too wide depending on the heteroskedasticity's direction) while the robust ones track the true sampling variability.

**P15.** As two predictors are made increasingly correlated (interpolate toward `x2 = x1 + tiny_noise`), track the condition number of `X^T X` and compare coefficients from the normal equations against QR decomposition — the QR solution should remain numerically stable (bounded relative error) noticeably longer than direct matrix inversion as collinearity increases.

</details>

## 37. Library exercises

**Cross-library comparison**

- **L1.** Fit the same model using pure Python, NumPy, statsmodels, and scikit-learn. Compare coefficients.
- **L2.** Create a statsmodels table with classical and HC3 standard errors.

**Pipelines and feature engineering**

- **L3.** Use formula syntax to fit a category-by-continuous interaction and graph predicted lines.
- **L4.** Build a scikit-learn pipeline with imputation, scaling, one-hot encoding, and ridge regression.

**Validation and model selection**

- **L5.** Compare random cross-validation with group cross-validation on clustered data.
- **L6.** Tune ridge, lasso, and elastic net and compare test RMSE.
- **L7.** Create a learning curve to diagnose whether more data may help.

**Diagnostics**

- **L8.** Create residual, Q-Q, scale-location, and influence plots.

**Classification workflows**

- **L9.** Fit logistic regression and select a threshold that gives at least 80% recall.
- **L10.** Compare ROC AUC with precision-recall AUC under severe class imbalance.
- **L11.** Use permutation importance and compare it with coefficients.

**Nonlinear extensions**

- **L12.** Fit a spline or generalized additive model and compare it with a quadratic.

<details markdown="1">
<summary>Answer key: Library exercises (L1-L12)</summary>

**Marking rubric.** Full credit requires running code, a printed comparison table or plot, and one sentence of interpretation — a plot with no discussion of what it shows earns partial credit only.

**L1.** Fit with your Appendix A functions, then `numpy.linalg.lstsq`, then `statsmodels.OLS`, then `sklearn.linear_model.LinearRegression`. All four coefficient vectors should agree to within `1e-8` on well-conditioned data — any larger discrepancy points to a bug in the from-scratch implementation, not in the libraries.

**L2.** `sm.OLS(y, X).fit()` gives classical SEs; `.fit(cov_type="HC3")` gives HC3. Print both tables side by side — the coefficients are identical, only the standard errors (and therefore t-statistics and p-values) change, which is exactly the point: robust covariance corrects inference, not the point estimates.

**L3.** `smf.ols("y ~ C(category) * x", data=df).fit()` fits a separate slope on `x` for each category level. Plot the fitted line for each category over the observed range of `x` and check whether the slopes visibly differ — that visible difference is the interaction.

**L4.** Use `ColumnTransformer` to route numeric columns through `SimpleImputer` + `StandardScaler` and categorical columns through `OneHotEncoder`, then feed both into `Ridge` inside a single `Pipeline`. Fit the pipeline only on the training fold in each cross-validation split so no preprocessing statistic leaks from validation data (see C14).

**L5.** Use `KFold` for random CV and `GroupKFold` (grouping by cluster/subject id) for group CV on the same clustered dataset. Random CV will typically report better (over-optimistic) performance because it lets information about a cluster leak between train and validation folds; group CV gives the honest estimate for predicting on an unseen cluster.

**L6.** Use `RidgeCV`, `LassoCV`, and `ElasticNetCV` (or manual `GridSearchCV`) with the same cross-validation folds for all three, then compare test-set RMSE. Also compare the number of nonzero coefficients — lasso and elastic net will typically zero out some features that ridge keeps small but nonzero.

**L7.** Use `sklearn.model_selection.learning_curve` to plot training and validation error against training-set size. Converging curves with a small gap suggest more data will not help much (the model is at its ceiling); a persistent large gap suggests more data (or a less flexible model) would help.

**L8.** Residuals-vs-fitted checks non-linearity and non-constant variance; Q-Q checks the normality assumption used for exact small-sample inference; scale-location (`sqrt(|standardized residual|)` vs fitted) checks heteroskedasticity more sensitively than the raw residual plot; leverage-vs-residual (with Cook's distance contours) flags influential points. Produce all four for one fitted model and describe what each one does or does not show.

**L9.** Use `predict_proba` and `sklearn.metrics.precision_recall_curve` to find the threshold where recall first reaches 0.80, then report the precision paid for that recall. Confirm that lowering the default 0.5 threshold is what buys the additional recall, and state the cost in false positives.

**L10.** Compute both curves with `roc_curve`/`roc_auc_score` and `precision_recall_curve`/`average_precision_score` on the same imbalanced dataset. ROC-AUC can look deceptively good under severe imbalance because it is dominated by the (easy) true-negative rate; PR-AUC is more sensitive to how the model handles the rare positive class — report both and explain the discrepancy if one appears.

**L11.** Use `sklearn.inspection.permutation_importance` and compare the ranking to the model's raw coefficients (standardized to be comparable). They can disagree meaningfully under multicollinearity — permutation importance measures the drop in performance from shuffling one feature *given the others are still present*, which can be small even for a feature with a large coefficient if a correlated feature can compensate for it.

**L12.** Fit `sklearn.preprocessing.SplineTransformer` (or a GAM via `pygam`) alongside a plain quadratic (`PolynomialFeatures(degree=2)`) on the same data, and compare fitted curves and out-of-sample error. Splines/GAMs can capture local curvature a global quadratic cannot, but check whether the improvement is real (cross-validated) or just a more flexible model overfitting the training data.

</details>

## 38. Suggested learning sequence


### Stage 1: Intuition

Study Sections 1-3. Be able to explain conditional means, fitted values, residuals, slope, and intercept without formulas.


### Stage 2: Derivation

Study Sections 3-6. Derive simple OLS and understand the multiple-regression normal equations.


### Stage 3: Uncertainty

Study Sections 5-9. Explain assumptions, standard errors, t tests, intervals, leverage, and influence.


### Stage 4: From-scratch coding

Study Sections 15-24. Run the companion module. Alter data and verify algebraic properties.


### Stage 5: Practical libraries

Study Sections 25-32. Reproduce the same model with NumPy, statsmodels, and scikit-learn.


### Stage 6: Extensions

Study transformations, regularization, logistic regression, and causal cautions.


### Stage 7: Independent project

Choose one real dataset. Write the question before modelling. Produce:

- a data audit;
- a baseline model;
- a justified specification;
- diagnostics;
- validation results;
- an interpretation in natural language;
- limitations and sensitivity checks;
- reproducible code.

## 39. Prompts for an AI tutor or NotebookLM

Use prompts like these after uploading the guide and companion code.


### Concept teaching

- “Teach me Section 3 Socratically. Ask one question at a time and do not reveal the answer until I attempt it.”
- “Explain the geometric meaning of OLS using only a two-predictor mental picture.”
- “Compare zero conditional mean, homoskedasticity, and normality. Tell me exactly which conclusions fail when each is violated.”
- “Explain the Frisch-Waugh-Lovell theorem with a development-programme example.”

### Mathematical practice

- “Derive the simple-regression slope step by step and stop after each algebraic transformation for me to confirm.”
- “Quiz me on dimensions of every matrix in multiple OLS.”
- “Give me five exercises on confidence intervals and prediction intervals, increasing in difficulty.”
- “Make me prove that residuals are orthogonal to the columns of X.”

### Code teaching

- “Walk through fit_ols in the pure-Python module line by line. For each block, connect code to the exact equation.”
- “Hide one function from the module and ask me to reimplement it from a specification and tests.”
- “Create debugging exercises by inserting one realistic bug at a time into matrix multiplication, QR, and inference code.”
- “Compare the pure-Python implementation with numpy.linalg.lstsq and explain numerical stability.”

### Interpretation practice

- “Give me a regression output table and ask me to write a correct substantive interpretation.”
- “Create examples where a p-value is small but practical importance is negligible.”
- “Give me examples of omitted-variable bias with positive and negative bias.”
- “Ask me whether a causal claim is justified in ten short research scenarios.”

### Project supervision

- “Act as a demanding statistical reviewer. Ask for my research question, data-generating process, validation design, diagnostics, and limitations.”
- “Review my model specification for leakage, bad controls, extrapolation, and incorrect coefficient interpretation.”
- “Create a study plan based on my answers, revisiting only concepts I fail.”

## 40. Glossary

**Adjusted R-squared:** An in-sample fit measure that penalizes the number of estimated parameters.

**Bias:** Difference between an estimator’s expected value and the target parameter.

**Categorical predictor:** A variable representing groups rather than numeric magnitude, usually encoded with indicator variables.

**Coefficient:** A parameter describing how the conditional outcome changes with a predictor under a specified model.

**Collinearity:** Linear association among predictors; perfect collinearity prevents unique estimation.

**Conditional mean:** The expected outcome at specified predictor values.

**Confidence interval:** An interval produced by a procedure with a stated long-run coverage rate under assumptions.

**Cook’s distance:** A measure of how strongly one observation influences the fitted regression.

**Cross-validation:** Repeated fitting and validation across data folds to estimate generalization performance or tune models.

**Design matrix:** Matrix containing predictor columns, transformations, indicators, interactions, and often an intercept.

**Disturbance:** Unobserved population error term in the model.

**Elastic net:** Regression using a mixture of L1 and L2 penalties.

**Endogeneity:** Correlation between a predictor and the disturbance, often violating zero conditional mean.

**Extrapolation:** Prediction outside the predictor range or support represented by the data.

**Fitted value:** Model prediction for an observed case.

**Heteroskedasticity:** Non-constant conditional disturbance variance.

**Influence:** Degree to which an observation changes fitted results.

**Interaction:** A term allowing the association of one predictor to vary with another.

**Lasso:** L1-penalized regression capable of setting coefficients exactly to zero.

**Leverage:** Unusualness of an observation’s predictor combination, measured by a hat-matrix diagonal.

**Likelihood:** Probability model for observed data viewed as a function of unknown parameters.

**Log odds:** log[p/(1-p)], the linear scale used by logistic regression.

**Mean squared error:** Average squared prediction error.

**Multicollinearity:** Strong linear dependence among multiple predictors.

**Normal equations:** X^T X beta_hat = X^T y, the first-order conditions for OLS.

**Ordinary least squares:** Coefficient estimates minimizing the sum of squared residuals.

**Outlier:** Observation unusual in the outcome direction relative to the model.

**Overfitting:** Learning sample-specific noise that does not generalize.

**p-value:** Under a null model, the probability of a test statistic at least as extreme as observed.

**Prediction interval:** Interval for a future individual outcome, including observation noise.

**QR decomposition:** Matrix factorization used for stable least-squares computation.

**R-squared:** Fraction of sample outcome variation represented by fitted values in an intercept model.

**Regularization:** Penalizing model complexity or coefficient magnitude to improve stability and generalization.

**Residual:** Observed outcome minus fitted value.

**Ridge regression:** L2-penalized regression that shrinks coefficients but usually does not set them to zero.

**Robust standard error:** Standard error designed to remain valid under specified departures such as heteroskedasticity.

**Standard error:** Estimated sampling standard deviation of an estimator.

**Student t distribution:** Reference distribution used for coefficient inference when variance is estimated.

**Variance inflation factor:** Measure of how predictor collinearity inflates a coefficient’s variance.

**Zero conditional mean:** Assumption that the disturbance averages zero at every predictor combination.


## 41. Further study

After mastering this guide, the natural next topics are:

- generalized linear models for counts, proportions, and non-normal outcomes;
- multilevel and mixed-effects models for grouped data;
- panel-data models and fixed effects;
- time-series regression and forecasting;
- survival analysis;
- causal inference and research design;
- generalized additive models and splines;
- quantile regression and robust regression;
- Bayesian regression;
- tree ensembles and modern machine learning;
- uncertainty quantification under model selection and distribution shift.

Useful foundational books include works on applied regression, econometrics, statistical learning, and linear-model theory. When studying from different traditions, pay attention to differences in notation and purpose: econometrics emphasizes identification and inference, statistics emphasizes probability models and uncertainty, and machine learning emphasizes generalization and computation.


# Part V. Research-Level Regression Practice


## 42. Weighted least squares and generalized least squares

Ordinary least squares gives every squared residual equal weight. Weighted least squares minimizes

```
sum_i w_i (y_i - x_i^T beta)^2
```

with positive weights w_i. If Var(epsilon_i | X) is proportional to 1/w_i and observations are otherwise independent, the estimator is efficient under the specified variance model:

```
beta_hat_WLS = (X^T W X)^(-1) X^T W y
```

Weights have different meanings and must not be conflated. Precision weights represent inverse conditional variance. Frequency weights represent repeated identical observations. Sampling weights represent unequal probabilities of selection and generally require survey-design methods rather than naive WLS. Importance weights change the target distribution in predictive learning.

Generalized least squares allows a non-diagonal covariance matrix Omega:

```
beta_hat_GLS = (X^T Omega^(-1) X)^(-1) X^T Omega^(-1) y
```

GLS is useful when errors have known or modelled correlation, such as repeated or time-ordered observations. Feasible GLS estimates nuisance parameters in Omega and then plugs them into the formula. Its validity depends on the covariance model; robust or cluster-robust inference may be preferable when that model is uncertain.


## 43. Robust covariance estimators in greater detail

Write the OLS residual for observation i as e_i and its leverage as h_i. The general sandwich form is

```
V_hat(beta_hat) = B M B, where B = (X^T X)^(-1)
```

The meat M changes across estimators. HC0 uses sum_i e_i^2 x_i x_i^T. HC1 multiplies HC0 by n/(n-k). HC2 replaces e_i^2 with e_i^2/(1-h_i). HC3 uses e_i^2/(1-h_i)^2 and resembles a leave-one-out correction.

Robust standard errors are not a cure for misspecified conditional means, endogeneity, dependence, or severe small-sample problems. They address a particular uncertainty problem: unknown heteroskedasticity under exogeneity and suitable asymptotics.

Cluster-robust covariance groups score contributions. For cluster g, let X_g and e_g denote its rows and residual vector. The meat is built from

```
sum_g X_g^T e_g e_g^T X_g
```

The effective asymptotic information comes mainly from the number of independent clusters, not merely the number of observations. With few clusters, use small-sample corrections, wild cluster bootstrap methods, or design-specific approaches.


## 44. Bootstrap reasoning

The bootstrap approximates a sampling distribution by repeated resampling and refitting. In the pairs bootstrap, sample rows (x_i, y_i) with replacement. In a residual bootstrap, hold X fixed and resample centered residuals under assumptions that make residual exchangeability credible. In a cluster bootstrap, sample whole clusters.

A valid bootstrap must mimic the dependence and data-generation structure relevant to the estimator. Randomly resampling individual rows is invalid when observations are clustered, spatially dependent, or sequential.

Common intervals include percentile intervals, basic intervals, and bias-corrected and accelerated intervals. Bootstrap methods are not automatically superior to analytic intervals; they can fail under non-smooth estimators, weak identification, boundary parameters, small numbers of clusters, or a resampling scheme that does not match the design.


## 45. Missing data and regression

Missingness is part of the data-generating process. Three classical labels are:

- MCAR: missingness is independent of observed and unobserved values;

- MAR: after conditioning on observed data, missingness does not depend on missing values;

- MNAR: missingness still depends on unobserved values after conditioning on observed data.

Complete-case analysis changes the target population and may be biased unless restrictive conditions hold. Single mean imputation understates uncertainty and distorts relationships. Predictive pipelines may use train-fold imputation for generalization, but inferential research often requires multiple imputation or explicit missingness models.

Multiple imputation creates several completed datasets, fits the analysis separately, and combines estimates and uncertainty using Rubin-style rules. The imputation model should contain the analysis variables, variables predictive of missingness, and transformations or interactions needed to preserve the intended analysis.

Sensitivity analysis is essential when MNAR mechanisms are plausible because the missingness assumption cannot usually be tested from observed data alone.


## 46. Survey data, sampling weights, and finite populations

Regression in a complex survey must respect stratification, clustering, and unequal inclusion probabilities. A sampling weight often approximates the number of population units represented by an observed unit. Applying those weights in an ordinary regression does not by itself produce correct standard errors.

Design-based inference treats population values as fixed and randomness as arising from the sampling design. Model-based inference treats outcomes as generated by a stochastic model. Modern analyses often combine these perspectives, but the target estimand and variance calculation must be explicit.

When results are intended to describe a population rather than optimize prediction in the observed sample, report the sampling design, weight construction, nonresponse adjustments, and variance method.


## 47. Model specification, causal diagrams, and bad controls

A regression specification is a scientific claim about which conditional relationship is useful. Variable selection cannot be reduced to a mechanical p-value rule.

For causal analysis, a directed acyclic graph can encode assumptions about causal direction and common causes. A sufficient adjustment set blocks non-causal backdoor paths without conditioning on descendants of treatment or colliders. The graph does not prove the assumptions; it makes them inspectable.

Predictive feature selection has a different goal. Variables that are descendants, proxies, or operational signals may improve prediction even when they are inappropriate causal controls. Always state whether the target is prediction, description, explanation, or intervention.


## 48. Model uncertainty and researcher degrees of freedom

A reported model is often one of many plausible specifications. If transformations, controls, exclusions, interactions, and outcomes were repeatedly changed after viewing results, ordinary standard errors ignore this search process.

Useful responses include:

- preregistration or a written analysis plan;

- a clearly labelled exploratory phase;

- multiverse or specification-curve analysis;

- holdout data for model selection;

- shrinkage and regularization;

- selective-inference methods when appropriate;

- transparent reporting of all consequential choices.

Sensitivity analysis is not a ritual of showing many nearly identical models. It should vary assumptions that could plausibly change the conclusion and explain why each alternative matters.


## 49. Reproducible regression research

A reproducible project should preserve:

- an immutable copy or checksum of source data;

- a data dictionary and provenance record;

- code that creates the analysis dataset from raw inputs;

- a fixed software environment or environment specification;

- random seeds and deterministic settings where possible;

- tests for key transformations and calculations;

- a record of exclusions and model-selection decisions;

- generated tables and figures produced from code rather than manual editing;

- a README explaining how to run the project.

Reproducibility is necessary but not sufficient. A perfectly reproducible analysis can still answer the wrong question or rely on invalid assumptions.


## 50. How to read a regression research paper

Read a paper in passes rather than from the first line to the last.


### Pass 1: Orientation

Identify the research question, target population, outcome, main predictors or intervention, unit of observation, data source, and headline claim. Read the abstract, figures, tables, and conclusion. Do not yet accept the authors' interpretation.


### Pass 2: Identification and estimand

Write the estimand in your own words. Ask what comparison gives the coefficient its meaning. For causal claims, identify the intervention, counterfactual contrast, timing, and assumptions that make the comparison credible.


### Pass 3: Data generation

Inspect sampling, measurement, missingness, exclusions, repeated observations, clustering, and temporal ordering. Determine whether the available data represent the target population and whether the variables were measured before or after the exposure of interest.


### Pass 4: Model and computation

Write the model mathematically. Identify transformations, interactions, fixed effects, weights, penalties, and covariance estimator. Check whether standard errors respect clustering or dependence. Determine how hyperparameters and models were selected.


### Pass 5: Diagnostics and robustness

Look for residual checks, influence analysis, overlap or support, collinearity, calibration, validation, and sensitivity to reasonable alternatives. Absence of a diagnostic is not proof of failure, but it identifies an unexamined risk.


### Pass 6: Reproduction

Attempt to reconstruct one key table or figure. Confirm sample size, variable definitions, coding, and coefficient interpretation. Reproduction frequently reveals ambiguities that passive reading misses.


### Pass 7: Critique and extension

Separate threats to internal validity, external validity, measurement validity, statistical conclusion validity, and computational reproducibility. Propose one analysis that would most reduce uncertainty, rather than listing every imaginable limitation.

A paper-reading worksheet should contain: claim, estimand, design, model, assumptions, evidence, diagnostics, robustness, limitations, and your proposed extension.


## 51. A research-grade regression report structure

A strong report normally contains:

1. Question and estimand.
2. Institutional or substantive context.
3. Data source, sampling, and measurement.
4. Analysis plan and preprocessing.
5. Model specification and identification logic.
6. Estimation and uncertainty method.
7. Diagnostics and validation.
8. Main results in meaningful units.
9. Sensitivity and robustness analyses.
10. Limitations and scope of inference.
11. Reproducibility information.
12. Appendices with derivations, code, and supplementary results.

Lead with the scientific question, not the software command. A table of coefficients is evidence, not the argument itself.


## 52. Pure-Python research extensions

The following functions build on the companion module. They are educational implementations for small datasets.


### 52.1 Weighted least squares

```python
def fit_wls(X, y, weights, add_intercept=True):
    if len(X) != len(y) or len(y) != len(weights):
        raise ValueError("X, y, and weights must have equal row counts")
    if any(w <= 0 for w in weights):
        raise ValueError("weights must be positive")

    design = add_intercept_column(X) if add_intercept else X
    sqrt_w = [math.sqrt(float(w)) for w in weights]
    Xw = [
        [sqrt_w[i] * float(value) for value in design[i]]
        for i in range(len(design))
    ]
    yw = [sqrt_w[i] * float(y[i]) for i in range(len(y))]
    coefficients = least_squares_qr(Xw, yw)
    fitted = matvec(design, coefficients)
    residuals = [float(a) - b for a, b in zip(y, fitted)]
    return coefficients, fitted, residuals
```


### 52.2 HC0 and HC3 covariance

```python
def robust_covariance(model, kind="HC3"):
    kind = kind.upper()
    if kind not in {"HC0", "HC1", "HC2", "HC3"}:
        raise ValueError("kind must be HC0, HC1, HC2, or HC3")

    X = model.design_matrix
    bread = model.xtx_inverse
    n = model.n_observations
    k = model.n_parameters
    meat = [[0.0 for _ in range(k)] for _ in range(k)]

    for row, residual in zip(X, model.residuals):
        h = dot(row, matvec(bread, row))
        if kind == "HC0" or kind == "HC1":
            scale = residual ** 2
        elif kind == "HC2":
            scale = residual ** 2 / max(1.0 - h, 1e-15)
        else:
            scale = residual ** 2 / max((1.0 - h) ** 2, 1e-15)

        for a in range(k):
            for b in range(k):
                meat[a][b] += scale * row[a] * row[b]

    covariance = matmul(matmul(bread, meat), bread)
    if kind == "HC1":
        factor = n / (n - k)
        covariance = [[factor * value for value in row] for row in covariance]
    return covariance
```


### 52.3 Pairs bootstrap

```python
def bootstrap_ols_pairs(X, y, repetitions=1000, seed=42):
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if len(X) != len(y):
        raise ValueError("X and y must have equal row counts")

    rng = random.Random(seed)
    n = len(y)
    coefficient_draws = []
    for _ in range(repetitions):
        indices = [rng.randrange(n) for _ in range(n)]
        X_star = [X[i] for i in indices]
        y_star = [y[i] for i in indices]
        try:
            coefficient_draws.append(fit_ols(X_star, y_star).coefficients)
        except ValueError:
            # A bootstrap sample can be rank deficient. Record and continue.
            continue
    if not coefficient_draws:
        raise RuntimeError("all bootstrap fits failed")
    return coefficient_draws
```

The possibility of rank-deficient bootstrap samples should be reported rather than hidden. Frequent failure is evidence that the design is fragile.


## 53. Further research directions

After the core material, deepen understanding through the following themes:

- high-dimensional linear models, sparsity, and post-selection inference;

- semiparametric regression and partially linear models;

- quantile and robust regression;

- nonparametric regression, kernels, splines, and generalized additive models;

- hierarchical and multilevel models;

- panel data, fixed effects, and correlated random effects;

- causal identification, doubly robust estimation, and heterogeneous effects;

- conformal prediction and distribution-free predictive intervals;

- Bayesian regression and prior sensitivity;

- distribution shift, transportability, and domain adaptation;

- measurement error and latent-variable models;

- privacy-preserving and federated estimation.

The next formal topic in this curriculum is generalized linear models, which replaces the constant-variance Gaussian response assumption with a distributional family and a link between the conditional mean and a linear predictor.


## 54. Expanded glossary


### Asymptotic distribution

A limiting probability distribution that approximates an estimator's sampling distribution as sample size grows. Asymptotic results justify many standard errors and tests without exact normal errors, but their quality depends on effective sample size, leverage, dependence, parameter dimension, and regularity conditions.


### Average partial effect

The sample or population average of an observation-specific marginal effect. In nonlinear models, a coefficient is not generally a constant effect on the outcome scale, so averaging derivative or discrete-change effects can provide a more interpretable estimand.


### Bootstrap

A resampling framework that approximates sampling variation by repeatedly reconstructing datasets and refitting the analysis. The resampling unit must match the design: rows for independent observations, clusters for clustered data, or blocks for dependent sequences.


### Calibration

Agreement between predicted values or probabilities and observed outcomes at corresponding prediction levels. Calibration is distinct from ranking and discrimination. A model can rank cases well while systematically overpredicting or underpredicting.


### Causal estimand

A precisely defined contrast between potential outcomes under specified interventions, such as an average treatment effect. A regression coefficient is a causal estimand only when model, design, timing, and identification assumptions justify that interpretation.


### Cluster-robust covariance

A sandwich covariance estimator that permits arbitrary residual dependence within predefined clusters while requiring suitable independence across clusters. Reliability depends strongly on the number and balance of clusters.


### Condition number

A measure of sensitivity of a matrix problem to perturbations. In regression, a large condition number may reflect near collinearity or incompatible scales and warns that coefficient estimates can be numerically unstable.


### Confounding

A mixing of the association of interest with differences caused by common determinants of exposure and outcome. Confounding is a causal concept and cannot be diagnosed solely from correlations or p-values in observed data.


### Cross-fitting

A sample-splitting procedure in which nuisance models are trained on one part of the data and predictions are formed on another. It reduces overfitting bias in modern semiparametric and machine-learning estimators.


### Data-generating process

The substantive and probabilistic mechanism by which units enter the sample, variables are measured, treatments or exposures arise, and outcomes are generated. Statistical assumptions are claims about this process, not properties created by software.


### Degrees of freedom

A measure of independent information remaining after estimating parameters or imposing constraints. In simple classical OLS, residual degrees of freedom are n-k, but effective degrees of freedom become more subtle in smoothing and regularized models.


### Design effect

The ratio of actual sampling variance under a complex design to the variance under a simple random sample of the same nominal size. Clustering often raises the design effect and reduces effective sample size.


### Estimand

The population quantity the analysis intends to learn. Defining it forces clarity about population, outcome, intervention or predictor contrast, aggregation, and time horizon.


### Estimator

A rule that maps observed data to an estimate. OLS, ridge, lasso, maximum likelihood, and bootstrap-corrected procedures are estimators with different targets and sampling properties.


### Exchangeability

A symmetry condition under which joint probability is unchanged by permitted reorderings. Many resampling and causal methods rely on specific forms of exchangeability; it is weaker than identical physical cases and stronger than a visual resemblance in covariates.


### Fixed design

A theoretical framework that conditions on the observed predictor matrix X and treats only outcomes as random. Random-design analysis also averages over the distribution of predictors. The distinction affects interpretations of variance and generalization.


### Generalization error

Expected loss on new data drawn from a specified target distribution. It is not the same as training error and changes when the deployment population or measurement process changes.


### Identification

The property that an estimand is uniquely determined by the distribution of observed data under stated assumptions. An identified parameter can still be estimated imprecisely; an unidentified parameter cannot be recovered merely by collecting more of the same type of data.


### Influence function

A first-order description of how an estimator changes when the data distribution is infinitesimally contaminated at a point. It underlies robust statistics, sandwich variance estimators, and many semiparametric methods.


### Interaction

A model term indicating that the relationship between one predictor and the outcome depends on another predictor. Interaction is scale-dependent: absence on an additive scale does not imply absence on a multiplicative scale.


### Leverage

The degree to which an observation has an unusual predictor configuration relative to the design. High leverage creates the potential for influence but does not itself imply a large residual or data error.


### Loss function

A numerical penalty assigned to prediction or estimation error. Squared loss emphasizes large errors; absolute loss targets a conditional median; log loss evaluates probabilistic predictions. The chosen loss defines what optimality means.


### Measurement error

Discrepancy between a measured predictor or outcome and the underlying quantity of interest. Classical error in a predictor often attenuates a simple slope, but general measurement-error bias can have either direction and requires an explicit model.


### Model-assisted inference

A survey approach that uses a working model to improve precision while retaining design-based validity under the sampling scheme. It distinguishes the role of the model from fully model-based inference.


### Nonparametric bootstrap

Bootstrap resampling from the empirical distribution without specifying a full parametric model. It is flexible but still assumes the empirical resampling scheme represents the relevant data-generation structure.


### Orthogonality

A zero dot product between vectors. OLS residuals are orthogonal to every design-matrix column at the fitted solution. In modern estimation, orthogonal scores are constructed to reduce sensitivity to nuisance-parameter errors.


### Overlap or positivity

The requirement that relevant predictor or treatment combinations have adequate probability in the data. Lack of overlap makes comparisons dependent on extrapolation and can cause unstable weights or coefficients.


### Partialling out

Removing the linear contribution of control variables from a predictor and outcome before relating their remaining components. The Frisch-Waugh-Lovell theorem formalizes this operation for OLS.


### Power

The probability that a statistical test rejects a false null under a specified alternative. Power depends on effect size, noise, design, sample size, test, and significance level. Low power also makes statistically significant estimates prone to exaggeration.


### Preregistration

A time-stamped statement of hypotheses, outcomes, exclusions, and analysis decisions made before inspecting the relevant results. It does not eliminate judgement but helps distinguish confirmatory from exploratory analysis.


### Projection parameter

The coefficient vector giving the best linear mean-square approximation to an outcome even when the conditional mean is not exactly linear. This interpretation can make OLS meaningful under misspecification, though robust uncertainty is then important.


### Regularization path

The sequence of fitted coefficients as a penalty parameter varies. It reveals which coefficients shrink, enter, or leave the model and is more informative than a single tuned solution.


### Sandwich estimator

A covariance estimator with a bread-meat-bread form. The bread represents local curvature or sensitivity of estimating equations; the meat represents variability of observation or cluster score contributions.


### Sensitivity analysis

A systematic examination of how conclusions change under plausible alternative assumptions, definitions, specifications, or unmeasured bias. A good sensitivity analysis is tied to a concrete threat, not a random collection of models.


### Standardization

Transforming a variable by subtracting a center and dividing by a scale. It changes coefficient units and optimization geometry but does not create comparability of causal meaning across variables.


### Statistical functional

A quantity defined as a function of an underlying probability distribution, such as a mean, quantile, covariance, or regression projection. Thinking in functionals clarifies what an estimator targets.


### Support

The set of values or combinations that a variable can take with positive probability. Predictions outside observed or plausible support are extrapolations even if individual values look numerically close.


### Transportability

The problem of moving an estimated relationship or causal effect from a study population to a different target population. It requires assumptions about which mechanisms are stable and which population differences matter.


### Uncertainty interval

A general term for an interval intended to express uncertainty. Confidence, credible, prediction, tolerance, and conformal intervals have different definitions and should not be treated as interchangeable.


### Wild bootstrap

A bootstrap that multiplies residual-like quantities by random weights, often used for heteroskedastic or clustered regression. It preserves predictor values and can approximate null distributions more accurately than naive residual resampling in some settings.


## 55. Suggested foundational and research papers

Read the papers in stages. Begin with the problem and contribution, then reconstruct the mathematics and code.

1. Frisch, R., and Waugh, F. V. (1933). Partial Time Regressions as Compared with Individual Trends. Econometrica, 1(4), 387-401. A foundation for partialling out and the theorem later completed in modern form.
2. Lovell, M. C. (1963). Seasonal Adjustment of Economic Time Series and Multiple Regression Analysis. Journal of the American Statistical Association, 58(304), 993-1010. Develops the residualization result now called the Frisch-Waugh-Lovell theorem.
3. White, H. (1980). A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity. Econometrica, 48(4), 817-838. Foundational sandwich covariance work.
4. MacKinnon, J. G., and White, H. (1985). Some Heteroskedasticity-Consistent Covariance Matrix Estimators with Improved Finite Sample Properties. Journal of Econometrics, 29(3), 305-325. Introduces finite-sample variants related to HC2 and HC3.
5. Hoerl, A. E., and Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. Technometrics, 12(1), 55-67. The classical ridge-regression paper.
6. Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. Journal of the Royal Statistical Society: Series B, 58(1), 267-288. Introduces lasso shrinkage and exact zeros.
7. Zou, H., and Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net. Journal of the Royal Statistical Society: Series B, 67(2), 301-320. Combines L1 and L2 penalties and explains the grouping effect.
8. Efron, B., Hastie, T., Johnstone, I., and Tibshirani, R. (2004). Least Angle Regression. Annals of Statistics, 32(2), 407-499. Connects forward selection and lasso paths computationally.
9. Breusch, T. S., and Pagan, A. R. (1979). A Simple Test for Heteroscedasticity and Random Coefficient Variation. Econometrica, 47(5), 1287-1294. A classic score-type heteroskedasticity diagnostic.
10. Cook, R. D. (1977). Detection of Influential Observation in Linear Regression. Technometrics, 19(1), 15-18. Introduces Cook's distance.
11. Huber, P. J. (1964). Robust Estimation of a Location Parameter. Annals of Mathematical Statistics, 35(1), 73-101. Foundational robust-estimation ideas that extend naturally to regression.
12. Koenker, R., and Bassett, G. (1978). Regression Quantiles. Econometrica, 46(1), 33-50. Establishes quantile regression.
13. Rosenbaum, P. R., and Rubin, D. B. (1983). The Central Role of the Propensity Score in Observational Studies for Causal Effects. Biometrika, 70(1), 41-55. Important for understanding regression adjustment and design in observational studies.
14. Chernozhukov, V., Chetverikov, D., Demirer, M., et al. (2018). Double/Debiased Machine Learning for Treatment and Structural Parameters. The Econometrics Journal, 21(1), C1-C68. Introduces orthogonal scores and cross-fitting for modern causal estimation.
15. Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., and Wasserman, L. (2018). Distribution-Free Predictive Inference for Regression. Journal of the American Statistical Association, 113(523), 1094-1111. A key conformal-prediction paper.
16. Cattaneo, M. D., Jansson, M., and Newey, W. K. (2018). Inference in Linear Regression Models with Many Covariates and Heteroskedasticity. Journal of the American Statistical Association, 113(523), 1350-1361. Shows why standard robust covariance formulas can fail with many controls.
17. Buja, A., Berk, R., Brown, L., et al. (2019). Models as Approximations I: Consequences Illustrated with Linear Regression. Statistical Science, 34(4), 523-544. Reinterprets regression under model misspecification.
18. Berk, R., Brown, L., Buja, A., Zhang, K., and Zhao, L. (2013). Valid Post-Selection Inference. Annals of Statistics, 41(2), 802-837. Explains inferential consequences of selecting models after seeing data.

For each paper, create a one-page note containing: the research problem, the estimand, the main assumptions, the central derivation or algorithm, the simulation or empirical design, the main result, one limitation, and one extension you could study.


# Appendix A. Full Pure-Python Implementation

The following module is complete and executable. Save it as regression_from_scratch.py.

```python
"""Regression from scratch using only Python's standard library.

Educational implementation covering:
- descriptive statistics
- simple and multiple ordinary least squares (OLS)
- QR and normal-equation solvers
- coefficient inference using a from-scratch Student t distribution
- confidence and prediction intervals
- residual diagnostics, leverage, and Cook's distance
- train/test splitting and k-fold cross-validation
- gradient-descent linear regression
- ridge and lasso regression
- logistic regression by gradient descent

The emphasis is transparency rather than production-scale speed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, List, Sequence, Tuple, Optional, Dict, Any

Vector = List[float]
Matrix = List[List[float]]


# ---------------------------------------------------------------------------
# Basic descriptive statistics
# ---------------------------------------------------------------------------

def _as_float_vector(values: Iterable[float]) -> Vector:
    result = [float(v) for v in values]
    if not result:
        raise ValueError("Expected at least one value.")
    return result


def mean(values: Iterable[float]) -> float:
    xs = _as_float_vector(values)
    return sum(xs) / len(xs)


def variance(values: Iterable[float], sample: bool = True) -> float:
    xs = _as_float_vector(values)
    n = len(xs)
    ddof = 1 if sample else 0
    if n <= ddof:
        raise ValueError("Not enough observations for the requested variance.")
    xbar = sum(xs) / n
    return sum((x - xbar) ** 2 for x in xs) / (n - ddof)


def standard_deviation(values: Iterable[float], sample: bool = True) -> float:
    return math.sqrt(variance(values, sample=sample))


def covariance(x: Iterable[float], y: Iterable[float], sample: bool = True) -> float:
    xs = _as_float_vector(x)
    ys = _as_float_vector(y)
    if len(xs) != len(ys):
        raise ValueError("x and y must have the same length.")
    n = len(xs)
    ddof = 1 if sample else 0
    if n <= ddof:
        raise ValueError("Not enough observations for the requested covariance.")
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    return sum((a - xbar) * (b - ybar) for a, b in zip(xs, ys)) / (n - ddof)


def correlation(x: Iterable[float], y: Iterable[float]) -> float:
    xs = _as_float_vector(x)
    ys = _as_float_vector(y)
    sx = standard_deviation(xs)
    sy = standard_deviation(ys)
    if sx == 0.0 or sy == 0.0:
        raise ValueError("Correlation is undefined when a variable has zero variance.")
    return covariance(xs, ys) / (sx * sy)


# ---------------------------------------------------------------------------
# Matrix operations
# ---------------------------------------------------------------------------

def _validate_matrix(a: Matrix) -> Tuple[int, int]:
    if not a or not a[0]:
        raise ValueError("Matrix must be non-empty.")
    rows = len(a)
    cols = len(a[0])
    if any(len(row) != cols for row in a):
        raise ValueError("Matrix rows must all have the same length.")
    return rows, cols


def transpose(a: Matrix) -> Matrix:
    rows, cols = _validate_matrix(a)
    return [[float(a[i][j]) for i in range(rows)] for j in range(cols)]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length.")
    return sum(float(x) * float(y) for x, y in zip(a, b))


def matmul(a: Matrix, b: Matrix) -> Matrix:
    a_rows, a_cols = _validate_matrix(a)
    b_rows, b_cols = _validate_matrix(b)
    if a_cols != b_rows:
        raise ValueError("Incompatible matrix dimensions for multiplication.")
    bt = transpose(b)
    return [[dot(a[i], bt[j]) for j in range(b_cols)] for i in range(a_rows)]


def matvec(a: Matrix, x: Sequence[float]) -> Vector:
    rows, cols = _validate_matrix(a)
    if cols != len(x):
        raise ValueError("Incompatible matrix and vector dimensions.")
    return [dot(a[i], x) for i in range(rows)]


def identity(n: int) -> Matrix:
    if n <= 0:
        raise ValueError("n must be positive.")
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def add_intercept_column(x: Matrix) -> Matrix:
    _validate_matrix(x)
    return [[1.0] + [float(v) for v in row] for row in x]


def solve_linear_system(a: Matrix, b: Sequence[float], tolerance: float = 1e-12) -> Vector:
    """Solve A x = b by Gauss-Jordan elimination with partial pivoting."""
    n_rows, n_cols = _validate_matrix(a)
    if n_rows != n_cols:
        raise ValueError("A must be square.")
    if len(b) != n_rows:
        raise ValueError("b has incompatible length.")

    aug = [[float(a[i][j]) for j in range(n_cols)] + [float(b[i])] for i in range(n_rows)]

    for col in range(n_cols):
        pivot_row = max(range(col, n_rows), key=lambda r: abs(aug[r][col]))
        pivot = aug[pivot_row][col]
        if abs(pivot) < tolerance:
            raise ValueError("Matrix is singular or nearly singular.")
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
        raise ValueError("Matrix must be square.")
    inv_columns: Matrix = []
    eye = identity(n_rows)
    for j in range(n_cols):
        column = solve_linear_system(a, [eye[i][j] for i in range(n_rows)])
        inv_columns.append(column)
    return transpose(inv_columns)


def _vector_norm(x: Sequence[float]) -> float:
    return math.sqrt(dot(x, x))


def qr_decomposition(a: Matrix, tolerance: float = 1e-12) -> Tuple[Matrix, Matrix]:
    """Modified Gram-Schmidt QR decomposition for a tall or square matrix."""
    n, p = _validate_matrix(a)
    if n < p:
        raise ValueError("QR least squares requires rows >= columns.")

    columns = transpose([[float(v) for v in row] for row in a])
    q_columns: Matrix = []
    r = [[0.0 for _ in range(p)] for _ in range(p)]

    for j in range(p):
        v = columns[j][:]
        for i in range(j):
            r[i][j] = dot(q_columns[i], v)
            v = [v_k - r[i][j] * q_columns[i][k] for k, v_k in enumerate(v)]
        r[j][j] = _vector_norm(v)
        if r[j][j] < tolerance:
            raise ValueError("Design matrix is rank-deficient or nearly rank-deficient.")
        q_columns.append([value / r[j][j] for value in v])

    q = transpose(q_columns)
    return q, r


def back_substitution(r: Matrix, b: Sequence[float], tolerance: float = 1e-12) -> Vector:
    n_rows, n_cols = _validate_matrix(r)
    if n_rows != n_cols or len(b) != n_rows:
        raise ValueError("Incompatible dimensions for back substitution.")
    x = [0.0] * n_rows
    for i in range(n_rows - 1, -1, -1):
        if abs(r[i][i]) < tolerance:
            raise ValueError("Matrix is singular or nearly singular.")
        rhs = float(b[i]) - sum(r[i][j] * x[j] for j in range(i + 1, n_rows))
        x[i] = rhs / r[i][i]
    return x


def least_squares_qr(x: Matrix, y: Sequence[float]) -> Vector:
    n, _ = _validate_matrix(x)
    if len(y) != n:
        raise ValueError("X and y have incompatible dimensions.")
    q, r = qr_decomposition(x)
    qt_y = matvec(transpose(q), y)
    return back_substitution(r, qt_y)


def least_squares_normal_equation(x: Matrix, y: Sequence[float]) -> Vector:
    n, _ = _validate_matrix(x)
    if len(y) != n:
        raise ValueError("X and y have incompatible dimensions.")
    xt = transpose(x)
    xtx = matmul(xt, x)
    xty = matvec(xt, y)
    return solve_linear_system(xtx, xty)


# ---------------------------------------------------------------------------
# Probability functions needed for exact t-based inference
# ---------------------------------------------------------------------------

def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iterations = 200
    epsilon = 3.0e-14
    tiny = 1.0e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d

    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < epsilon:
            return h

    raise RuntimeError("Beta continued fraction did not converge.")


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if not (0.0 <= x <= 1.0):
        raise ValueError("x must be between 0 and 1.")
    if a <= 0.0 or b <= 0.0:
        raise ValueError("a and b must be positive.")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    bt = math.exp(log_bt)

    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _beta_continued_fraction(a, b, x) / a
    return 1.0 - bt * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_cdf(t: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive.")
    if t == 0.0:
        return 0.5
    v = float(degrees_of_freedom)
    x = v / (v + t * t)
    ib = regularized_incomplete_beta(x, v / 2.0, 0.5)
    return 1.0 - 0.5 * ib if t > 0 else 0.5 * ib


def student_t_ppf(probability: float, degrees_of_freedom: int) -> float:
    """Inverse Student t CDF found by symmetry and bisection."""
    if not (0.0 < probability < 1.0):
        raise ValueError("probability must lie strictly between 0 and 1.")
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive.")
    if probability == 0.5:
        return 0.0
    if probability < 0.5:
        return -student_t_ppf(1.0 - probability, degrees_of_freedom)

    low, high = 0.0, 1.0
    while student_t_cdf(high, degrees_of_freedom) < probability:
        high *= 2.0
        if high > 1e6:
            raise RuntimeError("Could not bracket t quantile.")

    for _ in range(120):
        mid = (low + high) / 2.0
        if student_t_cdf(mid, degrees_of_freedom) < probability:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


# ---------------------------------------------------------------------------
# OLS models and inference
# ---------------------------------------------------------------------------
@dataclass
class SimpleOLSResult:
    intercept: float
    slope: float
    fitted: Vector
    residuals: Vector
    sse: float
    sst: float
    r_squared: float

    def predict(self, x_new: Iterable[float]) -> Vector:
        return [self.intercept + self.slope * float(x) for x in x_new]


def simple_ols(x: Iterable[float], y: Iterable[float]) -> SimpleOLSResult:
    xs = _as_float_vector(x)
    ys = _as_float_vector(y)
    if len(xs) != len(ys):
        raise ValueError("x and y must have the same length.")
    if len(xs) < 2:
        raise ValueError("At least two observations are required.")
    xbar = mean(xs)
    ybar = mean(ys)
    sxx = sum((value - xbar) ** 2 for value in xs)
    if sxx == 0.0:
        raise ValueError("x must vary.")
    sxy = sum((a - xbar) * (b - ybar) for a, b in zip(xs, ys))
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    fitted = [intercept + slope * value for value in xs]
    residuals = [actual - prediction for actual, prediction in zip(ys, fitted)]
    sse = sum(r * r for r in residuals)
    sst = sum((value - ybar) ** 2 for value in ys)
    r_squared = 1.0 - sse / sst if sst > 0.0 else 1.0
    return SimpleOLSResult(intercept, slope, fitted, residuals, sse, sst, r_squared)


@dataclass
class OLSResult:
    coefficients: Vector
    fitted: Vector
    residuals: Vector
    n_observations: int
    n_parameters: int
    degrees_of_freedom: int
    sse: float
    sst: float
    mse: float
    rmse: float
    r_squared: float
    adjusted_r_squared: float
    covariance_matrix: Matrix
    standard_errors: Vector
    t_statistics: Vector
    p_values: Vector
    confidence_intervals: List[Tuple[float, float]]
    design_matrix: Matrix
    xtx_inverse: Matrix
    includes_intercept: bool

    def predict(self, x_new: Matrix, add_intercept: Optional[bool] = None) -> Vector:
        use_intercept = self.includes_intercept if add_intercept is None else add_intercept
        design = add_intercept_column(x_new) if use_intercept else x_new
        return matvec(design, self.coefficients)

    def prediction_intervals(
        self,
        x_new: Matrix,
        confidence: float = 0.95,
        include_observation_noise: bool = True,
    ) -> List[Tuple[float, float, float]]:
        if not (0.0 < confidence < 1.0):
            raise ValueError("confidence must lie between 0 and 1.")
        design = add_intercept_column(x_new) if self.includes_intercept else x_new
        predictions = matvec(design, self.coefficients)
        critical = student_t_ppf(0.5 + confidence / 2.0, self.degrees_of_freedom)
        intervals = []
        for row, prediction in zip(design, predictions):
            leverage_new = dot(row, matvec(self.xtx_inverse, row))
            variance_multiplier = leverage_new + (1.0 if include_observation_noise else 0.0)
            standard_error = math.sqrt(self.mse * variance_multiplier)
            margin = critical * standard_error
            intervals.append((prediction, prediction - margin, prediction + margin))
        return intervals


def fit_ols(
    x: Matrix,
    y: Sequence[float],
    add_intercept: bool = True,
    method: str = "qr",
    confidence: float = 0.95,
) -> OLSResult:
    n, p_raw = _validate_matrix(x)
    ys = [float(v) for v in y]
    if len(ys) != n:
        raise ValueError("X and y have incompatible dimensions.")
    design = add_intercept_column(x) if add_intercept else [[float(v) for v in row] for row in x]
    _, p = _validate_matrix(design)
    if n <= p:
        raise ValueError("OLS inference requires more observations than parameters.")

    if method == "qr":
        coefficients = least_squares_qr(design, ys)
    elif method == "normal":
        coefficients = least_squares_normal_equation(design, ys)
    else:
        raise ValueError("method must be 'qr' or 'normal'.")

    fitted = matvec(design, coefficients)
    residuals = [actual - prediction for actual, prediction in zip(ys, fitted)]
    ybar = mean(ys)
    sse = sum(r * r for r in residuals)
    sst = sum((value - ybar) ** 2 for value in ys)
    df = n - p
    mse = sse / df
    rmse = math.sqrt(mse)
    r_squared = 1.0 - sse / sst if sst > 0.0 else 1.0
    adjusted = 1.0 - (1.0 - r_squared) * (n - 1) / df

    xt = transpose(design)
    xtx = matmul(xt, design)
    xtx_inv = inverse(xtx)
    covariance_matrix = [[mse * value for value in row] for row in xtx_inv]
    standard_errors = [math.sqrt(max(covariance_matrix[j][j], 0.0)) for j in range(p)]
    t_statistics = [
        coefficients[j] / standard_errors[j] if standard_errors[j] > 0.0 else math.inf
        for j in range(p)
    ]
    p_values = [2.0 * (1.0 - student_t_cdf(abs(t), df)) for t in t_statistics]

    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must lie between 0 and 1.")
    critical = student_t_ppf(0.5 + confidence / 2.0, df)
    confidence_intervals = [
        (coefficients[j] - critical * standard_errors[j], coefficients[j] + critical * standard_errors[j])
        for j in range(p)
    ]

    return OLSResult(
        coefficients=coefficients,
        fitted=fitted,
        residuals=residuals,
        n_observations=n,
        n_parameters=p,
        degrees_of_freedom=df,
        sse=sse,
        sst=sst,
        mse=mse,
        rmse=rmse,
        r_squared=r_squared,
        adjusted_r_squared=adjusted,
        covariance_matrix=covariance_matrix,
        standard_errors=standard_errors,
        t_statistics=t_statistics,
        p_values=p_values,
        confidence_intervals=confidence_intervals,
        design_matrix=design,
        xtx_inverse=xtx_inv,
        includes_intercept=add_intercept,
    )


def ols_diagnostics(model: OLSResult) -> List[Dict[str, float]]:
    p = model.n_parameters
    diagnostics = []
    for i, (row, residual) in enumerate(zip(model.design_matrix, model.residuals)):
        leverage = dot(row, matvec(model.xtx_inverse, row))
        one_minus_h = max(1.0 - leverage, 1e-15)
        standardized = residual / math.sqrt(model.mse * one_minus_h)
        cooks_distance = (residual * residual / (p * model.mse)) * leverage / (one_minus_h ** 2)
        diagnostics.append(
            {
                "index": float(i),
                "residual": residual,
                "leverage": leverage,
                "standardized_residual": standardized,
                "cooks_distance": cooks_distance,
            }
        )
    return diagnostics


# ---------------------------------------------------------------------------
# Metrics and validation
# ---------------------------------------------------------------------------

def mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("Inputs must be non-empty and have equal length.")
    return sum(abs(float(a) - float(b)) for a, b in zip(y_true, y_pred)) / len(y_true)


def mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("Inputs must be non-empty and have equal length.")
    return sum((float(a) - float(b)) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)


def root_mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def r_squared_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("Inputs must be non-empty and have equal length.")
    ybar = mean(y_true)
    sse = sum((float(a) - float(b)) ** 2 for a, b in zip(y_true, y_pred))
    sst = sum((float(a) - ybar) ** 2 for a in y_true)
    if sst == 0.0:
        return 1.0 if sse == 0.0 else 0.0
    return 1.0 - sse / sst


def train_test_split(
    x: Matrix,
    y: Sequence[float],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[Matrix, Matrix, Vector, Vector]:
    n, _ = _validate_matrix(x)
    if len(y) != n:
        raise ValueError("X and y have incompatible dimensions.")
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("test_fraction must lie between 0 and 1.")
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_test = max(1, int(round(n * test_fraction)))
    test_set = set(indices[:n_test])
    x_train, x_test, y_train, y_test = [], [], [], []
    for i in range(n):
        if i in test_set:
            x_test.append([float(v) for v in x[i]])
            y_test.append(float(y[i]))
        else:
            x_train.append([float(v) for v in x[i]])
            y_train.append(float(y[i]))
    return x_train, x_test, y_train, y_test


def k_fold_indices(n: int, k: int = 5, seed: int = 42) -> List[List[int]]:
    if n <= 1:
        raise ValueError("n must exceed 1.")
    if not (2 <= k <= n):
        raise ValueError("k must be between 2 and n.")
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    folds = [[] for _ in range(k)]
    for position, index in enumerate(indices):
        folds[position % k].append(index)
    return folds


def cross_validate_ols(
    x: Matrix,
    y: Sequence[float],
    k: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    n, _ = _validate_matrix(x)
    if len(y) != n:
        raise ValueError("X and y have incompatible dimensions.")
    folds = k_fold_indices(n, k=k, seed=seed)
    fold_scores = []
    fold_rmse = []
    for validation_indices in folds:
        validation_set = set(validation_indices)
        x_train = [x[i] for i in range(n) if i not in validation_set]
        y_train = [float(y[i]) for i in range(n) if i not in validation_set]
        x_valid = [x[i] for i in validation_indices]
        y_valid = [float(y[i]) for i in validation_indices]
        model = fit_ols(x_train, y_train)
        predictions = model.predict(x_valid)
        fold_scores.append(r_squared_score(y_valid, predictions))
        fold_rmse.append(root_mean_squared_error(y_valid, predictions))
    return {
        "r_squared_by_fold": fold_scores,
        "mean_r_squared": sum(fold_scores) / len(fold_scores),
        "rmse_by_fold": fold_rmse,
        "mean_rmse": sum(fold_rmse) / len(fold_rmse),
    }


# ---------------------------------------------------------------------------
# Scaling and optimization-based linear regression
# ---------------------------------------------------------------------------
@dataclass
class Standardizer:
    means: Vector
    scales: Vector

    def transform(self, x: Matrix) -> Matrix:
        _, p = _validate_matrix(x)
        if p != len(self.means):
            raise ValueError("X has the wrong number of columns.")
        return [
            [
                (float(row[j]) - self.means[j]) / self.scales[j]
                for j in range(p)
            ]
            for row in x
        ]


def fit_standardizer(x: Matrix) -> Standardizer:
    n, p = _validate_matrix(x)
    means = [sum(float(x[i][j]) for i in range(n)) / n for j in range(p)]
    scales = []
    for j in range(p):
        variance_j = sum((float(x[i][j]) - means[j]) ** 2 for i in range(n)) / n
        scale = math.sqrt(variance_j)
        scales.append(scale if scale > 0.0 else 1.0)
    return Standardizer(means, scales)


@dataclass
class GradientDescentLinearResult:
    coefficients: Vector
    loss_history: Vector
    standardizer: Optional[Standardizer]

    def predict(self, x_new: Matrix) -> Vector:
        features = self.standardizer.transform(x_new) if self.standardizer else x_new
        design = add_intercept_column(features)
        return matvec(design, self.coefficients)


def fit_linear_gradient_descent(
    x: Matrix,
    y: Sequence[float],
    learning_rate: float = 0.05,
    epochs: int = 5000,
    standardize: bool = True,
    tolerance: float = 1e-10,
) -> GradientDescentLinearResult:
    n, p = _validate_matrix(x)
    if len(y) != n:
        raise ValueError("X and y have incompatible dimensions.")
    ys = [float(v) for v in y]
    scaler = fit_standardizer(x) if standardize else None
    features = scaler.transform(x) if scaler else [[float(v) for v in row] for row in x]
    design = add_intercept_column(features)
    coefficients = [0.0] * (p + 1)
    history = []
    previous_loss = math.inf

    for _ in range(epochs):
        predictions = matvec(design, coefficients)
        errors = [predictions[i] - ys[i] for i in range(n)]
        loss = sum(error * error for error in errors) / n
        history.append(loss)
        gradients = [
            (2.0 / n) * sum(errors[i] * design[i][j] for i in range(n))
            for j in range(p + 1)
        ]
        coefficients = [
            coefficients[j] - learning_rate * gradients[j]
            for j in range(p + 1)
        ]
        if abs(previous_loss - loss) < tolerance:
            break
        previous_loss = loss

    return GradientDescentLinearResult(coefficients, history, scaler)


# ---------------------------------------------------------------------------
# Regularization
# ---------------------------------------------------------------------------
@dataclass
class RegularizedLinearResult:
    coefficients: Vector
    standardizer: Optional[Standardizer]

    def predict(self, x_new: Matrix) -> Vector:
        features = self.standardizer.transform(x_new) if self.standardizer else x_new
        return matvec(add_intercept_column(features), self.coefficients)


def fit_ridge(
    x: Matrix,
    y: Sequence[float],
    alpha: float = 1.0,
    standardize: bool = True,
) -> RegularizedLinearResult:
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative.")
    n, p = _validate_matrix(x)
    if len(y) != n:
        raise ValueError("X and y have incompatible dimensions.")
    scaler = fit_standardizer(x) if standardize else None
    features = scaler.transform(x) if scaler else [[float(v) for v in row] for row in x]
    design = add_intercept_column(features)
    xt = transpose(design)
    xtx = matmul(xt, design)
    for j in range(1, p + 1):  # do not penalize intercept
        xtx[j][j] += alpha
    xty = matvec(xt, y)
    coefficients = solve_linear_system(xtx, xty)
    return RegularizedLinearResult(coefficients, scaler)


def _soft_threshold(value: float, threshold: float) -> float:
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


def fit_lasso(
    x: Matrix,
    y: Sequence[float],
    alpha: float = 0.1,
    max_iterations: int = 10000,
    tolerance: float = 1e-8,
) -> RegularizedLinearResult:
    """Lasso by coordinate descent on standardized predictors.

    Objective: (1 / (2n)) * SSE + alpha * sum(|beta_j|), excluding intercept.
    """
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative.")
    n, p = _validate_matrix(x)
    if len(y) != n:
        raise ValueError("X and y have incompatible dimensions.")
    scaler = fit_standardizer(x)
    z = scaler.transform(x)
    y_values = [float(v) for v in y]
    y_mean = sum(y_values) / n
    y_centered = [value - y_mean for value in y_values]
    beta = [0.0] * p
    predictions = [0.0] * n
    column_sq_means = [sum(z[i][j] ** 2 for i in range(n)) / n for j in range(p)]

    for _ in range(max_iterations):
        max_change = 0.0
        for j in range(p):
            old = beta[j]
            partial_residual = [
                y_centered[i] - predictions[i] + z[i][j] * old
                for i in range(n)
            ]
            rho = sum(z[i][j] * partial_residual[i] for i in range(n)) / n
            beta[j] = _soft_threshold(rho, alpha) / column_sq_means[j]
            change = beta[j] - old
            if change != 0.0:
                for i in range(n):
                    predictions[i] += z[i][j] * change
            max_change = max(max_change, abs(change))
        if max_change < tolerance:
            break

    return RegularizedLinearResult([y_mean] + beta, scaler)


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------

def sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


@dataclass
class LogisticRegressionResult:
    coefficients: Vector
    loss_history: Vector
    standardizer: Optional[Standardizer]

    def predict_proba(self, x_new: Matrix) -> Vector:
        features = self.standardizer.transform(x_new) if self.standardizer else x_new
        scores = matvec(add_intercept_column(features), self.coefficients)
        return [sigmoid(score) for score in scores]

    def predict(self, x_new: Matrix, threshold: float = 0.5) -> List[int]:
        if not (0.0 < threshold < 1.0):
            raise ValueError("threshold must lie between 0 and 1.")
        return [1 if probability >= threshold else 0 for probability in self.predict_proba(x_new)]


def binary_log_loss(y_true: Sequence[int], probabilities: Sequence[float]) -> float:
    if len(y_true) != len(probabilities) or not y_true:
        raise ValueError("Inputs must be non-empty and have equal length.")
    epsilon = 1e-15
    total = 0.0
    for y, probability in zip(y_true, probabilities):
        p = min(max(float(probability), epsilon), 1.0 - epsilon)
        total -= int(y) * math.log(p) + (1 - int(y)) * math.log(1.0 - p)
    return total / len(y_true)


def fit_logistic_gradient_descent(
    x: Matrix,
    y: Sequence[int],
    learning_rate: float = 0.1,
    epochs: int = 10000,
    l2: float = 0.0,
    standardize: bool = True,
    tolerance: float = 1e-10,
) -> LogisticRegressionResult:
    n, p = _validate_matrix(x)
    if len(y) != n:
        raise ValueError("X and y have incompatible dimensions.")
    labels = [int(v) for v in y]
    if any(v not in (0, 1) for v in labels):
        raise ValueError("Logistic regression labels must be 0 or 1.")
    if l2 < 0.0:
        raise ValueError("l2 must be non-negative.")

    scaler = fit_standardizer(x) if standardize else None
    features = scaler.transform(x) if scaler else [[float(v) for v in row] for row in x]
    design = add_intercept_column(features)
    coefficients = [0.0] * (p + 1)
    history = []
    previous_loss = math.inf

    for _ in range(epochs):
        scores = matvec(design, coefficients)
        probabilities = [sigmoid(score) for score in scores]
        loss = binary_log_loss(labels, probabilities) + (
            l2 * sum(value * value for value in coefficients[1:]) / (2.0 * n)
        )
        history.append(loss)

        gradients = []
        for j in range(p + 1):
            gradient = sum(
                (probabilities[i] - labels[i]) * design[i][j]
                for i in range(n)
            ) / n
            if j > 0:
                gradient += l2 * coefficients[j] / n
            gradients.append(gradient)

        coefficients = [
            coefficients[j] - learning_rate * gradients[j]
            for j in range(p + 1)
        ]
        if abs(previous_loss - loss) < tolerance:
            break
        previous_loss = loss

    return LogisticRegressionResult(coefficients, history, scaler)


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------
def demo() -> None:
    x = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
    y = [3.1, 4.9, 7.2, 9.1, 11.2, 12.8]
    model = fit_ols(x, y)
    print("OLS coefficients:", [round(v, 4) for v in model.coefficients])
    print("R-squared:", round(model.r_squared, 4))
    print("95% coefficient intervals:", [tuple(round(v, 4) for v in pair) for pair in model.confidence_intervals])
    print("Prediction interval at x=7:", tuple(round(v, 4) for v in model.prediction_intervals([[7.0]])[0]))


if __name__ == "__main__":
    demo()
```


# Appendix B. Full Library-Based Demonstration

The following script generates synthetic data and demonstrates NumPy, pandas, statsmodels, and scikit-learn workflows. Save it as regression_with_libraries.py.

```python
"""Practical regression workflows with NumPy, pandas, statsmodels, and scikit-learn.

The examples are self-contained and generate their own synthetic data.
Run with:
    python regression_with_libraries.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNetCV,
    LassoCV,
    LinearRegression,
    LogisticRegression,
    RidgeCV,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


def make_regression_data(seed: int = 42, n: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    experience = rng.uniform(0, 25, n)
    education = rng.integers(10, 21, n)
    region = rng.choice(["North", "Central", "South"], size=n, p=[0.3, 0.4, 0.3])
    training = rng.binomial(1, 0.45, n)
    noise_scale = 2.0 + 0.12 * experience  # deliberate heteroskedasticity
    noise = rng.normal(0, noise_scale, n)
    region_effect = pd.Series(region).map({"North": 4.0, "Central": 0.0, "South": -2.5}).to_numpy()
    income = (
        18.0
        + 1.7 * experience
        + 1.25 * education
        + 3.2 * training
        + 0.12 * experience * training
        + region_effect
        + noise
    )
    return pd.DataFrame(
        {
            "income": income,
            "experience": experience,
            "education": education,
            "training": training,
            "region": region,
        }
    )


def numpy_ols(df: pd.DataFrame) -> np.ndarray:
    x = df[["experience", "education", "training"]].to_numpy(dtype=float)
    y = df["income"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    # Prefer lstsq to explicitly forming inv(X.T @ X).
    beta_hat, residual_sum_squares, rank, singular_values = np.linalg.lstsq(design, y, rcond=None)
    print("\nNumPy OLS coefficients:", beta_hat)
    print("Rank:", rank)
    print("Singular values:", singular_values)
    return beta_hat


def statsmodels_inference(df: pd.DataFrame):
    model = smf.ols(
        "income ~ experience + education + training + experience:training + C(region)",
        data=df,
    ).fit()
    robust = model.get_robustcov_results(cov_type="HC3")

    print("\nStatsmodels classical OLS summary:")
    print(model.summary())
    print("\nHC3 robust standard errors:")
    print(robust.summary())

    bp_stat, bp_pvalue, f_stat, f_pvalue = het_breuschpagan(model.resid, model.model.exog)
    print("\nBreusch-Pagan p-value:", bp_pvalue)

    influence = model.get_influence()
    influence_frame = influence.summary_frame()
    print("\nLargest Cook's distances:")
    print(influence_frame.nlargest(5, "cooks_d")[["cooks_d", "hat_diag", "student_resid"]])

    vif_frame = pd.DataFrame(
        {
            "term": model.model.exog_names,
            "VIF": [
                variance_inflation_factor(model.model.exog, i)
                for i in range(model.model.exog.shape[1])
            ],
        }
    )
    print("\nVariance inflation factors:")
    print(vif_frame)
    return model, robust


def sklearn_mixed_data_pipeline(df: pd.DataFrame):
    x = df.drop(columns="income")
    y = df["income"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42
    )

    numeric_features = ["experience", "education", "training"]
    categorical_features = ["region"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessing),
            ("model", LinearRegression()),
        ]
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    print("\nScikit-learn test metrics:")
    print("MAE:", mean_absolute_error(y_test, predictions))
    print("RMSE:", mean_squared_error(y_test, predictions) ** 0.5)
    print("R-squared:", r2_score(y_test, predictions))

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        pipeline,
        x,
        y,
        cv=cv,
        scoring={
            "r2": "r2",
            "neg_rmse": "neg_root_mean_squared_error",
            "neg_mae": "neg_mean_absolute_error",
        },
        return_train_score=True,
    )
    print("\nCross-validation mean test R-squared:", scores["test_r2"].mean())
    print("Cross-validation mean test RMSE:", -scores["test_neg_rmse"].mean())
    return pipeline


def polynomial_and_regularized_models(df: pd.DataFrame):
    x = df[["experience", "education", "training"]]
    y = df["income"]

    polynomial = Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scale", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-3, 3, 60), cv=5)),
        ]
    )
    polynomial.fit(x, y)
    print("\nSelected polynomial ridge alpha:", polynomial.named_steps["model"].alpha_)

    ridge = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-4, 4, 100), cv=5)),
        ]
    )
    lasso = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("model", LassoCV(alphas=np.logspace(-4, 1, 100), cv=5, max_iter=20000)),
        ]
    )
    elastic_net = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                ElasticNetCV(
                    l1_ratio=[0.1, 0.5, 0.9, 1.0],
                    alphas=np.logspace(-4, 1, 100),
                    cv=5,
                    max_iter=20000,
                ),
            ),
        ]
    )
    for name, estimator in [("Ridge", ridge), ("Lasso", lasso), ("Elastic Net", elastic_net)]:
        estimator.fit(x, y)
        print(f"{name} selected parameters:", estimator.named_steps["model"].get_params())
    return polynomial, ridge, lasso, elastic_net


def make_classification_data(seed: int = 42, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.uniform(18, 75, n)
    income = rng.lognormal(mean=10.4, sigma=0.45, size=n)
    prior_contact = rng.binomial(1, 0.35, n)
    score = -7.0 + 0.045 * age + 0.000035 * income + 1.2 * prior_contact
    probability = 1.0 / (1.0 + np.exp(-score))
    purchased = rng.binomial(1, probability)
    return pd.DataFrame(
        {
            "age": age,
            "income": income,
            "prior_contact": prior_contact,
            "purchased": purchased,
        }
    )


def logistic_regression_workflow(df: pd.DataFrame):
    x = df.drop(columns="purchased")
    y = df["purchased"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )

    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=2000,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print("\nLogistic regression metrics:")
    print("ROC AUC:", roc_auc_score(y_test, probabilities))
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions, zero_division=0))
    return model


def main() -> None:
    regression_df = make_regression_data()
    print(regression_df.head())
    numpy_ols(regression_df)
    statsmodels_inference(regression_df)
    sklearn_mixed_data_pipeline(regression_df)
    polynomial_and_regularized_models(regression_df)

    classification_df = make_classification_data()
    logistic_regression_workflow(classification_df)


if __name__ == "__main__":
    main()
```


# Appendix C. Minimal verification tests

```python
from regression_from_scratch import (
    simple_ols,
    fit_ols,
    student_t_cdf,
    student_t_ppf,
    fit_ridge,
    fit_lasso,
    fit_logistic_gradient_descent,
)


def close(a, b, tolerance=1e-7):
    return abs(a - b) <= tolerance


# Exact straight line y = 1 + 2x
x = [1, 2, 3, 4, 5]
y = [3, 5, 7, 9, 11]
simple = simple_ols(x, y)
assert close(simple.intercept, 1.0)
assert close(simple.slope, 2.0)
assert close(simple.r_squared, 1.0)

# Multiple OLS reproduces exact linear relationship.
X = [[1, 0], [2, 1], [3, 0], [4, 1], [5, 0], [6, 1]]
y = [3 + 2*a - 4*b for a, b in X]
multiple = fit_ols(X, y)
assert all(close(a, b) for a, b in zip(multiple.coefficients, [3, 2, -4]))

# t CDF and inverse agree.
for df in [1, 2, 5, 30, 100]:
    for probability in [0.025, 0.5, 0.975]:
        quantile = student_t_ppf(probability, df)
        assert close(student_t_cdf(quantile, df), probability, tolerance=1e-8)

# Ridge and lasso produce predictions.
ridge = fit_ridge(X, y, alpha=1.0)
lasso = fit_lasso(X, y, alpha=0.01)
assert len(ridge.predict([[7, 0]])) == 1
assert len(lasso.predict([[7, 0]])) == 1

# Logistic probabilities are valid.
logit = fit_logistic_gradient_descent(
    [[0], [1], [2], [3], [4], [5]],
    [0, 0, 0, 1, 1, 1],
    epochs=5000,
)
p = logit.predict_proba([[2.5]])[0]
assert 0.0 <= p <= 1.0

print("All verification tests passed.")
```


# Appendix D. Research-extension verification exercises

A fifth tier, Research (R), sits above Library exercises and draws on Part V. Attempt these only after the C/D/P/L stack for a topic is solid; each one asks you to verify a research-grade claim computationally rather than take it on faith.

- **R1.** Simulate heteroskedastic data and compare classical, HC0, HC1, HC2, and HC3 standard errors over repeated samples.
- **R2.** Construct a dataset with ten clusters and show how row-level bootstrap intervals differ from cluster-bootstrap intervals.
- **R3.** Simulate measurement error in X and trace attenuation as reliability changes.
- **R4.** Compare pairs and residual bootstraps under homoskedastic and heteroskedastic errors.
- **R5.** Create a missing-at-random mechanism, compare complete-case analysis with multiple imputation using a library, and explain why single mean imputation fails.
- **R6.** Reproduce the main numerical claim from one paper in Section 55.
- **R7.** Write a specification curve for one substantive coefficient across defensible control sets and transformations.
- **R8.** Use singular values to construct increasingly ill-conditioned design matrices and compare normal equations, QR, and SVD.
- **R9.** Develop a simulation in which predictive performance remains strong while an individual coefficient becomes unstable under collinearity.
- **R10.** Write a research report using the structure in Section 51 and include a computational reproducibility appendix.


# Final perspective

Regression is most powerful when its four layers remain connected:

- **Substantive layer:** What question and decision matter?
- **Mathematical layer:** What relationship is being assumed and optimized?
- **Statistical layer:** What uncertainty and identification assumptions support the conclusion?
- **Computational layer:** How are data transformations, fitting, diagnostics, and validation implemented without leakage or numerical error?

A strong analyst connects the question, specification, assumptions, mathematics, code, validation, interpretation, and limitations.
