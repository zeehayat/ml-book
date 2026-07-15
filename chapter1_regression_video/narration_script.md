# Chapter 1 Video Tutorial — Narration Script

## 01. Linear & Logistic Regression

Welcome to Chapter One: Linear Regression and Logistic Models. In this video tutorial, we will walk through the mathematical foundations, statistical properties, diagnostic mitigations, and implementation ecosystems of classical regression analysis. We will start from simple ordinary least squares and build up to generalized linear boundaries and regularized optimization. Let us begin.

## 02. What you will be able to do

By the end of this chapter, you will be able to derive the ordinary least squares slope and intercept by hand and express multiple regression using projection geometry. You will also be able to run model diagnostic tests, apply Ridge and Lasso regularizations, and train a binary logistic regression classifier from scratch using gradient descent.

## 03. Ordinary Least Squares (OLS) Math

Let us start with Ordinary Least Squares. For a single predictor, the slope is the covariance of x and y divided by the variance of x, and the intercept shifts the line to pass through the sample means. In multiple regression with several predictors, we write the system in matrix form as y equals X beta plus epsilon. Minimizing the sum of squared residuals yields the famous closed-form solution: beta hat equals X transpose X inverse times X transpose y.

## 04. OLS Geometric Projection

Geometrically, ordinary least squares is an orthogonal projection. The outcome vector y lives in an n-dimensional space, while the columns of the design matrix X span a lower-dimensional subspace. The hat matrix H projects y orthogonally onto this subspace to yield the fitted values y hat. The residual vector e represents the projection error, which is mathematically orthogonal to the column space of X, meaning X transpose times e equals zero.

## 05. The Gauss-Markov Theorem

The Gauss-Markov Theorem defines the optimal properties of the OLS estimator. If the relationship is linear, predictors are strictly exogenous, the design matrix has full column rank, and the errors are spherical, meaning homoskedastic and uncorrelated, then OLS is the Best Linear Unbiased Estimator, or BLUE. This means no other linear unbiased estimator can achieve a lower variance.

## 06. Uncertainty & Inference

To conduct statistical inference, we must quantify our uncertainty. We estimate the residual variance s squared by dividing the sum of squared residuals by the degrees of freedom. This variance scale determines the covariance matrix of our coefficients. We can then test individual coefficient significance using t-statistics and evaluate joint linear restrictions using F-statistics.

## 07. Model Diagnostics & Mitigations

Before trusting our inference, we must run diagnostics to check for violations of OLS assumptions. If the errors are heteroskedastic, standard errors are biased and must be corrected using White robust standard errors. Multicollinearity is detected when the Variance Inflation Factor exceeds ten. Outliers with high leverage are identified using Cook's distance.

## 08. Regularization: Ridge vs. Lasso

When multicollinearity is present or parameters exceed observations, OLS variance explodes. Regularization introduces a bias to reduce variance. Ridge regression adds an L2 penalty, shrinking coefficients close to zero. Lasso regression adds an L1 penalty, which drives redundant coefficients to exactly zero, performing automatic feature selection.

## 09. Logistic Regression Mechanics

For binary outcomes, linear regression fails because it can predict values outside zero and one. Logistic regression models the probability of success using the Sigmoid activation function. This maps the linear predictor to a probability between zero and one. The link function is the log-odds, also called the logit link, which remains linear.

## 10. Log-Likelihood & Optimization

We estimate logistic coefficients by maximizing the Bernoulli likelihood, which is equivalent to minimizing the Binary Cross-Entropy loss. Since no analytical closed-form solution exists, we optimize the convex loss function iteratively using Gradient Descent. The gradient simplifies beautifully to X transpose times the probability difference vector, divided by sample size.

## 11. Implementation Ecosystem

Our curriculum maintains two tracks. The scratch track builds linear solvers and gradient descent from first principles in pure Python to expose the raw mechanics. The library track uses statsmodels for detailed statistical diagnostics and scikit-learn for high-performance predictive pipelines. We always verify our scratch solvers against these mature libraries.

## 12. Chapter 1 Summary

Let us summarize Chapter One. Ordinary Least Squares projects y orthogonally onto the design space. Statistical inference is valid only when exogeneity and spherical error properties hold. Diagnostics are necessary to check for heteroskedasticity and collinearity. Regularizations introduce minor bias to stabilize variance. And logistic regression models binary probabilities, requiring gradient descent optimization.
