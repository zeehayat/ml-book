# Generalized Linear Models: A Mathematical and Python Guide

*Extending Regression to Counts, Proportions, Rates, and Beyond*

Prepared as a self-study and AI-tutor source. Second guide in the Zero-to-Research-Level Machine Learning sequence, following *Regression: A Mathematical and Python Guide*.

---

## How to use this guide

**Purpose.** This is a self-contained source for learning generalized linear models (GLMs) through mathematics, intuition, pure Python, and modern Python libraries. It is designed to be read directly or uploaded as a source to an AI study tool such as NotebookLM. It assumes you have completed, or are working alongside, the companion *Regression* guide: ordinary least squares, the normal equations, gradient descent, residual diagnostics, and logistic regression are treated as known and are extended here rather than re-derived from zero.

**Meaning of "pure Python."** In Part II, "pure Python" means no third-party numerical, statistical, or machine-learning packages. Only the Python standard library is used. The point is not to compete with statsmodels or scikit-learn. The point is to expose every calculation — including the iterative reweighting step that makes GLM fitting different from ordinary least squares — so that nothing is a black box.

**Why this topic, right after regression.** Ordinary least squares assumes a continuous outcome whose conditional mean is linear in the predictors and whose errors are (approximately) normal with constant variance. Enormous classes of real outcomes violate this: counts (number of customer complaints per week), proportions (fraction of a batch that fails inspection), binary outcomes (already covered as logistic regression in the Regression guide, but here generalized), waiting times and other strictly positive continuous outcomes, and rates. Generalized linear models are the smallest, most direct extension of everything you already know about OLS: they keep a *linear predictor* (the familiar `Xβ`) but connect it to the outcome through a *link function*, and they replace the constant-variance normal error assumption with a member of the *exponential family* of distributions whose variance is allowed to depend on its mean. Linear regression and logistic regression turn out to be the two GLMs you already know; Poisson, gamma, and negative binomial regression are new members of the same family.

**Scope.** The guide covers:

- the exponential family of distributions and why it unifies GLMs;
- link functions, canonical links, and how to choose one;
- maximum likelihood estimation for GLMs and the score equations;
- Iteratively Reweighted Least Squares (IRLS), the algorithm that fits every GLM in this guide;
- deviance, the GLM generalization of the residual sum of squares;
- Poisson regression for counts, including offsets and exposure;
- overdispersion, quasi-Poisson, and negative binomial regression;
- binomial regression for grouped proportion data;
- gamma regression for strictly positive continuous outcomes;
- diagnostics: Pearson and deviance residuals, leverage, and influence for GLMs;
- model comparison via AIC, BIC, and likelihood ratio tests;
- regularized GLMs (ridge, lasso, elastic net penalties on the GLM likelihood);
- implementations from scratch and then with statsmodels and scikit-learn;
- exercises, teaching prompts, a glossary, suggested research papers, and how to read them.

**Contents at a glance.**

- Part I, Chapters 1–15: the exponential family, link functions, maximum likelihood, IRLS, deviance, the Poisson/binomial/gamma families, overdispersion, diagnostics, model selection, and regularized GLMs.
- Part II, Chapters 16–26: pure-Python implementation using only the standard library, including a single generic IRLS solver reused across all three families.
- Part III, Chapters 27–36: statsmodels, scikit-learn, diagnostic plots, workflows, and common mistakes.
- Part IV, Chapters 37–46: exercises, learning sequence, AI-tutor prompts, glossary, further study, research papers, and how to read them.
- Appendices A–C: complete executable code and verification tests.

---

# Part I. Mathematical Foundations

## 1. What GLMs generalize, and why

### 1.1 The limits of ordinary least squares

Ordinary least squares models a continuous outcome `y` as `E[y | x] = xᵀβ`, with errors assumed independent, homoskedastic, and (for exact inference) normal. Three things break this picture in common practice:

- **The outcome's natural range is restricted.** Counts are non-negative integers; proportions lie in `[0, 1]`; strictly positive durations and amounts cannot go below zero. A linear predictor `xᵀβ` ranges over all real numbers, so predicting the outcome directly with `xᵀβ` can produce impossible values (negative counts, probabilities above 1).
- **The variance is not constant.** For count data, variance typically grows with the mean (a process averaging 100 events per period fluctuates more in absolute terms than one averaging 2 events per period). Treating this variance as constant, as OLS does, produces standard errors that are wrong in a predictable, correctable direction.
- **The error distribution is not well approximated by the normal.** A count outcome with a mean near zero cannot have symmetric, unbounded normal errors; a binary outcome has only two possible error values for any fitted probability.

Generalized linear models solve all three problems with one shared idea: keep the linear predictor `η = xᵀβ`, but connect it to the outcome's mean through a **link function** `g`, so that `g(E[y | x]) = η`, and replace the normal-with-constant-variance assumption with a distribution from the **exponential family**, whose variance is an explicit function of its mean.

### 1.2 Three components of a GLM

Every GLM is fully specified by three pieces:

1. **Random component.** A distribution for `y` from the exponential family (Gaussian, Bernoulli/Binomial, Poisson, Gamma, and others), specifying how the outcome varies around its mean `μ`.
2. **Systematic component.** The linear predictor `η = Xβ`, built from the design matrix exactly as in the Regression guide — the same encoding of continuous predictors, categorical indicators, interactions, and transformations applies unchanged.
3. **Link function.** A monotonic, differentiable function `g` connecting the mean of the random component to the systematic component: `g(μ) = η`, equivalently `μ = g⁻¹(η)`.

Ordinary least squares is the GLM with a Gaussian random component and the **identity link**, `g(μ) = μ`. Logistic regression is the GLM with a Bernoulli random component and the **logit link**, `g(μ) = log[μ / (1 − μ)]`. Nothing about fitting either of these models requires new machinery once you see them this way — they are two points in the same family this guide develops in general.

### 1.3 When you need a GLM instead of OLS

Use a GLM, and pick the family in parentheses, when the outcome is: a count with no natural upper bound (Poisson or negative binomial); a proportion of successes out of a known number of trials (binomial); a binary indicator (binomial with one trial, i.e. logistic regression, already covered); a strictly positive, right-skewed continuous quantity such as an insurance claim amount or a service time (gamma, or sometimes inverse Gaussian). If the outcome is continuous, roughly symmetric, and plausibly normal after any necessary transformation, ordinary least squares remains the right default — a GLM is a generalization, not a universal replacement.

## 2. The exponential family of distributions

### 2.1 Canonical form

A distribution belongs to the (one-parameter) exponential family if its density or probability mass function can be written as

```
f(y; θ, φ) = exp[ (yθ − b(θ)) / a(φ) + c(y, φ) ]
```

where `θ` is the **canonical (natural) parameter**, `φ` is a **dispersion parameter**, and `a`, `b`, `c` are known functions specific to the distribution. This form looks abstract, but it earns its keep: it guarantees, for any member of the family, closed-form expressions for the mean and variance in terms of `b`, and it guarantees that maximum likelihood estimation reduces to the same iterative algorithm (Chapter 6) regardless of which family you picked.

### 2.2 Mean and variance from the exponential family

For any distribution in canonical exponential-family form, two general identities hold:

```
E[y] = μ = b'(θ)                (the mean is the derivative of b with respect to θ)
Var(y) = a(φ) · b''(θ) = a(φ) · V(μ)
```

`V(μ)`, the **variance function**, is what makes each family behave differently and is the single most important object in this chapter. It expresses how spread grows with the mean, and it is exactly the quantity that determines the iterative weights used to fit the model in Chapter 6.

### 2.3 Four families used throughout this guide

| Family | Support of y | Variance function V(μ) | Canonical link | Canonical θ |
|---|---|---|---|---|
| Gaussian | all reals | 1 (constant) | identity: g(μ) = μ | θ = μ |
| Bernoulli/Binomial | {0,1} or count/n trials | μ(1 − μ) | logit: g(μ) = log[μ/(1−μ)] | θ = log[μ/(1−μ)] |
| Poisson | non-negative integers | μ | log: g(μ) = log(μ) | θ = log(μ) |
| Gamma | positive reals | μ² | inverse: g(μ) = 1/μ (log link used in practice) | θ = −1/μ |

Read this table as the master reference for the rest of Part I: once you fix a row (a family) and a link function, the log-likelihood, the score equations, the IRLS weights, and the deviance formula are all mechanically determined, as the next four chapters show.

## 3. Link functions

### 3.1 Canonical versus non-canonical links

The **canonical link** for a family is the link function `g` for which the canonical parameter `θ` equals the linear predictor `η` directly, i.e. `θ = η = Xβ`. Canonical links are the default choice because they give the GLM convenient mathematical properties (the score equations simplify, and for canonical links the log-likelihood is guaranteed concave, so Newton-Raphson-type fitting converges reliably). You are not required to use the canonical link — for example, probit and complementary log-log links are sometimes used instead of logit for binomial data, and identity or square-root links are occasionally used for Poisson data — but canonical links are the sensible default and are what this guide uses unless stated otherwise.

### 3.2 Common links and where they are used

- **Identity**, `g(μ) = μ`: ordinary least squares. `η` ranges over all reals, matching `μ`'s unrestricted range.
- **Logit**, `g(μ) = log[μ/(1−μ)]`: binomial/Bernoulli data. Maps `μ ∈ (0,1)` to `η ∈ (−∞, ∞)`.
- **Log**, `g(μ) = log(μ)`: Poisson and gamma data. Maps `μ ∈ (0, ∞)` to `η ∈ (−∞, ∞)`, and guarantees `μ = exp(η) > 0` however negative `η` becomes — this is why log is used for gamma in practice even though gamma's canonical link is the inverse link.
- **Inverse**, `g(μ) = 1/μ`: gamma's canonical link. Rarely used directly because it does not guarantee `μ > 0` for all `η`; practitioners default to the log link for gamma instead.

### 3.3 Interpreting coefficients under each link

Under the log link, a one-unit increase in a predictor multiplies the mean by `exp(β)`, holding other predictors fixed — coefficients are interpreted as proportional (percentage) changes, not additive ones, which is exactly the same interpretive move you already made for the log-odds coefficients of logistic regression in the Regression guide. Under the identity link, coefficients are additive changes in the mean, as in OLS. Always state, when reporting a GLM coefficient, which link produced it — an unlabeled GLM coefficient is not interpretable on its own.

## 4. The linear predictor and design matrix, revisited

Nothing changes here from the Regression guide: the design matrix `X` is built the same way, with an intercept column, continuous predictors, indicator-coded categorical predictors, interactions, and transformations. The only difference is that `η = Xβ` is no longer the model's direct prediction of `y` — it is an intermediate quantity that must be passed through `g⁻¹` to obtain the predicted mean `μ = g⁻¹(η)`. Keep the distinction between `η` (linear predictor, unrestricted range) and `μ` (mean of `y`, restricted to the family's support) clear throughout the rest of this guide; conflating them is the single most common source of interpretation errors in applied GLM work, and is revisited in Chapter 36.

## 5. Maximum likelihood estimation for GLMs

### 5.1 Log-likelihood

For `n` independent observations from a GLM, the log-likelihood in terms of the canonical parameters `θᵢ` is

```
ℓ(β) = Σᵢ [ (yᵢθᵢ − b(θᵢ)) / a(φ) + c(yᵢ, φ) ]
```

where each `θᵢ` depends on `β` through `μᵢ = g⁻¹(xᵢᵀβ)` and the relationship between `θ` and `μ` established in Chapter 2. There is no closed-form maximizer of `ℓ(β)` in general (unlike OLS, whose normal equations solve in closed form) because `g⁻¹` is nonlinear for every link except the identity link. This is precisely why GLMs need an iterative fitting algorithm, developed in Chapter 6.

### 5.2 Score equations

Differentiating `ℓ(β)` with respect to `β` and setting the result to zero gives the **score equations**:

```
Σᵢ  (yᵢ − μᵢ) / (a(φ) V(μᵢ)) · (∂μᵢ / ∂ηᵢ) · xᵢ  =  0
```

This is a system of `p` nonlinear equations (one per coefficient) in `β`, since `μᵢ` and `∂μᵢ/∂ηᵢ` both depend on `β`. For the canonical link specifically, `∂μᵢ/∂ηᵢ = V(μᵢ)` and the equation simplifies to `Σᵢ (yᵢ − μᵢ) xᵢ = 0` — the direct generalization of the OLS normal equations `Σᵢ (yᵢ − xᵢᵀβ) xᵢ = 0`, recovered exactly when the link is the identity and `V(μ) = 1`.

### 5.3 Fisher information and the role of weights

The Fisher information matrix for `β` is `I(β) = XᵀWX`, where `W` is a diagonal matrix of **iterative weights**,

```
wᵢ = (∂μᵢ / ∂ηᵢ)² / V(μᵢ)
```

`W` plays exactly the role that a constant `σ²` plays in OLS's `(XᵀX)⁻¹σ²` covariance formula, except it is not constant — it depends on the current fitted mean `μᵢ`, which depends on `β`, which is what we are trying to estimate. This circularity (the weights needed to estimate `β` depend on `β` itself) is resolved by iterating, which is the entire content of the next chapter.

## 6. Iteratively Reweighted Least Squares (IRLS)

### 6.1 Newton-Raphson and Fisher scoring

Newton-Raphson finds a root of the score equations by repeatedly updating `β` using the score vector and the (negative) Hessian of the log-likelihood. **Fisher scoring** is a close variant that replaces the observed Hessian with its expected value, the Fisher information `XᵀWX` from Chapter 5.3. For GLMs with a canonical link, Fisher scoring and Newton-Raphson coincide exactly; for non-canonical links they differ slightly, and Fisher scoring is used because it guarantees a positive-definite (hence invertible) information matrix at every step.

### 6.2 Deriving IRLS

Fisher scoring's update rule, `β⁽ᵗ⁺¹⁾ = β⁽ᵗ⁾ + (XᵀW⁽ᵗ⁾X)⁻¹ Xᵀ W⁽ᵗ⁾ (y − μ⁽ᵗ⁾) / (∂μ/∂η)⁽ᵗ⁾`, can be rewritten in a form that is exactly a weighted least squares problem. Define, at each iteration `t`, the **working response**

```
zᵢ⁽ᵗ⁾  =  ηᵢ⁽ᵗ⁾  +  (yᵢ − μᵢ⁽ᵗ⁾) · (∂ηᵢ/∂μᵢ)⁽ᵗ⁾
```

(a first-order Taylor expansion of `g(y)` around `μ⁽ᵗ⁾`), and the weights `wᵢ⁽ᵗ⁾` from Chapter 5.3 evaluated at the current fit. Then the Fisher scoring update is exactly the weighted least squares solution

```
β⁽ᵗ⁺¹⁾  =  (Xᵀ W⁽ᵗ⁾ X)⁻¹ Xᵀ W⁽ᵗ⁾ z⁽ᵗ⁾
```

This is **Iteratively Reweighted Least Squares**: at each step, form a working response `z` and weights `w` from the current fit, solve a weighted OLS problem for a new `β`, recompute `η`, `μ`, `z`, and `w` from the new `β`, and repeat until `β` stops changing. Every GLM in this guide — Poisson, binomial, gamma — is fit with the same IRLS loop; only the functions `μ = g⁻¹(η)`, `V(μ)`, and `∂η/∂μ` that plug into it change from family to family. This is exactly what Part II implements once, generically, and reuses three times.

### 6.3 Convergence

IRLS is initialized with a sensible starting `μ⁽⁰⁾` (commonly `μᵢ⁽⁰⁾ = yᵢ` for Gaussian, `μᵢ⁽⁰⁾ = yᵢ + 0.5` for Poisson to avoid `log(0)`, and similar small adjustments for other families) and iterates until the change in the log-likelihood, or equivalently the change in the deviance defined in Chapter 7, falls below a small tolerance. For canonical links the log-likelihood is concave in `β`, so IRLS converges to the unique global maximum from any reasonable starting point; for non-canonical links convergence is usually still reliable in practice but is not theoretically guaranteed.

## 7. Deviance and goodness of fit

### 7.1 Definition

The **deviance** is the GLM generalization of the residual sum of squares, defined as

```
D(y, μ̂)  =  2 [ ℓ(y; y) − ℓ(y; μ̂) ]  =  2 Σᵢ [ ℓᵢ(saturated) − ℓᵢ(fitted) ]
```

where `ℓ(y; y)` is the log-likelihood of the **saturated model** — the model with one free parameter per observation, which fits each `yᵢ` exactly — and `ℓ(y; μ̂)` is the log-likelihood of the fitted model. Deviance is always non-negative and equals zero only for a perfect fit. For the Gaussian family with the identity link, deviance reduces exactly to the residual sum of squares `Σᵢ(yᵢ − μ̂ᵢ)²`, so everything in this chapter specializes to familiar OLS quantities when applied to that one row of the family table.

### 7.2 Per-family deviance formulas

```
Gaussian:   dᵢ = (yᵢ − μ̂ᵢ)²
Poisson:    dᵢ = 2 [ yᵢ log(yᵢ / μ̂ᵢ) − (yᵢ − μ̂ᵢ) ]         (with yᵢ log(yᵢ/μ̂ᵢ) := 0 when yᵢ = 0)
Binomial:   dᵢ = 2 [ yᵢ log(yᵢ/μ̂ᵢ) + (nᵢ − yᵢ) log((nᵢ−yᵢ)/(nᵢ−μ̂ᵢ)) ]   (grouped, nᵢ trials)
Gamma:      dᵢ = 2 [ −log(yᵢ/μ̂ᵢ) + (yᵢ − μ̂ᵢ)/μ̂ᵢ ]
```

`D = Σᵢ dᵢ` is the total deviance. Each `dᵢ` is non-negative and is the GLM analogue of a squared residual.

### 7.3 Deviance residuals

The **deviance residual** for observation `i` is `rᵢ = sign(yᵢ − μ̂ᵢ) · √dᵢ`, so that `Σᵢ rᵢ² = D`, exactly paralleling how ordinary residuals sum-of-squares to the residual sum of squares in OLS. Deviance residuals are the primary diagnostic tool for GLMs, used in residual-versus-fitted plots exactly as in Chapter 9 of the Regression guide, and are covered further in Chapter 12 of this guide.

### 7.4 Likelihood ratio tests via deviance

Because deviance is a monotone transform of the log-likelihood, the difference in deviance between a smaller (nested) model and a larger model is directly a **likelihood ratio test statistic**: `D_small − D_large ~ χ²(df)` under the null hypothesis that the extra parameters in the larger model are all zero, where `df` is the number of extra parameters. This is the GLM generalization of the F-test for nested linear models covered in Chapter 6.4 of the Regression guide, and it is the standard way to test whether a predictor, or a group of predictors, belongs in a GLM.

## 8. Poisson regression for counts

### 8.1 Model specification

Poisson regression models a count outcome `y ∈ {0, 1, 2, …}` with `E[y | x] = μ = exp(xᵀβ)` (log link), and assumes `Var(y | x) = μ` — variance equal to the mean, the defining feature of the Poisson distribution. The log-likelihood contribution of one observation is `yᵢ log(μᵢ) − μᵢ − log(yᵢ!)`, and the last term does not depend on `β`, so it can be dropped when maximizing.

### 8.2 Interpreting coefficients as rate ratios

Because the link is log, `exp(βⱼ)` is the multiplicative change in the expected count for a one-unit increase in predictor `j`, holding other predictors fixed — this is called a **rate ratio** or **incidence rate ratio**. A coefficient of `βⱼ = 0.10` means `exp(0.10) ≈ 1.105`, a 10.5% increase in the expected count per unit increase in that predictor.

### 8.3 Offset terms and exposure

Counts are frequently observed over varying amounts of exposure — different time periods, population sizes, or areas. An **offset** incorporates this directly into the linear predictor as a term with a fixed coefficient of 1: `η = log(exposure) + xᵀβ`, equivalently `μ = exposure · exp(xᵀβ)`, so the model predicts a *rate* (`μ`/exposure) rather than a raw count, while the log-likelihood is still computed on the raw counts. Forgetting the offset when exposure varies across observations is one of the most common Poisson regression mistakes and is revisited in Chapter 36.

## 9. Overdispersion and quasi-likelihood

### 9.1 Detecting overdispersion

The Poisson assumption `Var(y) = μ` (variance equals mean, dispersion parameter fixed at 1) is frequently violated in practice, almost always in the direction of the variance exceeding the mean — **overdispersion** — due to unmodeled heterogeneity or clustering. Overdispersion is detected by comparing the residual deviance (or the Pearson chi-squared statistic, `Σᵢ (yᵢ − μ̂ᵢ)²/μ̂ᵢ`) to its degrees of freedom `n − p`; a ratio substantially above 1 signals overdispersion. Fitting a Poisson model when the true dispersion exceeds 1 leaves point estimates of `β` unbiased but produces standard errors that are too small, inflating apparent statistical significance — directly analogous to what heteroskedasticity does to OLS standard errors in Chapter 9.7 of the Regression guide.

### 9.2 Quasi-Poisson

The **quasi-Poisson** approach keeps the Poisson mean-variance relationship's shape but multiplies it by an estimated dispersion parameter `φ̂ = Pearson χ² / (n − p)`, then scales all standard errors by `√φ̂`. It requires no new distributional assumption and no new fitting algorithm — only a post-hoc correction to standard errors after an ordinary Poisson IRLS fit.

### 9.3 Negative binomial regression

The **negative binomial** distribution is a genuine alternative distribution (not merely a correction) with `Var(y) = μ + α μ²` for a dispersion parameter `α > 0` estimated from the data, so variance grows faster than the mean rather than equal to it. It is fit by (a small extension of) the same IRLS machinery, alternating between updating `β` by IRLS at a fixed `α` and updating `α` by a separate one-dimensional maximum likelihood step. Negative binomial regression is generally preferred over quasi-Poisson when overdispersion is substantively believed to reflect real extra variability (e.g. unobserved heterogeneity across groups) rather than pure estimation noise.

## 10. Binomial regression for grouped and ungrouped proportion data

### 10.1 Bernoulli versus grouped binomial data

The Regression guide's logistic regression is the special case of binomial regression where each observation is one trial (`nᵢ = 1`, `yᵢ ∈ {0,1}`). Binomial regression generalizes this to **grouped data**: each row represents `nᵢ` trials with `yᵢ` successes (e.g. `yᵢ` defective items out of `nᵢ` inspected), modeled as `yᵢ ~ Binomial(nᵢ, μᵢ)` with `E[yᵢ/nᵢ] = μᵢ = g⁻¹(xᵢᵀβ)` under the logit link. The coefficients, their interpretation as log-odds ratios, and the fitting algorithm are unchanged from logistic regression; only the likelihood and deviance formulas (Chapter 7.2) account for `nᵢ > 1`.

### 10.2 Why this still matters after the Regression guide

Grouped binomial data is common in practice (batch inspection results, survey data reported as counts per stratum, clinical trial arms reported as event counts per group) and fitting it correctly as grouped binomial, rather than expanding it into `nᵢ` individual Bernoulli rows, produces identical point estimates but the correctly grouped deviance and Pearson statistics needed for the overdispersion diagnostics of Chapter 9.1 (binomial data can be overdispersed exactly as Poisson data can, and the same quasi-likelihood correction applies).

## 11. Gamma regression for positive continuous outcomes

### 11.1 When to use gamma regression versus log-transformed OLS

A common ad hoc approach to strictly positive, right-skewed outcomes (costs, durations, claim sizes) is to fit OLS to `log(y)` instead of `y`. This is convenient but changes the estimand: it models the mean of `log(y)`, not the log of the mean of `y`, and predictions must be exponentiated with a bias correction to estimate `E[y]` on the original scale. Gamma regression with a log link instead directly models `log(E[y]) = xᵀβ`, keeping the interpretation on the original, substantively meaningful scale, while allowing `Var(y) = φμ²` — variance growing with the square of the mean, appropriate for outcomes whose spread scales with their size (a common pattern for costs and durations).

### 11.2 Link functions for gamma

Although the inverse link `g(μ) = 1/μ` is gamma's canonical link, it does not constrain `μ` to be positive for all real `η`, which is inconvenient and can produce fitting failures. In practice the log link is used almost universally for gamma regression, at the minor cost of losing the canonical-link convenience described in Chapter 6.1; IRLS still applies unchanged, using the general (non-canonical) weight formula from Chapter 5.3.

## 12. Model diagnostics for GLMs

### 12.1 Pearson and deviance residuals

Two standardized residual types generalize OLS residuals to GLMs. The **Pearson residual** is `rᵢᴾ = (yᵢ − μ̂ᵢ) / √V(μ̂ᵢ)`, directly generalizing the idea of dividing by an estimated standard deviation. The **deviance residual**, defined in Chapter 7.3, is generally preferred for diagnostic plots because its distribution is closer to symmetric and normal-like for well-fitting models, making residual-versus-fitted and Q-Q plots (Chapter 9 of the Regression guide) more interpretable.

### 12.2 Working residuals

The **working residual**, `(yᵢ − μ̂ᵢ) · (∂ηᵢ/∂μᵢ)`, is the residual on the scale of the working response `z` from the IRLS algorithm (Chapter 6.2) — it is what OLS residuals would be if you ran OLS of `z` on `X` at convergence. It is used less often for reporting but is a useful bridge for understanding what IRLS is actually doing at each step.

### 12.3 Leverage and influence for GLMs

The GLM hat matrix is `H = W^(1/2) X (XᵀWX)⁻¹ Xᵀ W^(1/2)`, evaluated at the converged weights `W`, directly generalizing the OLS hat matrix `X(XᵀX)⁻¹Xᵀ` from Chapter 4.2 of the Regression guide (recovered when `W = I`). Its diagonal gives GLM leverage `hᵢᵢ`, and a GLM generalization of Cook's distance combines `hᵢᵢ` with the standardized deviance residual exactly as in Chapter 9.3–9.5 of the Regression guide.

### 12.4 Goodness-of-fit tests

For grouped or count data with reasonably large per-cell counts, the residual deviance and the Pearson chi-squared statistic can each be compared to a `χ²(n − p)` reference distribution as an overall goodness-of-fit check, in addition to their role in the overdispersion diagnostics of Chapter 9.1. For sparse count or binary data (many cells with `yᵢ` near 0), these asymptotic tests become unreliable and residual plots or held-out predictive checks (Chapter 13.3) are preferred.

## 13. Model selection and comparison

### 13.1 AIC and BIC

The **Akaike Information Criterion**, `AIC = −2ℓ(β̂) + 2p`, and the **Bayesian Information Criterion**, `BIC = −2ℓ(β̂) + p·log(n)`, both penalize log-likelihood by model complexity, generalizing the adjusted R-squared idea from Chapter 8.3 of the Regression guide to any GLM family (where a direct analogue of R-squared is not well defined, since deviance is not on a common scale across families). Lower AIC/BIC indicates a better complexity-adjusted fit; BIC penalizes additional parameters more heavily than AIC as sample size grows, and tends to favor smaller models.

### 13.2 Likelihood ratio tests for nested models

As established in Chapter 7.4, the deviance difference between two nested GLMs (fit with the same family and link, one model's predictors a subset of the other's) is a direct chi-squared test of whether the additional predictors improve fit beyond chance. This is the GLM generalization of the joint F-test in Chapter 6.4 of the Regression guide.

### 13.3 Cross-validation for GLMs

Held-out predictive performance is assessed the same way as in Chapter 11.2 of the Regression guide — k-fold cross-validation — but the scoring metric changes: mean squared error is not the natural choice for count or binary outcomes. Use the family's own deviance (or log-likelihood) computed on held-out data, or a task-specific metric (e.g. classification accuracy or AUC for binomial GLMs, mean absolute error for Poisson counts) as appropriate.

## 14. Regularized GLMs

### 14.1 Penalizing the GLM likelihood

Exactly as Chapter 12 of the Regression guide added ridge and lasso penalties to the OLS sum-of-squares objective, a GLM can be fit by maximizing a **penalized log-likelihood**:

```
ℓ_penalized(β) = ℓ(β) − λ [ (1−α)/2 · Σⱼ βⱼ² + α · Σⱼ |βⱼ| ]
```

with `α = 0` giving ridge-penalized GLMs, `α = 1` giving lasso-penalized GLMs, and intermediate `α` giving an elastic net penalty, mirroring Chapter 12.3 exactly. The intercept is conventionally left unpenalized, as in the Regression guide.

### 14.2 Fitting penalized GLMs: penalized IRLS

The IRLS algorithm of Chapter 6.2 extends directly: at each iteration, instead of solving an ordinary weighted least squares problem for the working response `z`, solve a *penalized* weighted least squares problem (ridge-penalized IRLS solves this in closed form at each step exactly as Chapter 12.1 of the Regression guide solves ridge; lasso-penalized IRLS solves it with coordinate descent at each step, exactly as Chapter 22.2 of the Regression guide solves lasso, nested one level deeper inside the IRLS loop). This is precisely the algorithm implemented from scratch in Chapter 24 of this guide.

## 15. GLMs and causal claims

Everything said about causal interpretation of OLS coefficients in Chapter 14 of the Regression guide applies unchanged to GLM coefficients, with one addition specific to nonlinear links: because a GLM coefficient measures an effect on the *link scale* (log-odds, log-rate, etc.), not on the mean scale directly, even a causally identified GLM coefficient does not translate into a constant effect on `μ` — the marginal effect on `μ` itself varies with the level of the predictors (this is the same phenomenon noted for logistic regression in Chapter 13.1 of the Regression guide, now recognized as a general property of nonlinear links, not a logistic-regression-specific quirk). Reporting only the link-scale coefficient, without also reporting predicted means or marginal effects at representative predictor values, is a common source of miscommunication in applied GLM work.

---

# Part II. GLMs from Scratch in Pure Python

## 16. Why implement GLMs yourself?

Everything in Part I collapses into one algorithm, IRLS (Chapter 6), reused across every family in the table in Chapter 2.3. Implementing it once, generically, and then supplying only the small family-specific pieces (link, inverse link, variance function, deviance formula) for Poisson, binomial, and gamma is the single best way to internalize that GLMs are not four unrelated models but one model with four interchangeable parts. This section builds that generic solver, then specializes it three times.

### 16.1 What "from scratch" reuses from the Regression guide

The linear algebra core — `transpose`, `matmul`, `matvec`, `solve_linear_system` by Gauss-Jordan elimination, `add_intercept_column` — is identical to Chapter 17 of the Regression guide and is not re-derived here; Appendix A includes a self-contained copy so this guide's code runs independently. What is new is everything downstream of forming `XᵀWX` and `XᵀWz`: the reweighting loop itself.

## 17. A minimal linear algebra core, revisited

The from-scratch GLM code needs exactly four linear algebra primitives beyond simple vector arithmetic: matrix transpose, matrix-matrix and matrix-vector multiplication, and a linear system solver. These are the same Gauss-Jordan-with-partial-pivoting routines from Chapter 17 of the Regression guide, included again in Appendix A so this module has no dependency on the other guide's file. If you have already implemented these for the Regression guide, treat this chapter as a one-line reminder: nothing here changes; a GLM's weighted normal equations `(XᵀWX)β = XᵀWz` are solved with exactly the `solve_linear_system` routine already written for OLS's `(XᵀX)β = Xᵀy`.

## 18. The generic IRLS solver

### 18.1 The family interface

Every family is represented as a small bundle of five functions matching the table in Chapter 2.3: `link(mu) → eta`, `link_inverse(eta) → mu`, `dlink_dmu(mu) → deta/dmu` (needed to build both the IRLS weight and the working response), `variance(mu) → V(mu)`, and `deviance_terms(y, mu) → per-observation dᵢ` from Chapter 7.2. A family also supplies a rule for a starting value `mu⁽⁰⁾`, chosen to avoid domain errors (e.g. `log(0)`) on the very first iteration.

### 18.2 The iteration loop

Each IRLS iteration performs exactly four steps, matching the derivation in Chapter 6.2: (1) compute `deta_dmu = dlink_dmu(mu)` and `V = variance(mu)` at the current `mu`; (2) form the iterative weights `w = prior_weight / (V · deta_dmu²)` and the working response `z = eta + (y − mu) · deta_dmu`; (3) solve the weighted normal equations `(XᵀWX) β = XᵀWz` for a new `β`; (4) recompute `eta = Xβ`, `mu = link_inverse(eta)`, and the deviance, and check whether the deviance changed by less than a small tolerance. Appendix A's `fit_glm` function implements exactly these four steps, generically, for any family object.

### 18.3 Numerical safeguards

Two safeguards matter in practice and are included in Appendix A: clamping `mu` away from the exact boundary of its support (e.g. keeping a Poisson `mu` at least `1e-10`, and a binomial `mu` strictly inside `(1e-10, 1 − 1e-10)`) before evaluating `variance(mu)` or `link(mu)`, since these boundary values produce division by zero or `log(0)`; and capping the number of iterations (25 is generous for a well-specified GLM) so a non-converging fit raises an explicit error rather than looping silently.

A specific, common way IRLS fails to converge deserves a direct warning: **(quasi-)complete separation** in binomial data. If a predictor (or combination of predictors) perfectly or near-perfectly separates the two outcome classes, the maximum likelihood coefficient for that predictor is unbounded — IRLS keeps pushing `β` toward infinity every iteration, `μ` keeps pressing against its clamped boundary, and the deviance stops changing meaningfully without the algorithm ever settling, which is exactly the `RuntimeError` Appendix A's `fit_glm` raises when `max_iterations` is exhausted. This is not a bug in the solver; it is the correct numerical symptom of a genuinely unbounded likelihood, and it is the same failure mode documented for rare-event or small-sample logistic regression in Firth (1993), listed in Chapter 45. If you hit it: check for a predictor that alone near-perfectly predicts the outcome, consider Firth's penalized-likelihood correction, or add a small ridge penalty (Chapter 24.1), which bounds the penalized likelihood and restores convergence.

## 19. Poisson regression from scratch

Poisson regression is the family with `link = log`, `link_inverse = exp`, `variance(mu) = mu`, `dlink_dmu(mu) = 1/mu`, and the deviance formula from Chapter 7.2, with the `y log(y/mu)` term defined as `0` when `y = 0` (its limiting value). Because the log link is Poisson's canonical link, `dmu_deta = mu` and the IRLS weight simplifies to `w = prior_weight · mu`, a useful closed-form check on Appendix A's more general implementation. An optional `offset` vector (Chapter 8.3) is added directly to `eta` before applying `link_inverse`, and is *not* multiplied by any coefficient.

## 20. Binomial (grouped) regression from scratch

Binomial regression is the family with `link = logit`, `variance(mu) = mu(1 − mu)`, `dlink_dmu(mu) = 1/(mu(1 − mu))`, prior weights equal to each observation's trial count `nᵢ` (Chapter 10.1), and the grouped-binomial deviance formula from Chapter 7.2. Setting every `nᵢ = 1` recovers exactly the from-scratch logistic regression of Chapter 23 of the Regression guide — running both implementations on the same 0/1 data and confirming identical coefficients is a strong correctness check on this chapter's code, done explicitly in Chapter 26.

## 21. Gamma regression from scratch

Gamma regression, using the log link rather than the canonical inverse link (Chapter 11.2), has `link = log`, `link_inverse = exp`, `variance(mu) = mu²`, `dlink_dmu(mu) = 1/mu`, and the deviance formula from Chapter 7.2. Because the log link is not gamma's canonical link, the IRLS weight does not simplify as neatly as Poisson's; Appendix A's generic solver handles this without modification, which is exactly the point of writing IRLS generically in Chapter 18 rather than re-deriving a family-specific shortcut each time.

## 22. Deviance, residuals, and diagnostics from scratch

Given a converged fit, three diagnostic quantities from Chapter 12 are computed directly from `y`, `mu`, and the family's `variance` function: **Pearson residuals**, `(y − mu) / sqrt(variance(mu))`; **deviance residuals**, `sign(y − mu) · sqrt(deviance_terms(y, mu))`; and **leverage**, the diagonal of `W^(1/2) X (XᵀWX)⁻¹ Xᵀ W^(1/2)` evaluated at the converged weights `W`, computed row-by-row without forming the full `n × n` hat matrix (exactly as the Regression guide's `ols_diagnostics` in Chapter 19.1 avoids forming the full hat matrix). A GLM-generalized Cook's distance combines leverage and the standardized deviance residual by direct analogy to Chapter 9.1 of the Regression guide's `ols_diagnostics`.

## 23. Overdispersion from scratch

The Pearson-based dispersion estimate `φ̂ = Σᵢ (yᵢ − μ̂ᵢ)² / V(μ̂ᵢ) / (n − p)` from Chapter 9.1 requires only quantities already computed during fitting. Quasi-Poisson standard errors (Chapter 9.2) are obtained by scaling the ordinary IRLS covariance matrix `(XᵀWX)⁻¹` by this `φ̂`. A simple from-scratch negative binomial fit (Chapter 9.3) alternates an ordinary Poisson-family IRLS update for `β` at a fixed dispersion `α`, with a one-dimensional bisection search for the `α` that maximizes the negative binomial log-likelihood at the current `β` — a small outer loop wrapped around the same `fit_glm` function from Chapter 18, not a new algorithm.

## 24. Regularized GLMs from scratch

### 24.1 Ridge-penalized IRLS

Ridge-penalized IRLS changes exactly one line of the Chapter 18.2 loop: step 3 solves `(XᵀWX + λΓ) β = XᵀWz` instead of `(XᵀWX) β = XᵀWz`, where `Γ` is the identity matrix with a zero in the intercept position (unpenalized intercept, exactly as in Chapter 12.1 of the Regression guide). Everything else — the weight and working-response updates, the convergence check — is untouched.

### 24.2 Lasso- and elastic-net-penalized IRLS

Lasso-penalized IRLS replaces the closed-form weighted least squares solve in step 3 with **weighted coordinate descent** (a direct generalization of the unweighted coordinate descent used for the Regression guide's from-scratch lasso in Chapter 22.2): holding all coefficients but `βⱼ` fixed, the optimal `βⱼ` for the weighted, penalized objective has a soft-thresholding closed form, `βⱼ ← S(ρⱼ, λα) / (Σᵢ wᵢ xᵢⱼ²)`, where `ρⱼ` is the weighted partial-residual correlation for feature `j` and `S` is the same soft-threshold function from Chapter 22.2 of the Regression guide. This inner coordinate-descent loop runs to convergence at each outer IRLS iteration, exactly the nested-loop structure used by production implementations such as `glmnet`.

## 25. Model comparison from scratch

AIC and BIC (Chapter 13.1) require only the converged log-likelihood, itself a direct byproduct of the deviance already computed each iteration (`ℓ = −D/2 + constant terms not depending on β`, so *differences* in AIC/BIC across nested models fit with the same family need only differences in deviance, sidestepping the family-specific constant entirely). The likelihood ratio test of Chapter 13.2 needs a chi-squared reference distribution; Appendix A implements a `chi2_sf` (survival function) using a regularized incomplete gamma function evaluated by series expansion or continued fraction depending on its arguments, the same numerical strategy Appendix A of the Regression guide uses for the Student t distribution.

## 26. A complete from-scratch workflow

Appendix A closes with a runnable `demo()` that: simulates Poisson count data with a known offset and two predictors; fits it with the from-scratch Poisson IRLS solver; reports coefficients, deviance, and the overdispersion ratio; fits the same coefficients again via ridge- and lasso-penalized IRLS at a moderate penalty and confirms the unpenalized and lightly-penalized coefficients are close; and cross-checks the grouped-binomial solver against the plain logistic regression solver from the Regression guide's Appendix A on identical 0/1 data, exactly as flagged in Chapter 20. Running this workflow end to end, and reading every intermediate print statement, is the fastest way to confirm your understanding of Part I is correct rather than merely familiar.

---

# Part III. GLMs with Python Libraries

## 27. Which library should do what?

**statsmodels** is the primary tool for GLM *inference*: it exposes every family and link from Chapter 2, reports standard errors, confidence intervals, p-values, deviance, AIC/BIC, and built-in overdispersion handling, and its API (`sm.GLM(y, X, family=...)`) mirrors the math of Part I almost line for line. **scikit-learn** is the primary tool for GLM *prediction pipelines*: `PoissonRegressor`, `GammaRegressor`, and `TweedieRegressor` integrate directly with `Pipeline`, `ColumnTransformer`, cross-validation, and built-in L2 regularization, but (as of the versions this guide targets) report point predictions rather than the full inferential output statsmodels provides. Use statsmodels when the question is "what is the effect of X, and how uncertain am I?" and scikit-learn when the question is "what will happen next, evaluated by held-out predictive accuracy?" — the same division of labor as Chapter 25 of the Regression guide, carried over unchanged.

## 28. statsmodels GLM: families and links

```python
import statsmodels.api as sm

X = sm.add_constant(df[["experience", "education"]])
model = sm.GLM(df["claims"], X, family=sm.families.Poisson()).fit()
print(model.summary())
```

`sm.families` exposes `Gaussian`, `Binomial`, `Poisson`, `Gamma`, `InverseGaussian`, `NegativeBinomial`, and `Tweedie`, each accepting an optional `link=` argument (`sm.families.links.Log()`, `sm.families.links.Logit()`, `sm.families.links.Identity()`, and others) so that non-canonical links (Chapter 3.1), such as a log link for gamma (Chapter 11.2), are one keyword argument away from the canonical default.

## 29. statsmodels: Poisson and negative binomial workflows

```python
poisson_model = sm.GLM(
    df["claims"], X, family=sm.families.Poisson(),
    offset=np.log(df["exposure"]),   # Chapter 8.3
).fit()

pearson_chi2 = poisson_model.pearson_chi2
dispersion = pearson_chi2 / poisson_model.df_resid   # Chapter 9.1
print("Dispersion:", dispersion)

quasi_poisson = sm.GLM(
    df["claims"], X, family=sm.families.Poisson(),
).fit(scale=dispersion)   # Chapter 9.2, rescaled standard errors

negbin_model = sm.GLM(
    df["claims"], X, family=sm.families.NegativeBinomial(),
).fit()
```

`statsmodels.discrete.discrete_model.NegativeBinomial` is an alternative entry point that also estimates the dispersion parameter `α` (Chapter 9.3) by maximum likelihood jointly with `β`, rather than treating it as fixed.

## 30. statsmodels: binomial regression with grouped data

```python
binomial_model = sm.GLM(
    endog=df[["successes", "failures"]],   # two-column endog: successes, failures
    exog=X,
    family=sm.families.Binomial(),
).fit()
```

Passing a two-column `endog` of successes and failures (rather than a single proportion column) is how statsmodels represents grouped binomial data (Chapter 10.1) and correctly incorporates the trial counts `nᵢ` into the deviance and Pearson statistics.

## 31. statsmodels: gamma regression

```python
gamma_model = sm.GLM(
    df["claim_amount"], X,
    family=sm.families.Gamma(link=sm.families.links.Log()),   # Chapter 11.2
).fit()
print("Estimated dispersion:", gamma_model.scale)
```

## 32. scikit-learn: PoissonRegressor, GammaRegressor, TweedieRegressor

```python
from sklearn.linear_model import PoissonRegressor, GammaRegressor, TweedieRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

poisson_pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", PoissonRegressor(alpha=1e-3)),   # alpha: L2 penalty strength, Chapter 33
])
poisson_pipeline.fit(X_train, y_train)
predicted_rate = poisson_pipeline.predict(X_test)
```

`TweedieRegressor(power=...)` is a single estimator spanning Gaussian (`power=0`), Poisson (`power=1`), gamma (`power=2`), and inverse Gaussian (`power=3`) by varying one continuous parameter that controls the variance function's exponent — a practical, code-level illustration of the exponential family's unifying role from Chapter 2.

## 33. Regularized GLMs with libraries

```python
# statsmodels: elastic-net-penalized GLM (Chapter 14)
regularized = sm.GLM(df["claims"], X, family=sm.families.Poisson()).fit_regularized(
    alpha=0.05, L1_wt=0.5,   # L1_wt=1 is pure lasso, L1_wt=0 is pure ridge
)

# scikit-learn: built-in L2 penalty via `alpha`
ridge_like = PoissonRegressor(alpha=0.5).fit(X_train, y_train)
```

statsmodels' `fit_regularized` does not, in general, report standard errors (penalized estimates are biased by construction, and standard inferential theory does not directly apply — see Chapter 12 of the Regression guide's discussion of the same issue for ridge and lasso). Use regularized GLMs for prediction and variable selection; use unpenalized `fit()` for inference on a final, chosen model.

## 34. Diagnostic plots for GLMs with matplotlib

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(poisson_model.fittedvalues, poisson_model.resid_deviance)
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_xlabel("Fitted values"); axes[0].set_ylabel("Deviance residuals")

import scipy.stats as stats
stats.probplot(poisson_model.resid_deviance, dist="norm", plot=axes[1])
```

Deviance residuals (Chapter 7.3), not raw `y − ŷ`, are the correct residual to use in GLM diagnostic plots, for the same reason ordinary residuals are preferred over raw errors in Chapter 9.1 of the Regression guide — they are normalized to be comparable across observations with different variances.

## 35. End-to-end GLM checklist

1. State the outcome's natural support (counts, proportions, positive continuous) and pick a family from Chapter 2.3 accordingly — do not default to Gaussian OLS by habit.
2. Choose a link function (Chapter 3), defaulting to the canonical link unless there is a specific reason not to.
3. Check for an offset (Chapter 8.3) whenever the outcome is a count observed over varying exposure.
4. Fit the model and inspect the residual deviance versus its degrees of freedom (Chapter 9.1) before trusting any standard error.
5. If overdispersed, apply a quasi-likelihood correction or refit with negative binomial (Chapter 9).
6. Examine deviance residuals, leverage, and influence (Chapter 12) exactly as you would OLS residuals.
7. Compare nested models with a likelihood ratio test on the deviance (Chapter 13.2), not with a raw deviance value in isolation.
8. Validate predictive performance with cross-validation using a family-appropriate scoring metric (Chapter 13.3), not mean squared error by default.
9. Report coefficients with their link explicitly stated, and supplement log-scale or logit-scale coefficients with predicted means or marginal effects at representative predictor values (Chapter 15).
10. If regularizing, refit the final selected model unpenalized to obtain valid standard errors before making inferential claims (Chapter 33).

## 36. Common mistakes and corrections

| Mistake | Why it is wrong | Correction |
|---|---|---|
| Fitting OLS to a count outcome | Ignores the mean-variance relationship (Chapter 2.2); can predict negative counts | Use Poisson or negative binomial regression (Chapter 8–9) |
| Forgetting an offset for varying exposure | Silently treats a rate as a raw count | Add `offset = log(exposure)` (Chapter 8.3) |
| Trusting Poisson standard errors without checking dispersion | Overdispersion (very common) makes them too small | Check Pearson χ²/df; use quasi-Poisson or negative binomial (Chapter 9.1) |
| Interpreting a log-link coefficient as an additive effect | Log-link coefficients are multiplicative (rate ratios), not additive (Chapter 8.2) | Exponentiate and describe as a percentage or ratio change |
| Expanding grouped binomial data into individual 0/1 rows before diagnostics | Point estimates match, but deviance and overdispersion diagnostics change (Chapter 10.2) | Fit and diagnose on the grouped `(successes, failures)` representation |
| Comparing AIC across models fit with different families | AIC is only comparable within a fixed family's likelihood constants (Chapter 13.1) | Compare AIC only among models sharing the same family and link |
| Using raw residuals instead of deviance residuals in diagnostic plots | Raw residuals are not comparable in scale across observations with different variances | Use deviance (or Pearson) residuals (Chapter 12.1) |
| Reporting only a link-scale coefficient with no substantive translation | A log-odds or log-rate coefficient is not directly interpretable to a non-technical audience | Report exponentiated coefficients and/or predicted means at representative values (Chapter 15) |
| Treating a penalized GLM's standard errors as valid | Penalized estimates are biased; standard formulas do not apply (Chapter 33) | Refit the selected model unpenalized for inference, or use bootstrap |

---

# Part IV. Exercises, Reference, and Further Study

## 37. Conceptual questions

1. Explain, without equations, why OLS applied directly to a count outcome can predict impossible (negative) values, and why a GLM with a log link cannot.
2. What are the three components that fully specify a GLM? Give the values of all three for logistic regression and for Poisson regression.
3. Why does the variance function `V(μ)` differ across families, and what real-world pattern does the Poisson variance function `V(μ) = μ` capture?
4. Explain, in one paragraph, why IRLS is needed instead of a closed-form solution like the OLS normal equations.
5. What does it mean for a link to be "canonical," and why is the canonical link the default choice?
6. Why does overdispersion leave point estimates unbiased but invalidate standard errors? Connect this to heteroskedasticity in OLS.
7. Explain, using the offset concept, the difference between modeling a count and modeling a rate.
8. Why is deviance, not raw residual sum of squares, the right generalization of "unexplained variation" for a GLM?

## 38. Derivation exercises

1. Starting from the exponential-family canonical form in Chapter 2.1, derive `E[y] = b'(θ)` for the Poisson distribution and confirm it matches the familiar Poisson mean.
2. Derive the Poisson deviance formula in Chapter 7.2 from the general deviance definition in Chapter 7.1.
3. Show that, for the canonical link, the IRLS weight in Chapter 6.2 simplifies to `w = V(μ)`, and confirm this for Poisson (`w = μ`) and Bernoulli (`w = μ(1−μ)`).
4. Show that Poisson regression's score equation (Chapter 5.2) reduces to `Σᵢ(yᵢ − μᵢ)xᵢ = 0`, the same form as the OLS normal equations.
5. Derive the working response `z` (Chapter 6.2) for the log link and confirm it matches Appendix A's implementation.
6. Show that the Gaussian family with the identity link reduces IRLS to a single, non-iterative step that is exactly the OLS normal equations.

## 39. Pure-Python coding exercises

1. Implement the `variance`, `link`, `link_inverse`, and `dlink_dmu` functions for the Gaussian family and confirm `fit_glm` reproduces the from-scratch OLS coefficients from the Regression guide on the same data.
2. Extend Appendix A's Poisson family to support a log-link negative binomial with a fixed, user-supplied dispersion `α`, and confirm the score equations still hold at convergence.
3. Implement a simple grid search over `α` (Chapter 23) that maximizes the negative binomial log-likelihood, and compare the selected `α` to `statsmodels`'s estimate on the same simulated data.
4. Add an `offset` parameter to `fit_glm` and confirm that fitting with a constant offset of zero reproduces the no-offset result exactly.
5. Implement the GLM leverage and Cook's-distance-analogue diagnostics from Chapter 22 and verify they reduce to the plain OLS versions from the Regression guide when the family is Gaussian.
6. Implement weighted coordinate descent for the lasso-penalized IRLS inner loop (Chapter 24.2) and confirm that setting all IRLS weights to 1 reproduces the plain lasso coordinate descent from the Regression guide.

## 40. Library exercises

1. Fit the same Poisson model with `statsmodels` and with `sklearn.linear_model.PoissonRegressor`; compare coefficients and explain any differences due to regularization defaults.
2. Simulate overdispersed count data, fit both a plain Poisson GLM and a negative binomial GLM with `statsmodels`, and compare their AIC values.
3. Fit a grouped binomial model in `statsmodels` using the two-column `endog` interface (Chapter 30) and confirm the coefficients match a Bernoulli-expanded logistic regression fit on the same underlying data.
4. Fit a `TweedieRegressor` at `power = 0, 1, 2` on the same dataset and describe how the predictions and implied variance assumptions change.
5. Use `fit_regularized` in `statsmodels` to select a sparse Poisson model, then refit the selected variables unpenalized and compare standard errors to a naive read of the regularized fit's coefficients.

## 41. Suggested learning sequence

**Stage 1: Bridge from regression.** Confirm you can explain, from memory, why linear regression and logistic regression are both GLMs (Chapter 1.2) before reading further.

**Stage 2: Exponential family fluency.** Work through Chapter 2 for all four families until you can write down `V(μ)` and the canonical link for each without looking it up.

**Stage 3: IRLS derivation.** Derive IRLS yourself (Chapter 6) starting only from the score equations (Chapter 5.2) — do not read Chapter 6.2's answer until you have attempted it.

**Stage 4: From-scratch coding.** Implement the generic `fit_glm` solver and all three families (Chapters 18–21) before looking at Appendix A's reference implementation.

**Stage 5: Diagnostics and overdispersion.** Fit a deliberately overdispersed simulated Poisson dataset and walk through the full diagnostic sequence in Chapter 35's checklist.

**Stage 6: Practical libraries.** Reproduce every code example in Part III on your own dataset.

**Stage 7: Independent project.** Pick a real or realistic count, proportion, or positive-continuous outcome; carry it through the full checklist in Chapter 35, including a regularized variant and a final unpenalized refit.

## 42. Prompts for an AI tutor or NotebookLM

**Concept teaching.** "Teach me generalized linear models starting from what I already know about OLS and logistic regression. Use the exponential family table from Chapter 2.3 as the spine of the explanation, and quiz me after each family before moving to the next."

**Mathematical practice.** "Give me a small dataset and have me derive, by hand, the working response and IRLS weights for one iteration of Poisson regression. Check my arithmetic at each step."

**Code teaching.** "Walk me through the generic `fit_glm` function in Appendix A line by line, and after each function, have me predict what would break if that function were removed or given the wrong family object."

**Interpretation practice.** "Give me a fitted Poisson regression summary table and ask me to state each coefficient's effect as a rate ratio and as a plain-language sentence, without using the words 'log' or 'link' in my answer."

**Diagnostic practice.** "Give me a residual deviance, degrees of freedom, and Pearson chi-squared value, and ask me whether the model is overdispersed and what I should do next."

**Project supervision.** "Act as my GLM research supervisor. Before I fit anything, ask me to state the outcome's support, the family and link I am choosing and why, whether an offset is needed, and how I will check for overdispersion."

## 43. Glossary

**Canonical link.** The link function for which the canonical (natural) parameter of the exponential family equals the linear predictor directly, θ = η. Canonical links make the score equations simplify and guarantee a concave log-likelihood, which is why they are the default choice (Chapter 3.1) even though other links remain valid for the same family.

**Canonical parameter.** The parameter θ appearing in the exponential family's canonical form (Chapter 2.1) that is linear in the sufficient statistic y. It is related to the mean μ through θ = (b')⁻¹(μ), and it is what the linear predictor η equals exactly when the canonical link is used.

**Deviance.** The GLM generalization of the residual sum of squares, defined as twice the difference between the saturated model's log-likelihood and the fitted model's log-likelihood. It reduces exactly to the residual sum of squares for Gaussian data under the identity link, and its per-family formulas (Chapter 7.2) are used for goodness of fit, model comparison, and residual diagnostics throughout this guide.

**Deviance residual.** The signed square root of an observation's individual deviance contribution, `sign(y − μ̂) · √dᵢ`. Deviance residuals sum-of-squares exactly to the total deviance and are generally preferred over raw or Pearson residuals for diagnostic plots because their distribution is closer to symmetric and normal-like for a well-fitting model.

**Dispersion parameter.** The parameter φ (or a(φ) in the canonical exponential-family form) scaling an observation's variance beyond the variance function, Var(y) = a(φ)·V(μ). It is fixed at 1 by assumption for Poisson and binomial models but is estimated from the data for Gaussian, gamma, and quasi-likelihood models, and its misestimation (as in unmodeled overdispersion) is the primary cause of incorrect GLM standard errors.

**Exponential family.** A broad class of probability distributions whose density or mass function can be written in the canonical form of Chapter 2.1. It unifies the Gaussian, Bernoulli/binomial, Poisson, and gamma distributions under one mathematical structure, which is what allows a single fitting algorithm (IRLS) and a single set of diagnostic formulas to apply, with only a handful of family-specific functions changing, across every model in this guide.

**Fisher information.** The expected value of the negative Hessian of the log-likelihood, `I(β) = XᵀWX` for a GLM, where W is the diagonal matrix of iterative weights. It measures how sharply peaked the likelihood is around its maximum and, inverted, gives the asymptotic covariance matrix of the maximum likelihood coefficient estimates.

**Fisher scoring.** An iterative maximum likelihood algorithm that updates parameters using the Fisher information in place of the observed Hessian used by plain Newton-Raphson. For GLMs, Fisher scoring is algebraically identical to Iteratively Reweighted Least Squares (Chapter 6.1–6.2), and the two names are frequently used interchangeably in the GLM literature.

**Gamma regression.** A GLM for strictly positive, right-skewed continuous outcomes, with variance function V(μ) = μ², typically fit with a log link rather than its canonical inverse link (Chapter 11). It directly models the mean of the outcome on its original scale, unlike the common ad hoc alternative of applying OLS to a log-transformed outcome.

**Grouped binomial data.** Data in which each row represents a count of successes out of a known number of trials, `yᵢ` out of `nᵢ`, rather than a single 0/1 outcome per row. Binomial regression on grouped data generalizes the Bernoulli logistic regression of the Regression guide, and correctly incorporates `nᵢ` into the deviance and overdispersion diagnostics (Chapter 10).

**Iteratively Reweighted Least Squares (IRLS).** The algorithm that fits every GLM in this guide, alternating between computing a working response and iterative weights from the current fit, and solving a weighted least squares problem for an updated coefficient vector, until convergence (Chapter 6). It reduces to a single, non-iterative step for Gaussian data under the identity link, recovering the ordinary least squares normal equations exactly.

**Likelihood ratio test.** A hypothesis test comparing two nested GLMs by their deviance difference, which follows a chi-squared distribution under the null hypothesis that the additional parameters in the larger model are all zero (Chapter 13.2). It is the GLM generalization of the joint F-test used for nested linear models.

**Linear predictor.** The quantity η = Xβ, formed from the design matrix exactly as in OLS, before being passed through the inverse link function to produce the mean μ. Keeping the linear predictor (unrestricted range) distinct from the mean (restricted to the outcome's natural support) is essential to correctly interpreting any GLM.

**Link function.** The monotonic, differentiable function g connecting a GLM's mean μ to its linear predictor η, via g(μ) = η. The link function is what allows a linear predictor with an unrestricted range to produce a mean that respects the outcome's natural support (Chapter 3), and it determines how coefficients should be interpreted (additively, multiplicatively, or otherwise).

**Log link.** The link function g(μ) = log(μ), mapping a mean restricted to positive values onto an unrestricted linear predictor. It is the canonical link for Poisson regression and the conventional (non-canonical) link for gamma regression, and it makes coefficients interpretable as multiplicative (percentage) effects via exponentiation.

**Logit link.** The link function g(μ) = log[μ/(1−μ)], mapping a mean restricted to (0,1) onto an unrestricted linear predictor. It is the canonical link for binomial and Bernoulli (logistic) regression, and its coefficients are interpretable as log-odds ratios.

**Negative binomial regression.** A GLM for count data with variance function V(μ) = μ + αμ², allowing variance to grow faster than the mean, unlike Poisson's V(μ) = μ. It is the preferred remedy for overdispersion (Chapter 9.3) when the extra variability is believed to reflect genuine unobserved heterogeneity rather than pure estimation noise.

**Offset.** A term added directly to the linear predictor with a fixed coefficient of 1, most commonly `log(exposure)` in Poisson regression, used to model a rate rather than a raw count when observations are measured over varying amounts of exposure (Chapter 8.3).

**Overdispersion.** The situation in which an outcome's true variance exceeds what the assumed family's variance function predicts at the fitted mean, most commonly encountered with Poisson and binomial models. It leaves coefficient point estimates unbiased but produces standard errors that are too small, inflating apparent statistical significance, unless corrected via quasi-likelihood or a richer family such as negative binomial (Chapter 9).

**Pearson residual.** A standardized residual, `(y − μ̂) / √V(μ̂)`, dividing the raw residual by an estimate of its standard deviation under the fitted model. It is closely related to the deviance residual and is the basis of the Pearson chi-squared statistic used for overdispersion diagnostics (Chapter 9.1).

**Poisson regression.** A GLM for count outcomes with variance function V(μ) = μ (variance equal to mean) and, canonically, a log link. It is the default starting model for unbounded count data, with coefficients interpretable as multiplicative rate ratios (Chapter 8).

**Prior weights.** Known, fixed weights (distinct from the iterative weights computed during IRLS) reflecting, for example, the number of trials underlying a grouped binomial observation. They enter the IRLS weight formula multiplicatively and are held constant throughout fitting, unlike the iterative weights, which are recomputed at every iteration.

**Quasi-likelihood.** An estimation approach that specifies only the mean-variance relationship of an outcome (rather than its full distribution) and estimates a dispersion parameter to rescale standard errors accordingly, without requiring a fully specified probability model. Quasi-Poisson regression (Chapter 9.2) is the most common quasi-likelihood GLM, used as a lightweight correction for overdispersion.

**Rate ratio.** The multiplicative effect of a one-unit predictor increase on a Poisson-modeled mean, equal to `exp(β)` under the log link. It is the count-data analogue of the odds ratio in logistic regression, and is the standard way to report Poisson regression coefficients in plain language (Chapter 8.2).

**Saturated model.** The model with one free parameter per observation, fitting every `yᵢ` exactly and achieving the maximum possible log-likelihood for the data. It is a reference point, not a usable model, and appears in the definition of deviance (Chapter 7.1) as the benchmark against which a fitted model's log-likelihood is compared.

**Score equations.** The system of equations obtained by setting the gradient of the log-likelihood to zero, generalizing the OLS normal equations to any GLM (Chapter 5.2). They have no closed-form solution for nonlinear links, which is why IRLS is needed to solve them iteratively.

**Variance function.** The function V(μ) expressing how an exponential-family distribution's variance depends on its mean, `Var(y) = a(φ)·V(μ)` (Chapter 2.2). It is the single characteristic that most distinguishes one GLM family from another, and it directly determines the IRLS weights used to fit the model.

**Working response.** The pseudo-outcome `z = η + (y − μ)·(∂η/∂μ)` constructed at each IRLS iteration from a first-order Taylor expansion of the link function, on which a weighted least squares fit is performed to update the coefficients (Chapter 6.2). It is the mechanism that turns a nonlinear maximum-likelihood problem into a sequence of linear weighted-least-squares problems.

## 44. Further study

After mastering this guide, the natural next topics are:

- generalized additive models (GAMs), which extend the linear predictor to a sum of smooth, nonparametric functions of the predictors while keeping the GLM's family and link structure;
- mixed-effects and multilevel GLMs, adding random effects for grouped or hierarchical data;
- generalized estimating equations (GEE), an alternative to mixed models for correlated (e.g. longitudinal or clustered) data that relaxes the full-distribution assumption in favor of a working correlation structure;
- survival analysis and the Cox proportional hazards model, which shares deep mathematical structure with GLMs but centers on time-to-event outcomes and censoring;
- zero-inflated and hurdle models, for count data with more zeros than a Poisson or negative binomial model predicts;
- Bayesian GLMs, replacing maximum likelihood point estimates with full posterior distributions over β;
- Tweedie models and compound Poisson-gamma distributions, unifying Poisson and gamma regression for outcomes that are a mixture of exact zeros and positive continuous values (common in insurance claims data);
- double/debiased machine learning applied to GLM-family outcomes, extending the causal-inference cautions of Chapter 15 to modern high-dimensional settings.

Two of these — generalized additive models and mixed-effects models — are the most direct next steps mathematically, since both keep the exponential-family and link-function machinery of this guide intact and change only the structure of the linear predictor.

## 45. Suggested research papers

**Nelder, J. A., & Wedderburn, R. W. M. (1972). Generalized linear models. Journal of the Royal Statistical Society, Series A, 135(3), 370-384.** The founding paper of this entire guide, unifying linear, logistic, and Poisson regression under the exponential-family and IRLS framework developed in Chapters 2–6.

**Wedderburn, R. W. M. (1974). Quasi-likelihood functions, generalized linear models, and the Gauss-Newton method. Biometrika, 61(3), 439-447.** Introduces quasi-likelihood estimation (Chapter 9.2), relaxing the need for a fully specified distribution to just a mean-variance relationship.

**McCullagh, P., & Nelder, J. A. (1989). Generalized Linear Models (2nd ed.). Chapman and Hall.** The standard reference textbook for this entire guide; read it after this guide for a fully rigorous, comprehensive treatment of every family, link, and diagnostic covered here.

**Cameron, A. C., & Trivedi, P. K. (1986). Econometric models based on count data: Comparisons and applications of some estimators and tests. Journal of Applied Econometrics, 1(1), 29-53.** A foundational comparison of Poisson, quasi-Poisson, and negative binomial models for overdispersed count data (Chapter 9).

**Hilbe, J. M. (2011). Negative Binomial Regression (2nd ed.). Cambridge University Press.** A book-length treatment of negative binomial regression that goes well beyond the from-scratch sketch in Chapter 23 of this guide.

**Firth, D. (1993). Bias reduction of maximum likelihood estimates. Biometrika, 80(1), 27-38.** Addresses a known small-sample bias in maximum likelihood GLM estimates (especially severe for logistic regression with rare events or separation), relevant whenever the from-scratch IRLS solver in Chapter 18 is applied to a small or highly imbalanced dataset.

**Park, T., & Casella, G. (2008). The Bayesian lasso. Journal of the American Statistical Association, 103(482), 681-686.** A Bayesian perspective on the lasso penalty introduced for GLMs in Chapter 14, useful once the point-estimate regularization of this guide starts to feel restrictive.

**Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. Journal of Statistical Software, 33(1), 1-22.** The paper behind the `glmnet` algorithm, describing exactly the nested IRLS-plus-coordinate-descent structure implemented from scratch in Chapter 24.2 of this guide, at production scale.

**Smyth, G. K. (1989). Generalized linear models with varying dispersion. Journal of the Royal Statistical Society, Series B, 51(1), 47-60.** Extends the GLM framework to let the dispersion parameter itself vary with predictors, relevant when the constant-dispersion assumption of Chapter 9 does not hold.

**Jørgensen, B. (1987). Exponential dispersion models. Journal of the Royal Statistical Society, Series B, 49(2), 127-162.** A more general mathematical treatment of the exponential family (Chapter 2) that includes the Tweedie family (Chapter 32) as a special case, connecting Poisson and gamma regression into one continuous model class.

## 46. How to read a research paper

The three-pass strategy, reading-order recommendation, and evaluation questions from Chapter 43 of the Regression guide apply here without modification and are not repeated in full. This chapter adds only what is specific to reading GLM and statistical-modeling papers.

**Locate the three components before anything else.** Before attempting to follow a GLM paper's argument, identify its random component (which family, and is it standard or a novel extension), its systematic component (is the linear predictor standard, or does the paper propose a new structure such as a GAM or mixed-effects extension), and its link function. Papers that propose "a new model" are frequently proposing a new choice for exactly one of these three components while keeping the other two standard; identifying which one immediately clarifies what is actually new.

**Separate the estimation method from the model.** A paper may propose a new *model* (e.g. a new variance function) or a new *estimation method* for an existing model (e.g. a faster or more robust way to fit standard Poisson regression). Conflating these two contributions is a common source of confusion; ask explicitly, "if I already trust the model, do I need this paper's method, and vice versa?"

**Check what the paper assumes about dispersion.** Because overdispersion (Chapter 9) is the single most common practical failure mode in applied GLM work, check early whether a paper's asymptotic or simulation results assume correctly specified dispersion, and whether its claimed standard errors or confidence intervals would survive a dispersion check like the one in Chapter 35's checklist.

**Reproduce the working example on a toy dataset.** Because every GLM in this guide is fit by the same IRLS algorithm (Chapter 6), you can validate your understanding of almost any GLM paper's method by implementing its family-specific pieces (link, variance function, deviance) as a new family object plugged into the generic `fit_glm` solver from Appendix A, and confirming it reproduces a simple worked example from the paper.

---

# Appendix A. Full Pure-Python Implementation

The following module is complete, executable, and has been run end to end as part of writing this guide (see the demo output referenced in Chapter 26). Save it as `glm_from_scratch.py`. It has no dependencies beyond the Python standard library.

```python
"""Generalized linear models from scratch using only Python's standard library.

Educational implementation covering:
- a minimal linear algebra core (transpose, matmul, matvec, linear solve)
- a generic Iteratively Reweighted Least Squares (IRLS) solver
- Gaussian, Poisson, grouped-binomial, and Gamma GLM families
- deviance, Pearson residuals, deviance residuals, leverage, Cook's distance
- overdispersion (Pearson-based dispersion, quasi-likelihood rescaling)
- negative binomial regression via a golden-section search over the dispersion
- ridge- and lasso-penalized IRLS
- AIC/BIC (deviance-based) and a chi-squared-based likelihood ratio test

The emphasis is transparency rather than production-scale speed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, List, Sequence, Tuple, Optional, Dict, Any

Vector = List[float]
Matrix = List[List[float]]


# ---------------------------------------------------------------------------
# Minimal linear algebra core (same routines as the Regression guide)
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
    eye = identity(n_rows)
    inv_columns: Matrix = []
    for j in range(n_cols):
        column = solve_linear_system(a, [eye[i][j] for i in range(n_rows)])
        inv_columns.append(column)
    return transpose(inv_columns)


# ---------------------------------------------------------------------------
# Regularized incomplete gamma function and the chi-squared distribution
# ---------------------------------------------------------------------------

def _gamma_series(a: float, x: float, max_iterations: int = 200, epsilon: float = 3e-14) -> float:
    """Lower regularized incomplete gamma P(a, x) via series expansion (x < a + 1)."""
    if x <= 0.0:
        return 0.0
    gln = math.lgamma(a)
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(max_iterations):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * epsilon:
            break
    return total * math.exp(-x + a * math.log(x) - gln)


def _gamma_continued_fraction(a: float, x: float, max_iterations: int = 200, epsilon: float = 3e-14) -> float:
    """Upper regularized incomplete gamma Q(a, x) via continued fraction (x >= a + 1)."""
    tiny = 1e-300
    gln = math.lgamma(a)
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, max_iterations + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def regularized_gamma_p(a: float, x: float) -> float:
    if x < 0.0 or a <= 0.0:
        raise ValueError("a must be positive and x must be non-negative.")
    if x == 0.0:
        return 0.0
    if x < a + 1.0:
        return _gamma_series(a, x)
    return 1.0 - _gamma_continued_fraction(a, x)


def regularized_gamma_q(a: float, x: float) -> float:
    return 1.0 - regularized_gamma_p(a, x)


def chi2_sf(x: float, degrees_of_freedom: int) -> float:
    """Chi-squared survival function 1 - CDF, used for likelihood ratio tests."""
    if x < 0.0:
        return 1.0
    return regularized_gamma_q(degrees_of_freedom / 2.0, x / 2.0)


def likelihood_ratio_test(deviance_small: float, deviance_large: float, df_difference: int) -> Tuple[float, float]:
    statistic = deviance_small - deviance_large
    p_value = chi2_sf(statistic, df_difference)
    return statistic, p_value


# ---------------------------------------------------------------------------
# GLM families
# ---------------------------------------------------------------------------

def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


@dataclass
class Family:
    name: str
    link: Callable[[float], float]
    link_inverse: Callable[[float], float]
    dlink_dmu: Callable[[float], float]
    variance: Callable[[float], float]
    deviance_term: Callable[[float, float, float], float]
    init_mu: Callable[[Sequence[float], Sequence[float]], Vector]
    mu_min: float
    mu_max: float
    fixed_dispersion: bool  # True for Poisson/Binomial (dispersion assumed 1)


def _clamp_mu(value: float, family: Family) -> float:
    if value < family.mu_min:
        return family.mu_min
    if value > family.mu_max:
        return family.mu_max
    return value


def gaussian_family() -> Family:
    return Family(
        name="gaussian",
        link=lambda mu: mu,
        link_inverse=lambda eta: eta,
        dlink_dmu=lambda mu: 1.0,
        variance=lambda mu: 1.0,
        deviance_term=lambda y, mu, w: w * (y - mu) ** 2,
        init_mu=lambda y, w: [float(v) for v in y],
        mu_min=-1e300,
        mu_max=1e300,
        fixed_dispersion=False,
    )


def poisson_family() -> Family:
    def deviance_term(y: float, mu: float, w: float) -> float:
        if y <= 0.0:
            return 2.0 * w * mu
        return 2.0 * w * (y * math.log(y / mu) - (y - mu))

    return Family(
        name="poisson",
        link=lambda mu: math.log(mu),
        link_inverse=lambda eta: math.exp(min(eta, 700.0)),
        dlink_dmu=lambda mu: 1.0 / mu,
        variance=lambda mu: mu,
        deviance_term=deviance_term,
        init_mu=lambda y, w: [max(float(v), 0.0) + 0.1 for v in y],
        mu_min=1e-10,
        mu_max=1e300,
        fixed_dispersion=True,
    )


def binomial_family() -> Family:
    """Grouped binomial: y is the observed proportion, prior weight is the trial count."""

    def deviance_term(y: float, mu: float, w: float) -> float:
        term1 = y * math.log(y / mu) if y > 0.0 else 0.0
        term2 = (1.0 - y) * math.log((1.0 - y) / (1.0 - mu)) if y < 1.0 else 0.0
        return 2.0 * w * (term1 + term2)

    def init_mu(y: Sequence[float], w: Sequence[float]) -> Vector:
        return [(float(yi) * float(wi) + 0.5) / (float(wi) + 1.0) for yi, wi in zip(y, w)]

    return Family(
        name="binomial",
        link=lambda mu: math.log(mu / (1.0 - mu)),
        link_inverse=_sigmoid,
        dlink_dmu=lambda mu: 1.0 / (mu * (1.0 - mu)),
        variance=lambda mu: mu * (1.0 - mu),
        deviance_term=deviance_term,
        init_mu=init_mu,
        mu_min=1e-10,
        mu_max=1.0 - 1e-10,
        fixed_dispersion=True,
    )


def gamma_family() -> Family:
    def deviance_term(y: float, mu: float, w: float) -> float:
        return 2.0 * w * (-math.log(y / mu) + (y - mu) / mu)

    return Family(
        name="gamma",
        link=lambda mu: math.log(mu),
        link_inverse=lambda eta: math.exp(min(eta, 700.0)),
        dlink_dmu=lambda mu: 1.0 / mu,
        variance=lambda mu: mu ** 2,
        deviance_term=deviance_term,
        init_mu=lambda y, w: [max(float(v), 1e-6) for v in y],
        mu_min=1e-10,
        mu_max=1e300,
        fixed_dispersion=False,
    )


def negative_binomial_family(alpha: float) -> Family:
    """Negative binomial with fixed dispersion alpha; Var(mu) = mu + alpha * mu^2."""

    def deviance_term(y: float, mu: float, w: float) -> float:
        inv_alpha = 1.0 / alpha
        term1 = y * math.log(y / mu) if y > 0.0 else 0.0
        term2 = (y + inv_alpha) * math.log((y + inv_alpha) / (mu + inv_alpha))
        return 2.0 * w * (term1 - term2)

    return Family(
        name="negative_binomial",
        link=lambda mu: math.log(mu),
        link_inverse=lambda eta: math.exp(min(eta, 700.0)),
        dlink_dmu=lambda mu: 1.0 / mu,
        variance=lambda mu: mu + alpha * mu ** 2,
        deviance_term=deviance_term,
        init_mu=lambda y, w: [max(float(v), 0.0) + 0.1 for v in y],
        mu_min=1e-10,
        mu_max=1e300,
        fixed_dispersion=True,
    )


def _total_deviance(y: Sequence[float], mu: Sequence[float], pw: Sequence[float], family: Family) -> float:
    return sum(family.deviance_term(y[i], mu[i], pw[i]) for i in range(len(y)))


# ---------------------------------------------------------------------------
# Generic IRLS solver
# ---------------------------------------------------------------------------

@dataclass
class GLMResult:
    coefficients: Vector
    fitted_mu: Vector
    linear_predictor: Vector
    weights: Vector
    deviance: float
    null_deviance: float
    df_resid: int
    df_null: int
    n_observations: int
    n_parameters: int
    family: Family
    design_matrix: Matrix
    xtwx_inverse: Matrix
    covariance_matrix: Matrix
    standard_errors: Vector
    dispersion: float
    pearson_chi2: float
    prior_weights: Vector
    iterations: int

    def predict(self, x_new: Matrix, offset: Optional[Sequence[float]] = None, add_intercept: Optional[bool] = None) -> Vector:
        use_intercept = (len(self.design_matrix[0]) == len(x_new[0]) + 1) if add_intercept is None else add_intercept
        design = add_intercept_column(x_new) if use_intercept else [[float(v) for v in row] for row in x_new]
        off = [float(v) for v in offset] if offset is not None else [0.0] * len(design)
        eta = [dot(design[i], self.coefficients) + off[i] for i in range(len(design))]
        return [_clamp_mu(self.family.link_inverse(e), self.family) for e in eta]


def fit_glm(
    x: Matrix,
    y: Sequence[float],
    family: Family,
    prior_weights: Optional[Sequence[float]] = None,
    offset: Optional[Sequence[float]] = None,
    add_intercept: bool = True,
    max_iterations: int = 25,
    tolerance: float = 1e-8,
) -> GLMResult:
    n, p_raw = _validate_matrix(x)
    ys = [float(v) for v in y]
    if len(ys) != n:
        raise ValueError("X and y have incompatible dimensions.")
    pw = [float(v) for v in prior_weights] if prior_weights is not None else [1.0] * n
    off = [float(v) for v in offset] if offset is not None else [0.0] * n
    design = add_intercept_column(x) if add_intercept else [[float(v) for v in row] for row in x]
    _, p = _validate_matrix(design)
    if n <= p:
        raise ValueError("GLM fitting requires more observations than parameters.")

    mu = [_clamp_mu(m, family) for m in family.init_mu(ys, pw)]
    eta = [family.link(m) - off[i] for i, m in enumerate(mu)]
    coefficients = [0.0] * p
    deviance = _total_deviance(ys, mu, pw, family)
    weights_iter = [1.0] * n
    converged_at = max_iterations

    for iteration in range(1, max_iterations + 1):
        deta_dmu = [family.dlink_dmu(m) for m in mu]
        var = [family.variance(m) for m in mu]
        weights_iter = [
            pw[i] / (var[i] * deta_dmu[i] ** 2) if var[i] > 0.0 and deta_dmu[i] != 0.0 else 1e-12
            for i in range(n)
        ]
        working_response = [eta[i] + (ys[i] - mu[i]) * deta_dmu[i] for i in range(n)]

        sqrt_w = [math.sqrt(w) for w in weights_iter]
        design_w = [[sqrt_w[i] * design[i][j] for j in range(p)] for i in range(n)]
        z_w = [sqrt_w[i] * working_response[i] for i in range(n)]
        xt = transpose(design_w)
        xtwx = matmul(xt, design_w)
        xtwz = matvec(xt, z_w)
        coefficients = solve_linear_system(xtwx, xtwz)

        eta = [dot(design[i], coefficients) for i in range(n)]
        eta_total = [eta[i] + off[i] for i in range(n)]
        mu = [_clamp_mu(family.link_inverse(e), family) for e in eta_total]

        new_deviance = _total_deviance(ys, mu, pw, family)
        if abs(new_deviance - deviance) < tolerance * (abs(deviance) + tolerance):
            deviance = new_deviance
            converged_at = iteration
            break
        deviance = new_deviance
    else:
        raise RuntimeError("IRLS did not converge within max_iterations.")

    # Final weights and covariance matrix at convergence.
    deta_dmu = [family.dlink_dmu(m) for m in mu]
    var = [family.variance(m) for m in mu]
    weights_final = [
        pw[i] / (var[i] * deta_dmu[i] ** 2) if var[i] > 0.0 and deta_dmu[i] != 0.0 else 1e-12
        for i in range(n)
    ]
    sqrt_w = [math.sqrt(w) for w in weights_final]
    design_w = [[sqrt_w[i] * design[i][j] for j in range(p)] for i in range(n)]
    xtwx = matmul(transpose(design_w), design_w)
    xtwx_inv = inverse(xtwx)

    pearson_chi2 = sum(
        pw[i] * (ys[i] - mu[i]) ** 2 / var[i] for i in range(n) if var[i] > 0.0
    )
    df_resid = n - p
    dispersion = 1.0 if family.fixed_dispersion else pearson_chi2 / df_resid
    covariance_matrix = [[dispersion * v for v in row] for row in xtwx_inv]
    standard_errors = [math.sqrt(max(covariance_matrix[j][j], 0.0)) for j in range(p)]

    # Null (intercept-only) deviance for pseudo-R-squared style comparisons.
    null_design = [[1.0] for _ in range(n)]
    null_result_deviance = _fit_null_deviance(null_design, ys, family, pw, off)

    return GLMResult(
        coefficients=coefficients,
        fitted_mu=mu,
        linear_predictor=eta,
        weights=weights_final,
        deviance=deviance,
        null_deviance=null_result_deviance,
        df_resid=df_resid,
        df_null=n - 1,
        n_observations=n,
        n_parameters=p,
        family=family,
        design_matrix=design,
        xtwx_inverse=xtwx_inv,
        covariance_matrix=covariance_matrix,
        standard_errors=standard_errors,
        dispersion=dispersion,
        pearson_chi2=pearson_chi2,
        prior_weights=pw,
        iterations=converged_at,
    )


def _fit_null_deviance(
    null_design: Matrix, y: Sequence[float], family: Family, pw: Sequence[float], off: Sequence[float],
    max_iterations: int = 50, tolerance: float = 1e-10,
) -> float:
    """Fit an intercept-only GLM (one-column design already containing the ones column)."""
    n = len(y)
    mu = [_clamp_mu(m, family) for m in family.init_mu(y, pw)]
    eta = [family.link(m) - off[i] for i, m in enumerate(mu)]
    coefficients = [0.0]
    deviance = _total_deviance(y, mu, pw, family)
    for _ in range(max_iterations):
        deta_dmu = [family.dlink_dmu(m) for m in mu]
        var = [family.variance(m) for m in mu]
        w = [pw[i] / (var[i] * deta_dmu[i] ** 2) if var[i] > 0.0 else 1e-12 for i in range(n)]
        z = [eta[i] + (y[i] - mu[i]) * deta_dmu[i] for i in range(n)]
        weighted_sum_w = sum(w)
        weighted_sum_z = sum(w[i] * z[i] for i in range(n))
        coefficients = [weighted_sum_z / weighted_sum_w]
        eta = [coefficients[0] for _ in range(n)]
        eta_total = [eta[i] + off[i] for i in range(n)]
        mu = [_clamp_mu(family.link_inverse(e), family) for e in eta_total]
        new_deviance = _total_deviance(y, mu, pw, family)
        if abs(new_deviance - deviance) < tolerance * (abs(deviance) + tolerance):
            deviance = new_deviance
            break
        deviance = new_deviance
    return deviance


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def glm_diagnostics(result: GLMResult, y: Sequence[float]) -> List[Dict[str, float]]:
    design = result.design_matrix
    family = result.family
    mu = result.fitted_mu
    pw = result.prior_weights
    weights = result.weights
    xtwx_inv = result.xtwx_inverse
    p = result.n_parameters
    diagnostics = []
    for i, row in enumerate(design):
        sqrt_w = math.sqrt(weights[i])
        row_w = [sqrt_w * v for v in row]
        leverage = dot(row_w, matvec(xtwx_inv, row_w))
        var_mu = family.variance(mu[i])
        pearson_resid = (y[i] - mu[i]) / math.sqrt(var_mu / pw[i]) if var_mu > 0.0 and pw[i] > 0.0 else 0.0
        d = family.deviance_term(y[i], mu[i], pw[i])
        dev_resid = math.copysign(math.sqrt(max(d, 0.0)), y[i] - mu[i])
        one_minus_h = max(1.0 - leverage, 1e-12)
        std_dev_resid = dev_resid / math.sqrt(one_minus_h)
        cooks_distance = (pearson_resid ** 2 * leverage) / (p * one_minus_h ** 2)
        diagnostics.append({
            "index": float(i),
            "leverage": leverage,
            "pearson_residual": pearson_resid,
            "deviance_residual": dev_resid,
            "standardized_deviance_residual": std_dev_resid,
            "cooks_distance": cooks_distance,
        })
    return diagnostics


def pearson_dispersion(result: GLMResult) -> float:
    return result.pearson_chi2 / result.df_resid


def aic_bic(deviance: float, n_parameters: int, n_observations: int) -> Tuple[float, float]:
    """Deviance-based AIC/BIC. Valid for comparing nested models fit with the
    same family and link (the additive log-likelihood constant that deviance
    omits is identical across such models and cancels in any comparison)."""
    aic = deviance + 2.0 * n_parameters
    bic = deviance + n_parameters * math.log(n_observations)
    return aic, bic


# ---------------------------------------------------------------------------
# Negative binomial regression via a 1-D golden-section search over alpha
# ---------------------------------------------------------------------------

def negative_binomial_log_likelihood(y: Sequence[float], mu: Sequence[float], alpha: float) -> float:
    total = 0.0
    r = 1.0 / alpha
    for yi, mui in zip(y, mu):
        total += (
            math.lgamma(yi + r) - math.lgamma(r) - math.lgamma(yi + 1.0)
            + r * math.log(r / (r + mui))
            + yi * math.log(mui / (r + mui))
        )
    return total


def fit_negative_binomial(
    x: Matrix, y: Sequence[float], prior_weights: Optional[Sequence[float]] = None,
    offset: Optional[Sequence[float]] = None, add_intercept: bool = True,
    lower: float = 1e-3, upper: float = 5.0, tolerance: float = 1e-4,
) -> Tuple[GLMResult, float]:
    golden = (math.sqrt(5.0) - 1.0) / 2.0

    def neg_ll_for_alpha(alpha: float) -> Tuple[float, GLMResult]:
        family = negative_binomial_family(alpha)
        result = fit_glm(x, y, family, prior_weights=prior_weights, offset=offset, add_intercept=add_intercept)
        ll = negative_binomial_log_likelihood(y, result.fitted_mu, alpha)
        return -ll, result

    a, b = lower, upper
    c = b - golden * (b - a)
    d = a + golden * (b - a)
    fc, result_c = neg_ll_for_alpha(c)
    fd, result_d = neg_ll_for_alpha(d)
    while abs(b - a) > tolerance:
        if fc < fd:
            b, d, fd, result_d = d, c, fc, result_c
            c = b - golden * (b - a)
            fc, result_c = neg_ll_for_alpha(c)
        else:
            a, c, fc, result_c = c, d, fd, result_d
            d = a + golden * (b - a)
            fd, result_d = neg_ll_for_alpha(d)
    return (result_c, c) if fc < fd else (result_d, d)


# ---------------------------------------------------------------------------
# Regularized GLMs: ridge- and lasso-penalized IRLS
# ---------------------------------------------------------------------------

def fit_glm_ridge(
    x: Matrix, y: Sequence[float], family: Family, alpha: float = 1.0,
    prior_weights: Optional[Sequence[float]] = None, offset: Optional[Sequence[float]] = None,
    add_intercept: bool = True, max_iterations: int = 25, tolerance: float = 1e-8,
) -> Tuple[Vector, Vector, float]:
    n, _ = _validate_matrix(x)
    ys = [float(v) for v in y]
    pw = [float(v) for v in prior_weights] if prior_weights is not None else [1.0] * n
    off = [float(v) for v in offset] if offset is not None else [0.0] * n
    design = add_intercept_column(x) if add_intercept else [[float(v) for v in row] for row in x]
    _, p = _validate_matrix(design)

    mu = [_clamp_mu(m, family) for m in family.init_mu(ys, pw)]
    eta = [family.link(m) - off[i] for i, m in enumerate(mu)]
    coefficients = [0.0] * p
    deviance = _total_deviance(ys, mu, pw, family)

    for _ in range(max_iterations):
        deta_dmu = [family.dlink_dmu(m) for m in mu]
        var = [family.variance(m) for m in mu]
        w = [pw[i] / (var[i] * deta_dmu[i] ** 2) if var[i] > 0.0 else 1e-12 for i in range(n)]
        z = [eta[i] + (ys[i] - mu[i]) * deta_dmu[i] for i in range(n)]
        sqrt_w = [math.sqrt(wi) for wi in w]
        design_w = [[sqrt_w[i] * design[i][j] for j in range(p)] for i in range(n)]
        z_w = [sqrt_w[i] * z[i] for i in range(n)]
        xt = transpose(design_w)
        xtwx = matmul(xt, design_w)
        for j in range(1, p):
            xtwx[j][j] += alpha
        xtwz = matvec(xt, z_w)
        coefficients = solve_linear_system(xtwx, xtwz)

        eta = [dot(design[i], coefficients) for i in range(n)]
        eta_total = [eta[i] + off[i] for i in range(n)]
        mu = [_clamp_mu(family.link_inverse(e), family) for e in eta_total]
        new_deviance = _total_deviance(ys, mu, pw, family)
        if abs(new_deviance - deviance) < tolerance * (abs(deviance) + tolerance):
            deviance = new_deviance
            break
        deviance = new_deviance

    return coefficients, mu, deviance


def _soft_threshold(value: float, threshold: float) -> float:
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


def fit_glm_lasso(
    x: Matrix, y: Sequence[float], family: Family, alpha: float = 0.1, l1_ratio: float = 1.0,
    prior_weights: Optional[Sequence[float]] = None, offset: Optional[Sequence[float]] = None,
    add_intercept: bool = True, max_outer_iterations: int = 25, max_inner_iterations: int = 500,
    outer_tolerance: float = 1e-8, inner_tolerance: float = 1e-8,
) -> Tuple[Vector, Vector, float]:
    n, _ = _validate_matrix(x)
    ys = [float(v) for v in y]
    pw = [float(v) for v in prior_weights] if prior_weights is not None else [1.0] * n
    off = [float(v) for v in offset] if offset is not None else [0.0] * n
    design = add_intercept_column(x) if add_intercept else [[float(v) for v in row] for row in x]
    _, p = _validate_matrix(design)

    mu = [_clamp_mu(m, family) for m in family.init_mu(ys, pw)]
    eta = [family.link(m) - off[i] for i, m in enumerate(mu)]
    coefficients = [0.0] * p
    deviance = _total_deviance(ys, mu, pw, family)
    l1_penalty = alpha * l1_ratio
    l2_penalty = alpha * (1.0 - l1_ratio)

    for _ in range(max_outer_iterations):
        deta_dmu = [family.dlink_dmu(m) for m in mu]
        var = [family.variance(m) for m in mu]
        w = [pw[i] / (var[i] * deta_dmu[i] ** 2) if var[i] > 0.0 else 1e-12 for i in range(n)]
        z = [eta[i] + (ys[i] - mu[i]) * deta_dmu[i] for i in range(n)]

        predictions = [dot(design[i], coefficients) for i in range(n)]
        for _ in range(max_inner_iterations):
            max_change = 0.0
            for j in range(p):
                old = coefficients[j]
                col_j = [design[i][j] for i in range(n)]
                weighted_col_sq = sum(w[i] * col_j[i] * col_j[i] for i in range(n))
                if weighted_col_sq < 1e-12:
                    continue
                partial_residual = [z[i] - predictions[i] + col_j[i] * old for i in range(n)]
                rho = sum(w[i] * col_j[i] * partial_residual[i] for i in range(n))
                if j == 0:
                    new_value = rho / weighted_col_sq
                else:
                    new_value = _soft_threshold(rho, l1_penalty) / (weighted_col_sq + l2_penalty)
                change = new_value - old
                if change != 0.0:
                    coefficients[j] = new_value
                    for i in range(n):
                        predictions[i] += col_j[i] * change
                max_change = max(max_change, abs(change))
            if max_change < inner_tolerance:
                break

        eta = [dot(design[i], coefficients) for i in range(n)]
        eta_total = [eta[i] + off[i] for i in range(n)]
        mu = [_clamp_mu(family.link_inverse(e), family) for e in eta_total]
        new_deviance = _total_deviance(ys, mu, pw, family)
        if abs(new_deviance - deviance) < outer_tolerance * (abs(deviance) + outer_tolerance):
            deviance = new_deviance
            break
        deviance = new_deviance

    return coefficients, mu, deviance


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def demo() -> None:
    random_seed = 7
    import random
    rng = random.Random(random_seed)

    # --- Poisson regression with an offset ---
    n = 400
    experience = [rng.uniform(0, 20) for _ in range(n)]
    training = [float(rng.randint(0, 1)) for _ in range(n)]
    exposure = [rng.uniform(0.5, 2.0) for _ in range(n)]
    true_rate = [math.exp(-1.0 + 0.05 * experience[i] + 0.6 * training[i]) for i in range(n)]
    counts = [float(rng.gammavariate(2.0, (true_rate[i] * exposure[i]) / 2.0) ) for i in range(n)]
    counts = [round(c) for c in counts]  # approximate Poisson-like counts via a noisy generator
    x_poisson = [[experience[i], training[i]] for i in range(n)]
    offset = [math.log(e) for e in exposure]

    poisson_result = fit_glm(x_poisson, counts, poisson_family(), offset=offset)
    print("Poisson coefficients (intercept, experience, training):",
          [round(c, 4) for c in poisson_result.coefficients])
    print("Poisson deviance:", round(poisson_result.deviance, 3),
          "df_resid:", poisson_result.df_resid)
    dispersion = pearson_dispersion(poisson_result)
    print("Pearson dispersion (near 1 => no strong overdispersion):", round(dispersion, 3))

    aic, bic = aic_bic(poisson_result.deviance, poisson_result.n_parameters, poisson_result.n_observations)
    print("Poisson AIC (deviance-based):", round(aic, 2), "BIC:", round(bic, 2))

    # --- Ridge- and lasso-penalized refits ---
    ridge_coef, _, ridge_dev = fit_glm_ridge(x_poisson, counts, poisson_family(), alpha=2.0, offset=offset)
    lasso_coef, _, lasso_dev = fit_glm_lasso(x_poisson, counts, poisson_family(), alpha=0.05, l1_ratio=1.0, offset=offset)
    print("Ridge-penalized coefficients:", [round(c, 4) for c in ridge_coef])
    print("Lasso-penalized coefficients:", [round(c, 4) for c in lasso_coef])

    # --- Diagnostics ---
    diags = glm_diagnostics(poisson_result, counts)
    worst = max(diags, key=lambda d: d["cooks_distance"])
    print("Largest Cook's distance observation:", int(worst["index"]),
          "value:", round(worst["cooks_distance"], 5))

    # --- Grouped binomial regression, cross-checked against Bernoulli logistic regression ---
    n_bin = 300
    age = [rng.uniform(18, 70) for _ in range(n_bin)]
    score = [-4.0 + 0.06 * a for a in age]
    prob = [_sigmoid(s) for s in score]
    successes = [1.0 if rng.random() < p else 0.0 for p in prob]
    x_bin = [[age[i]] for i in range(n_bin)]
    binomial_result = fit_glm(x_bin, successes, binomial_family(), prior_weights=[1.0] * n_bin)
    print("Grouped-binomial (n=1 trials) coefficients:", [round(c, 4) for c in binomial_result.coefficients])

    # --- Negative binomial fit on the same (possibly overdispersed) counts ---
    nb_result, alpha_hat = fit_negative_binomial(x_poisson, counts, offset=offset)
    print("Negative binomial alpha:", round(alpha_hat, 4))
    print("Negative binomial coefficients:", [round(c, 4) for c in nb_result.coefficients])

    # --- Likelihood ratio test: full Poisson model vs. intercept-only ---
    lr_stat, lr_p = likelihood_ratio_test(poisson_result.null_deviance, poisson_result.deviance,
                                            df_difference=poisson_result.n_parameters - 1)
    print("Likelihood ratio test statistic:", round(lr_stat, 3), "p-value:", round(lr_p, 6))


if __name__ == "__main__":
    demo()
```

# Appendix B. Full Library-Based Demonstration

The following script generates synthetic data and demonstrates statsmodels and scikit-learn GLM workflows for Poisson, negative binomial, grouped binomial, and gamma regression, plus regularized fits. Save it as `glm_with_libraries.py`. Run with:

```text
python glm_with_libraries.py
```

```python
"""Practical GLM workflows with statsmodels and scikit-learn.

The examples are self-contained and generate their own synthetic data.
Run with:
    python glm_with_libraries.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import NegativeBinomial

from sklearn.linear_model import PoissonRegressor, GammaRegressor, TweedieRegressor
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_poisson_deviance, mean_absolute_error


def make_poisson_data(seed: int = 42, n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    experience = rng.uniform(0, 20, n)
    training = rng.binomial(1, 0.4, n)
    exposure = rng.uniform(0.5, 2.0, n)
    rate = np.exp(-1.0 + 0.05 * experience + 0.6 * training)
    claims = rng.poisson(rate * exposure)
    return pd.DataFrame(
        {"claims": claims, "experience": experience, "training": training, "exposure": exposure}
    )


def make_overdispersed_poisson_data(seed: int = 7, n: int = 500) -> pd.DataFrame:
    """Counts generated with extra unobserved heterogeneity, so they are overdispersed
    relative to a plain Poisson model (Chapter 9)."""
    rng = np.random.default_rng(seed)
    experience = rng.uniform(0, 20, n)
    base_rate = np.exp(-0.5 + 0.04 * experience)
    heterogeneity = rng.gamma(shape=2.0, scale=0.5, size=n)  # mean 1, extra variance
    claims = rng.poisson(base_rate * heterogeneity)
    return pd.DataFrame({"claims": claims, "experience": experience})


def make_binomial_data(seed: int = 11, n_groups: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    batch_size = rng.integers(20, 80, n_groups)
    temperature = rng.uniform(150, 250, n_groups)
    logit_p = -6.0 + 0.03 * temperature
    p = 1.0 / (1.0 + np.exp(-logit_p))
    defects = rng.binomial(batch_size, p)
    return pd.DataFrame(
        {"defects": defects, "inspected": batch_size, "good": batch_size - defects, "temperature": temperature}
    )


def make_gamma_data(seed: int = 5, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    severity_score = rng.uniform(0, 10, n)
    region = rng.choice(["North", "South"], size=n)
    region_effect = pd.Series(region).map({"North": 0.1, "South": -0.1}).to_numpy()
    mean_claim = np.exp(6.0 + 0.15 * severity_score + region_effect)
    shape = 3.0
    claim_amount = rng.gamma(shape, mean_claim / shape)
    return pd.DataFrame({"claim_amount": claim_amount, "severity_score": severity_score, "region": region})


def poisson_workflow(df: pd.DataFrame):
    X = sm.add_constant(df[["experience", "training"]])
    model = sm.GLM(
        df["claims"], X, family=sm.families.Poisson(), offset=np.log(df["exposure"])
    ).fit()
    print("\nPoisson GLM summary:")
    print(model.summary())

    dispersion = model.pearson_chi2 / model.df_resid
    print("\nPearson dispersion (near 1 => no strong overdispersion):", round(dispersion, 3))

    quasi_poisson = sm.GLM(
        df["claims"], X, family=sm.families.Poisson(), offset=np.log(df["exposure"])
    ).fit(scale=dispersion)
    print("\nQuasi-Poisson standard errors (rescaled):")
    print(quasi_poisson.bse)
    return model


def negative_binomial_workflow(df: pd.DataFrame):
    X = sm.add_constant(df[["experience"]])
    nb_model = NegativeBinomial(df["claims"], X, loglike_method="nb2").fit(disp=False)
    print("\nNegative binomial coefficients:")
    print(nb_model.params)
    print("Estimated alpha:", round(nb_model.params["alpha"], 4))

    poisson_model = sm.GLM(df["claims"], X, family=sm.families.Poisson()).fit()
    print("\nAIC comparison (lower is better, same-family caution does not apply across families,")
    print("but Poisson-vs-NB comparison via AIC is standard practice for this specific pair):")
    print("Poisson AIC:", round(poisson_model.aic, 2), " Negative binomial AIC:", round(nb_model.aic, 2))
    return nb_model


def binomial_workflow(df: pd.DataFrame):
    X = sm.add_constant(df[["temperature"]])
    endog = df[["defects", "good"]]  # grouped binomial: successes, failures (Chapter 10.1, 30)
    model = sm.GLM(endog, X, family=sm.families.Binomial()).fit()
    print("\nGrouped binomial GLM summary:")
    print(model.summary())
    return model


def gamma_workflow(df: pd.DataFrame):
    model = smf.glm(
        "claim_amount ~ severity_score + C(region)",
        data=df,
        family=sm.families.Gamma(link=sm.families.links.Log()),
    ).fit()
    print("\nGamma GLM (log link) summary:")
    print(model.summary())
    print("Estimated dispersion:", round(model.scale, 4))
    return model


def sklearn_poisson_pipeline(df: pd.DataFrame):
    X = df[["experience", "training"]]
    y = df["claims"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("model", PoissonRegressor(alpha=1e-3)),
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    print("\nscikit-learn PoissonRegressor test mean Poisson deviance:",
          round(mean_poisson_deviance(y_test, np.clip(predictions, 1e-6, None)), 4))
    print("Test mean absolute error:", round(mean_absolute_error(y_test, predictions), 4))

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        pipeline, X, y, cv=cv,
        scoring="neg_mean_poisson_deviance",
        return_train_score=True,
    )
    print("Cross-validated mean Poisson deviance:", round(-scores["test_score"].mean(), 4))
    return pipeline


def sklearn_gamma_and_tweedie(df: pd.DataFrame):
    X = pd.get_dummies(df[["severity_score", "region"]], drop_first=True)
    y = df["claim_amount"]

    gamma_pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("model", GammaRegressor(alpha=1e-3)),
    ])
    gamma_pipeline.fit(X, y)
    print("\nscikit-learn GammaRegressor coefficients:", gamma_pipeline.named_steps["model"].coef_)

    for power in (0, 1, 1.5, 2, 3):
        tweedie = TweedieRegressor(power=power, link="log", alpha=1e-3, max_iter=2000)
        tweedie.fit(X, y)
        print(f"TweedieRegressor(power={power}) intercept:", round(tweedie.intercept_, 4))
    return gamma_pipeline


def regularized_glm_comparison(df: pd.DataFrame):
    X = sm.add_constant(df[["experience", "training"]])
    unpenalized = sm.GLM(df["claims"], X, family=sm.families.Poisson()).fit()

    for l1_wt in (0.0, 0.5, 1.0):
        penalized = sm.GLM(df["claims"], X, family=sm.families.Poisson()).fit_regularized(
            alpha=0.05, L1_wt=l1_wt, maxiter=200
        )
        print(f"\nfit_regularized(L1_wt={l1_wt}) coefficients:")
        print(penalized.params)

    print("\nUnpenalized coefficients for reference:")
    print(unpenalized.params)


def main() -> None:
    poisson_df = make_poisson_data()
    poisson_workflow(poisson_df)
    regularized_glm_comparison(poisson_df)
    sklearn_poisson_pipeline(poisson_df)

    overdispersed_df = make_overdispersed_poisson_data()
    negative_binomial_workflow(overdispersed_df)

    binomial_df = make_binomial_data()
    binomial_workflow(binomial_df)

    gamma_df = make_gamma_data()
    gamma_workflow(gamma_df)
    sklearn_gamma_and_tweedie(gamma_df)


if __name__ == "__main__":
    main()
```

# Appendix C. Minimal Verification Tests

```python
from glm_from_scratch import (
    fit_glm,
    gaussian_family,
    poisson_family,
    binomial_family,
    gamma_family,
    chi2_sf,
    regularized_gamma_p,
    likelihood_ratio_test,
    fit_glm_ridge,
    fit_glm_lasso,
    glm_diagnostics,
    aic_bic,
)
import math


def close(a, b, tolerance=1e-6):
    return abs(a - b) <= tolerance


# Gaussian GLM with the identity link must reproduce OLS exactly.
x = [[1.0], [2.0], [3.0], [4.0], [5.0]]
y = [3.0, 5.0, 7.0, 9.0, 11.0]  # y = 1 + 2x exactly
gaussian_result = fit_glm(x, y, gaussian_family())
assert close(gaussian_result.coefficients[0], 1.0, 1e-4)
assert close(gaussian_result.coefficients[1], 2.0, 1e-4)
assert close(gaussian_result.deviance, 0.0, 1e-6)

# Poisson regression recovers a known log-linear relationship closely on noiseless data.
x_p = [[float(v)] for v in range(1, 11)]
y_p = [math.exp(0.3 + 0.2 * v[0]) for v in x_p]
poisson_result = fit_glm(x_p, y_p, poisson_family())
assert close(poisson_result.coefficients[0], 0.3, 1e-2)
assert close(poisson_result.coefficients[1], 0.2, 1e-2)
assert poisson_result.deviance < 1e-4

# Grouped binomial with all trial counts equal to 1 (noisy, non-separable, to avoid
# the classic logistic-regression perfect-separation non-convergence failure mode).
import random
rng = random.Random(3)
x_b = [[float(v)] for v in range(-5, 6)] * 6
true_beta = [0.2, 0.5]
y_b = []
for row in x_b:
    eta = true_beta[0] + true_beta[1] * row[0]
    mu = 1.0 / (1.0 + math.exp(-eta))
    y_b.append(1.0 if rng.random() < mu else 0.0)
binomial_result = fit_glm(x_b, y_b, binomial_family())
predicted = binomial_result.predict(x_b)
assert all(0.0 <= p <= 1.0 for p in predicted)

# Gamma regression on noiseless log-linear positive data.
x_g = [[float(v)] for v in range(1, 11)]
y_g = [math.exp(1.0 + 0.15 * v[0]) for v in x_g]
gamma_result = fit_glm(x_g, y_g, gamma_family())
assert close(gamma_result.coefficients[0], 1.0, 1e-2)
assert close(gamma_result.coefficients[1], 0.15, 1e-2)

# Chi-squared survival function and regularized incomplete gamma agree with known values.
assert close(chi2_sf(0.0, 5), 1.0, 1e-9)
assert close(regularized_gamma_p(1.0, 0.0), 0.0, 1e-9)
assert close(regularized_gamma_p(1.0, 1e6), 1.0, 1e-6)
# For 1 degree of freedom, chi2_sf(3.841, 1) is approximately 0.05 (a well-known table value).
assert abs(chi2_sf(3.841, 1) - 0.05) < 1e-3

# Likelihood ratio test on Poisson null vs. full deviance is non-negative and well-formed.
stat, p_value = likelihood_ratio_test(poisson_result.null_deviance, poisson_result.deviance, 1)
assert stat >= 0.0
assert 0.0 <= p_value <= 1.0

# Ridge- and lasso-penalized IRLS produce valid, finite coefficient vectors.
ridge_coef, _, ridge_dev = fit_glm_ridge(x_p, y_p, poisson_family(), alpha=1.0)
lasso_coef, _, lasso_dev = fit_glm_lasso(x_p, y_p, poisson_family(), alpha=0.01)
assert all(math.isfinite(c) for c in ridge_coef)
assert all(math.isfinite(c) for c in lasso_coef)
assert ridge_dev >= 0.0
assert lasso_dev >= 0.0

# Diagnostics return one row per observation with non-negative Cook's distance.
diagnostics = glm_diagnostics(poisson_result, y_p)
assert len(diagnostics) == len(y_p)
assert all(d["cooks_distance"] >= 0.0 for d in diagnostics)

# AIC/BIC are finite and BIC penalizes at least as much as AIC for n >= 8.
aic, bic = aic_bic(poisson_result.deviance, poisson_result.n_parameters, poisson_result.n_observations)
assert math.isfinite(aic) and math.isfinite(bic)
assert bic >= aic

print("All verification tests passed.")
```

---

# Final perspective

Generalized linear models are best understood not as four separate techniques to memorize, but as one idea — a linear predictor, connected through a link function to the mean of a distribution whose variance is allowed to depend on that mean — applied four times. Every equation in Part I, from the exponential family's canonical form through IRLS to deviance, was written once and specialized, not rederived, for Poisson, binomial, and gamma outcomes. If you finish this guide able to write down a new family's link, variance function, and deviance formula and plug it into the same `fit_glm` solver from Appendix A, you have understood the actual content of the subject, not just its three most common examples.

A strong analyst carries the same four-layer discipline from the Regression guide forward: what question and decision matter; what relationship is being assumed and optimized; what uncertainty and identification assumptions support the conclusion; and how data transformations, fitting, diagnostics, and validation are implemented without leakage or numerical error — now with one addition specific to this guide, stated in Chapter 15: know which scale (link or mean) every reported number lives on, and translate deliberately between them before drawing a conclusion.
