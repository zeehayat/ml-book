# Agent Instructions: Frame and Complete the Next Topic in the Zero-to-Research ML Curriculum

## 1. Assignment

Create the next standalone volume after **Regression: A Mathematical, Statistical, and Python Guide**.

The required next topic is:

> **Generalized Linear Models (GLMs): from first principles to research-level practice**

The volume must be suitable for upload to NotebookLM and for guided self-study by a reader who begins with only elementary mathematics, introductory statistics, and basic Python.

Do not write this as a short chapter, overview, or collection of code recipes. Write it as a self-contained technical book that develops the reader from zero knowledge to the ability to read, reproduce, critique, and extend research using GLMs.

## 2. Required deliverables

Produce the following files:

1. `Generalized_Linear_Models_Mathematical_Statistical_and_Python_Guide.docx`
2. `glm_from_scratch.py`
3. `test_glm_from_scratch.py`
4. `glm_with_libraries.py`
5. `GLM_DATA_AND_PROJECTS.md`
6. A source bibliography in either BibTeX or Markdown with DOI or publisher links.

The DOCX is the primary teaching source. The Python files must be executable companions, not pseudocode.

## 3. Audience and pedagogical assumptions

Assume the learner:

- may not be fluent in algebraic notation;
- may know only basic probability;
- may not understand likelihood, score functions, Hessians, or matrix calculus;
- can read simple Python but may not understand numerical algorithms;
- has studied introductory linear regression but may need revision;
- intends eventually to read research papers and conduct publishable analysis.

Every advanced idea must therefore be introduced in layers:

1. intuitive purpose;
2. verbal definition;
3. small numerical example;
4. mathematical statement;
5. derivation;
6. pure-Python implementation;
7. library implementation;
8. diagnostics and failure modes;
9. research interpretation;
10. exercises and paper-reading connection.

Never assume that displaying a formula constitutes an explanation.

## 4. Non-negotiable structure

The book must follow this order.

### Part 0: Prerequisites rebuilt from scratch

Cover:

- random variables, PMFs, PDFs, expectation, variance, covariance;
- conditional distributions and conditional means;
- Bernoulli, binomial, Poisson, Gaussian, Gamma, inverse-Gaussian, and Tweedie intuition;
- logarithms, exponentials, odds, and log odds;
- derivatives, chain rule, gradients, Hessians, and curvature;
- vectors, matrices, dimensions, transposes, weighted cross-products, rank, and condition;
- floating-point arithmetic and numerical stability;
- basic Python functions, classes, validation, exceptions, testing, and reproducibility.

Include mastery checks before proceeding.

### Part I: Mathematical and statistical foundations

Cover in full:

- why ordinary linear regression is insufficient for constrained and heteroskedastic outcomes;
- the random, systematic, and link components;
- exponential-family and exponential-dispersion-family notation;
- canonical parameter and cumulant function;
- derivation of mean and variance from the cumulant function;
- canonical and noncanonical links;
- likelihood and log-likelihood;
- score, Hessian, observed information, and Fisher information;
- Newton-Raphson, Fisher scoring, and IRLS;
- full IRLS derivation, including working response and weights;
- model-based coefficient covariance;
- Wald, score, and likelihood-ratio tests;
- confidence intervals on coefficient and response scales;
- deviance, saturated models, null deviance, AIC, and BIC;
- dispersion and overdispersion;
- response, Pearson, deviance, working, Anscombe, and randomized quantile residuals;
- leverage, influence, case deletion, and local weighted geometry;
- offsets, exposure, frequency weights, analytic weights, and sampling weights;
- prediction of means versus prediction of new outcomes;
- sandwich covariance and its scope.

Every major equation must include dimensions and a derivation or derivation sketch.

### Part II: Major GLM families

Required chapters:

- Gaussian identity GLM as the bridge from OLS;
- Bernoulli and grouped-binomial logistic regression;
- probability, odds, odds ratios, marginal effects, and average marginal effects;
- complete and quasi-complete separation;
- rare events and small-sample bias;
- Poisson counts, rates, and exposure offsets;
- overdispersion and quasi-Poisson reasoning;
- negative-binomial models with explicit parameterization;
- zero-inflated and hurdle models;
- Gamma regression;
- inverse-Gaussian regression;
- Tweedie models and compound Poisson-Gamma interpretation;
- multinomial and ordinal extensions;
- alternative links: probit, complementary log-log, identity, inverse;
- family and link selection based on mechanism, support, diagnostics, and validation.

For every family, state:

- response support;
- PMF or PDF;
- mean and variance;
- common link;
- coefficient interpretation;
- log-likelihood;
- deviance contribution;
- typical diagnostics;
- common misuse;
- suitable research examples.

### Part III: From-scratch pure Python

Use only the Python standard library. No NumPy, pandas, SciPy, statsmodels, scikit-learn, or plotting packages in this part.

The implementation must include:

- matrix validation, transpose, dot product, matrix multiplication, matrix-vector multiplication;
- partial-pivot linear-system solver;
- matrix inverse for teaching and covariance calculation;
- weighted cross-products without constructing a full diagonal matrix;
- stable sigmoid, safe logarithm, normal CDF, and normal quantile;
- family objects or classes for Gaussian identity, binomial logit, Poisson log, and Gamma log;
- a generic IRLS or Fisher-scoring engine;
- offsets and prior weights;
- convergence criteria and objective history;
- coefficient covariance, standard errors, Wald z statistics, p-values, and intervals;
- fitted means, linear predictors, and multiple residual types;
- deviance, null deviance, log likelihood, AIC, and BIC;
- leverage and Cook-style influence;
- HC0, HC1, and HC3-style sandwich covariance;
- prediction methods;
- train/test splitting and k-fold validation;
- family-appropriate metrics;
- synthetic-data generators;
- verification tests.

The code must be executable and typed where useful. Validate response domains. Never silently return a model after nonconvergence.

### Part IV: Libraries

Use current official APIs and verify them against official documentation.

Required libraries:

- NumPy for array and linear-algebra demonstrations;
- SciPy for distributions and optimization;
- pandas for data auditing and feature construction;
- statsmodels for inference, family/link specification, robust covariance, GEE, and predictions;
- scikit-learn for pipelines, regularization, cross-validation, calibration, PoissonRegressor, GammaRegressor, TweedieRegressor, and LogisticRegression;
- matplotlib for diagnostics and explanatory plots.

Explain the division of labour among libraries. Explicitly warn that scikit-learn logistic regression is regularized by default and is not automatically equivalent to unpenalized maximum likelihood.

All preprocessing must be fitted within training folds. Include random, grouped, temporal, and spatial validation guidance.

### Part V: Research-level extensions

Cover:

- full distributional modelling versus mean estimating equations;
- GEE and population-average effects;
- GLMMs and cluster-specific effects;
- Bayesian GLMs and prior predictive checking;
- Firth bias reduction and penalized likelihood;
- robust GLMs;
- missing data and multiple imputation;
- survey, case-control, and outcome-dependent sampling;
- non-collapsibility;
- causal standardization, marginal risks, and average effects;
- high-dimensional GLMs and post-selection inference;
- external validation, calibration drift, and transportability;
- simulation-study design;
- reproducible research and transparent reporting;
- unresolved and active research questions.

Distinguish carefully between prediction, conditional association, marginal association, and causal effect.

### Part VI: Learning and research resources

Must contain:

- conceptual exercises;
- derivation exercises;
- pure-Python exercises;
- library exercises;
- debugging exercises;
- simulation projects;
- at least three complete research-project specifications;
- NotebookLM and Socratic tutor prompts;
- a staged learning plan;
- a comprehensive glossary;
- a detailed method for reading research papers;
- an annotated list of foundational and modern papers.

## 5. Required from-scratch derivations

At minimum, derive step by step:

1. Bernoulli log-likelihood.
2. Logistic score `X^T(y-p)`.
3. Logistic expected information `X^T W X`.
4. Poisson log-likelihood with a log link.
5. Poisson score.
6. General GLM score using the chain rule.
7. IRLS working response and weights.
8. Model-based covariance.
9. Binomial deviance.
10. Poisson deviance.
11. Delta-method uncertainty for a predicted mean.
12. Logistic marginal effects.
13. Standardized risk difference or risk ratio.
14. Sandwich covariance from estimating equations.

Do not skip algebra with phrases such as “after simplification” unless the omitted steps have already been demonstrated.

## 6. Code quality requirements

The pure-Python module must:

- run on current CPython without third-party packages;
- include docstrings and meaningful error messages;
- use deterministic seeds in demonstrations;
- avoid exact floating-point equality in tests;
- guard against `log(0)`, exponential overflow, singular matrices, invalid response support, and zero or negative exposure;
- expose convergence status and number of iterations;
- make the family and link explicit in result objects;
- separate model computation from display;
- include at least one independent verification against statsmodels in the test suite when libraries are available;
- pass all tests before delivery.

The library script must use pipelines, avoid data leakage, and report package versions.

## 7. Statistical quality requirements

For every modelling recommendation, state the assumption it depends on.

Do not make these errors:

- do not equate an odds ratio with a probability ratio;
- do not call Poisson regression appropriate merely because the outcome is a count;
- do not treat overdispersion correction as a complete model repair;
- do not infer a structural-zero class from excess zeros alone;
- do not compare quasi-likelihood models using ordinary AIC;
- do not treat convergence as evidence of correct specification;
- do not use observation-level resampling for clustered data;
- do not treat sampling weights as ordinary precision weights;
- do not attach classical p-values after data-driven penalized selection;
- do not make causal claims without identification assumptions.

## 8. Glossary standard

The glossary must not be a list of one-line definitions.

Each entry should include:

- plain-language meaning;
- mathematical meaning where relevant;
- role in GLMs;
- a common misconception or misuse;
- a cross-reference to the relevant chapter.

Include at least 60 terms, including: exponential family, natural parameter, cumulant, variance function, canonical link, score, Hessian, Fisher information, IRLS, working response, deviance, dispersion, quasi-likelihood, offset, exposure, separation, non-collapsibility, calibration, GEE, GLMM, sandwich covariance, randomized quantile residual, Tweedie, hurdle model, and zero inflation.

## 9. Research-paper section

Include foundational and modern papers. Verify bibliographic details using publisher pages, DOI records, or original PDFs.

Minimum required papers:

- Nelder and Wedderburn (1972), generalized linear models;
- Wedderburn (1974), quasi-likelihood;
- Wedderburn (1976), existence and uniqueness;
- Pregibon (1981), logistic diagnostics;
- Liang and Zeger (1986), GEE;
- Firth (1993), bias reduction;
- Dunn and Smyth (1996), randomized quantile residuals;
- Lambert (1992), zero-inflated Poisson;
- McCullagh (1980), ordinal regression;
- Breslow and Clayton (1993), GLMM approximation;
- Gneiting and Raftery (2007), proper scoring rules;
- a modern paper on penalized or high-dimensional GLMs;
- a modern paper on calibration or external validation;
- a modern paper on Bayesian GLMs or weakly informative priors.

For every paper, provide:

- complete citation and DOI;
- why it matters;
- prerequisites;
- what sections to read first;
- one equation or idea to reconstruct;
- one result to reproduce;
- one limitation or open question.

## 10. How to teach paper reading

Include a seven-pass method:

1. Outcome support and scientific question.
2. Estimand and interpretation scale.
3. Data-generation and sampling process.
4. Family, link, linear predictor, offset, weights, and dependence.
5. Estimation, uncertainty, and computation.
6. Diagnostics, calibration, validation, and sensitivity.
7. Reproduction, critique, and extension.

Provide a reusable paper-reading worksheet.

## 11. Research and source policy

Use primary sources whenever possible:

- original journal articles;
- official package documentation;
- official textbooks or monographs;
- authoritative statistical organization material.

Do not use unsourced blogs as authority for mathematical or statistical claims. A blog may be used only as optional intuition and must not replace a primary citation.

Mark any statement that is a recommendation rather than a theorem. Distinguish consensus practice from debated practice.

## 12. NotebookLM optimization

The DOCX must be easy for a retrieval-based tutor to consume:

- use descriptive headings;
- define notation locally before use;
- avoid references such as “as stated above” when a precise chapter reference is possible;
- repeat crucial assumptions in the chapter where they matter;
- keep code in complete, labelled blocks;
- include chapter summaries and mastery questions;
- include a glossary and explicit cross-references;
- use consistent terminology and symbols;
- put research citations in a clearly structured section.

## 13. Formatting requirements

Use a clean academic format:

- title and subtitle page;
- heading hierarchy with Word styles;
- page numbers;
- readable body font;
- monospace code style;
- equations displayed on separate centred lines;
- tables that repeat header rows when split;
- no clipped code or formulas;
- no citation-tool tokens or raw internal references;
- conventional ASCII punctuation where possible.

Render the DOCX to page images, inspect every page, correct layout defects, and render again before delivery.

## 14. Validation and acceptance tests

The work is accepted only if all of the following are true:

- the book is a separate file from the regression volume;
- prerequisite mathematics and Python are present;
- pure Python appears before third-party libraries;
- the generic IRLS implementation runs;
- Gaussian, logistic, Poisson, and Gamma examples converge on test data;
- results agree with an independent library within stated tolerance;
- invalid response values raise errors;
- exposure offsets are demonstrated correctly;
- separation and overdispersion are explained as failure modes;
- diagnostics, calibration, and validation are included;
- research extensions are substantive rather than a list of names;
- glossary entries are detailed;
- paper-reading guidance is operational;
- citations and DOI details have been checked;
- all code tests pass;
- the DOCX has been rendered and visually inspected page by page.

## 15. Suggested agent workflow

1. Audit the regression volume to avoid unnecessary duplication.
2. Create a chapter map and prerequisite dependency map.
3. Verify primary sources and current library APIs.
4. Implement and test the pure-Python engine before writing code explanations.
5. Write the mathematical chapters and connect them line by line to the implementation.
6. Add library workflows and reproduce the same examples.
7. Add diagnostics, failure cases, and research extensions.
8. Write exercises, glossary, paper-reading method, and annotated literature.
9. Run all code and record versions and tolerances.
10. Generate the DOCX, render it, inspect every page, fix defects, and repeat.
11. Deliver only final, tested files.

## 16. Ready-to-use handoff prompt

Use the following prompt when handing the task to another capable agent:

> Create a standalone research-level textbook titled “Generalized Linear Models: A Mathematical, Statistical, and Python Guide.” Follow every requirement in `NEXT_TOPIC_AGENT_INSTRUCTIONS.md`. The learner starts with elementary math, statistics, and Python. Teach prerequisites first, derive GLMs from likelihood and exponential-family principles, implement a generic IRLS engine in pure standard-library Python before using libraries, then cover diagnostics, major families, research extensions, paper reading, an expanded glossary, and an annotated research-paper list. Produce and test all companion Python files. Verify citations and current library APIs using primary sources. Render and inspect the final DOCX page by page before delivery. Do not merge this volume into the regression guide.
