# Regression: A Mathematical and Python Guide

*From First Principles to Practical Modelling*

> This file is a table of contents / quick-reference index. The full guide — with
> derivations, worked examples, executable pure-Python code, library workflows,
> exercises, a 40-term glossary, suggested research papers, and appendices — lives in
> [`Regression_Mathematical_and_Python_Guide.docx`](Regression_Mathematical_and_Python_Guide.docx).
> A browsable, notes-enabled version is in
> [`regression_guide.html`](regression_guide.html) (open it, or run
> `start_notes_server.bat` for the offline notes panel).

## How to use this guide

Self-contained source for learning regression through mathematics, intuition, pure
Python, and modern Python libraries. No prior calculus beyond basic derivatives is
assumed; the matrix chapters are easier with elementary linear algebra. "Pure Python"
in Part II means the Python standard library only — no NumPy, pandas, or any
third-party package — so every calculation is exposed rather than hidden behind a
library call.

## Part I — Mathematical Foundations (Ch. 1–14)

1. What regression is trying to do
2. Notation and data geometry
3. Simple linear regression
4. Multiple linear regression
5. Assumptions: what each one is needed for
6. Sampling uncertainty and statistical inference
7. Prediction and uncertainty
8. Measuring model fit
9. Residual diagnostics
10. Transformations, categories, interactions, and nonlinear terms
11. Generalization, validation, and leakage
12. Regularization (ridge, lasso, elastic net)
13. Logistic regression
14. Regression and causal claims

## Part II — Regression from Scratch in Pure Python (Ch. 15–24)

15. Why implement regression yourself?
16. Simple linear regression from scratch
17. Matrix operations without NumPy
18. Multiple OLS from scratch
19. Diagnostics from scratch
20. Gradient descent for linear regression
21. Train/test splitting and cross-validation from scratch
22. Ridge and lasso from scratch
23. Logistic regression from scratch
24. A complete from-scratch workflow

## Part III — Regression with Python Libraries (Ch. 25–33)

25. Which library should do what?
26. NumPy: OLS through linear algebra
27. pandas: preparing regression data
28. statsmodels for inference
29. scikit-learn for predictive workflows
30. Diagnostic plots with matplotlib
31. Logistic regression with libraries
32. End-to-end regression checklist
33. Common mistakes and corrections

## Part IV — Exercises and Teaching Sequence (Ch. 34–43)

34. Conceptual questions
35. Derivation exercises
36. Pure-Python coding exercises
37. Library exercises
38. Suggested learning sequence
39. Prompts for an AI tutor or NotebookLM
40. **Glossary** — 40 terms, each with a comprehensive multi-sentence definition
41. Further study — natural next topics (generalized linear models, mixed-effects
    models, panel data, time series, survival analysis, causal inference, GAMs,
    quantile/robust regression, Bayesian regression, tree ensembles)
42. **Suggested research papers** — 15 papers mapped to specific chapters, from
    Legendre (1805) and Gauss through Hoerl & Kennard (ridge), Tibshirani (lasso),
    Zou & Hastie (elastic net), and Nelder & Wedderburn (GLMs)
43. **How to read a research paper** — a three-pass reading strategy, questions to
    ask while reading, section reading order, and how to evaluate a claim

## Appendices

- **Appendix A** — Full pure-Python implementation (`regression_from_scratch.py`):
  descriptive statistics, OLS via QR and normal equations, a from-scratch Student t
  distribution, confidence/prediction intervals, diagnostics, cross-validation,
  gradient descent, ridge, lasso, and logistic regression — all executable.
- **Appendix B** — Full library-based demonstration (`regression_with_libraries.py`):
  NumPy, pandas, statsmodels, and scikit-learn end-to-end workflows.
- **Appendix C** — Minimal verification tests for the pure-Python implementation.

## Curriculum position

This is the first guide in a step-by-step, zero-to-research-level machine learning
sequence. The next guide, covering **Generalized Linear Models**, follows the same
template (From Scratch → Math → Libraries → Further Research → Glossary → Research
Papers → How to Read a Paper) and is specified in
[`NEXT_TOPIC_AGENT_BRIEF.md`](NEXT_TOPIC_AGENT_BRIEF.md).
