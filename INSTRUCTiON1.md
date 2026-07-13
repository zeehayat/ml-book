 # CLAUDE INSTRUCTIONS FOR REVISING THE ZERO-TO-RESEARCH MATHEMATICS, STATISTICS, PYTHON, REGRESSION, GLM, AND MACHINE-LEARNING MATERIAL

## 1. Your role

Act as a senior curriculum designer, statistician, mathematical educator, Python instructor, research-methods teacher, and technical editor.

You are revising a connected series of books intended to take a learner from very limited prior knowledge toward research-level competence in:

* mathematics;
* probability and statistics;
* Python programming;
* data analysis;
* linear regression;
* generalized linear models;
* research design;
* reproducible empirical research;
* selected machine-learning and computational topics.

The existing material is technically ambitious and contains many strong explanations, derivations, exercises, implementations, research extensions, and reading lists. Preserve those strengths.

Your task is not merely to add more advanced material. Your main task is to create the missing instructional bridge between an absolute beginner and the existing advanced content.

---

# 2. Files to revise

The current material includes at least the following:

1. `Regression_Mathematical_Statistical_and_Python_Guide_Updated.md`
2. `Generalized_Linear_Models_Mathematical_Statistical_and_Python_Guide.md`
3. `chapter_01.md`, currently titled or centred on tensors, memory layout, and compute hardware.

Treat these as parts of a larger curriculum rather than isolated documents.

---

# 3. Fundamental assessment to guide the revision

The current material is strong as:

* an advanced guide;
* a reference source;
* a bridge from introductory knowledge to research-oriented practice;
* a source for an AI tutor;
* a map of the knowledge required for research.

It is not yet sufficient as a literal “absolute beginner to research level” curriculum because:

* prerequisite mathematics is summarized rather than fully taught;
* prerequisite statistics is summarized rather than fully taught;
* Python fundamentals are assumed too early;
* advanced notation appears before the learner has practised the simpler notation;
* exercises frequently lack hints, worked solutions, feedback, and staged support;
* synthetic demonstrations are more common than complete real-data studies;
* research-level extensions are often surveys of topics rather than full instructional treatments;
* the tensor and hardware chapter assumes intermediate Python despite appearing as an opening chapter;
* the books contain breadth, but the learner needs a more gradual mastery pathway.

Revise the curriculum around this diagnosis.

---

# 4. Non-negotiable design principle

Do not confuse the following four things:

1. **Mentioning a topic**
2. **Explaining a topic**
3. **Teaching a learner to use a topic**
4. **Developing independent research competence in that topic**

The revised material must clearly distinguish these levels.

A topic should not be labelled “mastered” simply because it has been described.

For each major topic, provide a progression through:

1. intuition;
2. terminology;
3. notation;
4. hand-worked example;
5. guided exercise;
6. independent exercise;
7. pure-Python implementation where appropriate;
8. library implementation;
9. interpretation;
10. diagnostics and common errors;
11. real-data application;
12. research-level extension.

---

# 5. Define the target learner carefully

The curriculum should support two types of learners.

## Learner A: Absolute beginner

This learner may not know:

* how to rearrange an equation;
* what a function is;
* what a logarithm is;
* what a derivative is;
* what probability means formally;
* what a sample or population is;
* what variance or standard deviation means;
* what a matrix is;
* how to install or run Python;
* how a loop, function, list, or data frame works;
* how to read an error message;
* how to interpret a statistical table.

Do not assume these concepts.

## Learner B: Returning or partially prepared learner

This learner may already know some of the prerequisites and should be able to test out of basic sections through diagnostic assessments.

Design prerequisite tests and optional fast tracks so that prepared readers are not forced through every elementary lesson.

---

# 6. Create a new Foundation Volume

Do not try to repair the zero-level problem by inserting a few additional paragraphs into the regression and GLM books.

Create a substantial new preliminary volume provisionally titled:

# Foundations for Mathematical, Statistical, and Computational Research

This volume should come before regression.

It should be long enough to genuinely teach the prerequisites. It should not be a glossary or compressed review.

Organize it into the following parts.

---

## Part I: Learning how to learn technical material

Cover:

* how to use the books;
* how to use a notebook;
* how to study mathematics;
* why copying formulas is not mastery;
* productive struggle;
* how to check answers;
* how to use an AI tutor without becoming dependent on it;
* how to keep a learning journal;
* how to diagnose whether a difficulty is mathematical, statistical, computational, or conceptual;
* how to review using spaced repetition and cumulative practice.

Include a diagnostic pre-test and a recommended study pathway based on the results.

---

## Part II: Arithmetic and elementary algebra

Teach from the beginning:

* positive and negative numbers;
* fractions;
* decimals;
* percentages;
* ratios;
* powers and roots;
* scientific notation;
* order of operations;
* variables;
* constants;
* expressions;
* equations;
* inequalities;
* rearranging formulas;
* substitution;
* coordinate axes;
* slope;
* intercept;
* linear equations;
* systems of simple equations;
* summation notation.

For every topic include:

* a plain-language explanation;
* visual intuition;
* several hand-worked examples;
* common errors;
* short exercises;
* answers;
* cumulative review questions.

Do not proceed to regression notation until the learner can manipulate a simple linear equation confidently.

---

## Part III: Functions, graphs, logarithms, and exponentials

Teach:

* input and output;
* domain and range;
* function notation;
* graphs;
* linear and nonlinear functions;
* transformations;
* polynomials;
* exponentials;
* logarithms;
* inverse functions;
* log rules;
* growth rates;
* percentage change;
* interpreting log scales.

Connect these topics gradually to:

* log transformations;
* odds;
* logistic functions;
* multiplicative models;
* elasticity;
* likelihood calculations.

Do not introduce the logit or log link before the learner understands logarithms and inverse functions.

---

## Part IV: Calculus and optimization foundations

Teach only the calculus needed for the later books, but teach it properly.

Include:

* change and rate of change;
* average versus instantaneous change;
* limits through intuition;
* derivatives;
* derivative rules;
* chain rule;
* partial derivatives;
* gradients;
* second derivatives;
* curvature;
* minima and maxima;
* convexity intuition;
* numerical optimization;
* gradient descent;
* Hessian intuition.

For each topic connect the mathematics to later applications:

* minimizing squared errors;
* maximizing likelihood;
* logistic regression;
* Newton-Raphson;
* Fisher scoring;
* IRLS;
* regularization.

Include graph-based and numerical examples before symbolic derivations.

---

## Part V: Linear algebra foundations

Teach:

* scalars;
* vectors;
* matrices;
* dimensions;
* rows and columns;
* dot products;
* matrix addition;
* matrix multiplication;
* transpose;
* identity matrix;
* inverse;
* systems of equations;
* linear combinations;
* span;
* independence;
* rank;
* orthogonality;
* projection;
* eigenvalues and eigenvectors;
* singular values;
* condition number;
* QR and SVD intuition.

Do not merely define these terms. Use small numerical matrices that learners calculate by hand.

Connect each concept explicitly to regression:

* design matrix;
* coefficient vector;
* fitted-value vector;
* normal equations;
* projection;
* residual orthogonality;
* multicollinearity;
* numerical stability.

---

## Part VI: Probability foundations

Teach:

* uncertainty;
* events;
* sample spaces;
* probability rules;
* complements;
* unions and intersections;
* conditional probability;
* independence;
* Bayes’ rule;
* random variables;
* discrete and continuous variables;
* probability mass;
* probability density;
* cumulative probability;
* expectation;
* variance;
* covariance;
* correlation;
* conditional expectation;
* law of total expectation;
* law of total variance.

Introduce distributions gradually:

* Bernoulli;
* binomial;
* normal;
* Poisson;
* Gamma;
* Student t;
* chi-square;
* F distribution.

For each distribution explain:

* what kind of outcome it represents;
* parameters;
* support;
* mean;
* variance;
* shape;
* common misuse;
* simulation in Python;
* later modelling applications.

---

## Part VII: Statistical reasoning

Teach:

* population and sample;
* census and sample;
* parameter;
* statistic;
* estimand;
* estimator;
* estimate;
* sampling variation;
* standard error;
* bias;
* variance;
* consistency;
* sampling distribution;
* law of large numbers;
* central limit theorem;
* confidence intervals;
* hypothesis tests;
* p-values;
* Type I and Type II errors;
* statistical power;
* effect size;
* practical significance;
* multiple comparisons;
* uncertainty communication.

Use repeated simulation to show how sampling distributions arise.

Include common misconceptions and require the learner to correct them.

---

## Part VIII: Python from absolute zero

Teach Python as an independent subject before requiring statistical programming.

Include:

* installing Python;
* using notebooks;
* using scripts;
* printing;
* variables;
* numbers;
* strings;
* Boolean values;
* lists;
* tuples;
* dictionaries;
* indexing;
* slicing;
* conditions;
* loops;
* comprehensions;
* functions;
* arguments;
* return values;
* scope;
* modules;
* imports;
* reading and writing files;
* exceptions;
* debugging;
* type hints;
* classes only when needed;
* testing;
* random seeds;
* package environments;
* reproducibility.

Every code example must be executable.

Never introduce advanced Python constructs such as:

* dataclasses;
* generators;
* decorators;
* custom data models;
* `__getitem__`;
* iterators;
* type aliases;
* recursion;
* context managers;

without first teaching them or marking them as optional advanced material.

---

## Part IX: NumPy, pandas, and visualization

Teach:

* NumPy arrays;
* array shape;
* dimensions;
* dtypes;
* indexing;
* slicing;
* vectorization;
* broadcasting;
* missing values;
* pandas Series and DataFrames;
* reading CSV files;
* selecting rows and columns;
* filtering;
* sorting;
* grouping;
* merging;
* reshaping;
* handling categories;
* handling missingness;
* basic data validation;
* matplotlib;
* scatterplots;
* histograms;
* boxplots;
* line plots;
* labels and scales;
* misleading graphs.

Use small real datasets rather than only invented lists.

---

## Part X: Introduction to research practice

Before regression, teach:

* what a research question is;
* descriptive versus predictive versus explanatory versus causal questions;
* unit of analysis;
* target population;
* variables and constructs;
* measurement;
* validity;
* reliability;
* sampling;
* data-generating processes;
* observational and experimental studies;
* confounding;
* selection;
* missing data;
* ethical research;
* documentation;
* reproducibility;
* research notebooks;
* data dictionaries.

End the Foundation Volume with a complete beginner-level research project.

---

# 7. Revise the Regression Guide

Keep the regression guide as the main modelling volume, but revise it under the following instructions.

## 7.1 Change its stated prerequisites

Do not claim that it independently teaches all mathematics, statistics, and Python from absolute zero.

State clearly that it assumes either:

* completion of the Foundation Volume; or
* successful completion of a prerequisite diagnostic test.

Part 0 of the regression guide may remain as a concise review and reference, but should not be presented as a replacement for the Foundation Volume.

---

## 7.2 Preserve the strongest features

Preserve and strengthen:

* the distinction between description, prediction, explanation, and causality;
* estimand, estimator, estimate, and standard error;
* disturbance versus residual;
* simple and multiple regression;
* matrix geometry;
* assumptions and what each assumption supports;
* robust standard errors;
* prediction intervals;
* diagnostics;
* transformations;
* categorical predictors;
* interactions;
* validation;
* regularization;
* causal cautions;
* pure-Python implementations;
* statsmodels and scikit-learn workflows;
* paper-reading guidance;
* research extensions.

Do not remove technical depth merely to make the book more accessible. Add scaffolding before the depth.

---

## 7.3 Add more worked examples

For each major regression concept, add at least one complete worked example.

At minimum include:

1. calculation by hand;
2. pure-Python calculation;
3. library calculation;
4. interpretation in words;
5. diagnostic assessment;
6. common incorrect interpretation.

Topics requiring full worked examples include:

* simple regression;
* multiple regression;
* omitted-variable bias;
* confidence intervals;
* hypothesis testing;
* joint F tests;
* prediction intervals;
* categorical variables;
* interactions;
* polynomial terms;
* log transformations;
* heteroskedasticity;
* cluster dependence;
* multicollinearity;
* influence;
* cross-validation;
* ridge;
* lasso;
* logistic regression;
* causal versus associational interpretation.

---

## 7.4 Add real-data case studies

Include several complete, cumulative case studies.

Suggested structure:

### Case Study 1: Descriptive relationship

A small, clean dataset.

Purpose:

* scatterplots;
* correlation;
* simple regression;
* residuals;
* uncertainty;
* careful interpretation.

### Case Study 2: Multiple regression

A realistic social, economic, health, agricultural, educational, or development dataset.

Purpose:

* data cleaning;
* categorical variables;
* transformations;
* confounding;
* model specification;
* robust standard errors;
* sensitivity.

### Case Study 3: Prediction

Purpose:

* train/test split;
* cross-validation;
* pipelines;
* leakage;
* baseline comparison;
* RMSE and MAE;
* regularization.

### Case Study 4: Clustered or grouped data

Purpose:

* dependence;
* cluster-robust uncertainty;
* grouped validation;
* multilevel alternatives.

### Case Study 5: Research replication

Reproduce one table or figure from a published paper using openly available data.

For each case study include:

* research question;
* data source;
* data dictionary;
* cleaning log;
* analysis plan;
* complete code;
* output;
* interpretation;
* limitations;
* alternative specifications;
* final research-style report.

---

## 7.5 Add solutions and feedback

For every exercise tier include:

* a hint;
* a stronger hint;
* expected intermediate result;
* complete solution;
* explanation of common mistakes;
* marking rubric.

Separate exercise questions and solutions so instructors can hide the solutions.

Do not merely provide final numerical answers. Explain reasoning.

---

## 7.6 Introduce mastery checks

At the end of every major section include:

* “You should now be able to…”;
* five conceptual questions;
* three calculations;
* one coding task;
* one interpretation task;
* one diagnostic task.

Define a suggested mastery threshold before moving on.

---

## 7.7 Expand research methods surrounding regression

Add more explicit treatment of:

* measurement error;
* sample selection;
* missing-data mechanisms;
* multiple imputation;
* survey sampling;
* weighted estimands;
* clustered sampling;
* specification uncertainty;
* preregistration;
* exploratory versus confirmatory analysis;
* multiple testing;
* sensitivity analysis;
* external validity;
* ethics;
* transparent reporting.

Do not make every topic a full textbook, but clearly distinguish:

* awareness;
* working competence;
* advanced specialization.

---

# 8. Revise the Generalized Linear Models Guide

Treat the GLM guide as a second or advanced modelling volume.

## 8.1 State prerequisites honestly

The GLM guide should require:

* completion of the regression guide;
* basic probability distributions;
* likelihood;
* derivatives;
* matrix algebra;
* Python functions and NumPy;
* basic statistical inference.

Do not describe it as independently suitable for a learner starting from zero.

The prerequisite section may remain as review.

---

## 8.2 Slow the transition to exponential-family notation

Before presenting the general exponential-family form, build through:

1. Bernoulli likelihood;
2. binomial likelihood;
3. Poisson likelihood;
4. log-likelihood;
5. score;
6. curvature;
7. maximum likelihood;
8. link functions;
9. mean-variance relationships;
10. common structure across models.

Only then introduce the unified notation.

Explain every symbol in the exponential-family expression.

Provide at least one complete derivation with no skipped steps.

---

## 8.3 Build logistic regression in several layers

Use the following progression:

1. binary outcomes and probabilities;
2. problems with the linear probability model;
3. odds;
4. log odds;
5. logistic curve;
6. coefficient interpretation;
7. predicted probabilities;
8. Bernoulli likelihood;
9. maximum likelihood;
10. score;
11. Hessian or information;
12. IRLS;
13. diagnostics;
14. calibration;
15. discrimination;
16. classification thresholds;
17. separation;
18. rare events;
19. marginal effects;
20. causal cautions.

Add multiple worked numerical examples.

---

## 8.4 Build count models through substantive examples

Teach Poisson and negative-binomial models through real count and rate problems.

Explicitly distinguish:

* counts;
* rates;
* exposure;
* offsets;
* observation time;
* overdispersion;
* clustering;
* excess zeros;
* zero inflation;
* hurdle processes.

Do not present zero-inflated models merely as a technical reaction to many zeros. Require a substantive mechanism.

---

## 8.5 Separate core competence from advanced extensions

Create three clearly labelled levels.

### Core GLM competence

* logistic regression;
* grouped binomial;
* Poisson regression;
* offsets;
* Gamma with log link;
* likelihood;
* IRLS;
* deviance;
* residuals;
* robust covariance;
* prediction and interpretation.

### Intermediate research competence

* negative binomial;
* overdispersion;
* zero-modified models;
* marginal effects;
* calibration;
* clustered data;
* GEE;
* missing data;
* survey designs.

### Advanced specialization

* GLMMs;
* Bayesian GLMs;
* Firth correction;
* high-dimensional GLMs;
* post-selection inference;
* transportability;
* distributional regression.

Do not imply that a short section provides mastery of an advanced specialization.

---

## 8.6 Add complete case studies

Include at least:

1. binary health or programme-participation outcome;
2. count outcome with exposure;
3. overdispersed count outcome;
4. positive cost or duration outcome;
5. clustered or repeated outcome;
6. complete separation example;
7. replication of a published GLM analysis.

Each case must include diagnostics, interpretation, alternative models, and sensitivity analysis.

---

## 8.7 Expand computational diagnostics

Every iterative model should report:

* convergence status;
* number of iterations;
* objective history;
* warnings;
* rank problems;
* extreme fitted values;
* separation indicators;
* high leverage;
* influence;
* dispersion;
* calibration where relevant.

Never hide convergence failure.

---

# 9. Reposition and revise the Tensor and Compute Hardware Chapter

The tensor chapter is not suitable as Chapter 1 for an absolute beginner.

## 9.1 Move it to a separate track

Create a separate book or track provisionally titled:

# Machine Learning Systems and Computational Foundations

Place the tensor chapter after chapters on:

1. Python fundamentals;
2. lists and arrays;
3. NumPy;
4. basic linear algebra;
5. elementary computer architecture;
6. binary data and numeric types.

The tensor chapter may then become Chapter 3 or later.

---

## 9.2 State its true prerequisites

Require familiarity with:

* Python functions;
* lists and tuples;
* classes;
* methods;
* type hints;
* indexing and slicing;
* generators;
* dataclasses;
* `__getitem__`;
* `__setitem__`;
* recursion or nested traversal;
* set notation;
* summation and product notation;
* integer arithmetic;
* basic memory concepts.

Where any prerequisite is essential, either teach it before use or provide a prerequisite appendix.

---

## 9.3 Preserve the chapter’s technical strengths

Preserve:

* flat-buffer intuition;
* shape, strides, and offset;
* C and Fortran order;
* zero-copy views;
* slicing;
* transposition;
* negative strides;
* aliasing;
* contiguity;
* serialization;
* performance reasoning;
* pure-Python implementation;
* NumPy-backed implementation;
* project work.

---

## 9.4 Mark simplified hardware models explicitly

Avoid presenting illustrative performance calculations as universal hardware laws.

Revise statements such as:

* “stride greater than 16 means every access is a cache miss”;
* “column traversal is 13.9 times slower”;
* “all tensor frameworks use exactly the same representation”;
* “reshape does not move data”;

into more precise statements.

Use wording such as:

* “Under this simplified model…”;
* “In the absence of effective prefetching or reuse…”;
* “This example illustrates a possible upper-bound slowdown…”;
* “Many dense tensor libraries use this fundamental conceptual model…”;
* “A reshape can be metadata-only when the existing layout permits it; otherwise it may require copying.”

Discuss factors that complicate the model:

* hardware prefetching;
* multiple cache levels;
* cache associativity;
* vectorization;
* loop ordering;
* compiler optimization;
* TLB effects;
* dtype;
* working-set size;
* BLAS packing;
* GPU coalescing;
* asynchronous execution;
* sparse and distributed tensor formats.

Keep the simplified model, but label its assumptions.

---

## 9.5 Divide essential and specialist material

Create two routes.

### Essential route

For data scientists and researchers:

* what an array is;
* shape;
* dtype;
* indexing;
* slicing;
* views versus copies;
* broadcasting;
* memory size;
* basic contiguity;
* why vectorization matters.

### Systems route

For ML engineers:

* manual stride derivation;
* negative strides;
* custom tensor class;
* serialization format;
* cache models;
* GPU hierarchy;
* tensor parallelism;
* performance benchmarking.

Do not require all statistical researchers to implement a tensor library before learning regression.

---

# 10. Required lesson structure

Every major lesson should follow a consistent pattern.

## A. Why this matters

Give a concrete research or computational reason.

## B. Prerequisite check

List exactly what the learner must already understand.

## C. Intuition

Explain without formal notation.

## D. Vocabulary

Define terms precisely.

## E. Visual or numerical example

Use small values that can be checked by hand.

## F. Formal treatment

Introduce equations and derivations.

## G. Guided practice

Show partially completed examples.

## H. Independent practice

Provide exercises.

## I. Python implementation

Begin with the simplest executable form.

## J. Library implementation

Use appropriate standard libraries.

## K. Interpretation

Translate output into substantive language.

## L. Diagnostics and failure modes

Show what can go wrong.

## M. Research connection

Explain how the concept appears in papers.

## N. Mastery check

Test conceptual, mathematical, computational, and interpretive understanding.

## O. Solutions

Provide complete, explained solutions in a separate section or file.

---

# 11. Rules for mathematics

Follow these rules throughout.

1. Introduce one new symbol at a time.
2. Define every symbol immediately.
3. Show dimensions for matrix expressions.
4. Do not skip algebraic steps in beginner derivations.
5. Use small numerical examples before general formulas.
6. Explain what an equation means in words.
7. Distinguish identity, definition, assumption, estimator, and approximation.
8. State when a result is exact and when it is asymptotic.
9. State when a formula depends on an intercept.
10. State when a result depends on independence, exogeneity, homoskedasticity, normality, or correct specification.
11. Never use “obviously,” “simply,” or “trivially” for a step a beginner may not find obvious.
12. Add common algebra mistakes and corrections.

---

# 12. Rules for statistics

1. Always identify the estimand.
2. State the unit of observation.
3. State the target population.
4. Distinguish conditional and marginal quantities.
5. Distinguish association, prediction, and causation.
6. State the data-generating assumptions.
7. Explain what uncertainty measure is being reported.
8. Do not reduce inference to p-values.
9. Report effect sizes and uncertainty.
10. Discuss practical importance.
11. Distinguish model-based and robust uncertainty.
12. Distinguish a correct mean model from a correct full distribution.
13. Treat missingness, measurement, selection, and dependence as central research issues.
14. Use causal language only with an explicit identification strategy.

---

# 13. Rules for Python and code

1. All code must run.
2. Avoid placeholders such as `...` in purportedly complete examples.
3. State required Python and package versions.
4. Include environment setup instructions.
5. Use reproducible random seeds.
6. Validate inputs.
7. Return informative errors.
8. Separate calculation from display.
9. Include docstrings.
10. Include small tests.
11. Show expected output.
12. Explain each advanced Python construct before using it.
13. Prefer readable code over clever code.
14. Distinguish educational implementations from production-quality implementations.
15. Compare from-scratch results with mature libraries.
16. Include tests for edge cases, failure, nonconvergence, rank deficiency, missing values, and invalid input.
17. Do not claim verification merely because code runs on one example.

---

# 14. Verification requirements

For major algorithms, verify results using several methods.

## Regression code

Test against:

* known analytic examples;
* statsmodels;
* NumPy;
* simulated data;
* singular designs;
* near-collinear designs;
* zero-variance outcomes;
* small samples;
* high-leverage observations;
* heteroskedastic data;
* clustered data.

## GLM code

Test:

* Gaussian;
* binomial;
* Poisson;
* Gamma;
* invalid response domains;
* separation;
* rare events;
* extreme predictors;
* overdispersion;
* offsets;
* rank deficiency;
* nonconvergence;
* high leverage;
* comparison with statsmodels.

## Tensor code

Test:

* scalar indexing;
* negative indexing;
* slicing;
* empty slices;
* negative strides;
* transposes;
* non-contiguous views;
* mutation through views;
* overlapping aliases;
* reshape eligibility;
* serialization;
* deserialization;
* dtype handling;
* zero-length dimensions;
* invalid metadata;
* round-trip equality.

Provide a test report, not merely the code.

---

# 15. Real-data requirements

The revised curriculum must not rely mainly on artificial data.

Use openly accessible data with permission for educational use.

For each dataset provide:

* source;
* licence;
* download instructions;
* frozen copy or version identifier where legally permitted;
* data dictionary;
* cleaning decisions;
* missing-value report;
* inclusion and exclusion rules;
* reproducible analysis.

Select datasets from multiple substantive areas, such as:

* public policy;
* development;
* health;
* education;
* agriculture;
* economics;
* environment;
* organizational performance.

Use some examples relevant to developing-country and public-sector contexts rather than relying entirely on conventional American datasets.

---

# 16. Research-level endpoint

Define research level realistically.

A learner has reached the intended endpoint when they can independently:

1. formulate a researchable question;
2. define an estimand;
3. identify a suitable dataset or data-collection design;
4. understand how the data were generated;
5. clean and document the data;
6. choose a justified model;
7. derive or explain the model’s main logic;
8. implement a basic version;
9. fit it using a mature library;
10. diagnose problems;
11. use suitable uncertainty estimates;
12. validate predictions where relevant;
13. conduct sensitivity analyses;
14. distinguish association from causation;
15. interpret results in substantive units;
16. reproduce a published result;
17. write a transparent research report;
18. identify limitations and further evidence needed.

Research level does not mean that the learner has mastered every specialist method mentioned in the books.

Mark advanced topics as pathways for further specialization.

---

# 17. Capstone sequence

Create a cumulative capstone programme.

## Capstone 1: Data description

Clean and describe a dataset.

## Capstone 2: Simple empirical question

Use visualization, correlation, and simple regression.

## Capstone 3: Multiple regression study

Define controls, examine assumptions, and conduct sensitivity analysis.

## Capstone 4: Predictive modelling study

Use pipelines, cross-validation, regularization, and out-of-sample evaluation.

## Capstone 5: GLM study

Choose a family and link based on outcome support and mechanism.

## Capstone 6: Published-paper replication

Reproduce at least one principal result.

## Capstone 7: Independent research project

Produce:

* proposal;
* research question;
* estimand;
* data description;
* analysis plan;
* code;
* diagnostics;
* sensitivity analysis;
* results;
* limitations;
* reproducibility package;
* final paper.

Provide rubrics for every capstone.

---

# 18. Output requirements for this revision task

Do not immediately rewrite all files without first producing a revision architecture.

Complete the work in phases.

## Phase 1: Curriculum map

Produce:

* proposed volumes;
* chapter sequence;
* prerequisites;
* learning outcomes;
* approximate length;
* relationship between existing and new material.

## Phase 2: Gap analysis

For every existing chapter identify:

* what should be retained;
* what should be expanded;
* what should be moved;
* what should be removed or qualified;
* what prerequisites are missing;
* what exercises and case studies are needed.

## Phase 3: Detailed revision plan

Create a section-by-section change plan for each file.

Use a table with:

* current section;
* problem;
* proposed change;
* new content required;
* priority;
* dependency.

## Phase 4: Rewrite

Rewrite one part at a time.

Preserve strong existing passages unless a change is necessary.

Do not silently delete technical material.

## Phase 5: Exercises and solutions

Create exercise banks, solutions, and rubrics.

## Phase 6: Code validation

Run all code and tests.

Document:

* environment;
* package versions;
* passing tests;
* failing tests;
* unresolved limitations.

## Phase 7: Final editorial review

Check:

* progression;
* repetition;
* notation;
* terminology;
* code consistency;
* cross-references;
* factual precision;
* claims about research competence;
* accessibility;
* completeness.

---

# 19. Tone and style

Use a serious but encouraging teaching voice.

The reader should feel respected, not patronized.

Prefer:

* clear sentences;
* concrete examples;
* explicit assumptions;
* gradual progression;
* substantive interpretation;
* honest statements of limitation.

Avoid:

* unnecessary jargon;
* inflated claims;
* excessive motivational language;
* pretending a short section creates mastery;
* overly compressed derivations;
* presenting simplified hardware or statistical models as universal facts.

---

# 20. Final objective

The revised series should achieve two things simultaneously:

1. preserve the intellectual and technical ambition of the existing books;
2. make the pathway genuinely traversable by a motivated beginner.

The final curriculum should not merely expose a learner to advanced research language. It should progressively develop the learner’s ability to reason mathematically, analyse statistically, program competently, diagnose mistakes, interpret evidence, and conduct reproducible research independently.

