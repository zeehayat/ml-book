"""
Build a detailed PowerPoint presentation for Chapter 1 (Linear Regression &
Logistic Models) aimed at a group of starting students.

Each slide has:
  - a short, presentation-friendly bullet list
  - an "annotation" callout box highlighting one key insight
  - speaker notes (a fuller explanation script for whoever presents)
  - a clickable link to one external resource for further learning

Run:
    python3 build_chapter1_presentation.py

Produces:
    Chapter1_Regression_Presentation.pptx
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
NAVY = RGBColor(0x1B, 0x26, 0x4F)
TEAL = RGBColor(0x0D, 0x94, 0x88)
LIGHT_BG = RGBColor(0xF7, 0xF9, 0xFC)
INSIGHT_BG = RGBColor(0xEA, 0xF6, 0xF4)
TEXT_DARK = RGBColor(0x22, 0x22, 0x2A)
LINK_BLUE = RGBColor(0x1A, 0x5C, 0xB8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

# ---------------------------------------------------------------------------
# Content: one dict per slide
# ---------------------------------------------------------------------------
SLIDES = [
    {
        "title": "Chapter 1: Linear Regression & Logistic Models",
        "subtitle": "Zero-to-Research Machine Learning Handbook — A First Course for Starting Students",
        "bullets": [],
        "insight": None,
        "notes": (
            "Welcome the group. This deck walks through Chapter 1 of the handbook end to end: "
            "why regression matters, the math behind it, how to build it from scratch in pure "
            "Python, how to use it with real libraries, and how to avoid the most common mistakes. "
            "No prior statistics or calculus background is assumed — we build every idea from the "
            "ground up, the same way the written chapter does."
        ),
        "resource_label": "Course text: Chapter 1 (this handbook)",
        "resource_url": "https://en.wikipedia.org/wiki/Regression_analysis",
    },
    {
        "title": "Chapter 1 Roadmap",
        "bullets": [
            "Part 0 — Foundations: the algebra, logs, probability, and Python you need first",
            "Part I — Mathematical Foundations: what regression is and how it is derived",
            "Part II — From Scratch: building regression in pure Python, no libraries",
            "Part III — Libraries: NumPy, pandas, statsmodels, scikit-learn",
            "Part IV — Exercises: conceptual → derivation → coding → library tiers",
            "Part V — Research-Level Practice: what comes after you're comfortable",
        ],
        "insight": "Each part builds on the last — do not skip Part 0 even if it looks basic.",
        "notes": (
            "Emphasize that this is a genuinely staged curriculum, not a grab-bag of topics. "
            "Part 0 exists because many learners stall on regression not because regression is hard, "
            "but because a rearranged formula or a logarithm rule was never taught explicitly. "
            "Part II (from scratch) exists so that Part III's library calls are never a black box."
        ),
        "resource_label": "How to use this guide (full chapter)",
        "resource_url": "https://en.wikipedia.org/wiki/Regression_analysis",
    },
    {
        "title": "Math Refresher: Variables, Slope, and Intercept",
        "bullets": [
            "A variable is a symbol that can change: x (rainfall), y (crop yield)",
            "A line is written  y = a + b·x",
            "a is the intercept: the value of y when x = 0",
            "b is the slope: how much y changes per one-unit increase in x",
            "Rearranging: given y, a, b → solve for x = (y − a) / b",
        ],
        "insight": "Every regression coefficient you will ever fit is just a slope, generalized to many predictors at once.",
        "notes": (
            "Draw a simple x-y plane on the whiteboard if possible. Plug in two or three values of x "
            "and show how y changes. Make sure everyone can rearrange y = a + bx to solve for x before "
            "moving on — this exact algebra reappears when we invert a fitted model to answer "
            "'what input would produce this output?'"
        ),
        "resource_label": "Khan Academy: slope & linear equations",
        "resource_url": "https://www.khanacademy.org/math/algebra",
    },
    {
        "title": "Math Refresher: Logs, Exponentials, and Odds",
        "bullets": [
            "Exponential growth multiplies by a fixed factor each step: 2, 4, 8, 16, ...",
            "A logarithm undoes an exponential: log(exp(x)) = x",
            "Log rules: log(ab) = log(a)+log(b);  log(a/b) = log(a)−log(b);  log(a^c) = c·log(a)",
            "Odds = p / (1−p); log-odds (\"logit\") can be any real number",
            "Logistic regression models the logit as a straight line — same tool, new scale",
        ],
        "insight": "Logs turn products into sums — this is exactly why log-likelihood, not likelihood, is used to fit every model in this handbook.",
        "notes": (
            "This slide previews material students will need soon: log transformations for skewed "
            "data, and the logit link in logistic regression. Reassure them we return to this in "
            "depth right before logistic regression is introduced — this is just a preview so the "
            "vocabulary is not new later."
        ),
        "resource_label": "Wikipedia: Logarithm",
        "resource_url": "https://en.wikipedia.org/wiki/Logarithm",
    },
    {
        "title": "Statistics Refresher: Population, Sample, Mean, Variance",
        "bullets": [
            "Population: the entire group you want to learn about",
            "Sample: the smaller group you actually observed",
            "Mean: add the values, divide by the count",
            "Variance: how spread out values are around the mean",
            "Covariance: whether two variables move together",
            "Correlation: covariance rescaled to always fall between −1 and 1",
        ],
        "insight": "The simple regression slope is nothing more than covariance(x,y) divided by variance(x).",
        "notes": (
            "This is the single most useful one-line intuition in the whole chapter: "
            "slope = Cov(x,y) / Var(x). Everything about simple linear regression's slope formula "
            "follows from this. Emphasize that every number computed from a sample is an estimate, "
            "not the true population quantity — a theme that returns constantly."
        ),
        "resource_label": "Seeing Theory: visual intro to statistics",
        "resource_url": "https://seeing-theory.brown.edu/",
    },
    {
        "title": "Python Refresher: What You Need to Read the Code",
        "bullets": [
            "Lists store multiple values: x = [1, 2, 3, 4, 5]",
            "Loops repeat work: for value in x: ...",
            "Functions name reusable logic: def mean(values): return sum(values)/len(values)",
            "zip() pairs values from two lists together",
            "List comprehensions build a new list in one line",
        ],
        "insight": "You do not need advanced Python for Chapter 1 — lists, loops, and functions are enough to read every line of code.",
        "notes": (
            "Live-code a tiny example if possible: define x and y as lists, write a mean() function, "
            "and compute the mean of both. This is deliberately minimal — Chapter 1's pure-Python "
            "section uses nothing beyond what's on this slide."
        ),
        "resource_label": "Official Python tutorial",
        "resource_url": "https://docs.python.org/3/tutorial/",
    },
    {
        "title": "What Is Regression, Really?",
        "bullets": [
            "Description: summarize how y relates to x in the data you have",
            "Prediction: guess y for a new observation's x",
            "Explanation: estimate how much x influences y, holding other things fixed",
            "Causal inference: what happens to y if we intervene and change x?",
            "These four goals require different levels of assumption — don't mix them up",
        ],
        "insight": "A model that predicts well is not automatically a model that explains or proves causation.",
        "notes": (
            "This distinction is one students get wrong constantly in later work: a high R^2 or "
            "good test accuracy says nothing about whether x causes y. We return to this explicitly "
            "in the causal-caution slide near the end of the deck."
        ),
        "resource_label": "Wikipedia: Correlation does not imply causation",
        "resource_url": "https://en.wikipedia.org/wiki/Correlation_does_not_imply_causation",
    },
    {
        "title": "A Tiny Regression, By Hand",
        "bullets": [
            "x = [1, 2, 3, 4, 5],  y = [2, 4, 5, 4, 5]",
            "x̄ = 3,  ȳ = 4",
            "slope = Σ(xᵢ−x̄)(yᵢ−ȳ) / Σ(xᵢ−x̄)² = 6/10 = 0.6",
            "intercept = ȳ − slope·x̄ = 4 − 0.6(3) = 2.2",
            "fitted line:  ŷ = 2.2 + 0.6x",
        ],
        "insight": "At x=4, the prediction is 4.6; the observed value is 4, so the residual is −0.6.",
        "notes": (
            "Walk through every arithmetic step on the board — do not skip to the answer. This "
            "worked example contains the entire vocabulary of regression in one place: data, average, "
            "slope, intercept, prediction, and residual. Ask the group to recompute one point "
            "themselves before moving on."
        ),
        "resource_label": "StatQuest: Linear Regression, Clearly Explained (YouTube channel)",
        "resource_url": "https://www.youtube.com/@statquest",
    },
    {
        "title": "Simple Linear Regression: The Model",
        "bullets": [
            "Model:  y = β₀ + β₁x + ε",
            "ε (epsilon) is the unobservable disturbance in the true process",
            "e (the residual) is what we actually observe: e = y − ŷ",
            "Least squares chooses β₀, β₁ to minimize Σeᵢ²",
            "Why squared error? It penalizes large mistakes more, and has clean calculus",
        ],
        "insight": "ε and e are not the same thing — e is our best estimate of ε, not ε itself.",
        "notes": (
            "This distinction (epsilon vs. e) is exercise C1 in the chapter and trips students up "
            "constantly. Epsilon is a property of the true, unknown data-generating process; the "
            "residual e is computed from our fitted model and depends on how good our estimate is."
        ),
        "resource_label": "Wikipedia: Ordinary least squares",
        "resource_url": "https://en.wikipedia.org/wiki/Ordinary_least_squares",
    },
    {
        "title": "Deriving the Slope and Intercept",
        "bullets": [
            "Take the derivative of Σ(yᵢ−b₀−b₁xᵢ)² with respect to b₀ and b₁",
            "Set both derivatives to zero → the two \"normal equations\"",
            "Solving them gives:  b₁ = Sₓᵧ / Sₓₓ   and   b₀ = ȳ − b₁x̄",
            "Sₓᵧ = Σ(xᵢ−x̄)(yᵢ−ȳ);   Sₓₓ = Σ(xᵢ−x̄)²",
            "The fitted line always passes through the point (x̄, ȳ)",
        ],
        "insight": "This is not magic — it is one application of \"set the derivative to zero to find a minimum,\" the same rule from introductory calculus.",
        "notes": (
            "If the group has seen any calculus, connect this explicitly to finding a minimum of a "
            "parabola. If not, it's fine to state the result and verify it numerically against the "
            "worked example from two slides ago — plug in the numbers and confirm b1=0.6, b0=2.2."
        ),
        "resource_label": "3Blue1Brown: essence of calculus",
        "resource_url": "https://www.3blue1brown.com/topics/calculus",
    },
    {
        "title": "Multiple Linear Regression & the Design Matrix",
        "bullets": [
            "Real problems have many predictors: y = β₀ + β₁x₁ + β₂x₂ + ... + ε",
            "Stack all predictors into a design matrix X (rows = observations, columns = predictors)",
            "Model in matrix form:  y = Xβ + ε",
            "Least squares solution:  β̂ = (XᵀX)⁻¹Xᵀy  — the \"normal equations\"",
            "Each βⱼ is interpreted \"holding all other predictors constant\"",
        ],
        "insight": "Matrix notation is not a new idea — it is simple regression's algebra, just packing many x's into one object.",
        "notes": (
            "Reassure the group that (X^T X)^-1 X^T y looks intimidating but is exactly the same "
            "optimization as simple regression, generalized. This is a good moment to connect back "
            "to the linear algebra foundations (vectors, matrix multiplication, transpose) from Part 0."
        ),
        "resource_label": "Wikipedia: Design matrix",
        "resource_url": "https://en.wikipedia.org/wiki/Design_matrix",
    },
    {
        "title": "The Geometry of OLS",
        "bullets": [
            "Xb̂eta is the projection of y onto the space spanned by X's columns",
            "The residual vector e is orthogonal (perpendicular) to every column of X: Xᵀe = 0",
            "This is why an intercept forces the residuals to sum to exactly zero",
            "Geometrically: OLS finds the closest point to y that X can possibly reach",
        ],
        "insight": "\"Closest point\" is measured in squared Euclidean distance — this is exactly why the method is called least squares.",
        "notes": (
            "If a whiteboard is available, sketch y as a vector sticking out of the plane spanned by "
            "the columns of X, with the fitted value as its shadow (projection) on that plane, and "
            "the residual as the vertical vector connecting them. This picture makes 'X^T e = 0' "
            "intuitive rather than a memorized fact."
        ),
        "resource_label": "Wikipedia: Projection (linear algebra)",
        "resource_url": "https://en.wikipedia.org/wiki/Projection_(linear_algebra)",
    },
    {
        "title": "The Assumptions Behind OLS",
        "bullets": [
            "Linearity: the true relationship is (approximately) linear in the parameters",
            "Independence: observations don't influence each other",
            "No perfect multicollinearity: predictors aren't exact linear combinations of each other",
            "Zero conditional mean: E[ε | X] = 0 — no systematic error given the predictors",
            "Homoskedasticity: constant error variance across all values of X",
            "Normality: needed for exact small-sample inference (less critical with large n)",
        ],
        "insight": "Each assumption supports one specific guarantee — break one, and only that guarantee is at risk, not the whole model.",
        "notes": (
            "Go assumption by assumption and name what breaks if it's violated: heteroskedasticity "
            "breaks standard errors, not unbiasedness; omitted-variable bias breaks zero conditional "
            "mean; perfect multicollinearity breaks the matrix inversion entirely. This framing "
            "prevents the common all-or-nothing misunderstanding of assumptions."
        ),
        "resource_label": "Wikipedia: Gauss–Markov theorem",
        "resource_url": "https://en.wikipedia.org/wiki/Gauss%E2%80%93Markov_theorem",
    },
    {
        "title": "Sampling Uncertainty: Standard Errors, CIs, and Tests",
        "bullets": [
            "Every coefficient estimate would differ slightly in a fresh sample — that's sampling variation",
            "The standard error measures how much a coefficient would typically vary across samples",
            "A confidence interval reports a plausible range for the true coefficient",
            "A p-value asks: how surprising is this data if the true effect were zero?",
            "95% confidence describes the reliability of the *procedure*, not this one interval",
        ],
        "insight": "A small p-value is evidence against 'no effect' — it is not proof the effect is large or important.",
        "notes": (
            "This is the most commonly misunderstood idea in applied statistics. Spend real time "
            "here. Emphasize the difference between statistical significance (a small p-value) and "
            "practical/substantive importance (the size of the effect in real-world units)."
        ),
        "resource_label": "Wikipedia: Confidence interval",
        "resource_url": "https://en.wikipedia.org/wiki/Confidence_interval",
    },
    {
        "title": "Measuring Fit: R² and Adjusted R²",
        "bullets": [
            "R² = 1 − (SS_residual / SS_total): proportion of variance explained by the model",
            "R² = 0 means the model does no better than predicting the mean every time",
            "Adding any predictor can only increase (never decrease) raw R²",
            "Adjusted R² penalizes extra predictors that don't earn their keep",
            "A low R² does not mean a coefficient is unimportant (see Exercise C7)",
        ],
        "insight": "A randomized trial can have a huge, precisely estimated effect and still a low R², if individual outcomes are just noisy.",
        "notes": (
            "Use a concrete example: a clinical trial where treatment has a real, large, statistically "
            "significant effect, but human health outcomes vary so much between individuals that R^2 "
            "stays low. This breaks the common 'R^2 close to 1 = good model' shortcut thinking."
        ),
        "resource_label": "Wikipedia: Coefficient of determination",
        "resource_url": "https://en.wikipedia.org/wiki/Coefficient_of_determination",
    },
    {
        "title": "Residual Diagnostics",
        "bullets": [
            "Residuals vs. fitted values: look for curved patterns (non-linearity) or funnel shapes (heteroskedasticity)",
            "Q-Q plot: checks whether residuals follow a normal distribution",
            "Leverage: how unusual a point's predictor values are",
            "Cook's distance: how much removing one point would change the fitted model",
            "A model that 'runs' without errors can still badly violate its assumptions — always plot the residuals",
        ],
        "insight": "Diagnostics are not optional extra credit — they are how you find out whether the assumptions from two slides ago actually hold.",
        "notes": (
            "If time allows, show real residual plots (Anscombe's quartet is an excellent, classic "
            "example: four datasets with identical summary statistics but wildly different shapes "
            "and residual patterns). Emphasize that a computed R^2 alone can never reveal this."
        ),
        "resource_label": "Wikipedia: Anscombe's quartet",
        "resource_url": "https://en.wikipedia.org/wiki/Anscombe%27s_quartet",
    },
    {
        "title": "Multicollinearity",
        "bullets": [
            "Occurs when predictors are highly correlated with each other",
            "Individual coefficients become unstable and hard to interpret",
            "But predicted values (Xb̂eta) usually remain reliable",
            "Variance Inflation Factor (VIF) quantifies how much a coefficient's variance is inflated",
            "VIF above roughly 5–10 is commonly flagged as concerning",
        ],
        "insight": "Multicollinearity damages interpretation far more than it damages prediction — know which goal you actually care about.",
        "notes": (
            "Give a concrete intuition: if two predictors are nearly identical (e.g., height in cm "
            "and height in inches), the model cannot tell which one 'deserves credit' for an effect, "
            "so their individual coefficients can swing wildly with small data changes, even though "
            "the combined prediction stays stable."
        ),
        "resource_label": "Wikipedia: Variance inflation factor",
        "resource_url": "https://en.wikipedia.org/wiki/Variance_inflation_factor",
    },
    {
        "title": "Regularization: Ridge, Lasso, Elastic Net",
        "bullets": [
            "Ridge adds a penalty λΣβⱼ² — shrinks coefficients toward zero, keeps them all",
            "Lasso adds a penalty λΣ|βⱼ| — can shrink coefficients to exactly zero (feature selection)",
            "Elastic Net blends both penalties",
            "Regularization requires standardized features — otherwise the penalty is unfair across scales",
            "More regularization = more bias, less variance = a deliberate trade-off, not free lunch",
        ],
        "insight": "Ridge is smooth and keeps correlated predictors together; Lasso's sharp corners tend to arbitrarily pick one from a correlated group and zero out the rest.",
        "notes": (
            "This geometric difference (smooth ridge ball vs. cornered lasso diamond) is exactly why "
            "lasso can perform variable selection and ridge cannot. If you have access to a picture "
            "of the L1 diamond vs. L2 circle constraint regions, show it here."
        ),
        "resource_label": "Wikipedia: Regularization (mathematics)",
        "resource_url": "https://en.wikipedia.org/wiki/Regularization_(mathematics)",
    },
    {
        "title": "Logistic Regression: From Probability to Log-Odds",
        "bullets": [
            "Outcome y is binary (0 or 1) — a straight line can't stay between 0 and 1",
            "Model the log-odds instead:  logit(p) = log(p/(1−p)) = β₀ + β₁x",
            "Invert to get a probability:  p = 1 / (1 + exp(−(β₀+β₁x)))",
            "exp(βⱼ) is the multiplicative change in the ODDS, not the probability, per unit of x",
            "Fit by maximum likelihood, not least squares — no closed-form solution, needs iteration",
        ],
        "insight": "Logistic regression is still, at its core, a straight line — just measured on the log-odds scale instead of the raw probability scale.",
        "notes": (
            "Return explicitly to the logs/odds refresher slide from earlier in the deck. Many "
            "students find logistic regression hard only because odds and log-odds were never made "
            "concrete — once that's solid, the rest of logistic regression is a direct analogy to "
            "linear regression."
        ),
        "resource_label": "Wikipedia: Logistic regression",
        "resource_url": "https://en.wikipedia.org/wiki/Logistic_regression",
    },
    {
        "title": "Evaluating Classification Models",
        "bullets": [
            "Confusion matrix: true/false positives and negatives at a chosen threshold",
            "Precision: of predicted positives, how many were correct?",
            "Recall: of actual positives, how many did we find?",
            "Accuracy is misleading when classes are imbalanced (e.g., 2% positive rate)",
            "ROC-AUC and precision-recall AUC summarize performance across all thresholds",
        ],
        "insight": "A model that always predicts 'no' can score 98% accuracy on a 2%-positive dataset while being completely useless.",
        "notes": (
            "Walk through a concrete confusion matrix with small numbers on the board. Compute "
            "accuracy, precision, and recall by hand for an imbalanced example so students feel the "
            "gap between 'high accuracy' and 'good classifier' directly."
        ),
        "resource_label": "Wikipedia: Precision and recall",
        "resource_url": "https://en.wikipedia.org/wiki/Precision_and_recall",
    },
    {
        "title": "Regression From Scratch, in Pure Python",
        "bullets": [
            "No NumPy, pandas, or scikit-learn — only the Python standard library",
            "Implements: mean/variance, simple & multiple OLS, gradient descent, ridge, lasso",
            "Also includes: Student-t distribution, confidence & prediction intervals",
            "Train/test split and k-fold cross-validation, all built by hand",
            "Goal: nothing is a black box — every library call later maps to code you've already written",
        ],
        "insight": "If you can explain every line of the from-scratch module, you can never be fooled by a library's default settings.",
        "notes": (
            "This is the philosophical heart of Part II. Encourage the group to actually run the "
            "from-scratch code and modify it — change the learning rate, break the assumptions "
            "deliberately, and watch what happens to the fitted coefficients."
        ),
        "resource_label": "Real Python: linear regression in Python",
        "resource_url": "https://realpython.com/linear-regression-in-python/",
    },
    {
        "title": "Regression With Python Libraries",
        "bullets": [
            "NumPy: the linear algebra engine underneath almost everything else",
            "pandas: loading, cleaning, and preparing tabular data",
            "statsmodels: rich statistical inference — coefficients, SEs, p-values, diagnostics",
            "scikit-learn: fast, consistent pipelines for prediction and cross-validation",
            "Rule of thumb: statsmodels for inference, scikit-learn for prediction pipelines",
        ],
        "insight": "Different libraries exist because 'explain the data' and 'predict on new data' are different jobs, even when the underlying model is the same.",
        "notes": (
            "If a live environment is available, demo fitting the exact same small dataset in both "
            "statsmodels (show the coefficient table with p-values) and scikit-learn (show a "
            "cross-validated prediction pipeline) so students see the two philosophies directly."
        ),
        "resource_label": "scikit-learn: linear models user guide",
        "resource_url": "https://scikit-learn.org/stable/modules/linear_model.html",
    },
    {
        "title": "Common Mistakes and Corrections",
        "bullets": [
            "“The coefficient is significant, so the effect is important” → report magnitude & units too",
            "“Scale the full dataset, then split” → fit scaling only inside the training fold",
            "“Use the test set until the model looks good” → test set becomes training info if reused",
            "“A 0.5 threshold is always correct” → choose thresholds from costs and prevalence",
            "“Delete every outlier” → investigate first; unusual points can be genuine and informative",
        ],
        "insight": "Almost every 'common mistake' in this chapter is a data-leakage or overclaiming mistake, not an arithmetic one.",
        "notes": (
            "Ask the group if they've seen or made any of these mistakes themselves — this usually "
            "generates good discussion. The 'scale before splitting' mistake is worth a live "
            "demonstration if time allows, since it is extremely common and easy to miss."
        ),
        "resource_label": "scikit-learn: data leakage guidance",
        "resource_url": "https://scikit-learn.org/stable/common_pitfalls.html",
    },
    {
        "title": "Practicing What You've Learned: The Exercise Tiers",
        "bullets": [
            "Conceptual (C): explain the idea in plain language, no formula allowed",
            "Derivation (D): reconstruct the mathematics from first principles, on paper",
            "Pure-Python coding (P): implement it using only the standard library",
            "Library (L): reproduce the result with NumPy, statsmodels, and scikit-learn",
            "Work one topic through all four tiers before moving to the next topic",
        ],
        "insight": "If you can't answer a topic's Conceptual question, do not attempt its Derivation exercise yet — the tiers are a ladder, not a menu.",
        "notes": (
            "Point the group to the chapter's own answer keys for every tier. Suggest picking one "
            "topic (e.g. ridge regression) and working it through C -> D -> P -> L as a full practice "
            "loop before the next study session."
        ),
        "resource_label": "Chapter 1 exercises with answer keys",
        "resource_url": "https://en.wikipedia.org/wiki/Regression_analysis",
    },
    {
        "title": "Causal Caution: Correlation Is Not Causation",
        "bullets": [
            "Ice cream sales strongly predict drowning deaths — both driven by hot weather",
            "A confounder is an unmeasured factor that drives both x and y",
            "Omitted-variable bias: leaving out a confounder correlated with x biases β̂",
            "High predictive accuracy does NOT imply a safe intervention (\"if we change x, y will change\")",
            "Causal claims need a causal design: randomization, natural experiments, or explicit assumptions",
        ],
        "insight": "A regression coefficient answers 'how are x and y associated in this data?' — not 'what happens if I change x?' unless you can defend that extra assumption.",
        "notes": (
            "This closes the loop back to the very first 'four goals of regression' slide. Make sure "
            "the group leaves understanding that prediction and causal inference are genuinely "
            "different tasks requiring different evidence."
        ),
        "resource_label": "Wikipedia: Confounding",
        "resource_url": "https://en.wikipedia.org/wiki/Confounding",
    },
    {
        "title": "Wrap-Up and Where to Go Next",
        "bullets": [
            "You now have the vocabulary: slope, residual, assumption, standard error, regularization",
            "You've seen regression built twice — once from scratch, once with libraries",
            "You know how to evaluate a model honestly, and how to avoid the most common mistakes",
            "Next: Chapter 2 extends this exact framework to counts, rates, and other outcome types (GLMs)",
            "Practice loop: pick a topic, work C → D → P → L, then move to the next topic",
        ],
        "insight": "Everything in Chapter 2 (and beyond) is this same regression framework, generalized — master this chapter and the rest gets easier, not harder.",
        "notes": (
            "End on encouragement: regression is the single most important model family in this "
            "handbook because nearly everything later either is a regression or borrows its core "
            "ideas (a linear predictor, a loss function, gradient-based fitting). Invite questions."
        ),
        "resource_label": "Generalized Linear Models (Wikipedia, Chapter 2 preview)",
        "resource_url": "https://en.wikipedia.org/wiki/Generalized_linear_model",
    },
]


def add_footer_link(slide, prs, label, url):
    box = slide.shapes.add_textbox(Inches(0.5), SLIDE_H - Inches(0.55), SLIDE_W - Inches(1.0), Inches(0.4))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = f"\U0001F517 Further reading: {label}"
    run.font.size = Pt(13)
    run.font.italic = True
    run.font.color.rgb = LINK_BLUE
    run.hyperlink.address = url


def add_slide_number(slide, n, total):
    box = slide.shapes.add_textbox(SLIDE_W - Inches(1.3), SLIDE_H - Inches(0.55), Inches(1.0), Inches(0.4))
    tf = box.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    run = p.add_run()
    run.text = f"{n} / {total}"
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)


def set_notes(slide, text):
    notes_slide = slide.notes_slide
    notes_slide.notes_text_frame.text = text


def build():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    blank_layout = prs.slide_layouts[6]

    total = len(SLIDES)

    for idx, s in enumerate(SLIDES, start=1):
        slide = prs.slides.add_slide(blank_layout)

        # Background
        bg = slide.background
        bg.fill.solid()
        bg.fill.fore_color.rgb = LIGHT_BG

        # Header band
        header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, Inches(1.15))
        header.fill.solid()
        header.fill.fore_color.rgb = NAVY
        header.line.fill.background()
        header.shadow.inherit = False

        title_box = header.text_frame
        title_box.word_wrap = True
        title_box.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = title_box.paragraphs[0]
        p.text = s["title"]
        p.font.size = Pt(30) if idx > 1 else Pt(34)
        p.font.bold = True
        p.font.color.rgb = WHITE
        title_box.margin_left = Inches(0.5)

        if idx == 1:
            # Title slide: subtitle + big centered treatment
            sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(3.0), SLIDE_W - Inches(1.6), Inches(1.5))
            tf = sub_box.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            run = p.add_run()
            run.text = s["subtitle"]
            run.font.size = Pt(22)
            run.font.color.rgb = TEAL
            run.font.italic = True

            accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(2.7), Inches(3.0), Pt(4))
            accent.fill.solid()
            accent.fill.fore_color.rgb = TEAL
            accent.line.fill.background()
        else:
            # Bullets
            body_top = Inches(1.5)
            body_box = slide.shapes.add_textbox(Inches(0.7), body_top, SLIDE_W - Inches(1.4), Inches(4.3))
            tf = body_box.text_frame
            tf.word_wrap = True
            for i, bullet in enumerate(s["bullets"]):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                p.text = "•  " + bullet
                p.font.size = Pt(19)
                p.font.color.rgb = TEXT_DARK
                p.space_after = Pt(12)

            # Annotation / insight callout
            if s["insight"]:
                callout_top = Inches(1.5) + Inches(0.62) * min(len(s["bullets"]), 6) + Inches(0.3)
                callout_top = min(callout_top, Inches(5.55))
                callout = slide.shapes.add_shape(
                    MSO_SHAPE.ROUNDED_RECTANGLE,
                    Inches(0.7), callout_top, SLIDE_W - Inches(1.4), Inches(1.05),
                )
                callout.fill.solid()
                callout.fill.fore_color.rgb = INSIGHT_BG
                callout.line.color.rgb = TEAL
                callout.line.width = Pt(1.25)
                callout.shadow.inherit = False
                ctf = callout.text_frame
                ctf.word_wrap = True
                ctf.vertical_anchor = MSO_ANCHOR.MIDDLE
                ctf.margin_left = Inches(0.2)
                ctf.margin_right = Inches(0.2)
                p = ctf.paragraphs[0]
                run = p.add_run()
                run.text = "\U0001F4A1 Key insight: " + s["insight"]
                run.font.size = Pt(15)
                run.font.italic = True
                run.font.bold = True
                run.font.color.rgb = NAVY

        add_footer_link(slide, prs, s["resource_label"], s["resource_url"])
        add_slide_number(slide, idx, total)
        set_notes(slide, s["notes"])

    out_path = "Chapter1_Regression_Presentation.pptx"
    prs.save(out_path)
    print(f"Saved {out_path} with {total} slides.")


if __name__ == "__main__":
    build()
