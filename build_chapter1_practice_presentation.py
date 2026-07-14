"""
Build a "concept + exercise" PowerPoint presentation for Chapter 1
(Linear Regression & Logistic Models).

Each content slide covers exactly one concept, then poses one exercise to
practice it immediately. Speaker notes hold the exercise's worked answer
(visible to a presenter/self-learner, not printed on the slide itself).
The final slide is a comprehensive mastery checklist spanning the chapter.

Run:
    python3 build_chapter1_practice_presentation.py

Produces:
    Chapter1_Regression_Practice_Presentation.pptx
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
NAVY = RGBColor(0x1B, 0x26, 0x4F)
AMBER = RGBColor(0xB8, 0x6A, 0x00)
AMBER_BG = RGBColor(0xFD, 0xF2, 0xE2)
LIGHT_BG = RGBColor(0xF7, 0xF9, 0xFC)
TEXT_DARK = RGBColor(0x22, 0x22, 0x2A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
MASTERY_BG = RGBColor(0xEA, 0xF6, 0xF4)
TEAL = RGBColor(0x0D, 0x94, 0x88)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

# ---------------------------------------------------------------------------
# Content: one concept + one exercise per slide
# ---------------------------------------------------------------------------
SLIDES = [
    {
        "title": "Chapter 1: Linear Regression & Logistic Models",
        "subtitle": "One concept, one exercise per slide — practice as you go",
        "concept": [],
        "exercise": None,
        "notes": "Welcome slide. Explain the format: every following slide teaches one idea, then immediately asks you to use it.",
    },
    {
        "title": "Concept: What Regression Is Trying to Do",
        "concept": [
            "Description: summarize how y relates to x in the data you have",
            "Prediction: guess y for a new observation's x",
            "Explanation: estimate how much x influences y, holding other things fixed",
            "Causal inference: what happens to y if we intervene and change x?",
        ],
        "exercise": "A retailer fits a model of sales vs. advertising spend to decide whether to increase next quarter's ad budget. Which of the four goals above does this require — and is a plain regression coefficient enough to justify the decision?",
        "notes": "Answer: this is a causal question (\"what happens if we intervene\"), not just prediction or association. A regression coefficient alone is not enough — it requires a causal design or explicit identifying assumptions (see the causal-caution slide later in this deck).",
    },
    {
        "title": "Concept: Slope and Intercept",
        "concept": [
            "A line is written y = a + b·x",
            "a (intercept): the value of y when x = 0",
            "b (slope): how much y changes per one-unit increase in x",
            "Rearranged: x = (y − a) / b",
        ],
        "exercise": "A line has intercept a = 5 and slope b = -2. What is y when x = 3? What value of x gives y = 1?",
        "notes": "Answer: y = 5 + (-2)(3) = -1. For y=1: 1 = 5 - 2x -> 2x = 4 -> x = 2.",
    },
    {
        "title": "Concept: Mean, Variance, and Covariance",
        "concept": [
            "Mean: add the values, divide by the count",
            "Variance: how spread out values are around the mean",
            "Covariance: whether two variables move together",
            "The simple regression slope is Cov(x,y) / Var(x)",
        ],
        "exercise": "For x = [1, 2, 3] and y = [2, 4, 6], compute x̄, ȳ, and the slope of the regression line using slope = Cov(x,y)/Var(x).",
        "notes": "Answer: x̄=2, ȳ=4. Cov(x,y) = mean of (xi-x̄)(yi-ȳ) = [(-1)(-2)+(0)(0)+(1)(2)]/3 = 4/3 (or 2 with the (n-1) sample convention). Var(x) using the same convention gives slope = 2 exactly — y is exactly 2x here, so the fit should recover slope=2 precisely.",
    },
    {
        "title": "Concept: Least Squares and Residuals",
        "concept": [
            "Residual: e = y − ŷ (observed minus predicted)",
            "OLS chooses the line minimizing the sum of squared residuals Σe²",
            "Squaring penalizes large mistakes more and has clean calculus",
            "ε (true disturbance) ≠ e (observed residual) — e is only an estimate of ε",
        ],
        "exercise": "A fitted line gives ŷ = 4.6 at x = 4. The observed value is y = 4. What is the residual? Is the model over- or under-predicting at this point?",
        "notes": "Answer: e = y - ŷ = 4 - 4.6 = -0.6. The model is over-predicting (residual is negative).",
    },
    {
        "title": "Concept: Multiple Regression and the Design Matrix",
        "concept": [
            "y = β₀ + β₁x₁ + β₂x₂ + ... + ε — many predictors, one model",
            "Stack predictors into a design matrix X; solve β̂ = (XᵀX)⁻¹Xᵀy",
            "Each βⱼ is interpreted \"holding all other predictors constant\"",
        ],
        "exercise": "A model predicts house price from square footage (x1) and number of bedrooms (x2), giving β1 = 150 and β2 = 5000. What does β1 = 150 mean in plain language?",
        "notes": "Answer: holding the number of bedrooms constant, each additional square foot is associated with a $150 increase in predicted price.",
    },
    {
        "title": "Concept: The Assumptions Behind OLS",
        "concept": [
            "Linearity, independence, no perfect multicollinearity",
            "Zero conditional mean: E[ε|X] = 0 — no systematic error given predictors",
            "Homoskedasticity: constant error variance across X",
            "Normality: needed for exact small-sample inference",
        ],
        "exercise": "A residual-vs-fitted plot shows a clear funnel shape (residuals fan out as fitted values increase). Which assumption is violated, and which part of the analysis does it put at risk — the coefficients themselves, or the standard errors?",
        "notes": "Answer: heteroskedasticity (non-constant error variance) is violated. Coefficients remain unbiased, but the classical standard errors (and therefore CIs and p-values) become invalid — use robust standard errors instead.",
    },
    {
        "title": "Concept: Standard Errors, Confidence Intervals, p-values",
        "concept": [
            "A coefficient's standard error measures how much it would vary across fresh samples",
            "A 95% CI is a range of plausible values built from the estimate ± SE",
            "A p-value asks: how surprising is this data if the true effect were zero?",
            "Statistical significance ≠ practical importance",
        ],
        "exercise": "A coefficient estimate is 0.8 with a 95% CI of [0.75, 0.85]. A second coefficient is 12.0 with a 95% CI of [-3.0, 27.0]. Which estimate is more precise, and which is more likely to be practically important if both are statistically significant?",
        "notes": "Answer: the first estimate (0.8) is far more precise (narrow CI). The second, despite a much larger point estimate, is imprecise and includes zero-adjacent values — precision and magnitude are separate questions.",
    },
    {
        "title": "Concept: R² and Measuring Fit",
        "concept": [
            "R² = 1 − (SS_residual / SS_total): proportion of variance explained",
            "R² = 0 means no better than predicting the mean every time",
            "Adding any predictor can only increase raw R² — use adjusted R² to penalize this",
            "Low R² does not mean a coefficient is unimportant",
        ],
        "exercise": "A model has SS_total = 500 and SS_residual = 100. Compute R². If a second predictor is added and SS_residual drops to 95, did the model necessarily improve in a meaningful way?",
        "notes": "Answer: R² = 1 - 100/500 = 0.80. Adding the predictor raised R² slightly (to 0.81) but raw R² always rises or stays flat when adding predictors — check adjusted R² or cross-validated performance before concluding real improvement.",
    },
    {
        "title": "Concept: Residual Diagnostics",
        "concept": [
            "Residuals vs. fitted: look for curves (non-linearity) or funnels (heteroskedasticity)",
            "Q-Q plot: checks whether residuals are normally distributed",
            "Leverage and Cook's distance: flag unusual or influential points",
            "A model that runs without error can still badly violate its assumptions",
        ],
        "exercise": "Anscombe's Quartet is four datasets with identical means, variances, correlation, and regression line — but very different shapes. What single diagnostic step would reveal the difference that summary statistics hide?",
        "notes": "Answer: plotting the data and the residuals. Summary statistics (R², coefficients) can be identical while the underlying relationship (linear, curved, outlier-driven) is completely different — always plot before trusting a fit.",
    },
    {
        "title": "Concept: Multicollinearity",
        "concept": [
            "Highly correlated predictors make individual coefficients unstable",
            "Predicted values usually remain reliable even when coefficients are shaky",
            "VIF (Variance Inflation Factor) quantifies the instability",
            "VIF above roughly 5–10 is commonly flagged as concerning",
        ],
        "exercise": "Two predictors — height in centimeters and height in inches — are both included in a regression. Predict what will happen to their individual coefficients and standard errors, and explain why the model's overall predictions are still likely to be fine.",
        "notes": "Answer: the two predictors are perfectly (or near-perfectly) collinear, so their coefficients become unstable/arbitrarily split between them with inflated standard errors — but since together they carry the same information, the combined prediction (and R²) is unaffected.",
    },
    {
        "title": "Concept: Ridge and Lasso Regularization",
        "concept": [
            "Ridge adds a penalty λΣβⱼ² — shrinks coefficients toward zero, keeps them all",
            "Lasso adds a penalty λΣ|βⱼ| — can shrink coefficients to exactly zero",
            "Regularization requires standardized features first",
            "More regularization = more bias, less variance",
        ],
        "exercise": "You have 100 candidate predictors and suspect only 5 genuinely matter. Would ridge or lasso be more useful for identifying which 5? Why?",
        "notes": "Answer: lasso — its L1 penalty's corner geometry can drive irrelevant coefficients to exactly zero, effectively performing feature selection. Ridge shrinks all coefficients toward zero but essentially never sets them to exactly zero.",
    },
    {
        "title": "Concept: Logistic Regression and Log-Odds",
        "concept": [
            "For binary outcomes, model the log-odds: logit(p) = log(p/(1−p)) = β₀+β₁x",
            "Invert to get a probability: p = 1 / (1 + exp(−(β₀+β₁x)))",
            "exp(βⱼ) is the multiplicative change in ODDS, not probability, per unit of x",
        ],
        "exercise": "A probability of an event is p = 0.8. Compute the odds. Then, if a coefficient's exp(β) = 1.5, what happens to the odds (not the probability) when x increases by 1?",
        "notes": "Answer: odds = 0.8/0.2 = 4 (4-to-1 in favor). If exp(beta)=1.5, the odds multiply by 1.5 for a one-unit increase in x — the odds become 6, though the corresponding change in probability is smaller and depends on the starting point.",
    },
    {
        "title": "Concept: Evaluating Classification Models",
        "concept": [
            "Precision: of predicted positives, how many were correct?",
            "Recall: of actual positives, how many did we find?",
            "Accuracy is misleading when classes are imbalanced",
            "ROC-AUC and PR-AUC summarize performance across all thresholds",
        ],
        "exercise": "Out of 100 emails, a spam filter flags 20 as spam. Of those, 15 are genuinely spam. There were 25 genuine spam emails total. Compute precision and recall.",
        "notes": "Answer: precision = 15/20 = 0.75 (75% of flagged emails were truly spam). Recall = 15/25 = 0.60 (60% of actual spam was caught).",
    },
    {
        "title": "Concept: Cross-Validation and Data Leakage",
        "concept": [
            "Train/test split: hold out data to estimate out-of-sample performance",
            "K-fold cross-validation: repeat the split k times for a more stable estimate",
            "Leakage: letting test-set information influence training",
            "Fit scaling/imputation only on the training fold — never on the full dataset first",
        ],
        "exercise": "A colleague standardizes the entire dataset (mean/variance from all rows) before doing a train/test split, then reports high test accuracy. What is wrong with this workflow?",
        "notes": "Answer: this is data leakage — the scaler's mean/variance were computed using test-set rows, so information from the test set leaked into training. The reported test accuracy is overly optimistic. Fix: fit the scaler on the training fold only, then apply it to the test fold.",
    },
    {
        "title": "Concept: From-Scratch vs. Library Implementations",
        "concept": [
            "Pure Python: every calculation explicit — nothing is a black box",
            "NumPy: vectorized linear algebra underneath almost everything else",
            "statsmodels: rich inference — coefficients, SEs, p-values, diagnostics",
            "scikit-learn: fast, consistent pipelines for prediction",
        ],
        "exercise": "You need to report a coefficient's p-value and confidence interval for a research paper. Which library is the right tool, and why not the other one?",
        "notes": "Answer: statsmodels — it is built for statistical inference (SEs, p-values, CIs) and reports them directly. scikit-learn is optimized for prediction pipelines and does not report inferential statistics by default.",
    },
    {
        "title": "Concept: Common Mistakes",
        "concept": [
            "\"Significant\" ≠ \"important\" — always report magnitude and units",
            "Scaling before splitting causes leakage",
            "Reusing the test set repeatedly turns it into training information",
            "Deleting every outlier without investigating first",
        ],
        "exercise": "A model reports p < 0.001 for a coefficient of 0.0003 on an outcome measured in millions of dollars. Is this result practically important? What else would you want to know?",
        "notes": "Answer: not necessarily — statistical significance with a huge sample size can produce a tiny, practically irrelevant p-value. You'd want the coefficient's real-world magnitude in context (0.0003 per unit of what, over what realistic range of x?) and a confidence interval, not just the p-value.",
    },
    {
        "title": "Concept: Causal Caution — Correlation Is Not Causation",
        "concept": [
            "A confounder is an unmeasured factor driving both x and y",
            "Omitted-variable bias: leaving out a confounder correlated with x biases β̂",
            "High predictive accuracy does not imply a safe intervention",
            "Causal claims need a causal design, not just a regression coefficient",
        ],
        "exercise": "Ice cream sales strongly predict drowning deaths across months. Identify the confounder, and explain why reducing ice cream sales would not reduce drownings.",
        "notes": "Answer: the confounder is hot weather, which independently increases both ice cream sales and swimming (and therefore drowning risk). The association is real but not causal — intervening on ice cream sales would not change drowning rates.",
    },
    {
        "title": "Mastery Check: Are You Ready to Move On?",
        "concept": [
            "If you can answer every question below confidently, you have mastered Chapter 1.",
        ],
        "exercise": None,
        "mastery_questions": [
            "Explain the difference between epsilon (the true disturbance) and e (the observed residual).",
            "Derive the simple regression slope b1 = Sxy/Sxx from the normal equations.",
            "State all five core OLS assumptions and name which guarantee each one supports.",
            "Explain why a prediction interval is always wider than a confidence interval for the mean.",
            "Give an example where R² is low but a coefficient is still substantively important.",
            "Explain why near multicollinearity damages coefficient interpretation but not necessarily prediction.",
            "Explain why ridge requires feature scaling and why lasso can produce exactly-zero coefficients.",
            "Convert a probability to odds and to log-odds, and explain what exp(beta) means in logistic regression.",
            "Explain why accuracy is a misleading metric under severe class imbalance.",
            "Explain why fitting a scaler on the full dataset before splitting causes data leakage.",
            "Implement simple linear regression from scratch in pure Python and verify it against a known example.",
            "Fit the same model in statsmodels and scikit-learn and explain why you'd choose one over the other.",
            "Give a concrete example where a regression shows strong association but no plausible causal path.",
            "Explain, in one sentence each, what a p-value is and what it is NOT.",
        ],
        "notes": "This is a self-check slide. If any question causes hesitation, return to that concept's slide and its exercise before moving to Chapter 2.",
    },
]


def add_slide_number(slide, n, total):
    box = slide.shapes.add_textbox(SLIDE_W - Inches(1.3), SLIDE_H - Inches(0.5), Inches(1.0), Inches(0.4))
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
        bg = slide.background
        bg.fill.solid()
        bg.fill.fore_color.rgb = LIGHT_BG

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
        p.font.size = Pt(30) if idx > 1 else Pt(32)
        p.font.bold = True
        p.font.color.rgb = WHITE
        title_box.margin_left = Inches(0.5)

        if idx == 1:
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

        elif "mastery_questions" in s:
            body_box = slide.shapes.add_textbox(Inches(0.6), Inches(1.35), SLIDE_W - Inches(1.2), Inches(0.5))
            tf = body_box.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            run = p.add_run()
            run.text = s["concept"][0]
            run.font.size = Pt(15)
            run.font.italic = True
            run.font.color.rgb = TEAL

            box = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(1.85), SLIDE_W - Inches(1.2), Inches(5.35)
            )
            box.fill.solid()
            box.fill.fore_color.rgb = MASTERY_BG
            box.line.color.rgb = TEAL
            box.line.width = Pt(1.25)
            box.shadow.inherit = False
            ctf = box.text_frame
            ctf.word_wrap = True
            ctf.margin_left = Inches(0.25)
            ctf.margin_right = Inches(0.25)
            ctf.margin_top = Inches(0.15)
            for i, q in enumerate(s["mastery_questions"]):
                p = ctf.paragraphs[0] if i == 0 else ctf.add_paragraph()
                p.text = f"{i+1}.  {q}"
                p.font.size = Pt(12.5)
                p.font.color.rgb = NAVY
                p.space_after = Pt(6)
                p.alignment = PP_ALIGN.LEFT

        else:
            body_box = slide.shapes.add_textbox(Inches(0.7), Inches(1.4), SLIDE_W - Inches(1.4), Inches(3.0))
            tf = body_box.text_frame
            tf.word_wrap = True
            for i, bullet in enumerate(s["concept"]):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                p.text = "•  " + bullet
                p.font.size = Pt(18)
                p.font.color.rgb = TEXT_DARK
                p.space_after = Pt(10)

            ex_top = Inches(1.4) + Inches(0.56) * len(s["concept"]) + Inches(0.35)
            ex_top = min(ex_top, Inches(5.1))
            ex_box = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), ex_top, SLIDE_W - Inches(1.4), Inches(1.7)
            )
            ex_box.fill.solid()
            ex_box.fill.fore_color.rgb = AMBER_BG
            ex_box.line.color.rgb = AMBER
            ex_box.line.width = Pt(1.25)
            ex_box.shadow.inherit = False
            etf = ex_box.text_frame
            etf.word_wrap = True
            etf.vertical_anchor = MSO_ANCHOR.MIDDLE
            etf.margin_left = Inches(0.25)
            etf.margin_right = Inches(0.25)
            p = etf.paragraphs[0]
            run = p.add_run()
            run.text = "✏️ Exercise: " + s["exercise"]
            run.font.size = Pt(15)
            run.font.bold = True
            run.font.color.rgb = AMBER

        add_slide_number(slide, idx, total)
        set_notes(slide, s["notes"])

    out_path = "Chapter1_Regression_Practice_Presentation.pptx"
    prs.save(out_path)
    print(f"Saved {out_path} with {total} slides.")


if __name__ == "__main__":
    build()
