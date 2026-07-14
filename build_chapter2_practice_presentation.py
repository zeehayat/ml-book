"""
Build a "concept + exercise" PowerPoint presentation for Chapter 2
(Generalized Linear Models).

Each content slide covers exactly one concept, then poses one exercise to
practice it immediately. Speaker notes hold the exercise's worked answer.
The final slide is a comprehensive mastery checklist spanning the chapter.

Run:
    python3 build_chapter2_practice_presentation.py

Produces:
    Chapter2_GLM_Practice_Presentation.pptx
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ---------------------------------------------------------------------------
# Palette (matches the Chapter 1 practice deck for a consistent series look)
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
        "title": "Chapter 2: Generalized Linear Models",
        "subtitle": "One concept, one exercise per slide — practice as you go",
        "concept": [],
        "exercise": None,
        "notes": "Welcome slide. GLMs extend everything from Chapter 1 to outcomes that are counts, proportions, rates, and positive continuous values — not just outcomes that are well-approximated by a normal, constant-variance error.",
    },
    {
        "title": "Concept: The Three Components of a GLM",
        "concept": [
            "Random component: the response distribution (e.g. Bernoulli, Poisson, Gamma)",
            "Systematic component: the linear predictor η = Xβ (same as OLS)",
            "Link component: a function g connecting the mean to the linear predictor, g(μ) = η",
            "OLS is the special case: identity link, Gaussian random component",
        ],
        "exercise": "A model predicts number of customer complaints per week (a count, always ≥ 0). Explain why an ordinary linear regression's identity link is a poor systematic choice here, in terms of what values Xβ can take versus what values the outcome can take.",
        "notes": "Answer: Xβ ranges over all real numbers (including negative), but a count can never be negative. An identity link lets the model predict impossible negative counts. A log link (η = log(μ)) guarantees the predicted mean stays positive, since μ = exp(η) > 0 always.",
    },
    {
        "title": "Concept: The Exponential Family",
        "concept": [
            "A distribution family with the form f(y;θ,φ) = exp[(yθ − b(θ))/a(φ) + c(y,φ)]",
            "θ is the natural parameter; b(θ) determines the mean and variance",
            "Normal, Bernoulli, Poisson, and Gamma are all members of this one family",
            "This unification is why one algorithm (IRLS) fits every GLM in this chapter",
        ],
        "exercise": "Why does writing every response distribution in the same exponential-family form make it possible to build one shared fitting algorithm instead of a different one for Poisson, logistic, and Gamma regression?",
        "notes": "Answer: because the score equations and Fisher information take the same algebraic form for every exponential-family member (differing only in the variance function V(mu) and link), a single IRLS routine parameterized by the family and link can fit all of them — exactly the design choice made in this chapter's from-scratch implementation.",
    },
    {
        "title": "Concept: Link Functions and Canonical Links",
        "concept": [
            "A link function g maps the mean μ (restricted range) to η (any real number)",
            "Canonical link: the link equals the natural parameter θ, which simplifies the score equations",
            "Gaussian → identity; Poisson → log; Binomial → logit",
            "Non-canonical links are valid too (e.g. probit for binomial) — just less algebraically tidy",
        ],
        "exercise": "Match each outcome type to its canonical link: (a) a continuous, unbounded outcome, (b) a count, (c) a binary outcome. Choose from: identity, log, logit.",
        "notes": "Answer: (a) continuous unbounded -> identity (Gaussian). (b) count -> log (Poisson). (c) binary -> logit (Binomial/logistic).",
    },
    {
        "title": "Concept: Iteratively Reweighted Least Squares (IRLS)",
        "concept": [
            "GLMs have no closed-form solution (except OLS) — fitting is iterative",
            "Each step solves a weighted least squares problem on a \"working response\" z",
            "Weights come from the variance function; they change every iteration",
            "IRLS is Newton-Raphson / Fisher scoring applied to the GLM likelihood",
        ],
        "exercise": "Why can't Poisson regression be solved in one step the way OLS can, even though both use a linear predictor Xβ?",
        "notes": "Answer: OLS's normal equations are linear in beta because the Gaussian log-likelihood's gradient is linear in beta. Poisson regression's log link makes the mean mu = exp(Xbeta) a non-linear function of beta, so the score equations are non-linear and must be solved iteratively (IRLS), re-linearizing around the current estimate at each step.",
    },
    {
        "title": "Concept: Deviance",
        "concept": [
            "Deviance generalizes the residual sum of squares to any GLM family",
            "D = 2 × (log-likelihood of the saturated model − log-likelihood of the fitted model)",
            "Saturated model: one parameter per observation, a perfect (but useless) fit",
            "Smaller deviance = better fit, analogous to smaller SSE in OLS",
        ],
        "exercise": "Why is the saturated model — which fits the data perfectly — not itself a useful model, even though it appears in the deviance formula as the benchmark?",
        "notes": "Answer: the saturated model has as many parameters as observations, so it has zero degrees of freedom and generalizes to no new data at all — it exists purely as a fixed reference point to measure how much log-likelihood the fitted (parsimonious) model gives up, not as a candidate model itself.",
    },
    {
        "title": "Concept: Poisson Regression for Counts",
        "concept": [
            "Models counts (0, 1, 2, ...) with mean μ using the log link: log(μ) = Xβ",
            "Poisson assumption: variance equals the mean, Var(Y) = μ",
            "exp(βⱼ) is the multiplicative change in the expected COUNT per unit of xⱼ",
        ],
        "exercise": "A Poisson regression coefficient for \"years of experience\" is β = 0.05. Compute exp(β) and interpret it as a percentage change in expected count per additional year.",
        "notes": "Answer: exp(0.05) ≈ 1.051, i.e. each additional year of experience is associated with about a 5.1% increase in the expected count, holding other predictors constant.",
    },
    {
        "title": "Concept: Offsets and Exposure",
        "concept": [
            "An offset converts a count model into a RATE model: log(μ) = log(exposure) + Xβ",
            "The offset's coefficient is fixed at exactly 1 — not estimated",
            "Without an offset, a model can't distinguish \"more events\" from \"more time observed\"",
        ],
        "exercise": "Hospital A sees 40 infections in 1000 patient-days; Hospital B sees 30 infections in 500 patient-days. Without using an offset for patient-days, would a raw Poisson count model correctly identify which hospital has the worse infection RATE?",
        "notes": "Answer: no — without an offset, the model only sees counts 40 vs 30 and would treat Hospital A as \"worse,\" but the RATE is 40/1000=0.04 for A and 30/500=0.06 for B — Hospital B actually has the higher rate. An offset of log(patient-days) is required to model the rate correctly.",
    },
    {
        "title": "Concept: Overdispersion and the Negative Binomial",
        "concept": [
            "Overdispersion: observed variance exceeds what the Poisson assumption (Var=mean) predicts",
            "Common causes: unmeasured heterogeneity, clustering, excess zeros",
            "Negative binomial adds a dispersion parameter, allowing Var(Y) = μ + αμ²",
            "Ignoring overdispersion understates standard errors and overstates significance",
        ],
        "exercise": "A Poisson model's residual deviance is 350 on 200 degrees of freedom. Is this consistent with the Poisson assumption? What would you consider doing instead?",
        "notes": "Answer: the deviance-to-df ratio is 350/200 = 1.75, well above the expected ~1.0 for a well-fitting Poisson model — a sign of overdispersion. Consider a quasi-Poisson (inflates SEs by the dispersion estimate) or a negative binomial model, which estimates the extra variance explicitly.",
    },
    {
        "title": "Concept: Binomial Regression for Proportions",
        "concept": [
            "Models grouped proportions: k successes out of n trials per row",
            "Different from logistic regression on individual 0/1 outcomes only in data shape",
            "Uses the same logit link and Bernoulli-family likelihood, weighted by n",
        ],
        "exercise": "A dataset records, for each of 50 factories, the number of defective units out of that factory's daily production run. Why is grouped binomial regression more appropriate here than treating each unit as its own row in a plain logistic regression?",
        "notes": "Answer: both give the same coefficient estimates, but grouped binomial regression directly encodes the trial size n per row, making per-group residuals, leverage, and goodness-of-fit diagnostics operate at the natural (factory) unit of analysis rather than requiring one row per individual unit.",
    },
    {
        "title": "Concept: Gamma Regression for Positive Continuous Outcomes",
        "concept": [
            "Models strictly positive, continuous, often right-skewed outcomes (cost, duration)",
            "Typically paired with a log link: log(μ) = Xβ",
            "Gamma's variance is proportional to μ² — larger predicted means have larger absolute spread",
        ],
        "exercise": "Insurance claim amounts are always positive, right-skewed, and larger claims tend to have much larger variability than small claims. Why does this description point toward Gamma regression rather than OLS on the raw claim amount?",
        "notes": "Answer: OLS assumes constant error variance regardless of the predicted mean (homoskedasticity) and allows negative predictions — both violated here. Gamma regression keeps predictions positive (via the log link) and lets variance scale with the square of the mean, matching the described heteroskedastic, right-skewed claims data.",
    },
    {
        "title": "Concept: Model Diagnostics — Pearson and Deviance Residuals",
        "concept": [
            "Raw residuals (y − μ̂) aren't comparable across GLM families with different variance structures",
            "Pearson residual: (y − μ̂) / sqrt(V(μ̂)) — standardized by the variance function",
            "Deviance residual: signed square root of each observation's deviance contribution",
            "Both should show no pattern against fitted values if the model is well-specified",
        ],
        "exercise": "Why can't you just plot raw residuals (y − ŷ) for a Poisson model the same way you would for OLS, and expect them to look randomly scattered even when the model fits well?",
        "notes": "Answer: Poisson variance grows with the mean, so raw residuals will naturally fan out for larger predicted counts even under a correctly specified model — this isn't heteroskedasticity to fix, it's the expected mean-variance relationship. Pearson or deviance residuals standardize for this, so a well-fitting model shows no pattern in THEM specifically.",
    },
    {
        "title": "Concept: Model Comparison — AIC, BIC, and Likelihood Ratio Tests",
        "concept": [
            "AIC = -2×log-likelihood + 2×(number of parameters) — lower is better",
            "BIC penalizes extra parameters more heavily for large samples",
            "Likelihood ratio test compares two NESTED models via their deviance difference",
            "Only compare AIC/BIC across models fit to the exact same data and same response",
        ],
        "exercise": "Model A (3 parameters) has log-likelihood -200. Model B (5 parameters, nesting Model A) has log-likelihood -197. Compute AIC for both. Does the extra complexity in Model B look worthwhile by AIC?",
        "notes": "Answer: AIC_A = -2(-200)+2(3) = 406. AIC_B = -2(-197)+2(5) = 404. Model B has the lower (better) AIC, so the two extra parameters were worth their complexity cost by this criterion — though a likelihood ratio test (2*(197-200)=-6... note: use |difference|=6 against a chi-square with 2 df) should also be checked for statistical significance.",
    },
    {
        "title": "Concept: Regularized GLMs",
        "concept": [
            "Ridge/lasso/elastic-net penalties extend directly to the GLM log-likelihood",
            "Same motivation as Chapter 1: control overfitting, handle many/correlated predictors",
            "Penalty is added to the negative log-likelihood, not the sum of squared errors",
            "Feature standardization is still required before penalizing",
        ],
        "exercise": "You are fitting a logistic regression with 500 candidate predictors and only 300 observations. Why is ordinary (unpenalized) maximum likelihood estimation likely to fail here, and what class of fix does this chapter recommend?",
        "notes": "Answer: with more predictors than observations (or even just many relative to n), the likelihood surface can be flat or the MLE may not exist / be wildly unstable (related to separation, Section on rare events). A regularized GLM (ridge, lasso, or elastic net penalty on the log-likelihood) stabilizes the estimates and, with lasso, can perform variable selection.",
    },
    {
        "title": "Concept: Separation in Logistic Regression",
        "concept": [
            "Complete separation: a predictor (or combination) perfectly predicts the outcome",
            "The MLE for that coefficient runs off to infinity — the algorithm may not converge",
            "Common with rare events, small samples, or too many predictors",
            "Fix: Firth's penalized-likelihood correction, or a regularized/Bayesian approach",
        ],
        "exercise": "A logistic regression coefficient's estimate is 45.2 with a standard error of 8.1×10⁶. What does this combination strongly suggest is happening, and why would simply reporting this coefficient's p-value be misleading?",
        "notes": "Answer: this pattern (huge coefficient, enormous standard error) is the classic signature of (quasi-)separation — the predictor perfectly or near-perfectly separates the outcome, so the MLE is trying to diverge to infinity. The Wald p-value computed from this absurd standard error is meaningless; the fix is Firth's penalized likelihood or a regularized/Bayesian alternative, not trusting the reported number.",
    },
    {
        "title": "Concept: GEE vs. GLMM for Clustered Data",
        "concept": [
            "Ordinary GLMs assume independent observations — often false for repeated/clustered data",
            "GEE (Generalized Estimating Equations): estimates population-average effects",
            "GLMM (Generalized Linear Mixed Model): estimates cluster-specific effects via random effects",
            "The two give numerically different coefficients for non-linear links — both are \"correct\", answering different questions",
        ],
        "exercise": "A researcher wants to know \"on average across the population, how does the treatment change the probability of recovery?\" versus a doctor who wants to know \"for THIS specific patient, given their cluster's random effect, how does treatment change their probability?\" Which approach (GEE or GLMM) answers each question?",
        "notes": "Answer: the population-average question is answered by GEE. The patient-specific (cluster-specific) question is answered by GLMM. This is the non-collapsibility distinction: for non-linear links, marginal (GEE) and conditional (GLMM) effects are not the same number, and both are legitimate depending on which question is actually being asked.",
    },
    {
        "title": "Concept: Calibration vs. Discrimination",
        "concept": [
            "Discrimination (e.g. ROC-AUC): does the model rank higher-risk cases above lower-risk ones?",
            "Calibration: does a predicted probability of 0.3 actually correspond to ~30% observed frequency?",
            "A model can discriminate perfectly while being badly calibrated, and vice versa",
            "Calibration plots (predicted vs. observed by bin) reveal problems ROC-AUC cannot",
        ],
        "exercise": "A model achieves an excellent ROC-AUC of 0.90, but every predicted probability is roughly double the true observed rate in each risk bin. Has this model failed at discrimination, calibration, or both? Is it safe to use its raw probabilities for a business decision that depends on the actual probability value (not just the ranking)?",
        "notes": "Answer: discrimination is excellent (AUC 0.90 — it ranks cases correctly), but calibration has failed badly (probabilities are systematically off by 2x). It's fine for ranking-based decisions but unsafe for any decision relying on the probability's literal value (e.g. expected-cost calculations) until it is recalibrated.",
    },
    {
        "title": "Mastery Check: Are You Ready to Move On?",
        "concept": [
            "If you can answer every question below confidently, you have mastered Chapter 2.",
        ],
        "exercise": None,
        "mastery_questions": [
            "Name the three components of a GLM and explain what each one contributes.",
            "Explain why OLS is a special case of the GLM framework rather than a separate method.",
            "Explain why GLM fitting requires an iterative algorithm (IRLS) instead of a closed-form solution.",
            "Define deviance and explain what the saturated model represents in that definition.",
            "Explain the role of an offset in a Poisson rate model, and why its coefficient is fixed at 1.",
            "Explain what overdispersion is, how to detect it, and two ways to correct for it.",
            "Explain why Pearson or deviance residuals are used instead of raw residuals for GLM diagnostics.",
            "State the formula for AIC and explain what it trades off against BIC.",
            "Describe the signature (in coefficients and standard errors) of separation in logistic regression, and one fix.",
            "Explain the difference between a GEE population-average effect and a GLMM cluster-specific effect.",
            "Explain the difference between discrimination and calibration, with an example where they diverge.",
            "Fit a Poisson regression with an offset in statsmodels and interpret exp(beta) correctly.",
            "Explain why regularizing a GLM's log-likelihood requires the same feature standardization as Chapter 1's ridge/lasso.",
            "Give an example of an outcome type well-suited to Gamma regression, and explain why OLS would be a poor fit for it.",
        ],
        "notes": "This is a self-check slide. If any question causes hesitation, return to that concept's slide and its exercise before moving to Chapter 3.",
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

    out_path = "Chapter2_GLM_Practice_Presentation.pptx"
    prs.save(out_path)
    print(f"Saved {out_path} with {total} slides.")


if __name__ == "__main__":
    build()
