"""
Build a "concept + exercise" PowerPoint presentation for Chapter 6
(Non-Linear Spaces & Regularization: SVMs and Decision Trees).

Run:
    python3 build_chapter6_practice_presentation.py

Produces:
    Chapter6_SVM_Trees_Practice_Presentation.pptx
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

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

SLIDES = [
    {
        "title": "Chapter 6: Non-Linear Spaces & Regularization",
        "subtitle": "One concept, one exercise per slide — practice as you go",
        "concept": [],
        "exercise": None,
        "notes": "Welcome slide. Two very different responses to the same problem — linear models can't separate every dataset — meet here: Support Vector Machines and Decision Trees.",
    },
    {
        "title": "Concept: The Maximum-Margin Hyperplane",
        "concept": [
            "An SVM finds the separating hyperplane with the WIDEST possible margin",
            "Primal objective: minimize (1/2)||w||² subject to yᵢ(w·xᵢ+b) ≥ 1 for every point",
            "A wider margin means a more confident, more generalizable decision boundary",
        ],
        "exercise": "Why does minimizing ||w||² (rather than maximizing something directly) correspond to maximizing the margin?",
        "notes": "Answer: for a fixed set of support-vector constraints, the geometric margin equals 2/||w|| — so making ||w|| as small as possible while still satisfying the constraints is mathematically equivalent to making the margin as wide as possible.",
    },
    {
        "title": "Concept: Soft Margins and the Parameter C",
        "concept": [
            "Real data is rarely perfectly separable — slack variables ξᵢ ≥ 0 allow margin violations",
            "Objective adds a penalty: (1/2)||w||² + C·Σξᵢ",
            "Large C: few violations tolerated, narrower margin, more overfitting risk",
            "Small C: more violations tolerated, wider margin, more regularization",
        ],
        "exercise": "You train an SVM and it overfits badly, achieving perfect training accuracy but poor test accuracy. Should you increase or decrease C?",
        "notes": "Answer: decrease C — a smaller C tolerates more margin violations, effectively increasing regularization and widening the margin, which should reduce overfitting.",
    },
    {
        "title": "Concept: The Dual Problem and the Kernel Trick",
        "concept": [
            "The SVM dual depends on training data ONLY through pairwise inner products xᵢ·xⱼ",
            "A kernel K(xᵢ,xⱼ) can replace this inner product with one computed in a much higher-dimensional space",
            "The high-dimensional coordinates are NEVER actually computed — only the kernel values are",
            "This is how SVMs draw curved decision boundaries using only a linear-separator algorithm",
        ],
        "exercise": "Explain why the kernel trick lets an SVM separate data that is not linearly separable in its original feature space, without ever explicitly transforming the data.",
        "notes": "Answer: the kernel function implicitly computes what the inner product WOULD be after mapping the data into a higher-dimensional space where it might become linearly separable — since the entire algorithm only ever needs these inner products (never the raw high-dimensional coordinates), the transformation is 'free' computationally.",
    },
    {
        "title": "Concept: Mercer Kernels — Linear, Polynomial, RBF",
        "concept": [
            "Linear kernel: K(x,z) = x·z — no transformation, a standard linear SVM",
            "Polynomial kernel: K(x,z) = (x·z + c)^d — curved, polynomial-degree boundaries",
            "RBF (Gaussian) kernel: K(x,z) = exp(-γ||x-z||²) — infinitely flexible, local similarity",
        ],
        "exercise": "A classic XOR-style dataset (two diagonal clusters of each class) cannot be separated by any straight line. Which kernel would you reach for first, and why?",
        "notes": "Answer: the RBF kernel — it measures local similarity rather than assuming any fixed polynomial shape, and is the standard first choice for non-linearly-separable data like XOR, where no simple linear or low-degree-polynomial boundary exists.",
    },
    {
        "title": "Concept: Shannon Entropy and Information Gain",
        "concept": [
            "Entropy H(S) = -Σ pₖ log₂(pₖ) measures how mixed a node's classes are",
            "H=0 means a node is perfectly pure (one class only); H is maximal when classes are balanced",
            "Information gain = parent entropy − weighted average of children's entropy",
            "A decision tree picks, at every node, the split that MAXIMIZES information gain",
        ],
        "exercise": "A node has 8 examples: 4 of class A, 4 of class B. A candidate split produces two children, each perfectly pure (all A or all B). What is the information gain?",
        "notes": "Answer: parent entropy H = -0.5*log2(0.5) - 0.5*log2(0.5) = 1.0 (maximally mixed). Each child has entropy 0 (pure), so the weighted average child entropy is 0. Information gain = 1.0 - 0 = 1.0 — the maximum possible gain for a binary split.",
    },
    {
        "title": "Concept: Why Decision Trees Overfit Without Limits",
        "concept": [
            "An unconstrained tree keeps splitting until every leaf is perfectly pure",
            "This means it can memorize the training data completely, including its noise",
            "Regularize via max_depth, min_samples_split, min_samples_leaf, or pruning",
        ],
        "exercise": "A decision tree trained with no depth limit achieves 100% training accuracy but only 65% test accuracy. What's the most direct first fix to try?",
        "notes": "Answer: set a max_depth (or min_samples_leaf) limit and re-tune it via cross-validation — the 100% training accuracy with a large train-test gap is the classic signature of an unconstrained tree memorizing noise.",
    },
    {
        "title": "Concept: Random Forests — Averaging Away Variance",
        "concept": [
            "Train many trees, each on a bootstrap-resampled subset of the data",
            "Each split considers only a random subset of features (typically √D)",
            "Averaging many decorrelated trees' predictions reduces variance without increasing bias much",
        ],
        "exercise": "Why does training many trees on the SAME full dataset with the SAME features (no bootstrapping, no feature subsampling) fail to give the variance-reduction benefit of a real random forest?",
        "notes": "Answer: without randomization (bootstrap sampling and feature subsampling), every tree would tend to make very similar splits and errors — their errors would be highly CORRELATED, so averaging them provides little variance reduction. The randomization is what decorrelates the trees' mistakes.",
    },
    {
        "title": "Concept: L1 vs. L2 Regularization Geometry",
        "concept": [
            "L1 (lasso) constraint region is a diamond with corners on the coordinate axes",
            "L2 (ridge) constraint region is a smooth sphere/circle",
            "The loss contours are more likely to touch a DIAMOND's corner — producing exact zeros",
            "A smooth sphere has no corners — coefficients shrink but rarely hit exactly zero",
        ],
        "exercise": "Using the geometric picture (not the calculus), explain in one or two sentences why lasso can perform feature selection but ridge cannot.",
        "notes": "Answer: because the L1 diamond's corners lie exactly on the coordinate axes (where one coefficient is zero), the loss function's contour lines are geometrically likely to first touch the constraint region AT a corner — setting that coefficient to exactly zero. The L2 circle has no such corners, so the tangent point almost never lands exactly on an axis.",
    },
    {
        "title": "Concept: SVM Margin Maximization Is Also Regularization",
        "concept": [
            "SVM's (1/2)||w||² term is structurally identical to L2 (ridge) regularization",
            "Margin maximization, decision-tree depth limits, and weight decay all share ONE principle:",
            "Penalize model complexity to prevent memorizing noise — the same idea, three different geometries",
        ],
        "exercise": "Name the three chapter concepts (from this deck and Chapter 1) that are all instances of the exact same 'penalize complexity' principle, just applied in different model families.",
        "notes": "Answer: SVM margin maximization (via ||w||^2), L2/ridge regularization on linear/logistic regression coefficients, and decision tree depth/leaf-size limits — all trade a bit of training fit for better generalization by constraining the model's effective complexity.",
    },
    {
        "title": "Mastery Check: Are You Ready to Move On?",
        "concept": [
            "If you can answer every question below confidently, you have mastered Chapter 6.",
        ],
        "exercise": None,
        "mastery_questions": [
            "Explain why minimizing ||w||^2 is equivalent to maximizing the SVM's margin.",
            "Explain the role of the parameter C and how it trades off margin width against violations.",
            "Explain the kernel trick and why it avoids ever computing high-dimensional coordinates explicitly.",
            "Compare linear, polynomial, and RBF kernels and when you'd choose each.",
            "Compute Shannon entropy and information gain for a given node split by hand.",
            "Explain why an unconstrained decision tree always reaches 100% training accuracy.",
            "Explain why random forests need BOTH bootstrap sampling and feature subsampling to reduce variance.",
            "Explain, geometrically, why lasso produces sparse solutions and ridge does not.",
            "Explain why SVM margin maximization, ridge regularization, and tree depth limits are the same underlying idea.",
            "Implement a simplified SMO solver or a recursive decision tree from scratch.",
        ],
        "notes": "This is a self-check slide. If any question causes hesitation, return to that concept's slide and its exercise before moving to Chapter 7.",
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
        p.font.size = Pt(28) if idx > 1 else Pt(32)
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
                p.font.size = Pt(14.5)
                p.font.color.rgb = NAVY
                p.space_after = Pt(9)
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

    out_path = "Chapter6_SVM_Trees_Practice_Presentation.pptx"
    prs.save(out_path)
    print(f"Saved {out_path} with {total} slides.")


if __name__ == "__main__":
    build()
