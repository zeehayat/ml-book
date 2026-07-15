"""
Build a "concept + exercise" PowerPoint presentation for Chapter 9
(Foundations of Neural Spaces: Perceptrons & Activation Functions).

Run:
    python3 build_chapter9_practice_presentation.py

Produces:
    Chapter9_Perceptrons_Practice_Presentation.pptx
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
        "title": "Chapter 9: Foundations of Neural Spaces",
        "subtitle": "One concept, one exercise per slide — practice as you go",
        "concept": [],
        "exercise": None,
        "notes": "Welcome slide. Before deep learning, there was a single neuron. This chapter builds the perceptron, proves its guarantees and its limits, and derives the activation functions that let networks escape those limits.",
    },
    {
        "title": "Concept: The Perceptron",
        "concept": [
            "Computes a linear score s = w·x + b, then applies a hard threshold: predict 1 if s ≥ 0, else 0",
            "Rosenblatt's update rule: when misclassified, wⱼ ← wⱼ + η(t−prediction)xⱼ",
            "The update nudges the decision boundary toward the misclassified point",
        ],
        "exercise": "A perceptron with w=[0,0], b=0 sees the example x=[1,2], target=1. It predicts 0 (since score=0 is not ≥ threshold... assume threshold requires strictly positive). Using learning_rate=1, compute the updated weights and bias.",
        "notes": "Answer: error = target - prediction = 1 - 0 = 1. w_new = [0,0] + 1*1*[1,2] = [1,2]. b_new = 0 + 1*1 = 1.",
    },
    {
        "title": "Concept: The Perceptron Convergence Theorem",
        "concept": [
            "IF the training data is linearly separable, the perceptron is GUARANTEED to converge",
            "Bound: at most R²/γ² updates, where R is the data's radius and γ is the margin",
            "A smaller margin (harder-to-separate data) means a LARGER guaranteed bound on updates needed",
        ],
        "exercise": "Why does the theorem's bound grow without limit as the margin γ shrinks toward zero? What does this predict about training time on nearly-inseparable data?",
        "notes": "Answer: the bound R^2/gamma^2 has gamma in the denominator (squared), so as gamma approaches 0, the bound approaches infinity — nearly-inseparable data (a tiny margin) can require an enormous number of updates before convergence, even though convergence is still technically guaranteed as long as SOME positive margin exists.",
    },
    {
        "title": "Concept: The Linear Stacking Collapse Theorem",
        "concept": [
            "Stacking two linear layers with NO activation between them: W₂(W₁x) = (W₂W₁)x",
            "This is just ONE linear layer with combined weights — no extra representational power",
            "This holds no matter how many linear-only layers you stack — depth alone accomplishes nothing without non-linearity",
        ],
        "exercise": "A colleague builds a 10-layer neural network but forgets to add any activation functions between the layers. How many DISTINCT linear regions can this network actually represent, no matter how it's trained?",
        "notes": "Answer: exactly the same as a single linear layer — the entire 10-layer stack collapses algebraically into one combined weight matrix and bias, so it can only represent one single hyperplane decision boundary, identical in power to a single perceptron.",
    },
    {
        "title": "Concept: Why the Perceptron Cannot Learn XOR",
        "concept": [
            "XOR: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0 — the positive and negative classes are NOT linearly separable",
            "No single straight line can separate the two classes",
            "Minsky and Papert's 1969 proof of this fact triggered the first \"AI winter\"",
        ],
        "exercise": "Plot the four XOR points mentally on a 2D grid. Explain in one sentence why no single straight line can separate the two classes.",
        "notes": "Answer: the positive-class points (0,1) and (1,0) sit on opposite corners from each other, and so do the negative-class points (0,0) and (1,1) — the two classes are arranged diagonally interleaved, so any straight line that puts one positive point on the correct side inevitably misclassifies at least one other point.",
    },
    {
        "title": "Concept: Escaping XOR — The Hidden Layer",
        "concept": [
            "Add ONE hidden layer with a non-linear activation between the layers",
            "The hidden layer \"folds\" the input space so the previously inseparable classes become separable",
            "A hand-built 2-neuron ReLU hidden layer can solve XOR exactly, with no training at all — just the right weights",
        ],
        "exercise": "Two ReLU hidden units compute h1 = ReLU(x1+x2) and h2 = ReLU(x1+x2−1). The output is h1 − 2·h2. Evaluate this for input (1,1) and check it matches XOR's expected output of 0.",
        "notes": "Answer: h1 = ReLU(1+1) = 2. h2 = ReLU(1+1-1) = ReLU(1) = 1. Output = 2 - 2*1 = 0 — correctly matches XOR(1,1)=0.",
    },
    {
        "title": "Concept: Sigmoid and Tanh — Derivatives and Saturation",
        "concept": [
            "Sigmoid: σ(z) = 1/(1+e⁻ᶻ), derivative σ'(z) = σ(z)(1−σ(z)), max value 0.25 at z=0",
            "Tanh: derivative 1 − tanh²(z), max value 1 at z=0",
            "Both derivatives approach exactly ZERO as |z| grows large — this is \"saturation\"",
        ],
        "exercise": "A neuron's pre-activation value is z=15 under a sigmoid activation. Is this neuron's gradient likely to be useful for learning, or effectively dead? Why?",
        "notes": "Answer: effectively dead — sigmoid saturates hard for large |z|; at z=15, sigmoid(z) is indistinguishable from 1.0 in floating point, and its derivative sigma(z)(1-sigma(z)) is essentially zero, meaning almost no gradient signal will flow backward through this neuron.",
    },
    {
        "title": "Concept: The Vanishing Gradient Problem",
        "concept": [
            "Backpropagating through many saturating (sigmoid/tanh) layers multiplies many small derivatives together",
            "Sigmoid's derivative maxes out at 0.25 — through 10 layers, the gradient can shrink by a factor of up to 0.25¹⁰",
            "This geometric decay made networks deeper than a couple of layers effectively untrainable for two decades",
        ],
        "exercise": "If every layer's sigmoid derivative happens to be at its maximum value of 0.25, roughly what fraction of the original gradient survives after passing back through 5 layers?",
        "notes": "Answer: 0.25^5 = 1/1024 ≈ 0.001, less than 0.1% of the original gradient — and this is the BEST case (derivative at its maximum); in practice with typical pre-activation values, the decay is often far worse. This is the mathematical origin of the vanishing gradient problem.",
    },
    {
        "title": "Concept: Why ReLU Breaks the Vanishing Gradient",
        "concept": [
            "ReLU(z) = max(0, z) — derivative is EXACTLY 1 for z>0, not a shrinking fraction",
            "Stacking many ReLU layers does NOT geometrically shrink the gradient the way sigmoid does",
            "Trade-off: ReLU's derivative is exactly 0 for z≤0 — a \"dying ReLU\" can get permanently stuck",
        ],
        "exercise": "Explain why ReLU solves the vanishing-gradient problem for POSITIVE pre-activations, but introduces a different failure mode for NEGATIVE ones.",
        "notes": "Answer: for z>0, ReLU's derivative is exactly 1, so gradients pass through unchanged (no geometric shrinkage across depth, unlike sigmoid). But for z≤0, the derivative is exactly 0 — if a neuron's weights/inputs keep it permanently in this negative regime, it stops receiving any gradient signal at all and can never recover ('dying ReLU'), a different but real pathology.",
    },
    {
        "title": "Mastery Check: Are You Ready to Move On?",
        "concept": [
            "If you can answer every question below confidently, you have mastered Chapter 9.",
        ],
        "exercise": None,
        "mastery_questions": [
            "Write the Rosenblatt perceptron update rule and apply it by hand to one misclassified example.",
            "State the Perceptron Convergence Theorem and explain what R and gamma represent.",
            "Prove (in outline) the Linear Stacking Collapse Theorem for a zero-bias two-layer network.",
            "Explain why XOR is not linearly separable and why this defeated the original perceptron.",
            "Explain how adding a hidden layer with a non-linear activation solves XOR.",
            "Derive the sigmoid derivative and state its maximum value and where it occurs.",
            "Explain the vanishing gradient problem as a consequence of the chain rule and saturating derivatives.",
            "Explain why ReLU avoids vanishing gradients on its positive side, and describe the 'dying ReLU' failure mode.",
            "Explain why a bias term is necessary for a perceptron to represent decision boundaries not through the origin.",
            "Implement a perceptron from scratch and empirically verify it converges on AND/OR but not on XOR.",
        ],
        "notes": "This is a self-check slide. If any question causes hesitation, return to that concept's slide and its exercise before moving to Chapter 10.",
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

    out_path = "Chapter9_Perceptrons_Practice_Presentation.pptx"
    prs.save(out_path)
    print(f"Saved {out_path} with {total} slides.")


if __name__ == "__main__":
    build()
