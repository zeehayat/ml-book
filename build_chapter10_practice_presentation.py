"""
Build a "concept + exercise" PowerPoint presentation for Chapter 10
(Deep Feedforward Networks & Backpropagation).

Run:
    python3 build_chapter10_practice_presentation.py

Produces:
    Chapter10_Backprop_Practice_Presentation.pptx
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
        "title": "Chapter 10: Deep Feedforward Networks & Backpropagation",
        "subtitle": "One concept, one exercise per slide — practice as you go",
        "concept": [],
        "exercise": None,
        "notes": "Welcome slide. This chapter is where every earlier idea — the chain rule from Chapter 4, activation functions from Chapter 9 — combines into one trainable deep network, built and verified from scratch.",
    },
    {
        "title": "Concept: The Multidimensional Chain Rule",
        "concept": [
            "A deep network is a chain of matrix-valued functions, not scalar ones",
            "The same chain rule from Chapter 4 applies — just with matrices and vectors instead of scalars",
            "Backpropagation is not a new algorithm — it is the chain rule applied systematically, layer by layer",
        ],
        "exercise": "Why is it accurate to say 'backpropagation is not a new optimization algorithm' when it clearly makes deep learning possible?",
        "notes": "Answer: backpropagation is a specific, efficient application of the multivariable chain rule to compute gradients through a layered computation — it is a DIFFERENTIATION technique, not an optimization algorithm; the actual optimization (updating weights) is done separately by SGD or a variant, using the gradients backprop supplies.",
    },
    {
        "title": "Concept: Backward Pass for a Linear Layer",
        "concept": [
            "Forward: Z = X·W + b",
            "dW = Xᵀ · dZ",
            "db = sum of dZ across all rows (the batch dimension)",
            "dX = dZ · Wᵀ  (needed to keep propagating backward into earlier layers)",
        ],
        "exercise": "A linear layer has X shape (32, 10), W shape (10, 5). If dZ has shape (32, 5), what are the shapes of dW, db, and dX? Do they match W, b, and X respectively?",
        "notes": "Answer: dW = X^T @ dZ -> (10,32)@(32,5) = (10,5), matching W's shape. db = sum over rows of dZ -> shape (5,), matching b. dX = dZ @ W^T -> (32,5)@(5,10) = (32,10), matching X's shape. All three gradients match the shape of the quantity they're the gradient of, as they must.",
    },
    {
        "title": "Concept: Backward Pass for ReLU and Sigmoid",
        "concept": [
            "ReLU backward: grad_output * (input > 0) — pass gradient through only where the input was positive",
            "Sigmoid backward: grad_output * output * (1 - output) — uses the CACHED forward output, not the input",
            "Every activation's backward rule needs something cached from its forward pass",
        ],
        "exercise": "A ReLU layer's cached input was [-2, 3, 0, 5]. The incoming gradient is [1, 1, 1, 1]. What is the outgoing gradient?",
        "notes": "Answer: [0, 1, 0, 1] — gradient passes through only where input > 0 (strictly). Note the input of exactly 0 also blocks the gradient here since the mask is (input > 0), not (input >= 0).",
    },
    {
        "title": "Concept: Softmax + Cross-Entropy — The Elegant Simplification",
        "concept": [
            "Softmax converts logits to probabilities; cross-entropy measures how wrong those probabilities are",
            "Computed separately, their combined backward pass would be a messy Jacobian product",
            "Computed TOGETHER, the gradient simplifies beautifully to: dLogits = (1/B)(P − Y)",
            "P = predicted probabilities, Y = one-hot true labels, B = batch size",
        ],
        "exercise": "For one example with true class 1 (of 3 classes) and predicted probabilities P=[0.2, 0.5, 0.3], what is the gradient with respect to the logits (ignore the 1/B batch averaging for this single example)?",
        "notes": "Answer: Y = [0, 1, 0] (one-hot for class 1). Gradient = P - Y = [0.2-0, 0.5-1, 0.3-0] = [0.2, -0.5, 0.3] — notice the correct class's gradient is negative (pushing its logit up) while incorrect classes get positive gradient (pushing their logits down).",
    },
    {
        "title": "Concept: Gradient Flow Pathologies",
        "concept": [
            "Vanishing gradients: products of many small Jacobians shrink toward zero across depth",
            "Exploding gradients: products of many large Jacobians grow without bound across depth",
            "Both make deep networks difficult or impossible to train without countermeasures",
            "Initialization, activation choice, normalization, and residual connections are all responses to this",
        ],
        "exercise": "A 20-layer sigmoid network trains fine for the first few layers near the output but the earliest layers' weights barely change at all during training. Which gradient pathology is this, and why does it hit early layers hardest?",
        "notes": "Answer: vanishing gradients — the gradient signal must pass back through all 20 layers to reach the earliest ones, and each sigmoid layer's derivative (max 0.25) shrinks the signal multiplicatively; by the time it reaches the earliest layers, it has been multiplied by many small factors and is nearly zero.",
    },
    {
        "title": "Concept: Xavier Initialization",
        "concept": [
            "Goal: keep the variance of activations roughly CONSTANT as signal passes through each layer",
            "Derived from Var(z) = n_in · Var(x) · Var(w) — solving for the weight variance that keeps Var(z)≈Var(x)",
            "Result: Var(W) = 2 / (n_in + n_out) — designed for symmetric activations like tanh",
        ],
        "exercise": "Why does Xavier initialization's variance formula include BOTH n_in and n_out, rather than just n_in?",
        "notes": "Answer: it balances the variance-preservation requirement for the FORWARD pass (which depends on n_in) against the equivalent requirement for the BACKWARD pass (which depends on n_out, since the backward pass is itself a matrix multiply by W^T) — averaging the two keeps both directions reasonably well-scaled.",
    },
    {
        "title": "Concept: He Initialization",
        "concept": [
            "Designed specifically for ReLU, which zeros out roughly half its inputs",
            "Because ReLU discards half the signal, the remaining variance must be compensated for",
            "Result: Var(W) = 2 / n_in — exactly double Xavier's forward-only term",
        ],
        "exercise": "Explain, in one sentence, why He initialization uses exactly 2/n_in rather than Xavier's more complex 2/(n_in+n_out) formula.",
        "notes": "Answer: because ReLU zeroes out roughly half of its inputs on average, the variance surviving through the layer is roughly halved — doubling the weight variance (2/n_in rather than 1/n_in) exactly compensates for that loss, keeping the forward signal's variance stable specifically for ReLU networks.",
    },
    {
        "title": "Concept: The Module Contract",
        "concept": [
            "Every layer implements the same three methods: forward(), backward(), parameters()",
            "forward() computes outputs AND caches whatever backward() will need",
            "Sequential.backward() walks layers in REVERSE order — the chain rule runs output-to-input",
            "This uniform contract is what lets layers compose into arbitrarily deep networks",
        ],
        "exercise": "A Sequential model has layers [Linear, ReLU, Linear]. In what order does forward() call them, and in what order does backward() call them?",
        "notes": "Answer: forward() calls them in order: Linear, then ReLU, then Linear. backward() calls them in REVERSE order: the second Linear first, then ReLU, then the first Linear — because the chain rule must start at the output (closest to the loss) and work backward toward the input.",
    },
    {
        "title": "Concept: Numerical Gradient Checking",
        "concept": [
            "Central difference: f'(θ) ≈ [f(θ+h) − f(θ−h)] / (2h) — more accurate than the one-sided version",
            "Compare this numerical estimate against your analytical (backprop) gradient",
            "Use RELATIVE error, not absolute — absolute error is misleading when gradients are tiny",
            "A correct float64 implementation should show relative error below about 1e-7",
        ],
        "exercise": "Your gradient check reports analytical=0.0031, numerical=0.0029. Is a relative-error tolerance of 1e-7 likely to be satisfied? What does this suggest about your backward pass?",
        "notes": "Answer: no — the relative error here is roughly |0.0031-0.0029|/max(1,0.0031,0.0029) ≈ 0.0002/0.0031 ≈ 0.065, or about 6.5%, vastly larger than 1e-7. This suggests a real bug in the backward pass, not floating-point noise — 1e-7-level tolerances expect near-exact agreement.",
    },
    {
        "title": "Concept: Common Bug — Missing Batch Averaging",
        "concept": [
            "If the FORWARD loss averages over the batch (divides by B), the BACKWARD gradient must too",
            "Forgetting grad /= batch_size makes the effective learning rate scale with batch size",
            "This is a silent bug — training still runs, just with a batch-size-dependent effective step size",
        ],
        "exercise": "You forget to divide the cross-entropy gradient by batch_size. If you then double your batch size, what happens to the effective step size of each parameter update, all else equal?",
        "notes": "Answer: without dividing by batch_size, the accumulated gradient (summed rather than averaged over the batch) roughly doubles when batch size doubles — so the effective step size (learning_rate * gradient) also roughly doubles, silently changing training dynamics whenever batch size changes.",
    },
    {
        "title": "Concept: Common Bug — Mutating Cached Inputs",
        "concept": [
            "Backward methods depend on values CACHED during the forward pass",
            "If a cached array is mutated in place before backward() runs, the gradient computation reads corrupted data",
            "Treat every cached tensor as read-only until backward() has consumed it",
        ],
        "exercise": "A custom training loop applies an in-place data augmentation to the input array AFTER the forward pass but BEFORE calling backward(). Why is this dangerous if the layer cached a reference to that same array?",
        "notes": "Answer: if the layer's forward() stored a reference (not a copy) to the input array, mutating that array afterward changes what backward() sees — the cached 'input' is no longer the value that was actually used to compute the forward output, so the resulting gradient will be computed against the wrong (mutated) values.",
    },
    {
        "title": "Concept: Diagnosing a Training Run",
        "concept": [
            "Activation histograms: watch for ReLUs stuck at all-zero, or sigmoids saturated at 0/1",
            "Gradient norms per layer: watch for early layers with vastly smaller norms than later layers",
            "Loss curves: flat-and-high (underfitting), diverging train/val gap (overfitting), NaN (instability)",
            "None of these prove correctness — but they narrow down WHERE to look for a bug",
        ],
        "exercise": "A network's loss suddenly becomes NaN partway through training. Name two of the most likely causes to check first.",
        "notes": "Answer: (1) the learning rate is too high, causing an update to overshoot into a region with extreme values; (2) a numerically unsafe operation such as log(0) or exp(large_number) inside the loss or an activation, without the stability guards this chapter's sigmoid/softmax implementations use.",
    },
    {
        "title": "Mastery Check: Are You Ready to Move On?",
        "concept": [
            "If you can answer every question below confidently, you have mastered Chapter 10.",
        ],
        "exercise": None,
        "mastery_questions": [
            "Explain why backpropagation is the chain rule applied systematically, not a separate algorithm.",
            "Derive dW, db, and dX for a linear layer Z = XW + b and verify their shapes match W, b, and X.",
            "Derive the backward rule for ReLU and for sigmoid, and explain what each one caches from its forward pass.",
            "Derive why the combined softmax + cross-entropy gradient simplifies to (P - Y) / B.",
            "Explain vanishing and exploding gradients and why they hit early layers of a deep network hardest.",
            "Derive Xavier initialization's variance formula and explain why it involves both n_in and n_out.",
            "Derive He initialization's variance formula and explain why it differs from Xavier by a factor of 2.",
            "Explain the Module contract (forward/backward/parameters) and why Sequential's backward runs in reverse order.",
            "Explain why numerical gradient checking uses relative error and central differences, and state a typical tolerance.",
            "Diagnose the training symptom caused by forgetting to divide the loss gradient by batch size.",
            "Diagnose a bug caused by mutating a cached forward-pass value before backward() runs.",
            "Implement a complete Linear + ReLU + Softmax-CrossEntropy MLP from scratch and verify it with a numerical gradient check.",
            "Train the scratch MLP on the two-spiral dataset and confirm a linear model could not have solved it.",
        ],
        "notes": "This is a self-check slide — and the final one in the series. If any question causes hesitation, return to that concept's slide and its exercise. Completing this chapter closes the Zero-to-Research curriculum's core sequence.",
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
        p.font.size = Pt(26) if idx > 1 else Pt(30)
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
            run.font.size = Pt(14.5)
            run.font.bold = True
            run.font.color.rgb = AMBER

        add_slide_number(slide, idx, total)
        set_notes(slide, s["notes"])

    out_path = "Chapter10_Backprop_Practice_Presentation.pptx"
    prs.save(out_path)
    print(f"Saved {out_path} with {total} slides.")


if __name__ == "__main__":
    build()
