"""
Build a "concept + exercise" PowerPoint presentation for Chapter 4
(The Core Optimization Engine: Automatic Differentiation).

Run:
    python3 build_chapter4_practice_presentation.py

Produces:
    Chapter4_Autograd_Practice_Presentation.pptx
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
        "title": "Chapter 4: Automatic Differentiation",
        "subtitle": "One concept, one exercise per slide — practice as you go",
        "concept": [],
        "exercise": None,
        "notes": "Welcome slide. Automatic differentiation is the mechanism behind every .backward() call in every deep learning framework. This chapter builds it from scratch.",
    },
    {
        "title": "Concept: Why Automatic Differentiation?",
        "concept": [
            "Numerical differentiation (finite differences) needs one extra forward pass PER parameter",
            "Symbolic differentiation produces expressions that explode in size with composition depth",
            "Automatic differentiation computes exact gradients for ALL parameters in about one extra pass",
            "AD is neither purely numerical nor purely symbolic — it applies the chain rule mechanically to a recorded computation",
        ],
        "exercise": "A model has 7 billion parameters. Roughly how many forward passes would numerical differentiation need to compute the full gradient? Why is this completely impractical?",
        "notes": "Answer: about 7 billion + 1 forward passes (one perturbation per parameter). At any realistic throughput this would take years per gradient step — reverse-mode AD instead computes the full gradient in about one backward pass, regardless of parameter count.",
    },
    {
        "title": "Concept: The Computational Graph",
        "concept": [
            "Every computation is a Directed Acyclic Graph (DAG) of elementary operations",
            "Leaf nodes: inputs and parameters. Root node: the final scalar output (the loss)",
            "Each node stores its forward value AND, after backward(), its gradient",
            "Each node also stores a local backward rule — the chain-rule recipe for its own operation",
        ],
        "exercise": "For f = (a + b) * c, draw the computation as a graph: how many nodes are there, and which are leaves?",
        "notes": "Answer: 4 nodes total — three leaves (a, b, c) and one interior/root node representing the sum (a+b), and the final multiply node. More precisely: a, b, c are leaves; (a+b) is an interior node; the final product is the root. That's 5 nodes if you count the intermediate sum explicitly.",
    },
    {
        "title": "Concept: The Multivariable Chain Rule (Summation Form)",
        "concept": [
            "If a node v feeds into several downstream nodes w1...wk, its gradient is a SUM:",
            "∂L/∂v = Σᵢ (∂L/∂wᵢ) · (∂wᵢ/∂v)",
            "This single equation is the entire mathematical content of backpropagation",
            "The sum is why gradient contributions must be ACCUMULATED, not overwritten",
        ],
        "exercise": "A variable x is used twice in f = x*x (i.e., x feeds two multiplication inputs of the same node). Using the summation form, explain in words why df/dx is NOT simply the derivative of one usage alone.",
        "notes": "Answer: x contributes to the output through two separate paths (both factors of the product), so its total gradient must sum the contribution from each path: d(x*x)/dx = x (from first factor) + x (from second factor) = 2x — exactly the summation form applied to a node with two downstream consumers.",
    },
    {
        "title": "Concept: Local Backward Rules",
        "concept": [
            "Every elementary operation (add, multiply, exp, ...) has a simple, known local derivative",
            "Addition: gradient passes through unchanged to both inputs",
            "Multiplication: each input's gradient is (incoming gradient) × (the OTHER input's value)",
            "Complex functions are just long chains of these simple rules",
        ],
        "exercise": "For out = a * b, if the incoming gradient at 'out' is 5, and a=3, b=4, what are the gradients that flow back to a and to b?",
        "notes": "Answer: grad_a = incoming_grad * b = 5*4 = 20. grad_b = incoming_grad * a = 5*3 = 15 — each input receives the incoming gradient scaled by the OTHER input's forward value.",
    },
    {
        "title": "Concept: Topological Sort — The Ordering Guarantee",
        "concept": [
            "The backward pass must visit every node AFTER all of its downstream consumers",
            "This ordering is exactly the reverse of a topological sort of the forward graph",
            "Without correct ordering, a node's gradient could be read before it's fully accumulated",
            "Implemented via recursive depth-first search with a visited set",
        ],
        "exercise": "In the graph a → c, b → c, c → d, why must d's gradient be fully computed before c's backward rule runs, and why must c's gradient be fully computed before a's or b's backward rule runs?",
        "notes": "Answer: c's gradient depends on d's gradient (via the chain rule), so d must go first. Similarly, a and b's gradients depend on c's (now-complete) gradient, so c must be fully processed before a or b. This is exactly reverse topological order: d, then c, then a and b.",
    },
    {
        "title": "Concept: Forward Mode (JVP) vs. Reverse Mode (VJP)",
        "concept": [
            "Forward mode: one pass computes the gradient with respect to ONE input — needs N passes for N inputs",
            "Reverse mode: one pass computes the gradient with respect to ALL inputs at once — needs M passes for M outputs",
            "Neural network training: N (parameters) is huge, M (the scalar loss) is 1",
            "Decision rule: use reverse mode when N ≫ M, forward mode when M ≫ N",
        ],
        "exercise": "A function has 3 inputs and 1,000,000 outputs. Which mode (forward or reverse) is cheaper, and why?",
        "notes": "Answer: forward mode — with only 3 inputs, forward mode needs just 3 passes to get the full Jacobian, while reverse mode would need 1,000,000 passes (one per output). This is the M ≫ N regime, the mirror image of neural network training.",
    },
    {
        "title": "Concept: The Reverse Accumulation Algorithm",
        "concept": [
            "1. Run the forward pass, recording every operation and its inputs",
            "2. Topologically sort the resulting graph",
            "3. Initialize the root node's gradient to 1 (∂L/∂L = 1)",
            "4. Walk the sorted nodes in REVERSE order, applying each local backward rule",
        ],
        "exercise": "Why does the algorithm initialize the root's gradient to exactly 1, rather than 0 or the loss value itself?",
        "notes": "Answer: the root IS the loss L, so ∂L/∂L = 1 by definition — this is the starting seed that the chain rule then multiplies through every downstream (really, upstream in the original graph) local derivative to produce every other gradient.",
    },
    {
        "title": "Concept: Gradient Accumulation at Shared Nodes",
        "concept": [
            "When a node is consumed by multiple downstream operations, EVERY backward closure must use +=",
            "Using = instead of += silently discards all but the last contribution",
            "This is a silent bug: no exception, no NaN — just a mathematically wrong gradient",
            "Directly required by the summation form of the chain rule from earlier in this deck",
        ],
        "exercise": "A buggy implementation writes self.grad = other.data * out.grad instead of self.grad += other.data * out.grad inside a multiply node's backward closure. For f = x*x, what wrong gradient value would this produce at x=3 (the correct answer is 6)?",
        "notes": "Answer: with overwriting, only the LAST backward path to update x.grad survives — since both paths compute the same local value (x=3) here, the bug happens to produce grad=3 (only one path's contribution survives) instead of the correct 6 (both paths summed). The bug is invisible unless checked against a known answer or finite differences.",
    },
    {
        "title": "Concept: zero_grad and Gradient Accumulation Across Steps",
        "concept": [
            "Gradients accumulate (+=) WITHIN one backward() call — that's required, not a bug",
            "But gradients also persist ACROSS multiple backward() calls unless explicitly reset",
            "Forgetting to zero gradients between training steps silently sums gradients across steps",
            "This is exactly what optimizer.zero_grad() exists to prevent",
        ],
        "exercise": "You call backward() twice in a row on two different batches without calling zero_grad() in between. What happens to the parameter gradients, and how would this show up as a training symptom?",
        "notes": "Answer: the second batch's gradients are ADDED on top of the first batch's, effectively doubling (and further compounding on more steps) the gradient magnitude — this typically shows up as unstable, oscillating, or diverging training that a smaller learning rate seems to 'fix' without addressing the real bug.",
    },
    {
        "title": "Concept: Complexity of a Backward Pass",
        "concept": [
            "Forward pass: O(|V|) — one visit per node, each doing O(c) work",
            "Backward pass: O(|E|) — work per node is proportional to its number of downstream consumers",
            "For most graphs, |E| = O(|V|), so a full backward pass costs about the same as one forward pass",
            "This is why reverse-mode AD is 'free' relative to computing the loss itself",
        ],
        "exercise": "A network's forward pass takes 10 milliseconds. Roughly how long should you expect the backward pass to take, and why?",
        "notes": "Answer: roughly the same order of magnitude, often cited as about 2x the forward pass in practice (one backward pass costs about the same asymptotic order as the forward pass, O(|V|) vs O(|E|)~O(|V|)) — NOT proportional to the number of parameters, which is what makes training large models tractable at all.",
    },
    {
        "title": "Concept: Gradient Checkpointing — Trading Time for Memory",
        "concept": [
            "Naively, EVERY node's forward value must stay in memory until backward() consumes it",
            "Peak memory scales as O(depth) for a deep network — this can exceed available memory",
            "Checkpointing stores only every √L-th activation, recomputing the rest during backward()",
            "Optimal trade-off: memory drops to O(√L) at the cost of one extra forward pass",
        ],
        "exercise": "A 100-layer network stores every activation naively, needing memory for 100 layers. Using the √L checkpointing strategy, roughly how many activations need to be held in memory at once instead?",
        "notes": "Answer: roughly sqrt(100) = 10 — checkpointing every ~10th layer and recomputing the ~10 layers between checkpoints on demand during the backward pass, trading one extra forward pass for a ~10x memory reduction.",
    },
    {
        "title": "Mastery Check: Are You Ready to Move On?",
        "concept": [
            "If you can answer every question below confidently, you have mastered Chapter 4.",
        ],
        "exercise": None,
        "mastery_questions": [
            "Explain why automatic differentiation is neither purely numerical nor purely symbolic differentiation.",
            "Draw the computational graph for a small expression and identify its leaves, interior nodes, and root.",
            "State the multivariable chain rule in summation form and explain why it requires gradient accumulation.",
            "Derive the local backward rule for multiplication and for addition.",
            "Explain why the backward pass must run in reverse topological order.",
            "Explain the difference between forward-mode (JVP) and reverse-mode (VJP) autodiff, and when to use each.",
            "Walk through the reverse accumulation algorithm's four steps from memory.",
            "Explain why using '=' instead of '+=' in a backward closure produces a silent, wrong gradient.",
            "Explain the difference between within-call gradient accumulation and across-step accumulation, and why zero_grad() is needed.",
            "State the time complexity of one backward pass and explain why it does not scale with the number of parameters directly.",
            "Explain the gradient checkpointing trade-off and derive why sqrt(L) is the optimal checkpoint spacing.",
            "Implement a minimal scalar autograd engine (Value class) supporting +, *, and backward().",
        ],
        "notes": "This is a self-check slide. If any question causes hesitation, return to that concept's slide and its exercise before moving to Chapter 5.",
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
                p.font.size = Pt(13.5)
                p.font.color.rgb = NAVY
                p.space_after = Pt(7)
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

    out_path = "Chapter4_Autograd_Practice_Presentation.pptx"
    prs.save(out_path)
    print(f"Saved {out_path} with {total} slides.")


if __name__ == "__main__":
    build()
