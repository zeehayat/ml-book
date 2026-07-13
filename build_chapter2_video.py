r"""Build the narrated Chapter 2 automatic-differentiation tutorial.

Run:
    .venv\Scripts\python.exe build_chapter2_video.py
"""

from __future__ import annotations

import json
import re
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw
from imageio_ffmpeg import get_ffmpeg_exe

from build_chapter1_video import (
    BG, CYAN, FPS, GREEN, GRID, H, MUTED, ORANGE, PANEL, PANEL_2, PINK,
    W, WHITE, draw_bullets, fit_text, font, rounded,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "chapter2_video"
SLIDES_DIR = OUT / "slides"
NARRATION_DIR = OUT / "narration"
AUDIO_DIR = OUT / "audio"
VIDEO_PATH = OUT / "Chapter2_Automatic_Differentiation_Tutorial.mp4"


SLIDES = [
    {
        "title": "The Core Optimization Engine",
        "kicker": "CHAPTER 2  •  VIDEO TUTORIAL",
        "subtitle": "Automatic differentiation, computational graphs, and backpropagation from scratch",
        "kind": "title",
        "narration": """Welcome to Chapter Two: The Core Optimization Engine, Automatic Differentiation. Every neural network learns by answering one question: if a parameter changed slightly, how would the loss change? Automatic differentiation answers that question for every parameter by recording ordinary arithmetic and replaying the chain rule in reverse. We will build that mechanism from first principles, trace a complete example, and connect our tiny scalar engine to the architecture used by PyTorch at production scale.""",
    },
    {
        "title": "What you will learn",
        "kicker": "LEARNING MAP",
        "kind": "bullets",
        "bullets": [
            "Build and label a computational graph",
            "Run forward evaluation and reverse gradient propagation",
            "Apply the multivariable chain rule at shared nodes",
            "Implement a scalar Value engine with topological ordering",
            "Extend backward rules to tensors and broadcasting",
        ],
        "narration": """By the end, you should be able to turn an arithmetic expression into a directed acyclic graph, annotate its forward values, and propagate exact gradients backward. You will see why gradients must accumulate when paths merge, why reverse topological order is essential, and why reverse mode fits neural networks with millions or billions of inputs and one scalar loss. We will then translate the mathematics into a small Value class and examine what changes when values become multidimensional tensors.""",
    },
    {
        "title": "Chapter 1 plus gradient state",
        "kicker": "THE BRIDGE FROM TENSORS TO AUTOGRAD",
        "kind": "bridge",
        "left": ["data / storage", "shape", "strides", "offset"],
        "right": ["grad", "_prev", "_backward", "requires_grad"],
        "narration": """Chapter One gave us a tensor as storage plus shape, strides, offset, and dtype. Autograd augments that object with gradient state. A scalar Value stores its forward data, the gradient of the final loss with respect to that data, pointers to the Values that produced it, and a backward function containing the local derivative rule. A production tensor keeps the Chapter One storage model and adds equivalent graph metadata. Autograd is therefore not separate magic layered above tensors; it is graph state woven into tensor operations.""",
    },
    {
        "title": "Why derivatives must scale",
        "kicker": "THE OPTIMIZATION PROBLEM",
        "kind": "scale",
        "formula": "θ ← θ − η · ∂L/∂θ",
        "bullets": [
            "Hand derivation becomes unmanageable as architectures change",
            "Finite differences need one forward pass per parameter",
            "Symbolic differentiation suffers expression swell",
            "Reverse-mode AD gets every parameter gradient in one backward pass",
        ],
        "narration": """Gradient descent updates each parameter by subtracting the learning rate times the derivative of loss with respect to that parameter. The formula is simple; obtaining the derivatives is the challenge. Hand derivation does not survive deep, changing architectures. Finite differences require a new forward evaluation for each parameter and introduce approximation error. Symbolic differentiation can create enormous expressions. Reverse-mode automatic differentiation records elementary operations, then computes all parameter gradients in one backward traversal whose cost is a small multiple of the forward computation.""",
    },
    {
        "title": "Automatic differentiation is its own method",
        "kicker": "THREE DIFFERENT IDEAS",
        "kind": "compare",
        "columns": [
            ("NUMERICAL", "perturb inputs", "approximate", PINK),
            ("SYMBOLIC", "rewrite expressions", "exact • may swell", ORANGE),
            ("AUTODIFF", "record operations", "exact local rules", GREEN),
        ],
        "narration": """It is important to separate three ideas. Numerical differentiation perturbs an input and observes a finite difference, so the answer is approximate. Symbolic differentiation rewrites an expression into another expression, which is exact but may grow explosively. Automatic differentiation evaluates the actual program on concrete values. Every elementary operation records a small local derivative recipe. Combining those recipes with the chain rule produces an exact derivative of the implemented computation, up to ordinary floating-point rounding. If the program is wrong, autograd faithfully differentiates the wrong program.""",
    },
    {
        "title": "A computational graph",
        "kicker": "WORKED EXAMPLE  •  f = (a + b)(b + 1)",
        "kind": "graph",
        "narration": """We will carry one example through the forward and backward passes. Let f equal a plus b, multiplied by b plus one. Introduce intermediate nodes u equals a plus b, and v equals b plus one. Then f equals u times v. The graph is directed because values flow from inputs to output. It is acyclic because an operation cannot depend on a result that has not yet been computed. Notice that b fans out into two branches. That shared input will force us to add gradient contributions during the backward pass.""",
    },
    {
        "title": "Forward pass: evaluate and record",
        "kicker": "a = 2  •  b = 3",
        "kind": "forward",
        "narration": """Set a to two and b to three. The forward pass computes u equals five, v equals four, and f equals twenty. Alongside each result, the engine records its parents and a backward closure. The multiplication node remembers that its local derivative with respect to u is v, and with respect to v is u. The addition nodes remember derivative one for each input. This collection of nodes and local recipes is often called a tape or Wengert list: the breadcrumb trail needed to retrace the execution.""",
    },
    {
        "title": "Backward pass: begin with one",
        "kicker": "SEED THE OUTPUT GRADIENT",
        "kind": "backward",
        "narration": """The backward pass begins by setting the output gradient to one, because the derivative of f with respect to itself is one. At the multiplication f equals u times v, the arriving gradient is multiplied by each local partial derivative. U receives one times v, which is four. V receives one times u, which is five. These numbers are adjoints: u dot means partial f over partial u, and v dot means partial f over partial v. We then continue backward through the two addition nodes.""",
    },
    {
        "title": "Shared nodes require accumulation",
        "kicker": "THE MOST IMPORTANT IMPLEMENTATION DETAIL",
        "kind": "accumulate",
        "formula": "∂f/∂b = 4·1 + 5·1 = 9",
        "narration": """Variable b influences f along two distinct paths. Through u equals a plus b, it receives contribution four times one. Through v equals b plus one, it receives contribution five times one. The total derivative is the sum, nine. Variable a appears on only one path, so its derivative is four. This is the multivariable chain rule in graph form: whenever paths merge at a node, add their contributions. A backward closure must use plus equals, not assignment. Overwriting would silently discard one path and return the wrong gradient.""",
    },
    {
        "title": "The multivariable chain rule",
        "kicker": "THE FUNDAMENTAL EQUATION",
        "kind": "formula",
        "formula": "∂L/∂x = Σⱼ (∂L/∂yⱼ) · (∂yⱼ/∂x)",
        "caption": "incoming gradient × local derivative, summed over every outgoing path",
        "narration": """Formally, if x influences the loss through several downstream variables y sub j, the derivative of loss with respect to x is the sum over j of the upstream derivative with respect to y sub j times the local derivative of y sub j with respect to x. Every backward rule implements this same pattern: multiply the gradient arriving from downstream by a local derivative, then accumulate the result into the parent. The global derivative emerges without any node needing to understand the entire function.""",
    },
    {
        "title": "Local backward rules",
        "kicker": "A SMALL DERIVATIVE LIBRARY",
        "kind": "rules",
        "rules": [
            ("w = u + v", "ū += w̄", "v̄ += w̄"),
            ("w = u · v", "ū += v·w̄", "v̄ += u·w̄"),
            ("w = uⁿ", "ū += n·uⁿ⁻¹·w̄", ""),
            ("w = tanh(u)", "ū += (1−w²)·w̄", ""),
            ("w = ReLU(u)", "ū += 1[u>0]·w̄", ""),
        ],
        "narration": """An autograd engine needs a derivative library for primitive operations. Addition sends the incoming gradient unchanged to both inputs. Multiplication scales it by the other operand. A power applies n times u to the n minus one. Tanh can reuse its forward output w, giving one minus w squared. ReLU passes the gradient only when its input was positive. Complicated neural networks are compositions of these simple rules. Supporting a new primitive means defining its forward computation and one correct vector-Jacobian product for backward.""",
    },
    {
        "title": "Why reverse topological order matters",
        "kicker": "ORDERING GUARANTEE",
        "kind": "topo",
        "narration": """A node must not run its backward rule until every downstream consumer has contributed to its gradient. A topological sort places every parent before every child in forward order. Reversing that list places children before parents for backward. We first perform a depth-first traversal from the output, append each node after visiting its parents, seed the output gradient, and execute backward closures in reversed order. This guarantees that a shared node such as b has received all contributions before it propagates further upstream.""",
    },
    {
        "title": "Forward mode versus reverse mode",
        "kicker": "CHOOSE THE DIRECTION BY GRAPH SHAPE",
        "kind": "modes",
        "left": ["one input direction", "propagate tangents forward", "best when inputs ≪ outputs", "computes J·v"],
        "right": ["one output direction", "propagate adjoints backward", "best when inputs ≫ outputs", "computes vᵀ·J"],
        "narration": """Forward mode propagates input sensitivities alongside values and efficiently computes a Jacobian-vector product. It is attractive when there are few inputs and many outputs. Reverse mode propagates output sensitivities backward and computes a vector-Jacobian product. Neural network training has millions or billions of parameters but usually one scalar loss. That is many inputs and one output, so one reverse pass produces the complete parameter gradient. This dimensional asymmetry, not a special property of neural networks, is why backpropagation uses reverse mode.""",
    },
    {
        "title": "The scalar Value object",
        "kicker": "PURE-PYTHON ENGINE",
        "kind": "code",
        "code": """class Value:
    def __init__(self, data, _children=()):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._backward = lambda: None

    def backward(self):
        topo = topological_sort(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()""",
        "narration": """The scalar engine begins with a Value object. Data stores the forward scalar. Grad starts at zero and accumulates the derivative of the final output with respect to this node. Prev stores parent nodes, while backward is a closure installed by the operation that created the Value. Calling backward topologically sorts the reachable graph, seeds the output gradient with one, and invokes every closure in reverse order. The graph is constructed dynamically as ordinary Python operators execute, which is the define-by-run model.""",
    },
    {
        "title": "A multiplication closure",
        "kicker": "LOCAL RULE CAPTURES FORWARD VALUES",
        "kind": "code",
        "code": """def __mul__(self, other):
    out = Value(self.data * other.data,
                (self, other))

    def _backward():
        self.grad  += other.data * out.grad
        other.grad += self.data  * out.grad

    out._backward = _backward
    return out""",
        "narration": """Multiplication creates an output Value and a closure that captures the two inputs and the output. During backward, the left input receives the right input's forward value times the output gradient. The right input receives the left value times the output gradient. Both updates use plus equals because either input may receive contributions from other branches. The closure architecture is powerful: each output node carries exactly the recipe needed to send its accumulated adjoint one step backward.""",
    },
    {
        "title": "From scalars to tensors",
        "kicker": "BROADCASTING CHANGES BACKWARD SHAPES",
        "kind": "broadcast",
        "formula": "y = x + b     x:(3,4)   b:(4,)   y:(3,4)",
        "narration": """Moving from scalars to tensors adds shape-aware backward rules. Suppose a three-by-four tensor x is added to a length-four bias b. Broadcasting repeats b across three rows during the forward pass. Backward must undo that repetition: x receives the output gradient unchanged, while b receives a sum across the broadcasted row axis. A helper often called sum to shape reduces extra leading axes and any axis whose original size was one. Matrix multiplication, transpose, reshape, and reductions each need similarly correct shape transformations.""",
    },
    {
        "title": "The tape has a memory cost",
        "kicker": "ACTIVATION MEMORY",
        "kind": "memory",
        "bullets": [
            "Forward stores intermediates needed by backward",
            "Backward is O(graph work); tape memory is O(saved activations)",
            "Checkpointing stores fewer activations and recomputes forward regions",
            "Trade extra compute for lower peak memory",
        ],
        "narration": """Reverse mode is computationally efficient, but it must retain forward information until backward consumes it. Multiplication needs its input values. Activations such as tanh often save an output or mask. In deep networks, these saved tensors can exceed parameter memory. Gradient checkpointing reduces peak memory by saving only selected boundary activations. During backward, it recomputes missing forward regions, rebuilds their temporary tape, and then differentiates them. The trade is explicit: more computation in exchange for less memory.""",
    },
    {
        "title": "Production autograd uses the same architecture",
        "kicker": "FROM VALUE TO PYTORCH",
        "kind": "production",
        "items": [
            ("grad_fn", "the producing operation", CYAN),
            ("saved tensors", "closure state", GREEN),
            ("torch.compile", "capture and fuse graphs", ORANGE),
            ("distributed", "reduce parameter gradients", PINK),
        ],
        "narration": """PyTorch's grad function corresponds to our backward closure, and saved tensors correspond to captured forward values. Define-by-run makes Python control flow natural because the graph reflects the path actually executed. Torch compile can capture regions of that dynamic program, optimize and fuse operations, then execute generated kernels. Mixed-precision training adds loss scaling to protect small gradients. Distributed training computes local backward passes and then reduces parameter gradients across devices. The scale changes dramatically, but the graph and chain-rule architecture does not.""",
    },
    {
        "title": "Three failures that silently break learning",
        "kicker": "COMMON MISTAKES",
        "kind": "mistakes",
        "bullets": [
            "Using = instead of += at a shared node",
            "Forgetting zero_grad() between optimization steps",
            "Mutating saved forward values before backward",
            "Running backward in an order that misses contributions",
            "Keeping old graphs alive and leaking activation memory",
        ],
        "narration": """Autograd bugs are dangerous because code may run while learning fails. Overwriting rather than accumulating loses shared-path contributions. Forgetting zero grad unintentionally mixes gradients from different training steps. In-place mutation can make saved values inconsistent with the forward output they produced. Incorrect traversal order propagates a partial gradient too early. Retaining references to losses or graph nodes can keep entire tapes alive and exhaust memory. Unit-test tiny expressions against hand derivatives and finite-difference checks before trusting a larger engine.""",
    },
    {
        "title": "Mini project: train an XOR MLP",
        "kicker": "THE ENGINE LEARNS",
        "kind": "mlp",
        "narration": """The chapter's mini project composes scalar Values into neurons, layers, and a multilayer perceptron, then trains it on XOR. The forward pass produces predictions and a loss. Backward fills every parameter's grad field. Gradient descent updates each parameter and zeroes gradients before the next iteration. XOR is deliberately small but nonlinear: a single linear boundary cannot solve it. When the loss falls and predictions separate, you have evidence that graph construction, local derivatives, accumulation, ordering, and optimization all work together.""",
    },
    {
        "title": "The four-primitive mental model",
        "kicker": "CHAPTER 2 SUMMARY",
        "kind": "summary",
        "bullets": [
            "Record operations and dependencies during forward",
            "Store the local derivative rule with each result",
            "Traverse the graph in reverse topological order",
            "Multiply locally and accumulate at every parent",
            "Use reverse mode when many parameters feed one loss",
        ],
        "narration": """Compress automatic differentiation into four primitives. Record operations and dependencies during the forward pass. Store a local derivative recipe with every result. Traverse reachable nodes in reverse topological order. Multiply the incoming gradient by each local derivative and accumulate into parents. Reverse mode turns one scalar loss into gradients for every parameter with one backward traversal. Chapter One explained where tensor values live; Chapter Two explains how sensitivity flows between them. With this engine in place, Chapter Three can focus on optimization rather than derivative bookkeeping.""",
    },
]


def base_slide(slide, number):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    for x in range(0, W, 64):
        d.line((x, 0, x, H), fill="#0a1829")
    for y in range(0, H, 64):
        d.line((0, y, W, y), fill="#0a1829")
    d.rectangle((0, 0, 12, H), fill=GREEN)
    d.text((54, 34), slide.get("kicker", "CHAPTER 2"), font=font(18, bold=True), fill=GREEN)
    d.text((54, 65), slide["title"], font=font(40, bold=True), fill=WHITE)
    d.line((54, 122, 1226, 122), fill=GRID, width=2)
    d.text((54, 684), "ML SYSTEMS ENGINEERING HANDBOOK", font=font(14, bold=True), fill=MUTED)
    d.text((1164, 682), f"{number:02d}", font=font(18, bold=True), fill=MUTED)
    return im, d


def node(d, x, y, label, value=None, color=CYAN):
    d.ellipse((x - 46, y - 46, x + 46, y + 46), fill=PANEL_2, outline=color, width=4)
    d.text((x - 15, y - 26), label, font=font(25, bold=True), fill=WHITE)
    if value is not None:
        d.text((x - 24, y + 5), str(value), font=font(18, mono=True), fill=color)


def arrow(d, a, b, color=GRID, width=4):
    d.line((*a, *b), fill=color, width=width)
    angle = __import__("math").atan2(b[1] - a[1], b[0] - a[0])
    p1 = (b[0] - 16 * __import__("math").cos(angle - .45), b[1] - 16 * __import__("math").sin(angle - .45))
    p2 = (b[0] - 16 * __import__("math").cos(angle + .45), b[1] - 16 * __import__("math").sin(angle + .45))
    d.polygon([b, p1, p2], fill=color)


def graph_scene(d, values=False, gradients=False):
    pos = {"a": (120, 330), "b": (120, 500), "1": (120, 590), "u": (480, 290), "v": (480, 500), "f": (880, 390)}
    edges = [("a", "u"), ("b", "u"), ("b", "v"), ("1", "v"), ("u", "f"), ("v", "f")]
    for p, q in edges:
        arrow(d, pos[p], pos[q], GREEN if gradients else GRID)
    vals = {"a": 2, "b": 3, "1": 1, "u": 5, "v": 4, "f": 20}
    grads = {"a": "ā=4", "b": "b̄=9", "u": "ū=4", "v": "v̄=5", "f": "f̄=1"}
    for name, (x, y) in pos.items():
        node(d, x, y, name, vals[name] if values else None, GREEN if gradients else CYAN)
        if gradients and name in grads:
            d.text((x + 55, y - 15), grads[name], font=font(19, bold=True), fill=ORANGE)
    d.text((285, 220), "+", font=font(35, bold=True), fill=MUTED)
    d.text((285, 535), "+", font=font(35, bold=True), fill=MUTED)
    d.text((680, 380), "×", font=font(38, bold=True), fill=MUTED)


def render_slide(slide, idx):
    im, d = base_slide(slide, idx)
    kind = slide["kind"]
    if kind == "title":
        d.text((74, 170), "AUTOMATIC", font=font(78, bold=True), fill=GREEN)
        d.text((74, 260), "DIFFERENTIATION", font=font(65, bold=True), fill=WHITE)
        fit_text(d, slide["subtitle"], (78, 370, 740, 120), 28, MUTED)
        points = [(850, 230), (850, 430), (1040, 330), (1180, 330)]
        arrow(d, points[0], points[2], CYAN, 4)
        arrow(d, points[1], points[2], CYAN, 4)
        arrow(d, points[2], points[3], GREEN, 5)
        for (x, y), label, color in zip(points, ["x", "w", "×", "L"], [CYAN, CYAN, GREEN, ORANGE]):
            node(d, x, y, label, None, color)
        arrow(d, (1160, 410), (880, 500), PINK, 6)
        d.text((905, 525), "gradients flow backward", font=font(20, bold=True), fill=PINK)
    elif kind in ("bullets", "summary", "mistakes"):
        accent = GREEN if kind == "summary" else (PINK if kind == "mistakes" else CYAN)
        draw_bullets(d, slide["bullets"], size=26, accent=accent)
    elif kind == "bridge":
        for x, title, items, color in [(75, "TENSOR", slide["left"], CYAN), (680, "AUTOGRAD STATE", slide["right"], GREEN)]:
            rounded(d, (x, 170, x + 525, 565), 22, PANEL, color, 3)
            d.text((x + 28, 194), title, font=font(28, bold=True), fill=color)
            for i, item in enumerate(items):
                rounded(d, (x + 28, 250 + i * 70, x + 495, 302 + i * 70), 10, PANEL_2)
                d.text((x + 50, 262 + i * 70), item, font=font(23, mono=True), fill=WHITE)
        d.text((616, 350), "+", font=font(38, bold=True), fill=ORANGE)
    elif kind == "scale":
        rounded(d, (70, 162, 1210, 270), 22, PANEL, GREEN, 3)
        d.text((330, 195), slide["formula"], font=font(38, mono=True), fill=WHITE)
        draw_bullets(d, slide["bullets"], x=70, y=315, width=1140, size=22, accent=ORANGE)
    elif kind == "compare":
        for i, (name, action, result, color) in enumerate(slide["columns"]):
            x = 65 + i * 400
            rounded(d, (x, 175, x + 360, 535), 22, PANEL, color, 3)
            d.text((x + 25, 205), name, font=font(26, bold=True), fill=color)
            d.text((x + 25, 305), action, font=font(23), fill=WHITE)
            d.text((x + 25, 390), result, font=font(21, bold=True), fill=MUTED)
    elif kind in ("graph", "forward", "backward"):
        graph_scene(d, values=kind != "graph", gradients=kind == "backward")
    elif kind == "accumulate":
        node(d, 150, 360, "b", 3, PINK)
        node(d, 530, 240, "u", "grad 4", CYAN)
        node(d, 530, 490, "v", "grad 5", GREEN)
        arrow(d, (480, 260), (205, 340), CYAN, 5)
        arrow(d, (480, 470), (205, 380), GREEN, 5)
        rounded(d, (680, 240, 1185, 450), 20, PANEL, ORANGE, 3)
        d.text((725, 285), slide["formula"], font=font(29, mono=True), fill=WHITE)
        d.text((725, 355), "contributions ADD", font=font(28, bold=True), fill=ORANGE)
    elif kind == "formula":
        rounded(d, (70, 195, 1210, 355), 24, PANEL, GREEN, 3)
        d.text((145, 245), slide["formula"], font=font(35, mono=True), fill=WHITE)
        fit_text(d, slide["caption"], (190, 415, 950, 80), 28, MUTED)
    elif kind == "rules":
        y = 160
        for expr, left, right in slide["rules"]:
            rounded(d, (70, y, 1210, y + 80), 14, PANEL)
            d.text((95, y + 22), expr, font=font(22, mono=True), fill=WHITE)
            d.text((470, y + 22), left, font=font(22, mono=True), fill=GREEN)
            d.text((850, y + 22), right, font=font(22, mono=True), fill=CYAN)
            y += 94
    elif kind == "topo":
        labels = ["a,b,1", "u,v", "f"]
        xs = [190, 600, 1030]
        for x, label in zip(xs, labels):
            rounded(d, (x - 110, 250, x + 110, 350), 18, PANEL_2, CYAN, 3)
            d.text((x - 55, 282), label, font=font(25, mono=True), fill=WHITE)
        arrow(d, (305, 300), (485, 300), CYAN, 6)
        arrow(d, (715, 300), (915, 300), CYAN, 6)
        d.text((365, 390), "forward topological order", font=font(26, bold=True), fill=CYAN)
        arrow(d, (920, 490), (300, 490), PINK, 7)
        d.text((390, 525), "backward = reverse order", font=font(27, bold=True), fill=PINK)
    elif kind == "modes":
        for x, title, items, color in [(70, "FORWARD MODE", slide["left"], CYAN), (660, "REVERSE MODE", slide["right"], GREEN)]:
            rounded(d, (x, 165, x + 550, 565), 22, PANEL, color, 3)
            d.text((x + 30, 195), title, font=font(27, bold=True), fill=color)
            for i, item in enumerate(items):
                d.text((x + 35, 270 + i * 67), "• " + item, font=font(22), fill=WHITE)
    elif kind == "code":
        rounded(d, (70, 155, 1210, 610), 20, "#0b1727", GRID)
        fit_text(d, slide["code"], (105, 182, 1050, 400), 22, GREEN, spacing=5)
    elif kind == "broadcast":
        rounded(d, (80, 175, 1200, 270), 20, PANEL, CYAN, 3)
        d.text((170, 205), slide["formula"], font=font(29, mono=True), fill=WHITE)
        for r in range(3):
            for c in range(4):
                x, y = 190 + c * 120, 335 + r * 75
                rounded(d, (x, y, x + 90, y + 52), 9, PANEL_2, GREEN)
                d.text((x + 23, y + 14), f"g{r}{c}", font=font(18, mono=True), fill=WHITE)
        d.text((755, 380), "∂L/∂b = sum over rows", font=font(25, bold=True), fill=ORANGE)
        d.text((790, 445), "[Σgᵢ0  Σgᵢ1  Σgᵢ2  Σgᵢ3]", font=font(21, mono=True), fill=WHITE)
    elif kind == "memory":
        draw_bullets(d, slide["bullets"], size=23, accent=ORANGE)
    elif kind == "production":
        for i, (name, desc, color) in enumerate(slide["items"]):
            x = 75 + (i % 2) * 580
            y = 175 + (i // 2) * 190
            rounded(d, (x, y, x + 530, y + 145), 20, PANEL, color, 3)
            d.text((x + 28, y + 25), name, font=font(27, bold=True), fill=color)
            d.text((x + 28, y + 80), desc, font=font(22), fill=WHITE)
    elif kind == "mlp":
        layers = [(150, 3, "INPUT"), (475, 4, "HIDDEN"), (800, 4, "HIDDEN"), (1110, 1, "OUTPUT")]
        positions = []
        for x, count, label in layers:
            pts = []
            for i in range(count):
                y = 245 + i * 90
                d.ellipse((x - 28, y - 28, x + 28, y + 28), fill=PANEL_2, outline=GREEN, width=3)
                pts.append((x, y))
            positions.append(pts)
            d.text((x - 45, 585), label, font=font(17, bold=True), fill=MUTED)
        for left, right in zip(positions, positions[1:]):
            for a in left:
                for b in right:
                    d.line((*a, *b), fill=GRID, width=2)
        d.text((945, 185), "loss ↓", font=font(28, bold=True), fill=ORANGE)
    return im


def write_sources():
    lines = ["# Chapter 2 Video Tutorial — Narration Script", ""]
    for i, slide in enumerate(SLIDES, 1):
        lines += [f"## {i:02d}. {slide['title']}", "", slide["narration"].strip(), ""]
    (OUT / "narration_script.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "lesson_plan.json").write_text(json.dumps(SLIDES, indent=2, ensure_ascii=False), encoding="utf-8")


def synthesize_audio():
    subprocess.run([
        "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
        str(ROOT / "render_chapter2_narration.ps1"), str(NARRATION_DIR), str(AUDIO_DIR),
    ], check=True)


def build_video():
    ffmpeg = get_ffmpeg_exe()
    segments = OUT / "segments"
    segments.mkdir(exist_ok=True)
    for i in range(1, len(SLIDES) + 1):
        subprocess.run([
            ffmpeg, "-y", "-loglevel", "error", "-loop", "1", "-framerate", str(FPS),
            "-i", str(SLIDES_DIR / f"{i:02d}.png"), "-i", str(AUDIO_DIR / f"{i:02d}.wav"),
            "-af", "apad=pad_dur=0.45", "-shortest", "-c:v", "libx264", "-preset", "veryfast",
            "-tune", "stillimage", "-pix_fmt", "yuv420p", "-r", str(FPS),
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100", str(segments / f"{i:02d}.mp4"),
        ], check=True)
    concat = segments / "concat.txt"
    concat.write_text("\n".join(f"file '{(segments / f'{i:02d}.mp4').as_posix()}'" for i in range(1, len(SLIDES) + 1)), encoding="utf-8")
    subprocess.run([
        ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
        "-c", "copy", "-movflags", "+faststart", str(VIDEO_PATH),
    ], check=True)


def write_timestamps():
    elapsed = 0.0
    ffmpeg = get_ffmpeg_exe()
    lines = ["# Chapter timestamps", ""]
    for i, slide in enumerate(SLIDES, 1):
        minutes, seconds = divmod(int(elapsed), 60)
        lines.append(f"- {minutes:02d}:{seconds:02d} — {slide['title']}")
        probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(OUT / "segments" / f"{i:02d}.mp4")], capture_output=True, text=True)
        match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", probe.stderr)
        if not match:
            raise RuntimeError(f"Cannot read duration for slide {i}")
        hours, mins, secs = match.groups()
        elapsed += int(hours) * 3600 + int(mins) * 60 + float(secs)
    lines += ["", f"Approximate runtime: {int(elapsed // 60)}:{int(elapsed % 60):02d}"]
    (OUT / "chapter_timestamps.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    for directory in (OUT, SLIDES_DIR, NARRATION_DIR, AUDIO_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    for i, slide in enumerate(SLIDES, 1):
        render_slide(slide, i).save(SLIDES_DIR / f"{i:02d}.png")
        (NARRATION_DIR / f"{i:02d}.txt").write_text(slide["narration"].strip(), encoding="utf-8-sig")
    write_sources()
    synthesize_audio()
    build_video()
    write_timestamps()
    print(f"Created: {VIDEO_PATH}")


if __name__ == "__main__":
    main()
