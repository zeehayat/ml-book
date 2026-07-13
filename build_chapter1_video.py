r"""Build a narrated Chapter 1 tutorial video and its reusable source assets.

Run with:
    .venv\Scripts\python.exe build_chapter1_video.py

The script intentionally uses only Pillow, MoviePy, and Windows SAPI so the
result can be rebuilt without a presentation application.
"""

from __future__ import annotations

import json
import re
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from imageio_ffmpeg import get_ffmpeg_exe


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "chapter1_video"
SLIDES_DIR = OUT / "slides"
NARRATION_DIR = OUT / "narration"
AUDIO_DIR = OUT / "audio"
VIDEO_PATH = OUT / "Chapter1_Tensors_and_Compute_Hardware_Tutorial.mp4"

W, H = 1280, 720
FPS = 12

BG = "#07111f"
PANEL = "#101f33"
PANEL_2 = "#152943"
WHITE = "#f5f8ff"
MUTED = "#a9b7ca"
CYAN = "#39d8ff"
GREEN = "#54e6a5"
ORANGE = "#ffb45b"
PINK = "#ff6fae"
GRID = "#29405e"

FONT_REG = Path(r"C:\Windows\Fonts\segoeui.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\segoeuib.ttf")
FONT_MONO = Path(r"C:\Windows\Fonts\consola.ttf")


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_MONO if mono else (FONT_BOLD if bold else FONT_REG)
    return ImageFont.truetype(str(path), size)


SLIDES = [
    {
        "title": "The Anatomy of a Tensor",
        "kicker": "CHAPTER 1  •  VIDEO TUTORIAL",
        "subtitle": "From flat memory to strides, views, caches, and GPU performance",
        "kind": "title",
        "narration": """Welcome to Chapter One: The Anatomy of a Tensor and Compute Hardware. In this tutorial, we will look beneath NumPy and PyTorch and build a hardware-level mental model of a tensor. By the end, shape, strides, slicing, transpose, contiguity, and memory performance should feel like consequences of one simple idea: multidimensional data lives in a flat buffer. Keep one question in mind throughout the lesson: when I ask for a logical element, which physical memory address must the computer read?""",
    },
    {
        "title": "What you will be able to do",
        "kicker": "LEARNING MAP",
        "bullets": [
            "Explain a tensor as buffer + dtype + shape + strides + offset",
            "Translate a multidimensional index into a flat buffer position",
            "Predict slicing and transpose without running Python",
            "Recognize when an operation is a view and when it must copy",
            "Connect access patterns to CPU cache and GPU bandwidth",
        ],
        "kind": "bullets",
        "narration": """Our goal is not to memorize framework behavior. It is to predict it. First, we will identify the small pieces of metadata that turn a byte buffer into a tensor. Then we will derive the flat-index formula and use it to understand row-major and column-major layouts. Next, we will transform metadata to create slices, transposes, and reversed views without moving data. Finally, we will connect memory layout to CPU caches, GPU bandwidth, implementation choices, and the most common production mistakes.""",
    },
    {
        "title": "Why not nested Python lists?",
        "kicker": "MOTIVATION",
        "kind": "pointers",
        "bullets": [
            "10,000 × 28 × 28 = 7.84 million pixels",
            "Nested lists store pointers to Python objects",
            "Each lookup may chase several unrelated addresses",
            "A tensor stores raw values in one dense byte buffer",
        ],
        "narration": """Suppose we store ten thousand grayscale images as nested Python lists. A lookup such as images three forty-seven, row five, column thirteen does not jump directly to a pixel. Python follows a pointer to an image list, another pointer to a row list, and another pointer to a Python number object. Each number also carries object metadata. A tensor removes that overhead. It stores the numerical values, and only the values, in one flat block. A few integers then describe the logical geometry. The result is smaller, predictable, and friendly to hardware that reads consecutive bytes efficiently.""",
    },
    {
        "title": "A tensor is a small contract",
        "kicker": "THE CORE MODEL",
        "kind": "components",
        "items": [
            ("BUFFER", "raw values", CYAN),
            ("DTYPE", "bytes per value", ORANGE),
            ("SHAPE", "logical extents", GREEN),
            ("STRIDES + OFFSET", "address rule", PINK),
        ],
        "narration": """At hardware level, a tensor is a contract between a flat buffer and some metadata. The buffer owns the raw values. The dtype says how to interpret bytes and how wide each element is. Shape gives the length of every logical axis. Strides say how far to jump in the buffer when an index increases by one on each axis. Finally, a base offset says where this particular view begins. Frameworks may add devices, gradients, and dispatch machinery, but this is the load-bearing representation.""",
    },
    {
        "title": "One formula explains every layout",
        "kicker": "ADDRESS TRANSLATION",
        "kind": "formula",
        "formula": "flat(i₀,…,iₙ₋₁) = offset + Σ iₖ · strideₖ",
        "example": "shape = (3, 4)   strides = (4, 1)   offset = 0\nindex (2, 1)  →  0 + 2·4 + 1·1 = 9",
        "narration": """Here is the chapter's central equation. To translate a multidimensional index, start at the view's offset. For every axis, multiply that axis index by its stride, then add the results. For a three by four row-major matrix, the strides are four and one. Index two, one maps to zero plus two times four plus one times one, which is buffer position nine. The boundary rule is equally important: every index on axis k must be at least zero and less than shape k. Shape controls valid logical indices; strides control physical movement.""",
    },
    {
        "title": "Row-major: the last axis moves fastest",
        "kicker": "C-CONTIGUOUS  •  SHAPE (3, 4)  •  STRIDES (4, 1)",
        "kind": "matrix",
        "order": "C",
        "narration": """In C-contiguous, or row-major, order, the last axis moves fastest. Our three by four matrix has stride one for columns because moving one column means moving one buffer element. The row stride is four because moving down one row skips the four elements in the previous row. In general, the final stride is one, and each earlier stride is the product of all dimensions to its right. Notice the highlighted path: walking left to right through a row visits consecutive addresses. That access pattern is ideal for cache lines.""",
    },
    {
        "title": "Column-major: the first axis moves fastest",
        "kicker": "F-CONTIGUOUS  •  SHAPE (3, 4)  •  STRIDES (1, 3)",
        "kind": "matrix",
        "order": "F",
        "narration": """Fortran-contiguous, or column-major, order uses the same buffer model with different strides. Now the row stride is one and the column stride is three. Walking downward through a column visits consecutive addresses. In general, the first stride is one, and each later stride is the product of dimensions to its left. Row-major and column-major are therefore not different kinds of tensor. They are two particular stride configurations. This is why one strided representation can support C, Fortran, NumPy, PyTorch, and many layouts in between.""",
    },
    {
        "title": "Slicing is an affine metadata update",
        "kicker": "ZERO-COPY VIEW",
        "kind": "slice",
        "formula": "new_offset = offset + start · stride\nnew_stride = stride · step",
        "narration": """A basic slice does not need to copy values. On a sliced axis, the new view starts later, so its offset becomes the old offset plus start times the old stride. If the step is greater than one, the new stride becomes the old stride times that step. The new shape is simply the number of selected positions. In this example, selecting every second column changes the column stride from one to two. Both the original tensor and the view still point to the same buffer, so writing through the view changes the original data.""",
    },
    {
        "title": "Transpose swaps metadata, not bytes",
        "kicker": "WHY .T IS FAST",
        "kind": "transpose",
        "narration": """Transpose is even simpler. For a two-dimensional tensor, swap the two shape entries and swap the two stride entries. A three by four tensor with strides four and one becomes a four by three view with strides one and four. The buffer remains untouched. That is why transposing a gigabyte-scale tensor can complete almost instantly: the operation changes a handful of integers, not a billion bytes. The tradeoff appears later, when a downstream operation traverses those bytes in an unfavorable order or requires contiguous input.""",
    },
    {
        "title": "Reshape asks a stricter question",
        "kicker": "VIEW OR COPY?",
        "kind": "decision",
        "bullets": [
            "Can the same address sequence be described by the new shape?",
            "YES → metadata-only view",
            "NO → reshape must copy, or view must reject",
            "Transpose often breaks C-contiguity without changing storage",
        ],
        "narration": """Reshape is not merely changing the labels on axes. A metadata-only reshape is valid only if the new shape can describe the same logical address sequence without rearranging storage. A contiguous tensor usually allows many reshapes because its addresses form one dense progression. A transposed view often does not. In PyTorch, view enforces this contiguity contract and may reject the operation. Reshape is more permissive: it returns a view when possible and silently allocates a contiguous copy when necessary. This difference matters in performance-sensitive loops.""",
    },
    {
        "title": "Negative strides reverse a view",
        "kicker": "SAME BUFFER  •  OPPOSITE DIRECTION",
        "kind": "reverse",
        "formula": "offset = 4,  stride = −1,  shape = (5,)",
        "narration": """A negative stride means that increasing a logical index moves backward in physical storage. To reverse a five-element vector without copying, begin at buffer position four, use shape five, and use stride negative one. Logical index zero reads position four; logical index four reads position zero. The subtle point is validation. Checking only the starting offset is unsafe. An implementation must compute the minimum and maximum reachable offsets across the entire shape, accounting for every positive and negative stride, and confirm that both lie inside the backing buffer.""",
    },
    {
        "title": "Why access order changes speed",
        "kicker": "CPU MEMORY HIERARCHY",
        "kind": "cache",
        "items": [
            ("REGISTERS", "~1 cycle", GREEN),
            ("L1 / L2 CACHE", "few cycles", CYAN),
            ("L3 CACHE", "tens of cycles", ORANGE),
            ("DRAM", "~100+ cycles", PINK),
        ],
        "narration": """Modern processors hide slow memory behind a hierarchy. Registers are tiny and fastest. L1 and L2 caches are small and close to each core. L3 is larger and slower. DRAM is vastly larger but can cost well over one hundred cycles to reach. When the CPU requests one float, hardware typically fetches a whole sixty-four-byte cache line, enough for sixteen float thirty-two values. If the next fifteen values are adjacent, that one miss pays for many useful reads. If a large stride jumps to a different cache line each time, most fetched bytes go unused.""",
    },
    {
        "title": "Contiguity is a performance property",
        "kicker": "CACHE-LINE UTILIZATION",
        "kind": "cachelines",
        "narration": """Compare these two walks. The contiguous walk consumes neighboring values from every fetched cache line. The strided walk touches one value, skips the rest of the line, and forces another memory request. Both loops perform the same number of additions, so big-O notation calls both linear. Hardware does not. Average memory access time can be modeled as hit time plus miss rate times miss penalty. A worse miss rate can create a large constant-factor slowdown. This is why loop order should match layout: iterate along the stride-one axis in the innermost loop.""",
    },
    {
        "title": "When should you call contiguous()?",
        "kicker": "COPY ONCE OR PAY REPEATEDLY",
        "kind": "breakeven",
        "formula": "copy_cost < repeats × (strided_cost − dense_cost)",
        "bullets": [
            "One traversal: the copy may cost more than it saves",
            "Repeated kernels: a one-time copy can win",
            "Measure end-to-end; copies consume time and memory bandwidth",
        ],
        "narration": """Calling contiguous materializes values in a new dense buffer. That costs linear time, consumes extra memory, and uses memory bandwidth. The copy is worthwhile only when its one-time cost is smaller than the repeated penalty of operating on the strided layout. For a single cheap traversal, keep the view. For many kernels, or for an operation whose implementation is dramatically faster on dense input, copying once may win. Treat contiguous as an explicit performance tradeoff, not a ritual. Profile the complete workload, including allocation and transfer costs.""",
    },
    {
        "title": "The same story continues on the GPU",
        "kicker": "GPU MEMORY HIERARCHY",
        "kind": "gpu",
        "items": [
            ("GLOBAL HBM", "large • high latency", PINK),
            ("L2 CACHE", "shared on device", ORANGE),
            ("SHARED MEMORY", "fast • per block", CYAN),
            ("REGISTERS", "fastest • per thread", GREEN),
        ],
        "narration": """A GPU has its own hierarchy. Large global HBM holds tensors. An on-device L2 cache reduces some traffic. Each streaming multiprocessor offers much faster shared memory for cooperating threads, plus registers for individual threads. Efficient kernels load global memory in coalesced, neighboring transactions, reuse tiles from shared memory, and keep intermediate values in registers. Strides influence whether neighboring threads request neighboring addresses. A logically correct layout can still waste most of a memory transaction if its access pattern is scattered.""",
    },
    {
        "title": "Arithmetic intensity predicts the bottleneck",
        "kicker": "ROOFLINE INTUITION",
        "kind": "intensity",
        "formula": "arithmetic intensity = operations ÷ bytes moved",
        "bullets": [
            "Low intensity → memory-bandwidth-bound",
            "High intensity → more likely compute-bound",
            "Layout, fusion, and reuse can reduce byte movement",
        ],
        "narration": """Arithmetic intensity is the number of useful operations performed per byte moved from memory. Elementwise addition has low intensity: it reads inputs, performs one addition, and writes an output. It is usually bandwidth-bound. Matrix multiplication can reuse loaded tiles for many multiply-adds, giving much higher intensity and a better chance of becoming compute-bound. Layout optimization matters because coalescing, caching, fusion, and reuse reduce unnecessary byte movement. Peak floating-point throughput is irrelevant when the kernel cannot feed the arithmetic units quickly enough.""",
    },
    {
        "title": "A minimal strided tensor implementation",
        "kicker": "IMPLEMENTATION CHECKLIST",
        "kind": "code",
        "code": """class TensorView:
    buffer: list[float]
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    offset: int

    def flat_index(self, index):
        return self.offset + sum(
            i * s for i, s in zip(index, self.strides)
        )""",
        "bullets": [
            "Validate rank and index bounds",
            "Views share the same buffer",
            "Slice/transpose return new metadata",
            "Validate min/max reachable offsets",
        ],
        "narration": """A minimal implementation needs a buffer, shape, strides, and offset. Element access validates the number and bounds of indices, computes the flat position with the central formula, and reads or writes the buffer. Slice and transpose return a new object that shares the same buffer but carries transformed metadata. Iteration must generate logical indices and translate each one. A robust implementation also handles zero-length dimensions, negative strides, axis insertion, and reachable-range validation. NumPy and PyTorch add optimized kernels and device machinery, but their view semantics grow from this same skeleton.""",
    },
    {
        "title": "Five mistakes to catch early",
        "kicker": "DEBUGGING GUIDE",
        "kind": "mistakes",
        "bullets": [
            "Assuming a slice owns independent data",
            "Assuming reshape is always zero-copy",
            "Computing a stride with the wrong product",
            "Validating only the base offset with negative strides",
            "Ignoring dtype width or iterating against the layout",
        ],
        "narration": """Here are the mistakes that cause the most confusion. First, a basic slice usually aliases the original buffer, so mutation is shared. Second, reshape may allocate when the address pattern is incompatible. Third, C-order strides multiply dimensions to the right, while Fortran order multiplies dimensions to the left. Fourth, negative strides require validating the whole reachable range. Fifth, byte offsets depend on dtype width, and performance depends on traversal order. When debugging, print shape, strides, offset, dtype, and whether the tensor is contiguous before inspecting values.""",
    },
    {
        "title": "The mental model to keep",
        "kicker": "CHAPTER 1 SUMMARY",
        "kind": "summary",
        "bullets": [
            "A tensor is a view over a flat typed buffer",
            "Address = offset + the dot product of index and strides",
            "Slice and transpose usually transform metadata",
            "Reshape is a view only when address order permits it",
            "Layout controls cache use, coalescing, and bandwidth",
        ],
        "narration": """Let us compress the chapter into five statements. A tensor is a logical view over a flat typed buffer. The physical address is the offset plus the dot product of indices and strides. Basic slicing and transpose usually create views by transforming metadata. Reshape is zero-copy only when the existing address sequence supports the new geometry. And layout is not cosmetic: it controls cache-line use, GPU coalescing, and memory bandwidth. With this foundation, automatic differentiation in Chapter Two becomes easier to reason about because every value and gradient ultimately lives in one of these strided buffers.""",
    },
]


def rounded(draw: ImageDraw.ImageDraw, box, radius=20, fill=PANEL, outline=None, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def fit_text(draw, text, box, size, color=WHITE, bold=False, spacing=8, align="left"):
    x, y, max_w, max_h = box
    f = font(size, bold=bold)
    avg = max(8, size * 0.53)
    lines = []
    for paragraph in text.splitlines() or [""]:
        lines.extend(textwrap.wrap(paragraph, width=max(1, int(max_w / avg))) or [""])
    line_h = size + spacing
    for line in lines[: max(1, int(max_h / line_h))]:
        draw.text((x, y), line, font=f, fill=color, align=align)
        y += line_h
    return y


def base_slide(slide, number):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    # subtle grid
    for x in range(0, W, 64):
        d.line((x, 0, x, H), fill="#0a1829", width=1)
    for y in range(0, H, 64):
        d.line((0, y, W, y), fill="#0a1829", width=1)
    d.rectangle((0, 0, 12, H), fill=CYAN)
    d.text((54, 34), slide.get("kicker", "CHAPTER 1"), font=font(18, bold=True), fill=CYAN)
    d.text((54, 65), slide["title"], font=font(40, bold=True), fill=WHITE)
    d.line((54, 122, 1226, 122), fill=GRID, width=2)
    d.text((54, 684), "ML SYSTEMS ENGINEERING HANDBOOK", font=font(14, bold=True), fill=MUTED)
    d.text((1164, 682), f"{number:02d}", font=font(18, bold=True), fill=MUTED)
    return im, d


def draw_bullets(d, bullets, x=78, y=166, width=1100, size=29, accent=CYAN):
    for n, item in enumerate(bullets):
        yy = y + n * 86
        rounded(d, (x, yy, x + width, yy + 64), 14, PANEL)
        d.ellipse((x + 20, yy + 21, x + 40, yy + 41), fill=accent)
        d.text((x + 58, yy + 14), item, font=font(size), fill=WHITE)


def draw_matrix(d, order):
    x0, y0, cw, ch = 92, 182, 145, 94
    strides = (4, 1) if order == "C" else (1, 3)
    for r in range(3):
        for c in range(4):
            flat = r * strides[0] + c * strides[1]
            x, y = x0 + c * cw, y0 + r * ch
            rounded(d, (x, y, x + cw - 12, y + ch - 12), 14, PANEL_2, GRID)
            d.text((x + 16, y + 14), f"A[{r},{c}]", font=font(20, bold=True), fill=WHITE)
            d.text((x + 16, y + 48), f"buffer[{flat}]", font=font(17, mono=True), fill=CYAN)
    rounded(d, (735, 178, 1180, 482), 20, PANEL)
    d.text((770, 210), "PHYSICAL BUFFER", font=font(20, bold=True), fill=MUTED)
    for i in range(12):
        col, row = i % 6, i // 6
        x, y = 770 + col * 62, 265 + row * 90
        rounded(d, (x, y, x + 51, y + 54), 9, PANEL_2, CYAN if i < 6 else GREEN)
        d.text((x + 15, y + 12), str(i), font=font(19, bold=True), fill=WHITE)
    d.text((93, 510), f"flat(r, c) = r·{strides[0]} + c·{strides[1]}", font=font(31, mono=True), fill=GREEN)


def render_slide(slide, idx):
    im, d = base_slide(slide, idx)
    kind = slide["kind"]
    if kind == "title":
        d.text((78, 178), "TENSORS", font=font(86, bold=True), fill=CYAN)
        d.text((78, 276), "ARE MEMORY LAYOUTS", font=font(57, bold=True), fill=WHITE)
        fit_text(d, slide["subtitle"], (82, 370, 760, 100), 29, MUTED)
        # buffer motif
        for i in range(8):
            x = 820 + (i % 4) * 88
            y = 190 + (i // 4) * 88
            rounded(d, (x, y, x + 70, y + 70), 12, PANEL_2, CYAN)
            d.text((x + 24, y + 19), str(i), font=font(23, bold=True), fill=WHITE)
        d.text((816, 390), "shape (2, 4)", font=font(24, mono=True), fill=GREEN)
        d.text((816, 432), "strides (4, 1)", font=font(24, mono=True), fill=PINK)
    elif kind == "bullets" or kind == "summary":
        draw_bullets(d, slide["bullets"], accent=GREEN if kind == "summary" else CYAN)
    elif kind == "pointers":
        rounded(d, (72, 166, 560, 586), 20, PANEL)
        labels = ["images", "image 347", "row 5", "float"]
        for i, label in enumerate(labels):
            x, y = 105 + i * 108, 255 + i * 55
            rounded(d, (x, y, x + 118, y + 55), 12, PANEL_2, PINK)
            d.text((x + 12, y + 15), label, font=font(17, bold=True), fill=WHITE)
            if i < 3:
                d.line((x + 105, y + 50, x + 138, y + 85), fill=PINK, width=4)
        draw_bullets(d, slide["bullets"], x=610, y=176, width=570, size=22, accent=GREEN)
    elif kind == "components":
        for i, (name, desc, color) in enumerate(slide["items"]):
            x = 74 + (i % 2) * 580
            y = 180 + (i // 2) * 190
            rounded(d, (x, y, x + 535, y + 150), 22, PANEL, color, 3)
            d.text((x + 30, y + 28), name, font=font(27, bold=True), fill=color)
            d.text((x + 30, y + 81), desc, font=font(25), fill=WHITE)
    elif kind == "formula":
        rounded(d, (74, 180, 1200, 310), 24, PANEL, CYAN, 3)
        d.text((108, 220), slide["formula"], font=font(34, mono=True), fill=WHITE)
        rounded(d, (74, 350, 1200, 548), 24, PANEL_2)
        fit_text(d, slide["example"], (108, 390, 1030, 130), 30, GREEN, spacing=16)
    elif kind == "matrix":
        draw_matrix(d, slide["order"])
    elif kind == "slice":
        for i in range(12):
            x = 80 + i * 88
            selected = i % 2 == 0
            rounded(d, (x, 190, x + 68, 258), 10, PANEL_2, GREEN if selected else GRID, 3)
            d.text((x + 22, 210), str(i), font=font(22, bold=True), fill=WHITE if selected else MUTED)
            if selected:
                d.line((x + 34, 265, x + 34, 305), fill=GREEN, width=3)
        rounded(d, (168, 318, 1100, 405), 18, PANEL)
        d.text((205, 343), "view = buffer[0 : 12 : 2]  →  positions 0,2,4,6,8,10", font=font(25, mono=True), fill=GREEN)
        fit_text(d, slide["formula"], (205, 450, 850, 110), 28, CYAN, spacing=12)
    elif kind == "transpose":
        rounded(d, (82, 180, 510, 500), 20, PANEL)
        rounded(d, (770, 180, 1190, 500), 20, PANEL)
        d.text((132, 220), "shape   (3, 4)", font=font(28, mono=True), fill=GREEN)
        d.text((132, 280), "strides (4, 1)", font=font(28, mono=True), fill=PINK)
        d.text((820, 220), "shape   (4, 3)", font=font(28, mono=True), fill=GREEN)
        d.text((820, 280), "strides (1, 4)", font=font(28, mono=True), fill=PINK)
        d.line((540, 330, 730, 330), fill=CYAN, width=8)
        d.polygon([(730, 330), (695, 310), (695, 350)], fill=CYAN)
        d.text((555, 365), "swap axes", font=font(23, bold=True), fill=CYAN)
        d.text((458, 545), "BUFFER UNCHANGED", font=font(30, bold=True), fill=WHITE)
    elif kind == "decision":
        rounded(d, (340, 160, 940, 245), 18, PANEL, CYAN, 3)
        d.text((390, 184), "Same address sequence?", font=font(31, bold=True), fill=WHITE)
        d.line((520, 245, 365, 342), fill=GREEN, width=5)
        d.line((760, 245, 915, 342), fill=PINK, width=5)
        rounded(d, (105, 340, 515, 460), 18, PANEL, GREEN, 3)
        rounded(d, (765, 340, 1175, 460), 18, PANEL, PINK, 3)
        d.text((150, 370), "YES  →  VIEW", font=font(32, bold=True), fill=GREEN)
        d.text((805, 370), "NO  →  COPY", font=font(32, bold=True), fill=PINK)
        fit_text(d, slide["bullets"][3], (260, 515, 800, 70), 25, MUTED)
    elif kind == "reverse":
        for i in range(5):
            x = 120 + i * 190
            rounded(d, (x, 190, x + 140, 300), 18, PANEL_2, CYAN)
            d.text((x + 52, 215), str(i), font=font(32, bold=True), fill=WHITE)
            d.text((x + 35, 265), f"buf[{i}]", font=font(17, mono=True), fill=MUTED)
        d.line((980, 350, 145, 350), fill=PINK, width=8)
        d.polygon([(145, 350), (180, 330), (180, 370)], fill=PINK)
        d.text((397, 378), "logical traversal", font=font(25, bold=True), fill=PINK)
        rounded(d, (295, 460, 985, 545), 18, PANEL)
        d.text((342, 486), slide["formula"], font=font(27, mono=True), fill=GREEN)
    elif kind == "cache" or kind == "gpu":
        items = slide["items"]
        for i, (name, desc, color) in enumerate(items):
            margin = i * 85
            x1, x2 = 170 + margin, 1110 - margin
            y1, y2 = 170 + i * 100, 250 + i * 100
            rounded(d, (x1, y1, x2, y2), 18, PANEL_2, color, 3)
            d.text((x1 + 28, y1 + 20), name, font=font(25, bold=True), fill=color)
            d.text((x2 - 225, y1 + 24), desc, font=font(20), fill=WHITE)
    elif kind == "cachelines":
        for row, (label, step, color) in enumerate([("CONTIGUOUS", 1, GREEN), ("STRIDED", 4, PINK)]):
            y = 200 + row * 190
            d.text((80, y), label, font=font(23, bold=True), fill=color)
            for i in range(16):
                x = 300 + i * 52
                active = i % step == 0
                rounded(d, (x, y - 7, x + 42, y + 42), 7, color if active else PANEL_2, GRID)
            d.text((300, y + 65), "one cache line" if row == 0 else "useful value scattered across lines", font=font(21), fill=MUTED)
    elif kind == "breakeven" or kind == "intensity":
        rounded(d, (80, 175, 1200, 295), 22, PANEL, CYAN, 3)
        d.text((118, 214), slide["formula"], font=font(31, mono=True), fill=WHITE)
        draw_bullets(d, slide["bullets"], x=80, y=340, width=1120, size=25, accent=ORANGE)
    elif kind == "code":
        rounded(d, (64, 156, 770, 605), 18, "#0b1727", GRID)
        fit_text(d, slide["code"], (92, 180, 640, 390), 21, GREEN, spacing=6)
        draw_bullets(d, slide["bullets"], x=805, y=175, width=390, size=19, accent=CYAN)
    elif kind == "mistakes":
        draw_bullets(d, slide["bullets"], size=25, accent=PINK)
    return im


def write_narration_script():
    lines = ["# Chapter 1 Video Tutorial — Narration Script", ""]
    for i, s in enumerate(SLIDES, 1):
        lines += [f"## {i:02d}. {s['title']}", "", s["narration"].strip(), ""]
    (OUT / "narration_script.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "lesson_plan.json").write_text(json.dumps(SLIDES, indent=2, ensure_ascii=False), encoding="utf-8")


def synthesize_audio():
    ps1 = ROOT / "render_chapter1_narration.ps1"
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1), str(NARRATION_DIR), str(AUDIO_DIR)],
        check=True,
    )


def write_chapter_timestamps():
    elapsed = 0.0
    ffmpeg = get_ffmpeg_exe()
    lines = ["# Chapter timestamps", ""]
    for i, slide in enumerate(SLIDES, 1):
        minutes, seconds = divmod(int(elapsed), 60)
        lines.append(f"- {minutes:02d}:{seconds:02d} — {slide['title']}")
        probe = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", str(OUT / "segments" / f"{i:02d}.mp4")],
            capture_output=True, text=True,
        )
        match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", probe.stderr)
        if not match:
            raise RuntimeError(f"Could not read duration for segment {i:02d}")
        hours, mins, secs = match.groups()
        elapsed += int(hours) * 3600 + int(mins) * 60 + float(secs)
    lines += ["", f"Approximate runtime: {int(elapsed // 60)}:{int(elapsed % 60):02d}"]
    (OUT / "chapter_timestamps.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_video():
    segments = OUT / "segments"
    segments.mkdir(exist_ok=True)
    ffmpeg = get_ffmpeg_exe()
    for i in range(1, len(SLIDES) + 1):
        segment = segments / f"{i:02d}.mp4"
        subprocess.run([
            ffmpeg, "-y", "-loglevel", "error",
            "-loop", "1", "-framerate", str(FPS), "-i", str(SLIDES_DIR / f"{i:02d}.png"),
            "-i", str(AUDIO_DIR / f"{i:02d}.wav"),
            "-af", "apad=pad_dur=0.45", "-shortest",
            "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage",
            "-pix_fmt", "yuv420p", "-r", str(FPS),
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
            str(segment),
        ], check=True)

    concat_file = segments / "concat.txt"
    concat_file.write_text(
        "\n".join(f"file '{(segments / f'{i:02d}.mp4').as_posix()}'" for i in range(1, len(SLIDES) + 1)),
        encoding="utf-8",
    )
    subprocess.run([
        ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", str(concat_file), "-c", "copy", "-movflags", "+faststart", str(VIDEO_PATH),
    ], check=True)


def main():
    for directory in (OUT, SLIDES_DIR, NARRATION_DIR, AUDIO_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    for i, slide in enumerate(SLIDES, 1):
        render_slide(slide, i).save(SLIDES_DIR / f"{i:02d}.png", quality=95)
        (NARRATION_DIR / f"{i:02d}.txt").write_text(slide["narration"].strip(), encoding="utf-8-sig")
    write_narration_script()
    synthesize_audio()
    build_video()
    write_chapter_timestamps()
    print(f"Created: {VIDEO_PATH}")


if __name__ == "__main__":
    main()
