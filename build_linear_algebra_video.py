import os
import json
import re
import subprocess
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "linear_algebra_video"
SLIDES_DIR = OUT / "slides"
AUDIO_DIR = OUT / "audio"
SEGS_DIR = OUT / "segments"
VIDEO_PATH = OUT / "Just_Enough_Linear_Algebra_for_ML.mp4"

W, H = 1280, 720
FPS = 12

# Professional Color Palette
BG = "#0b1329"
PANEL = "#1c2541"
PANEL_2 = "#222b4e"
WHITE = "#ffffff"
MUTED = "#a5b4fc"
CYAN = "#5bc0be"
GREEN = "#4ade80"
ORANGE = "#f97316"
PINK = "#f43f5e"
GRID = "#3a506b"

# Linux Font Paths
FONT_REG = Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf")
FONT_MONO = Path("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf")

# Fallbacks if Noto is missing
if not FONT_REG.exists():
    FONT_REG = Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf")
if not FONT_BOLD.exists():
    FONT_BOLD = Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf")
if not FONT_MONO.exists():
    FONT_MONO = Path("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf")

def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_MONO if mono else (FONT_BOLD if bold else FONT_REG)
    try:
        return ImageFont.truetype(str(path), size)
    except IOError:
        return ImageFont.load_default()

SLIDES = [
    {
        "title": "Just Enough Linear Algebra for ML",
        "kicker": "MATHEMATICAL PRIMER  •  BEGINNER LEVEL",
        "subtitle": "The foundational vector and matrix operations that power machine learning models",
        "kind": "title",
        "narration": "Welcome to Just Enough Linear Algebra for Machine Learning. This tutorial is designed for absolute beginners. We will skip the abstract proofs and focus entirely on the load-bearing concepts—vectors, matrices, multiplication rules, transformations, and factorization—that you need to understand how data moves through machine learning models."
    },
    {
        "title": "01. Vectors: The Coordinates of Data",
        "kicker": "CORE MATHEMATICAL OBJECTS",
        "kind": "formula",
        "formula": "v = [v₁, v₂, ..., v_d]ᵀ",
        "example": "Geometric: An arrow in space pointing to coordinates.\nAlgebraic: An ordered list of numbers.\n\nOperations:\nAddition: [1, 2] + [3, 4] = [4, 6]\nScalar Mult: 2 · [1, 3] = [2, 6]",
        "narration": "Let us start with vectors. Algebraically, a vector is simply an ordered list of numbers representing features of an observation, such as an house's price and size. Geometrically, it is an arrow in space pointing to specific coordinates. We can add vectors element-wise to combine features, or scale them to adjust magnitudes."
    },
    {
        "title": "02. Matrices: Grids & Datasets",
        "kicker": "MULTIDIMENSIONAL STORAGE",
        "kind": "pointers",
        "bullets": [
            "A matrix is a 2D grid of numbers with shape Row × Column",
            "A row vector represents a single observation (example)",
            "A column vector represents a single feature across observations",
            "Design Matrix X: Stacks all examples (n rows, d columns)",
        ],
        "narration": "A matrix is a two-dimensional grid of numbers, organized in rows and columns. In machine learning, we stack our data in a design matrix X, where each row represents a single example or observation, and each column represents a single feature or measurement across all examples."
    },
    {
        "title": "03. Matrix Multiplication (GEMM)",
        "kicker": "THE PRIMARY COMPUTATIONAL WORKHORSE",
        "kind": "formula",
        "formula": "(n × k) · (k × p)  =  (n × p)",
        "example": "Rule: Output cell (i, j) is the dot product of Row i of A and Column j of B.\n\nConstraint:\nInner dimensions must match (k = k).\nNon-commutative: A · B is not equal to B · A.",
        "narration": "Matrix multiplication is the core compute operation in machine learning. To multiply matrix A and B, the number of columns in A must equal the number of rows in B. The resulting cell at row i, column j is the dot product of A's i-th row and B's j-th column. Crucially, matrix multiplication is not commutative: changing the order changes the result."
    },
    {
        "title": "04. Special Matrices: Transpose & Identity",
        "kicker": "METADATA AND ALGEBRAIC ANCHORS",
        "kind": "components",
        "items": [
            ("TRANSPOSE (Xᵀ)", "Flips rows and columns: shape (n x d) becomes (d x n)", CYAN),
            ("IDENTITY (I)", "Diagonal of ones, zeros elsewhere: A · I = A", ORANGE),
            ("DOT PRODUCT (uᵀv)", "Measures alignment: uᵀv = 0 means orthogonal", GREEN),
            ("DIAGONAL MATRIX", "Scaling elements on the diagonal only", PINK),
        ],
        "narration": "There are several special matrix operations. The transpose, written as X transpose, flips the rows and columns, turning a row-major matrix into a column-major layout. The identity matrix acts like the number one in scalar math: multiplying any matrix by the identity leaves it unchanged. The dot product of two vectors measures their geometric alignment."
    },
    {
        "title": "05. Invertibility & Rank",
        "kicker": "SOLVABILITY CONDITIONS",
        "kind": "pointers",
        "bullets": [
            "Inverse Matrix A⁻¹ satisfies: A · A⁻¹ = I",
            "Only square matrices can have an inverse",
            "Rank: Number of linearly independent row or column directions",
            "Full Rank: Matrix contains no redundant information (invertible)",
        ],
        "narration": "To solve systems of equations, we use the inverse of a matrix, written as A inverse, which satisfies A times A inverse equals the identity matrix. An inverse exists only for square, full-rank matrices—meaning they have no redundant column directions. If a matrix lacks full rank, it is singular and cannot be inverted."
    },
    {
        "title": "06. Linear Transformations",
        "kicker": "GEOMETRIC PERSPECTIVE",
        "kind": "formula",
        "formula": "A · x = b",
        "example": "A acts as a function: warping, rotating, or scaling space.\nx is the input vector in the domain.\nb is the output vector in the codomain.\n\nSolving Ax = b is finding the input x that maps to target b.",
        "narration": "We can view matrix multiplication geometrically as a linear transformation. A matrix A acts as a function that warps, rotates, or scales space. When we write A times x equals b, we are asking: what input vector x in our domain gets mapped to the target vector b in our codomain? Solving this system is equivalent to inverting the transformation."
    },
    {
        "title": "07. Eigenvectors & Eigenvalues",
        "kicker": "DIRECTIONS OF PURE SCALING",
        "kind": "formula",
        "formula": "A · v = λ · v",
        "example": "v is the Eigenvector (direction remains unchanged by A).\nλ is the Eigenvalue (scalar scale factor of stretch or shrink).\n\nApplication in ML:\nLeading eigenvectors of covariance matrices define Principal Components.",
        "narration": "An eigenvector of a matrix A is a special direction that does not change its orientation when transformed by A; it only gets stretched or shrunk. The scale factor is called the eigenvalue. In machine learning, the leading eigenvectors of a dataset's covariance matrix point in the directions of maximum variance, forming the basis of Principal Component Analysis."
    },
    {
        "title": "08. Singular Value Decomposition (SVD)",
        "kicker": "MATRIX FACTORIZATION FOR DATA REDUCTION",
        "kind": "components",
        "items": [
            ("U (n x n)", "Orthogonal columns: left singular vectors", CYAN),
            ("Σ (n x d)", "Diagonal matrix: singular values (magnitudes)", ORANGE),
            ("Vᵀ (d x d)", "Orthogonal rows: right singular vectors", GREEN),
            ("LOW-RANK APPROX", "Keep leading singular values to compress data", PINK),
        ],
        "narration": "Singular Value Decomposition, or SVD, factorizes any rectangular matrix A into three matrices: U, Sigma, and V transpose. U and V transpose contain orthogonal vectors, while Sigma is a diagonal matrix containing singular values that measure scaling magnitudes. By keeping only the largest singular values, we can perform low-rank approximations to compress high-dimensional data."
    },
    {
        "title": "Linear Algebra Summary",
        "kicker": "SUMMARY & NEXT STEPS",
        "kind": "bullets",
        "bullets": [
            "Vectors represent data features; matrices represent grouped datasets",
            "Matrix multiplication requires outer and inner dimension compatibility",
            "Inverses require full column rank matrices to guarantee unique solutions",
            "Eigenvalues and SVD isolate maximum variance and compress data spaces",
            "With this math, linear regression and projections become intuitive",
        ],
        "narration": "Let us summarize. Vectors coordinate data features; matrices store datasets. Matrix multiplication computes combinations using row-by-column dot products. Inverting a matrix is solving a geometric projection, which requires full rank. And eigenvectors and SVD identify the principal directions of variance to compress data. With these essentials, you are ready for classical regression and deep learning."
    }
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
    # Draw subtle background grid lines
    for x in range(0, W, 64):
        d.line((x, 0, x, H), fill="#0f172a", width=1)
    for y in range(0, H, 64):
        d.line((0, y, W, y), fill="#0f172a", width=1)
    d.rectangle((0, 0, 12, H), fill=CYAN)
    d.text((54, 34), slide.get("kicker", "MATH PRIMER"), font=font(18, bold=True), fill=CYAN)
    d.text((54, 65), slide["title"], font=font(40, bold=True), fill=WHITE)
    d.line((54, 122, 1226, 122), fill=GRID, width=2)
    d.text((54, 684), "JUST ENOUGH LINEAR ALGEBRA", font=font(14, bold=True), fill=MUTED)
    d.text((1164, 682), f"{number:02d}", font=font(18, bold=True), fill=MUTED)
    return im, d

def draw_bullets(d, bullets, x=78, y=166, width=1100, size=24, accent=CYAN):
    for n, item in enumerate(bullets):
        yy = y + n * 86
        rounded(d, (x, yy, x + width, yy + 64), 14, PANEL)
        d.ellipse((x + 20, yy + 21, x + 40, yy + 41), fill=accent)
        d.text((x + 58, yy + 16), item, font=font(size), fill=WHITE)

def render_slide(slide, idx):
    im, d = base_slide(slide, idx)
    kind = slide["kind"]
    if kind == "title":
        d.text((78, 178), "LINEAR", font=font(86, bold=True), fill=CYAN)
        d.text((78, 276), "ALGEBRA PRIMER", font=font(57, bold=True), fill=WHITE)
        fit_text(d, slide["subtitle"], (82, 370, 760, 100), 28, MUTED)
        # Decorative visual motif: Vector projections
        d.line((850, 430, 1080, 200), fill=MUTED, width=4) # original vector
        d.line((850, 430, 1150, 430), fill=GRID, width=2)  # axis
        d.line((1080, 200, 1080, 430), fill=PINK, width=2, joint="round") # projection line
        d.ellipse((1070, 190, 1090, 210), fill=CYAN)
    elif kind == "bullets":
        draw_bullets(d, slide["bullets"], accent=GREEN if "summary" in slide["kicker"].lower() else CYAN)
    elif kind == "pointers":
        rounded(d, (72, 166, 560, 586), 20, PANEL)
        # Left side panel matrix grid diagram
        d.text((105, 205), "Matrix A (Row x Column)", font=font(22, bold=True), fill=MUTED)
        for r in range(3):
            for c in range(3):
                x = 130 + c * 110
                y = 260 + r * 90
                rounded(d, (x, y, x + 90, y + 70), 12, PANEL_2, GRID)
                d.text((x + 22, y + 22), f"a_{r}{c}", font=font(18, mono=True), fill=WHITE)
        draw_bullets(d, slide["bullets"], x=610, y=166, width=570, size=19, accent=GREEN)
    elif kind == "formula":
        rounded(d, (74, 180, 1200, 310), 24, PANEL, CYAN, 3)
        d.text((108, 220), slide["formula"], font=font(34, mono=True), fill=WHITE)
        rounded(d, (74, 350, 1200, 548), 24, PANEL_2)
        fit_text(d, slide["example"], (108, 380, 1030, 140), 25, GREEN, spacing=14)
    elif kind == "components":
        for i, (name, desc, color) in enumerate(slide["items"]):
            x = 74 + (i % 2) * 580
            y = 180 + (i // 2) * 190
            rounded(d, (x, y, x + 535, y + 150), 22, PANEL, color, 3)
            d.text((x + 30, y + 28), name, font=font(25, bold=True), fill=color)
            d.text((x + 30, y + 81), desc, font=font(21), fill=WHITE)
    return im

def write_narration_script():
    lines = ["# Just Enough Linear Algebra for ML — Narration Script", ""]
    for i, s in enumerate(SLIDES, 1):
        lines += [f"## {i:02d}. {s['title']}", "", s["narration"].strip(), ""]
    (OUT / "narration_script.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "lesson_plan.json").write_text(json.dumps(SLIDES, indent=2, ensure_ascii=False), encoding="utf-8")

def synthesize_audio():
    print("Synthesizing narration audios using gTTS...")
    for i, slide in enumerate(SLIDES, 1):
        print(f"  Generating slide {i:02d} audio...")
        tts = gTTS(text=slide["narration"], lang="en")
        tts.save(str(AUDIO_DIR / f"{i:02d}.mp3"))

def write_chapter_timestamps():
    elapsed = 0.0
    lines = ["# Chapter timestamps", ""]
    for i, slide in enumerate(SLIDES, 1):
        minutes, seconds = divmod(int(elapsed), 60)
        lines.append(f"- {minutes:02d}:{seconds:02d} — {slide['title']}")
        # Query duration of segment using ffmpeg
        segment_path = SEGS_DIR / f"{i:02d}.mp4"
        probe = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(segment_path)],
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
    print("Building video segments using FFmpeg...")
    for i in range(1, len(SLIDES) + 1):
        segment = SEGS_DIR / f"{i:02d}.mp4"
        # Combine image + mp3 audio into mp4 segment using ffmpeg
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-loop", "1", "-framerate", str(FPS), "-i", str(SLIDES_DIR / f"{i:02d}.png"),
            "-i", str(AUDIO_DIR / f"{i:02d}.mp3"),
            "-af", "apad=pad_dur=0.5", "-shortest",
            "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage",
            "-pix_fmt", "yuv420p", "-r", str(FPS),
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
            str(segment),
        ], check=True)

    # Concat segments
    concat_file = SEGS_DIR / "concat.txt"
    concat_file.write_text(
        "\n".join(f"file '{(SEGS_DIR / f'{i:02d}.mp4').as_posix()}'" for i in range(1, len(SLIDES) + 1)),
        encoding="utf-8",
    )
    print("Concatenating segments into final video...")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", str(concat_file), "-c", "copy", "-movflags", "+faststart", str(VIDEO_PATH),
    ], check=True)

def main():
    print("Initializing directories...")
    for directory in (OUT, SLIDES_DIR, AUDIO_DIR, SEGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    
    print("Rendering slides...")
    for i, slide in enumerate(SLIDES, 1):
        render_slide(slide, i).save(SLIDES_DIR / f"{i:02d}.png", quality=95)
        
    write_narration_script()
    synthesize_audio()
    build_video()
    write_chapter_timestamps()
    print(f"Successfully generated: {VIDEO_PATH}")

if __name__ == "__main__":
    main()
