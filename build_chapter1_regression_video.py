import os
import json
import re
import subprocess
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "chapter1_regression_video"
SLIDES_DIR = OUT / "slides"
AUDIO_DIR = OUT / "audio"
SEGS_DIR = OUT / "segments"
VIDEO_PATH = OUT / "Chapter1_Linear_and_Logistic_Regression_Tutorial.mp4"

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
        # Final fallback to PIL default font
        return ImageFont.load_default()

SLIDES = [
    {
        "title": "Linear & Logistic Regression",
        "kicker": "CHAPTER 1  •  VIDEO TUTORIAL",
        "subtitle": "Derivations, properties, diagnostics, regularizations, and implementation tracks",
        "kind": "title",
        "narration": "Welcome to Chapter One: Linear Regression and Logistic Models. In this video tutorial, we will walk through the mathematical foundations, statistical properties, diagnostic mitigations, and implementation ecosystems of classical regression analysis. We will start from simple ordinary least squares and build up to generalized linear boundaries and regularized optimization. Let us begin."
    },
    {
        "title": "What you will be able to do",
        "kicker": "LEARNING OBJECTIVES",
        "kind": "bullets",
        "bullets": [
            "Derive OLS slope and intercept for simple linear regression",
            "Solve multiple linear regression using matrix projection notation",
            "Verify model assumptions and run diagnostic mitigations",
            "Apply Ridge and Lasso regularization to control overfitting",
            "Train binary logistic regression using gradient descent optimization",
        ],
        "narration": "By the end of this chapter, you will be able to derive the ordinary least squares slope and intercept by hand and express multiple regression using projection geometry. You will also be able to run model diagnostic tests, apply Ridge and Lasso regularizations, and train a binary logistic regression classifier from scratch using gradient descent."
    },
    {
        "title": "Ordinary Least Squares (OLS) Math",
        "kicker": "MATHEMATICAL FOUNDATIONS",
        "kind": "formula",
        "formula": "y = Xβ + ε   →   β̂ = (XᵀX)⁻¹Xᵀy",
        "example": "Simple Regression:\nβ₁ = Cov(x, y) / Var(x)   and   β₀ = ȳ - β₁x̄\n\nMatrix Objective:\nMinimize SSR(β) = ||y - Xβ||²",
        "narration": "Let us start with Ordinary Least Squares. For a single predictor, the slope is the covariance of x and y divided by the variance of x, and the intercept shifts the line to pass through the sample means. In multiple regression with several predictors, we write the system in matrix form as y equals X beta plus epsilon. Minimizing the sum of squared residuals yields the famous closed-form solution: beta hat equals X transpose X inverse times X transpose y."
    },
    {
        "title": "OLS Geometric Projection",
        "kicker": "PROJECTION GEOMETRY",
        "kind": "pointers",
        "bullets": [
            "Outcome vector y is projected onto the column space of X",
            "Projected fitted values: ŷ = H y  where  H = X(XᵀX)⁻¹Xᵀ",
            "Residual vector e = y - ŷ is orthogonal to the column space",
            "Orthogonality condition: Xᵀe = 0"
        ],
        "narration": "Geometrically, ordinary least squares is an orthogonal projection. The outcome vector y lives in an n-dimensional space, while the columns of the design matrix X span a lower-dimensional subspace. The hat matrix H projects y orthogonally onto this subspace to yield the fitted values y hat. The residual vector e represents the projection error, which is mathematically orthogonal to the column space of X, meaning X transpose times e equals zero."
    },
    {
        "title": "The Gauss-Markov Theorem",
        "kicker": "ESTIMATOR PROPERTIES",
        "kind": "bullets",
        "bullets": [
            "Strict Exogeneity: E[ε | X] = 0 (no confounding)",
            "Spherical Errors: Constant variance & zero correlation across errors",
            "No Multicollinearity: X has full column rank",
            "Under these conditions, OLS is BLUE",
            "BLUE: Best Linear Unbiased Estimator (minimum variance)",
        ],
        "narration": "The Gauss-Markov Theorem defines the optimal properties of the OLS estimator. If the relationship is linear, predictors are strictly exogenous, the design matrix has full column rank, and the errors are spherical, meaning homoskedastic and uncorrelated, then OLS is the Best Linear Unbiased Estimator, or BLUE. This means no other linear unbiased estimator can achieve a lower variance."
    },
    {
        "title": "Uncertainty & Inference",
        "kicker": "STATISTICAL SIGNIFICANCE",
        "kind": "components",
        "items": [
            ("RESIDUAL VARIANCE", "s² = eᵀe / (n - k - 1)", CYAN),
            ("COVARIANCE MATRIX", "Var(β̂|X) = s²(XᵀX)⁻¹", ORANGE),
            ("t-STATISTIC", "t = β̂ⱼ / SE(β̂ⱼ)  ~ t_{n-k-1}", GREEN),
            ("F-STATISTIC", "Joint linear hypothesis testing", PINK),
        ],
        "narration": "To conduct statistical inference, we must quantify our uncertainty. We estimate the residual variance s squared by dividing the sum of squared residuals by the degrees of freedom. This variance scale determines the covariance matrix of our coefficients. We can then test individual coefficient significance using t-statistics and evaluate joint linear restrictions using F-statistics."
    },
    {
        "title": "Model Diagnostics & Mitigations",
        "kicker": "DEBUGGING ASSUMPTIONS",
        "kind": "pointers",
        "bullets": [
            "Heteroskedasticity: Non-constant variance (White robust SEs)",
            "Multicollinearity: VIF > 10 (Variable deletion or regularization)",
            "Non-Normality: Saturated errors (Log transforms / bootstrap)",
            "High Leverage: Cook's distance > 1 (Outlier inspection)",
        ],
        "narration": "Before trusting our inference, we must run diagnostics to check for violations of OLS assumptions. If the errors are heteroskedastic, standard errors are biased and must be corrected using White robust standard errors. Multicollinearity is detected when the Variance Inflation Factor exceeds ten. Outliers with high leverage are identified using Cook's distance."
    },
    {
        "title": "Regularization: Ridge vs. Lasso",
        "kicker": "SHRINKAGE TECHNIQUES",
        "kind": "formula",
        "formula": "Ridge: Objective + λ||β||₂²   vs.   Lasso: Objective + λ||β||₁",
        "example": "Ridge (L2): Shrinks coefficients close to zero (dense representation)\nLasso (L1): Shrinks coefficients to exactly zero (sparse feature selection)",
        "narration": "When multicollinearity is present or parameters exceed observations, OLS variance explodes. Regularization introduces a bias to reduce variance. Ridge regression adds an L2 penalty, shrinking coefficients close to zero. Lasso regression adds an L1 penalty, which drives redundant coefficients to exactly zero, performing automatic feature selection."
    },
    {
        "title": "Logistic Regression Mechanics",
        "kicker": "BINARY CLASSIFICATION",
        "kind": "formula",
        "formula": "p = σ(Xβ) = 1 / (1 + e⁻ˣᵝ)",
        "example": "Link function: Log-odds (logit link) is linear\nlog(p / (1 - p)) = Xβ",
        "narration": "For binary outcomes, linear regression fails because it can predict values outside zero and one. Logistic regression models the probability of success using the Sigmoid activation function. This maps the linear predictor to a probability between zero and one. The link function is the log-odds, also called the logit link, which remains linear."
    },
    {
        "title": "Log-Likelihood & Optimization",
        "kicker": "MAXIMUM LIKELIHOOD ESTIMATION",
        "kind": "formula",
        "formula": "β ← β - η ∇ J   where  ∇ J = Xᵀ(p - y) / n",
        "example": "BCE Loss:\nJ(β) = -1/n * Σ [y log(p) + (1-y) log(1-p)]",
        "narration": "We estimate logistic coefficients by maximizing the Bernoulli likelihood, which is equivalent to minimizing the Binary Cross-Entropy loss. Since no analytical closed-form solution exists, we optimize the convex loss function iteratively using Gradient Descent. The gradient simplifies beautifully to X transpose times the probability difference vector, divided by sample size."
    },
    {
        "title": "Implementation Ecosystem",
        "kicker": "SCRATCH VS. LIBRARIES",
        "kind": "pointers",
        "bullets": [
            "Scratch Track: Exposes transposes, LU solvers, and gradient updates in pure Python",
            "statsmodels: Used for detailed inference summaries and hypothesis tests",
            "scikit-learn: High-performance training, cross-validation, and pipelines",
            "Verification: Always check custom solvers against statsmodels and scikit-learn",
        ],
        "narration": "Our curriculum maintains two tracks. The scratch track builds linear solvers and gradient descent from first principles in pure Python to expose the raw mechanics. The library track uses statsmodels for detailed statistical diagnostics and scikit-learn for high-performance predictive pipelines. We always verify our scratch solvers against these mature libraries."
    },
    {
        "title": "Chapter 1 Summary",
        "kicker": "KEY TAKEAWAYS",
        "kind": "bullets",
        "bullets": [
            "OLS minimizes residual sum of squares and projects y onto X",
            "Inference depends on strict exogeneity and spherical error properties",
            "Diagnostics identify heteroskedasticity, multicollinearity, and leverage",
            "Regularization (L1/L2) trades small bias for reduced variance",
            "Logistic regression models log-odds and is optimized via gradient descent",
        ],
        "narration": "Let us summarize Chapter One. Ordinary Least Squares projects y orthogonally onto the design space. Statistical inference is valid only when exogeneity and spherical error properties hold. Diagnostics are necessary to check for heteroskedasticity and collinearity. Regularizations introduce minor bias to stabilize variance. And logistic regression models binary probabilities, requiring gradient descent optimization."
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
    d.text((54, 34), slide.get("kicker", "CHAPTER 1"), font=font(18, bold=True), fill=CYAN)
    d.text((54, 65), slide["title"], font=font(40, bold=True), fill=WHITE)
    d.line((54, 122, 1226, 122), fill=GRID, width=2)
    d.text((54, 684), "ML ZERO-TO-RESEARCH CURRICULUM", font=font(14, bold=True), fill=MUTED)
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
        d.text((78, 178), "REGRESSION &", font=font(86, bold=True), fill=CYAN)
        d.text((78, 276), "CLASSIFICATION", font=font(57, bold=True), fill=WHITE)
        fit_text(d, slide["subtitle"], (82, 370, 760, 100), 28, MUTED)
        # Decorative visual motif: Regression line
        d.line((820, 420, 1180, 200), fill=MUTED, width=2)
        d.line((820, 310, 1180, 310), fill=GRID, width=1)
        d.line((900, 200, 900, 450), fill=GRID, width=1)
        d.ellipse((880, 280, 900, 300), fill=PINK)
        d.ellipse((980, 320, 1000, 340), fill=GREEN)
        d.ellipse((1080, 240, 1100, 260), fill=CYAN)
    elif kind == "bullets":
        draw_bullets(d, slide["bullets"], accent=GREEN if "takeaways" in slide["kicker"].lower() else CYAN)
    elif kind == "pointers":
        rounded(d, (72, 166, 560, 586), 20, PANEL)
        # Left side panel diagram
        d.text((105, 205), "OLS Projections / Violations", font=font(22, bold=True), fill=MUTED)
        d.line((120, 480, 500, 280), fill=GREEN, width=4)
        # points
        d.ellipse((150, 430, 170, 450), fill=PINK)
        d.ellipse((280, 330, 300, 350), fill=ORANGE)
        d.ellipse((420, 340, 440, 360), fill=CYAN)
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
    lines = ["# Chapter 1 Video Tutorial — Narration Script", ""]
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
