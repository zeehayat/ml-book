"""
Build a "concept + exercise" PowerPoint presentation for Chapter 7
(Dimensionality Reduction & Latent Spaces: PCA and SVD).

Run:
    python3 build_chapter7_practice_presentation.py

Produces:
    Chapter7_PCA_SVD_Practice_Presentation.pptx
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
        "title": "Chapter 7: Dimensionality Reduction & Latent Spaces",
        "subtitle": "One concept, one exercise per slide — practice as you go",
        "concept": [],
        "exercise": None,
        "notes": "Welcome slide. PCA and the SVD are two views of the same underlying idea: finding the directions that capture the most structure in high-dimensional data.",
    },
    {
        "title": "Concept: What PCA Is Trying to Do",
        "concept": [
            "Find the direction in feature space along which the data has MAXIMUM variance",
            "That direction is the first principal component",
            "Subsequent components are the next-best directions, each ORTHOGONAL to all previous ones",
            "Projecting onto the top-k components compresses D features down to k with minimal information loss",
        ],
        "exercise": "A dataset has 1000 features, but the first 2 principal components explain 95% of the total variance. What does this suggest about the data's true underlying dimensionality?",
        "notes": "Answer: the data likely lives close to a 2-dimensional structure embedded in the 1000-dimensional space — most of the 1000 features are redundant or highly correlated with each other, and 2 well-chosen directions capture nearly all the meaningful variation.",
    },
    {
        "title": "Concept: The Covariance Matrix and Its Eigenvectors",
        "concept": [
            "Center the data first (subtract the mean from every feature)",
            "The covariance matrix C = (1/(N-1)) X̃ᵀX̃ captures how features vary together",
            "Principal components are exactly the EIGENVECTORS of C",
            "Each eigenvector's eigenvalue equals the variance captured along that direction",
        ],
        "exercise": "Why must the data be mean-centered before computing the covariance matrix for PCA?",
        "notes": "Answer: without centering, the covariance-like matrix would capture variation around the ORIGIN rather than around the data's own mean — the resulting 'principal directions' would be dominated by how far the data is from zero, not by the data's actual internal spread/structure.",
    },
    {
        "title": "Concept: Power Iteration — Finding the Dominant Eigenvector",
        "concept": [
            "Repeatedly apply v ← Cv / ||Cv|| — this converges to the eigenvector with the LARGEST eigenvalue",
            "Convergence rate depends on |λ₂/λ₁| — well-separated top eigenvalues converge fast",
            "After finding one component, DEFLATE it out to find the next: C' = C − λ₁v₁v₁ᵀ",
        ],
        "exercise": "A covariance matrix has top two eigenvalues λ1=10 and λ2=9.9 (very close together). Would you expect power iteration to converge quickly or slowly to the first principal component, and why?",
        "notes": "Answer: slowly — the convergence rate depends on the ratio |λ2/λ1| = 9.9/10 = 0.99, which is very close to 1, meaning each iteration barely amplifies the dominant direction relative to the second one. Nearly-equal top eigenvalues make the top component unstable and slow to isolate.",
    },
    {
        "title": "Concept: The SVD-Covariance Duality",
        "concept": [
            "Economy SVD: X̃ = U·Σ·Vᵀ (mean-centered data)",
            "The columns of V are EXACTLY the eigenvectors of the covariance matrix — no eigensolver needed",
            "The scores (projections) satisfy Z = X̃V = UΣ — read directly off the SVD factors",
            "This is why production PCA implementations use SVD rather than explicitly forming the covariance matrix",
        ],
        "exercise": "Forming the D×D covariance matrix costs O(ND²) and can be numerically unstable for large D. Why does computing the SVD directly on the N×D data matrix avoid this problem?",
        "notes": "Answer: the SVD operates directly on the (typically much smaller in one dimension) data matrix without ever forming or squaring it into a D×D covariance matrix — avoiding both the O(ND^2) cost of forming C and the numerical instability of squaring the condition number that comes from explicitly computing X^T X.",
    },
    {
        "title": "Concept: The Eckart-Young Theorem — Optimal Low-Rank Approximation",
        "concept": [
            "Truncating the SVD to the top k singular values gives the BEST possible rank-k approximation",
            "\"Best\" means minimal reconstruction error, in both Frobenius and spectral norm",
            "Fraction of variance explained by the top k components: Σᵢ₌₁ᵏ σᵢ² / Σⱼ σⱼ²",
        ],
        "exercise": "A matrix has singular values σ1=5, σ2=3, σ3=1. What fraction of the squared Frobenius norm does the rank-1 approximation (using only σ1) capture?",
        "notes": "Answer: total = 5^2+3^2+1^2 = 25+9+1 = 35. Rank-1 captures 25/35 ≈ 71.4% of the squared Frobenius norm.",
    },
    {
        "title": "Concept: Condition Number and Numerical Stability",
        "concept": [
            "Forming X̃ᵀX̃ SQUARES the condition number: κ(C) = κ(X̃)²",
            "A moderately ill-conditioned X̃ can become severely ill-conditioned once squared",
            "This costs real floating-point precision — computing the SVD directly avoids the squaring",
        ],
        "exercise": "A data matrix has condition number κ(X)=1000. What is the condition number of its covariance matrix, and roughly how many decimal digits of precision are lost by forming it explicitly (in float64, machine epsilon ≈ 1e-16)?",
        "notes": "Answer: κ(C) = 1000² = 1,000,000 = 10^6. Roughly 6 extra decimal digits of precision are lost (log10 of the condition number) compared to working with X directly via the SVD — a real, measurable numerical cost.",
    },
    {
        "title": "Concept: Randomized SVD for Large-Scale Data",
        "concept": [
            "Exact SVD costs O(N·D²) — impractical for very large D",
            "Randomized SVD projects onto a random low-dimensional sketch first, then computes a small SVD",
            "Cost drops to about O(N·D·k) for the top k components — dramatically cheaper when k ≪ D",
        ],
        "exercise": "You need the top 50 components of a matrix with D=10,000 features and N=1,000 rows. Roughly how much cheaper is randomized SVD (O(N·D·k)) than the exact SVD (O(N·D²)) here?",
        "notes": "Answer: exact cost ~ N*D^2 = 1000 * 10000^2 = 10^11. Randomized cost ~ N*D*k = 1000*10000*50 = 5*10^8. That's roughly a 200x reduction in operation count — exactly why randomized SVD is the practical choice when only a small number of components are actually needed.",
    },
    {
        "title": "Concept: Mistake — Forgetting to Center the Data",
        "concept": [
            "Running PCA on raw, uncentered data is a common and completely silent bug",
            "The resulting \"components\" are contaminated by the data's distance from the origin",
            "Always subtract the per-feature mean before computing covariance or running SVD for PCA",
        ],
        "exercise": "A colleague runs PCA on raw sensor readings that all happen to be large positive numbers (e.g., 1000-1010) and gets a strange first component pointing almost exactly along the direction of the sensors' average reading. What did they forget to do?",
        "notes": "Answer: mean-centering. Without it, the first 'principal component' is dominated by the large common offset (the mean itself) rather than the actual pattern of variation around that mean — a classic silent PCA bug.",
    },
    {
        "title": "Mastery Check: Are You Ready to Move On?",
        "concept": [
            "If you can answer every question below confidently, you have mastered Chapter 7.",
        ],
        "exercise": None,
        "mastery_questions": [
            "Explain what a principal component is and why each one must be orthogonal to the previous ones.",
            "Explain why PCA requires mean-centering the data first.",
            "Explain why the principal components are the eigenvectors of the covariance matrix.",
            "Explain how power iteration finds the dominant eigenvector and why the eigenvalue ratio controls its speed.",
            "State the SVD-covariance duality and explain why production PCA uses SVD instead of an explicit covariance matrix.",
            "State the Eckart-Young theorem and compute a rank-k approximation's captured variance fraction by hand.",
            "Explain why forming X^T X squares the condition number, and why this matters numerically.",
            "Explain when and why randomized SVD is preferred over the exact SVD.",
            "Diagnose a PCA result that looks wrong because the data was never mean-centered.",
            "Implement PCA from scratch via power iteration and verify it against a library's SVD-based PCA.",
        ],
        "notes": "This is a self-check slide. If any question causes hesitation, return to that concept's slide and its exercise before moving to Chapter 8.",
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

    out_path = "Chapter7_PCA_SVD_Practice_Presentation.pptx"
    prs.save(out_path)
    print(f"Saved {out_path} with {total} slides.")


if __name__ == "__main__":
    build()
