"""
Build a "concept + exercise" PowerPoint presentation for Chapter 8
(Graph Spaces & Spectral Clustering: PageRank and Spectral Clustering).

Run:
    python3 build_chapter8_practice_presentation.py

Produces:
    Chapter8_Graphs_Practice_Presentation.pptx
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
        "title": "Chapter 8: Graph Spaces & Spectral Clustering",
        "subtitle": "One concept, one exercise per slide — practice as you go",
        "concept": [],
        "exercise": None,
        "notes": "Welcome slide. Not all data lives in a flat Euclidean space — this chapter treats data as a graph and asks two questions: which nodes matter most (PageRank), and which nodes belong together (spectral clustering)?",
    },
    {
        "title": "Concept: PageRank — The Random Surfer Model",
        "concept": [
            "Imagine a surfer randomly following links forever — PageRank is the long-run fraction of time spent on each page",
            "A page's rank is proportional to the sum of the ranks of pages linking to it, divided by their out-degree",
            "This is a fixed point: π = Mπ — PageRank is the dominant eigenvector of the link matrix",
        ],
        "exercise": "Page A is linked to by two pages: a high-authority page with few outgoing links, and a low-authority page with hundreds of outgoing links. Which incoming link contributes more to A's PageRank, and why?",
        "notes": "Answer: the high-authority page with FEW outgoing links contributes more — each page distributes its own rank evenly across its outgoing links, so a link from a page with few outgoing links carries more of that page's rank than a link from a page whose rank is spread across hundreds of links.",
    },
    {
        "title": "Concept: The Damping Factor and Teleportation",
        "concept": [
            "Pure link-following can get trapped in cycles or dead ends",
            "The Google Matrix adds teleportation: with probability (1-d), jump to a uniformly random page",
            "Standard damping factor d = 0.85 — guarantees a unique, well-defined stationary ranking (Perron-Frobenius theorem)",
        ],
        "exercise": "Why is the damping factor necessary — what could go wrong with PageRank on a real web graph if surfers ALWAYS followed links (d=1) and never teleported?",
        "notes": "Answer: without teleportation, a surfer could get trapped forever in a closed loop of pages with no outgoing links to the rest of the graph (or stuck at a dead-end 'sink' page with zero outgoing links), and the resulting stationary distribution might not exist or might not be unique. Teleportation guarantees the Perron-Frobenius conditions hold.",
    },
    {
        "title": "Concept: Dangling Nodes",
        "concept": [
            "A dangling node has NO outgoing links — its column in the link matrix is all zeros",
            "Without special handling, probability mass simply vanishes at every dangling node",
            "Fix: redistribute a dangling node's rank uniformly across ALL nodes at every iteration",
        ],
        "exercise": "After several PageRank iterations without dangling-node handling, you notice the rank values no longer sum to 1.0 (they're slowly shrinking). What is the most likely cause?",
        "notes": "Answer: dangling nodes are leaking probability mass out of the system — since their column has no outgoing edges to distribute rank to, that mass simply disappears each iteration instead of being redistributed, causing the total to shrink below 1.0 over time.",
    },
    {
        "title": "Concept: The Graph Laplacian",
        "concept": [
            "L = D − A (degree matrix minus adjacency matrix)",
            "Quadratic form: fᵀLf = (1/2)Σᵢⱼ Aᵢⱼ(fᵢ−fⱼ)² — measures how 'non-smooth' f is over the graph",
            "L is always positive semidefinite — this quadratic form can never be negative",
            "The MULTIPLICITY of the zero eigenvalue equals the number of connected components",
        ],
        "exercise": "A graph has 3 completely separate, disconnected clusters (no edges between them). How many zero eigenvalues will its Laplacian have?",
        "notes": "Answer: exactly 3 — one zero eigenvalue per connected component. This is a direct, checkable consequence of the theorem connecting zero-eigenvalue multiplicity to connected component count.",
    },
    {
        "title": "Concept: The Fiedler Vector and RatioCut",
        "concept": [
            "Finding the best way to CUT a graph into two balanced, weakly-connected pieces is NP-hard",
            "Relaxing the discrete cut problem to a continuous one gives a Rayleigh quotient minimization",
            "The solution is the Fiedler vector — the eigenvector for the SECOND-smallest Laplacian eigenvalue",
            "Sign of each entry in the Fiedler vector suggests which side of the cut that node belongs to",
        ],
        "exercise": "Why do we use the SECOND-smallest eigenvalue (not the smallest) of the Laplacian to find a meaningful graph cut?",
        "notes": "Answer: the smallest eigenvalue is always 0, with the constant (all-ones) eigenvector — a trivial, uninformative 'solution' that doesn't cut the graph at all. The second-smallest eigenvalue's eigenvector (the Fiedler vector) is the first one that actually varies across the graph in a way that reveals a meaningful partition.",
    },
    {
        "title": "Concept: Spectral Clustering — The Full Pipeline",
        "concept": [
            "1. Build a similarity graph (e.g. k-nearest-neighbors) from raw feature data",
            "2. Compute the (normalized) graph Laplacian",
            "3. Extract the bottom-k eigenvectors — this becomes a new, k-dimensional embedding",
            "4. Run ordinary k-means IN this new embedded space",
        ],
        "exercise": "Why does spectral clustering run k-means on the LAPLACIAN EIGENVECTORS rather than on the original raw features?",
        "notes": "Answer: the original features may not separate the clusters with simple round/convex boundaries (e.g. interlocking rings) — the spectral embedding transforms the data into a new space where the graph-based cluster structure becomes linearly/round separable, which is exactly the kind of structure k-means (which assumes round clusters) can find.",
    },
    {
        "title": "Concept: Choosing k via the Spectral Gap",
        "concept": [
            "Sort the Laplacian's eigenvalues: λ₁ ≤ λ₂ ≤ ... ≤ λₙ",
            "A large GAP between λₖ and λₖ₊₁ suggests k is a good number of clusters",
            "A vanishing gap suggests the data doesn't cleanly support that many clusters",
        ],
        "exercise": "The sorted eigenvalues are: 0, 0.01, 0.02, 0.45, 0.9, 1.3, .... Where is the largest spectral gap, and what value of k does it suggest?",
        "notes": "Answer: the largest gap is between the 3rd eigenvalue (0.02) and the 4th (0.45) — a jump of 0.43, far larger than any other consecutive gap. This suggests k=3 clusters (the first 3 near-zero eigenvalues correspond to 3 well-separated/near-disconnected components).",
    },
    {
        "title": "Concept: Common Mistake — Disconnected Graphs Break Spectral Clustering",
        "concept": [
            "If the input graph has c > 1 connected components, the bottom c eigenvalues are ALL exactly zero",
            "The 'Fiedler vector' becomes ambiguous — any vector in that zero-eigenspace is equally valid",
            "Always check the number of near-zero eigenvalues before trusting the clustering result",
        ],
        "exercise": "You run spectral clustering expecting k=2 clusters, but the graph actually has 2 completely disconnected components already. What goes wrong, and how would you detect it?",
        "notes": "Answer: the bottom eigenspace is degenerate (both bottom eigenvalues are exactly zero), so the 'Fiedler vector' the algorithm extracts is an arbitrary choice within that degenerate space rather than a meaningful signal — you'd detect this by counting near-zero eigenvalues before trusting the result; here it correctly finds 2 components, but for larger unintended disconnection this can silently produce meaningless splits within a component.",
    },
    {
        "title": "Mastery Check: Are You Ready to Move On?",
        "concept": [
            "If you can answer every question below confidently, you have mastered Chapter 8.",
        ],
        "exercise": None,
        "mastery_questions": [
            "Explain the random-surfer interpretation of PageRank and why it is a dominant-eigenvector problem.",
            "Explain why the damping factor and teleportation are necessary for PageRank to be well-defined.",
            "Explain what a dangling node is and how its rank mass must be handled.",
            "Write the Laplacian's quadratic form and explain what it measures about a function on a graph.",
            "Explain why the number of zero eigenvalues of the Laplacian equals the number of connected components.",
            "Explain why the Fiedler vector uses the SECOND-smallest eigenvalue, not the smallest.",
            "Walk through all four steps of the spectral clustering pipeline from memory.",
            "Explain how to choose k using the spectral gap, with a worked numeric example.",
            "Diagnose why spectral clustering gives an ambiguous or wrong result on a disconnected graph.",
            "Implement PageRank via power iteration on a small graph from scratch.",
        ],
        "notes": "This is a self-check slide. If any question causes hesitation, return to that concept's slide and its exercise before moving to Chapter 9.",
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

    out_path = "Chapter8_Graphs_Practice_Presentation.pptx"
    prs.save(out_path)
    print(f"Saved {out_path} with {total} slides.")


if __name__ == "__main__":
    build()
