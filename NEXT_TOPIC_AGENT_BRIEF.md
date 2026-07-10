# Agent Brief: How to Produce the Next Guide in This Curriculum

This file is a self-contained handoff. A fresh agent with no memory of any prior
conversation should be able to read only this file (plus the existing guides it
references) and correctly produce the next book in the series. If you are that
agent: read this whole file before writing anything.

## 1. What this curriculum is

A step-by-step, **zero-to-research-level machine learning** curriculum, delivered as
a sequence of standalone "guides" (one per topic), each written as both a Markdown
source file and a matching `.docx` file. The reader starts from **zero** — no prior
math, statistics, or Python assumed at the start of guide 1 — and each subsequent
guide assumes only what earlier guides in the sequence covered, never more. The
guides are designed to be uploaded to **NotebookLM** (or a similar AI study tool) as
source material, so the AI tutor can teach the reader to research level using the
guide's own structure, exercises, and prompts.

**Guides completed so far, in order:**

1. `Regression_Mathematical_and_Python_Guide.md` / `.docx` — OLS, multiple regression,
   inference, diagnostics, regularization (ridge/lasso/elastic net), and logistic
   regression, from scratch in pure Python and then with NumPy/pandas/statsmodels/
   scikit-learn.
2. `Generalized_Linear_Models_Mathematical_and_Python_Guide.md` / `.docx` — the
   exponential family, link functions, IRLS, deviance, Poisson/binomial/gamma
   regression, overdispersion, negative binomial regression, and regularized GLMs,
   from scratch and with statsmodels/scikit-learn. Explicitly built as the direct
   generalization of guide 1 (OLS and logistic regression are shown to be two GLMs
   among several).

Both existing guides live in the same directory as this brief. **Read the most
recently completed guide in full before writing the next one** — it is your style
guide, your depth calibration, and the source of what the reader already knows.

## 2. Choosing the next topic — confirm before committing

Do not silently pick a topic and start writing a full guide. Writing one of these
guides is a large amount of work (the GLM guide alone is roughly 16,000 words plus
three fully executable, tested Python modules), and picking the wrong topic wastes
nearly all of it. **Ask the user to confirm the next topic before you start**, the
same way this brief's own predecessor guide was confirmed: present 3–4 reasonable
candidate topics with a one-line tradeoff each, recommend one, and let the user
choose or redirect.

The most recently completed guide's own "Further study" chapter lists the natural
next candidates it identified while it was being written — read that chapter first;
it is the best-informed source of what should come next, because it was written with
full knowledge of what the guide itself just covered. As of this brief, the
Generalized Linear Models guide's Chapter 44 names **generalized additive models
(GAMs)** and **mixed-effects/multilevel GLMs** as the two most direct next steps
(both keep the exponential-family and link-function machinery intact and change only
the structure or independence assumptions of the linear predictor), with GAMs edging
out as the smaller conceptual jump. Treat that as the default recommendation, not a
final decision.

## 3. Required document structure

Every guide in this series follows the same template, chapter-numbered continuously
within the guide (starting at 1), organized into four parts plus appendices. Copy
this structure exactly; only the content changes per topic.

```
# <Topic>: A Mathematical and Python Guide
*<subtitle: what it extends / what it's for>*
Prepared as a self-study and AI-tutor source. <Nth> guide in the Zero-to-Research-Level
Machine Learning sequence, following <list prior guide(s) by name>.

## How to use this guide
  - Purpose (self-contained, NotebookLM-ready, assumes prior guides' content as known)
  - Meaning of "pure Python" (standard library only, restated per guide)
  - Why this topic, right after the previous one (explicit bridge/motivation)
  - Scope (bullet list)
  - Contents at a glance (Part I-IV + Appendices, one line each)

# Part I. Mathematical Foundations
  Chapters covering: intuition for why this topic is needed beyond what came before;
  the core mathematical objects; the estimation/fitting principle (closed form,
  iterative, or otherwise) DERIVED, not asserted; model diagnostics; model
  comparison/selection; regularization if applicable; a chapter connecting back to
  causal-inference cautions established in guide 1.

# Part II. <Topic> from Scratch in Pure Python
  Chapters explaining, concisely, what the from-scratch implementation contains and
  WHY each piece is needed (do not paste the full code into the chapter text --
  that belongs in Appendix A only). Explicitly note what is reused unchanged from
  earlier guides' from-scratch code (e.g. linear algebra core, solvers) versus what
  is new for this topic.

# Part III. <Topic> with Python Libraries
  Chapters covering: which library does what (mirrors the division of labor already
  established in guide 1: statsmodels/similar for inference, scikit-learn for
  prediction pipelines); real, runnable code snippets per library feature; a
  diagnostic-plots chapter; an end-to-end checklist chapter; a common-mistakes table.

# Part IV. Exercises, Reference, and Further Study
  37+. Conceptual questions
  38+. Derivation exercises
  39+. Pure-Python coding exercises
  40+. Library exercises
  41+. Suggested learning sequence (staged, "Stage 1" through "Stage N")
  42+. Prompts for an AI tutor or NotebookLM (concept/math/code/interpretation/
       diagnostic/project-supervision prompts, each a literal quoted prompt string)
  43+. Glossary -- see Section 4 below, this is a hard requirement
  44+. Further study -- bullet list of natural next topics, written with the
       benefit of having just finished this guide (this is what the NEXT brief
       will read to pick the topic after this one)
  45+. Suggested research papers -- see Section 4 below, hard requirement
  46+. How to read a research paper -- see Section 4 below, hard requirement

# Appendix A. Full Pure-Python Implementation
# Appendix B. Full Library-Based Demonstration
# Appendix C. Minimal Verification Tests
# Final perspective
```

## 4. Non-negotiable content requirements

These were explicit, repeated instructions from the human running this curriculum.
Every guide must include all of them, in this depth, not as a token gesture:

- **A "From Scratch" section, then a "Math" section, then a "Libraries" section, in
  that order** (Part II is pure standard-library Python only, no exceptions — verify
  this by grepping the from-scratch appendix for `import` statements and confirming
  only `math`, `random`, `dataclasses`, and `typing` appear).
- **A glossary where every term gets a comprehensive, multi-sentence definition** —
  not a one-line dictionary gloss. The pattern used in both existing guides: sentence
  one defines the term precisely; sentence two explains why it matters or what
  breaks without it; sentence three (when useful) connects it to another glossary
  term or an earlier guide's concept. Aim for 25–40 terms.
- **A "Suggested research papers" section with real, correctly cited papers** —
  actual authors, years, journal/venue names, matched explicitly to the chapter(s)
  they support. Do not invent citations; only include papers you are confident
  actually exist with those details. Prioritize the paper that originated each
  major method covered in the guide.
- **A "How to read a research paper" section**, written fresh for the guide's own
  subject matter (do not just copy the previous guide's version verbatim — add at
  least one subsection specific to what is distinctive about reading papers in this
  particular topic, the way the GLM guide added a subsection on separating "new
  model" claims from "new estimation method" claims).
- **A "Further study" section** naming the natural next topics, to feed the next
  agent's Section 2 decision above.
- Every chapter should, where a direct analogue exists in a prior guide, **say so
  explicitly** ("this is the GLM generalization of Chapter X of the Regression
  guide") rather than re-deriving from zero. This is what makes the series compound
  rather than repeat itself.

## 5. Code must actually run — this is not optional

Both existing guides' appendices were written as standalone scripts, executed in a
real Python interpreter, and debugged before being embedded into the guide. Do the
same:

1. Write the from-scratch module (Appendix A content) as a standalone `.py` file
   first. Run it. Fix bugs. A real bug was caught and fixed this way while writing
   the GLM guide (IRLS failing to converge on quasi-separable data) — that fix
   became a genuinely useful "Numerical safeguards" warning in the guide text
   (Chapter 18.3 of the GLM guide is a good model for this: a caution grounded in
   an actual failure you hit, not a generic warning).
2. Write a minimal test script (Appendix C content) that imports from the module
   above and asserts on real numeric results (recover known coefficients on
   noiseless synthetic data close to the true values, confirm probabilities/
   residuals/etc. are in valid ranges, confirm a special case reduces to a simpler
   already-verified model). Run it; it must pass before you embed it.
3. Write the library-based script (Appendix B content) as a standalone `.py` file.
   Check whether `numpy`, `pandas`, `statsmodels`, and `scikit-learn` (and anything
   else you need, e.g. `scipy`, `matplotlib`) are already installed
   (`python -c "import numpy, pandas, statsmodels, sklearn"`). If not, install them
   (`pip install numpy pandas statsmodels scikit-learn scipy matplotlib`) — this
   works fine in this environment and is worth the one-time cost. Run the script.
   Fix any deprecation errors, API signature mismatches, or convergence warnings
   before embedding it.
4. Only after all three scripts run cleanly, embed their final, verified contents
   into the guide's Markdown under the three appendix headings.

Do not write appendix code directly into the Markdown and assume it is correct
without running it. Both appendices in the GLM guide needed at least one real fix
after the first run (a convergence warning in `fit_regularized`, a separation
failure in a hand-written test case) — expect the same here.

## 6. Producing the `.docx` file

This environment does **not** have `pandoc`, LibreOffice/`soffice`, or `python-docx`
pre-installed, so the normal docx-skill workflow (unzip an existing docx and edit
`document.xml` directly) only applies when *updating* an existing docx — it does not
help you *create* a new one from Markdown. `node` and `npm` **are** available.
The approach that worked for the GLM guide:

1. `npm install docx` in a scratch working directory.
2. Write a small Node.js script that reads the finished Markdown file line by line
   and converts it to a `docx` document using the `docx` npm package: `#`/`##`/`###`
   → Title/Heading1/Heading2/Heading3 (first `#` line only is the Title; subsequent
   single `#` lines are Heading1 for Part/Appendix headers); fenced code blocks
   (` ``` `) → one monospace `Paragraph` per source line (never collapse a code
   block into a single paragraph with embedded `\n` — that breaks readability, and
   is a mistake present in the very first Regression guide's docx that later needed
   correcting); Markdown tables (`| ... |` with a `---` separator row) → real `Table`
   objects with `columnWidths` on the table and `width` on every cell, both in DXA,
   never `PERCENTAGE`; `**bold**` and `` `inline code` `` → inline `TextRun`
   formatting; `- ` bullets and `1. ` numbered lists → real `numbering` config
   (never insert a literal bullet character); a leading italic subtitle line under
   the title → a styled subtitle paragraph.
3. Run the script, then sanity-check the output before considering it done:
   `python -c "import zipfile; z = zipfile.ZipFile('out.docx'); print(z.testzip())"`
   should print `None`; parse `word/document.xml` with `xml.dom.minidom.parseString`
   to confirm well-formedness; and extract all `Title`/`Heading1`/`Heading2`/
   `Heading3` paragraphs back out and confirm the chapter list matches the Markdown
   source's headings exactly (same count, same order, same numbering).
4. There is no `soffice`/`pdftoppm` available in this environment to visually render
   and screenshot the result, so structural verification (steps above) is the
   substitute for a visual check — do not skip it.

## 7. File naming and location

Place both files in the same directory as this brief and the existing guides
(alongside `Regression_Mathematical_and_Python_Guide.docx` and
`Generalized_Linear_Models_Mathematical_and_Python_Guide.docx`), named
`<Topic_With_Underscores>_Mathematical_and_Python_Guide.md` and the matching
`.docx`. Keep the Markdown file even after producing the docx — it is the editable
source of truth and is what future agents (including whoever writes the guide after
this one) will read first.

## 8. After finishing the next guide

Update this brief's Section 1 ("Guides completed so far") to add the new guide, and
update Section 2's default recommendation using the new guide's own "Further study"
chapter, exactly as this version of the brief was informed by the GLM guide. This
keeps the brief self-renewing across the whole curriculum rather than going stale
after one use.
