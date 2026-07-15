# Video Tutorial Project Status

Last updated: 2026-07-15

This file tracks the status of the textbook video tutorials. Read it before creating or modifying any chapter video.

---

## Completed work

### Chapter 1 — Linear Regression & Logistic Models

- **Source chapter:** OLS & Logistic Regression sections of the compiled handbook.
- **Final video:** `chapter1_regression_video/Chapter1_Linear_and_Logistic_Regression_Tutorial.mp4`
- **Runtime:** approximately 06:14
- **Format:** 1280×720, H.264 video, AAC mono narration, 12 fps
- **Lesson structure:** 12 narrated sections
- **Renderer:** `build_chapter1_regression_video.py`
- **Narration method:** Automated Google Text-to-Speech (gTTS)
- **Documentation:** `chapter1_regression_video/chapter1_regression_video_README.md`
- **Narration text:** `chapter1_regression_video/narration_script.md`
- **Chapter index:** `chapter1_regression_video/chapter_timestamps.md`
- **Structured content:** `chapter1_regression_video/lesson_plan.json`
- **Assets:** `chapter1_regression_video/slides/`, `chapter1_regression_video/audio/` (MP3), and `chapter1_regression_video/segments/`

The tutorial covers simple regression slope/intercept, matrix form multiple regression, OLS geometric orthogonal projections, Gauss-Markov theorem (BLUE), inference (standard errors, t/F tests), diagnostics (heteroskedasticity, VIF, Cook's distance), regularization (Ridge, Lasso, Elastic Net), logistic regression probability models, log-likelihood cross-entropy loss, gradient descent updates, and scratch vs. library tracks.

Build command:
```bash
python3 build_chapter1_regression_video.py
```

---

### Chapter 3 — The Anatomy of a Tensor & Compute Hardware
*(Formerly Chapter 1 in the video project)*

- **Source chapter:** `source_material/ch01_introduction/chapter_01.md`
- **Final video:** `chapter1_video/Chapter1_Tensors_and_Compute_Hardware_Tutorial.mp4`
- **Runtime:** approximately 11:35
- **Format:** 1280×720, H.264 video, AAC mono narration, 12 fps
- **Lesson structure:** 19 narrated sections
- **Renderer:** `build_chapter1_video.py`
- **Narration helper:** `render_chapter1_narration.ps1`
- **Documentation:** `chapter1_video_README.md`
- **Narration text:** `chapter1_video/narration_script.md`
- **Chapter index:** `chapter1_video/chapter_timestamps.md`
- **Structured content:** `chapter1_video/lesson_plan.json`
- **Assets:** `chapter1_video/slides/`, `chapter1_video/audio/`, and `chapter1_video/segments/`

The tutorial covers the flat-buffer tensor model, shape and stride arithmetic, C- and F-contiguous layouts, slicing, transpose, reshape, negative strides, CPU caches, GPU memory, arithmetic intensity, implementation, and common mistakes.

Rebuild command:
```bash
python3 build_chapter1_video.py
```

---

### Chapter 4 — The Core Optimization Engine (Automatic Differentiation)
*(Formerly Chapter 2 in the video project)*

- **Source chapter:** `source_material/chapter02_autograde/chapter_02.md`
- **Final video:** `chapter2_video/Chapter2_Automatic_Differentiation_Tutorial.mp4`
- **Runtime:** approximately 12:16
- **Format:** 1280×720, H.264 video, AAC mono narration, 12 fps
- **Lesson structure:** 21 narrated sections
- **Renderer:** `build_chapter2_video.py`
- **Narration helper:** `render_chapter2_narration.ps1`
- **Documentation:** `chapter2_video_README.md`
- **Narration text:** `chapter2_video/narration_script.md`
- **Chapter index:** `chapter2_video/chapter_timestamps.md`
- **Structured content:** `chapter2_video/lesson_plan.json`
- **Assets:** `chapter2_video/slides/`, `chapter2_video/audio/`, and `chapter2_video/segments/`

The tutorial covers differentiation at scale, numerical versus symbolic versus automatic differentiation, computational graphs, the tape, a complete worked forward/backward example, shared-node accumulation, the multivariable chain rule, local backward rules, topological sorting, forward versus reverse mode, the scalar `Value` engine, tensor broadcasting, activation memory, checkpointing, production autograd, common mistakes, and XOR training.

Rebuild command:
```bash
python3 build_chapter2_video.py
```

---

## Rendering environment

The active environment contains the packages used by the video builders:
- Pillow
- gTTS (Google Text-to-Speech)
- imageio and FFmpeg
- NumPy

The current builders use Pillow for slide images, gTTS for high-quality voice-over synthesis, and the system FFmpeg binary for final video concatenation. SAPI is not required in the Linux environment.
