# Just Enough Linear Algebra for ML Video Tutorial

This project generates a narrated slide tutorial covering the load-bearing linear algebra concepts required for machine learning models (vectors, matrices, multiplication, transformations, invertibility, rank, eigenvectors, and SVD).

## Build

To compile the slides, generate the narration audios, and build the video:
```bash
python3 build_linear_algebra_video.py
```

The script utilizes:
* **gTTS** (Google Text-to-Speech) for high-quality audio synthesis.
* **Pillow** (PIL) for drawing professional dark-themed slides.
* **FFmpeg** for segment encoding and final video concatenation.

## Output Structure

The results are generated in `linear_algebra_video/`:
* `Just_Enough_Linear_Algebra_for_ML.mp4` — The final compiled narrated video.
* `narration_script.md` — The complete voice-over script used for synthesis.
* `chapter_timestamps.md` — Navigation indexes and slide timestamps.
* `lesson_plan.json` — The structured JSON slide contents.
* `slides/` — Visual slide images (PNG format).
* `audio/` — Individual slide voice-over files (MP3 format).
