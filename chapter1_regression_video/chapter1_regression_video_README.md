# Chapter 1 Regression Video Tutorial

This project generates a narrated slide tutorial from the OLS and Logistic Regression sections of the handbook (Chapter 1).

## Build

To compile the slides, generate the narration audios, and build the video:
```bash
python3 build_chapter1_regression_video.py
```

The script utilizes:
* **gTTS** (Google Text-to-Speech) for high-quality audio synthesis.
* **Pillow** (PIL) for drawing professional dark-themed slides.
* **FFmpeg** for segment encoding and final video concatenation.

## Output Structure

The results are generated in `chapter1_regression_video/`:
* `Chapter1_Linear_and_Logistic_Regression_Tutorial.mp4` — The final compiled narrated video.
* `narration_script.md` — The complete voice-over script used for synthesis.
* `chapter_timestamps.md` — Chapter navigation indexes and slide timestamps.
* `lesson_plan.json` — The structured JSON slide contents.
* `slides/` — Visual slide images (PNG format).
* `audio/` — Individual slide voice-over files (MP3 format).
