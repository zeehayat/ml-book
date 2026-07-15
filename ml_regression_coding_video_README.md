# ML Regression for Beginners - Coding Video

This package creates a beginner-focused ML regression video that emphasizes coding rather than math or statistics.

## Final video

- `ml_regression_coding_video/ML_Regression_for_Beginners_Coding_Tutorial.mp4`
- Runtime: about 5 minutes 11 seconds
- Format: 1280x720 H.264 video with AAC narration

## Companion files

- `ml_regression_coding_video/narration_script.md` - full voice-over script.
- `ml_regression_coding_video/chapter_timestamps.md` - chapter markers.
- `ml_regression_coding_video/lesson_plan.json` - structured slide plan.
- `ml_regression_coding_video/slides/` - rendered slide PNGs.
- `ml_regression_coding_video/audio/` - generated offline TTS narration.
- `ml_regression_coding_examples_pure_python.py` - pure-Python regression examples.
- `ml_regression_coding_examples_libraries.py` - pandas, NumPy, scikit-learn, matplotlib, and optional statsmodels examples.

## Rebuild

```bash
python3 build_ml_regression_coding_video.py
```

The builder uses Pillow for slides, FFmpeg for video encoding, and FFmpeg's offline `flite` text-to-speech filter for narration.

## Run the examples

```bash
python3 ml_regression_coding_examples_pure_python.py
python3 ml_regression_coding_examples_libraries.py
```

The library example saves `ml_regression_actual_vs_predicted.png`.

