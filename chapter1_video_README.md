# Chapter 1 Video Tutorial

This project generates a narrated slide tutorial from
`source_material/ch01_introduction/chapter_01.md`.

## Build

```powershell
.\.venv\Scripts\python.exe .\build_chapter1_video.py
```

The build uses Windows SAPI for offline narration and MoviePy/FFmpeg for video
encoding. Output is written to `chapter1_video/`:

- `Chapter1_Tensors_and_Compute_Hardware_Tutorial.mp4` — final narrated video
- `narration_script.md` — editable voice-over script
- `chapter_timestamps.md` — navigation index for the completed lesson
- `lesson_plan.json` — structured slide content
- `slides/` — individual 1280×720 slide images
- `audio/` — per-slide narration WAV files

Edit the `SLIDES` list in `build_chapter1_video.py` to change content or timing,
then rerun the build.
