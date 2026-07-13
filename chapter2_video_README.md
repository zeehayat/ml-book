# Chapter 2 Video Tutorial

This project renders a narrated tutorial based on
`source_material/chapter02_autograde/chapter_02.md`.

## Build with automated narration

```powershell
.\.venv\Scripts\python.exe .\build_chapter2_video.py
```

## Rebuild with your own narration

Record one WAV file per slide, named `01.wav` through `21.wav`, and place the
files in `chapter2_video/audio/`. Recommended format: mono, 44.1 kHz, 16-bit
PCM. Then run:

```powershell
.\.venv\Scripts\python.exe -c "import build_chapter2_video as b; b.build_video(); b.write_timestamps()"
```

The output folder contains the MP4, editable narration script, chapter
timestamps, structured lesson plan, slides, recordings, and encoded segments.
