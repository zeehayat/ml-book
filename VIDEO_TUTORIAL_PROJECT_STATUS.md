# Video Tutorial Project Status

Last updated: 2026-07-13

This file is the resume point for the textbook video-tutorial work. Read it
before creating or modifying another chapter video.

## Completed work

### Chapter 1 — The Anatomy of a Tensor & Compute Hardware

- Source chapter: `source_material/ch01_introduction/chapter_01.md`
- Final video: `chapter1_video/Chapter1_Tensors_and_Compute_Hardware_Tutorial.mp4`
- Runtime: approximately 11:35
- Format: 1280×720, H.264 video, AAC mono narration, 12 fps
- Lesson structure: 19 narrated sections
- Renderer: `build_chapter1_video.py`
- Narration helper: `render_chapter1_narration.ps1`
- Documentation: `chapter1_video_README.md`
- Narration text: `chapter1_video/narration_script.md`
- Chapter index: `chapter1_video/chapter_timestamps.md`
- Structured content: `chapter1_video/lesson_plan.json`
- Editable assets: `chapter1_video/slides/`, `chapter1_video/audio/`, and
  `chapter1_video/segments/`

The tutorial covers the flat-buffer tensor model, shape and stride arithmetic,
C- and F-contiguous layouts, slicing, transpose, reshape, negative strides,
CPU caches, GPU memory, arithmetic intensity, implementation, and common
mistakes.

Build with automated narration:

```powershell
.\.venv\Scripts\python.exe .\build_chapter1_video.py
```

Rebuild after replacing the audio with a human voice:

```powershell
.\.venv\Scripts\python.exe -c "import build_chapter1_video as b; b.build_video(); b.write_chapter_timestamps()"
```

Human recordings should replace `chapter1_video/audio/01.wav` through
`19.wav`. Recommended format: mono, 44.1 kHz, 16-bit PCM WAV.

### Chapter 2 — The Core Optimization Engine (Automatic Differentiation)

- Source chapter: `source_material/chapter02_autograde/chapter_02.md`
- Final video: `chapter2_video/Chapter2_Automatic_Differentiation_Tutorial.mp4`
- Runtime: approximately 12:16
- Format: 1280×720, H.264 video, AAC mono narration, 12 fps
- Lesson structure: 21 narrated sections
- Renderer: `build_chapter2_video.py`
- Narration helper: `render_chapter2_narration.ps1`
- Documentation: `chapter2_video_README.md`
- Narration text: `chapter2_video/narration_script.md`
- Chapter index: `chapter2_video/chapter_timestamps.md`
- Structured content: `chapter2_video/lesson_plan.json`
- Editable assets: `chapter2_video/slides/`, `chapter2_video/audio/`, and
  `chapter2_video/segments/`

The tutorial covers differentiation at scale, numerical versus symbolic versus
automatic differentiation, computational graphs, the tape, a complete worked
forward/backward example, shared-node accumulation, the multivariable chain
rule, local backward rules, topological sorting, forward versus reverse mode,
the scalar `Value` engine, tensor broadcasting, activation memory,
checkpointing, production autograd, common mistakes, and XOR training.

Build with automated narration:

```powershell
.\.venv\Scripts\python.exe .\build_chapter2_video.py
```

Rebuild after replacing the audio with a human voice:

```powershell
.\.venv\Scripts\python.exe -c "import build_chapter2_video as b; b.build_video(); b.write_timestamps()"
```

Human recordings should replace `chapter2_video/audio/01.wav` through
`21.wav`. Recommended format: mono, 44.1 kHz, 16-bit PCM WAV.

## Rendering environment

The repository virtual environment contains the packages used by the video
builders:

- Pillow
- MoviePy
- imageio-ffmpeg
- NumPy and MoviePy's supporting packages

The current builders use Pillow for slide images, Windows SAPI for offline
automated narration, and the FFmpeg binary supplied by `imageio-ffmpeg` for
encoding. Microsoft PowerPoint is not available in this environment.

The Chapter 2 renderer imports shared drawing helpers and colors from
`build_chapter1_video.py`, so retain that file even if only Chapter 2 is being
rebuilt.

## Validation already performed

Both completed MP4 files were checked for:

- H.264 video and AAC audio streams
- 1280×720 resolution
- 12 fps playback
- Successful full-file FFmpeg decoding
- Legibility of sampled title, diagram, formula, code, and summary slides

## Next work

No Chapter 3 video has been created yet. The likely next source is:

`source_material/chapter03_covex_optimization/chapter03_Convex_Optimization.md`

Before building it:

1. Inspect all Chapter 3 headings and identify its core teaching progression.
2. Create a dedicated `build_chapter3_video.py`; reuse the established visual
   system but design chapter-specific diagrams.
3. Target roughly 18–22 narrated sections and a 10–15 minute runtime unless the
   user requests a different depth.
4. Generate the same output package: MP4, narration script, timestamps,
   `lesson_plan.json`, slides, WAV recordings, encoded segments, README, and a
   human-voice rebuild command.
5. Validate media streams, full decoding, duration, and representative frames.
6. Update this status file after completing or materially changing any video.

## Important workflow notes

- Do not overwrite human narration by running a complete automated build after
  the user replaces the WAV files. Invoke only `build_video()` and the relevant
  timestamp function.
- Slide duration automatically follows each WAV recording, with a short padded
  transition at the end.
- Preserve unrelated working-tree changes. At the time of this work, unrelated
  modifications already existed in `notes_server.py`, `regression_guide.html`,
  and `regression_notes.sqlite3`.
