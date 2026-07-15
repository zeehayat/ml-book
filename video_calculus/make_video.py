"""
Assemble slides/*.png + audio/*.mp3 into per-slide video clips, then
concatenate into the final MP4.
"""
import subprocess
import json
from script import SLIDES

FPS = 24
PAD_SECONDS = 0.6  # hold the slide briefly after narration ends


def get_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", path],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


def make_clip(i):
    img = f"slides/slide_{i:02d}.png"
    audio = f"audio/narration_{i:02d}.mp3"
    out = f"clips/clip_{i:02d}.mp4"
    duration = get_duration(audio) + PAD_SECONDS

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", img,
        "-i", audio,
        "-c:v", "libx264", "-tune", "stillimage",
        "-c:a", "aac", "-b:a", "160k",
        "-pix_fmt", "yuv420p",
        "-vf", f"fps={FPS}",
        "-t", f"{duration:.3f}",
        "-shortest",
        out,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"built {out} ({duration:.1f}s)")


def concatenate(total):
    list_path = "clips/concat_list.txt"
    with open(list_path, "w") as f:
        for i in range(1, total + 1):
            f.write(f"file 'clip_{i:02d}.mp4'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", "concat_list.txt",
        "-c", "copy",
        "Just_Enough_Calculus_for_ML.mp4",
    ]
    subprocess.run(cmd, check=True, capture_output=True, cwd="clips")
    # move final file up
    subprocess.run(["mv", "clips/Just_Enough_Calculus_for_ML.mp4", "."], check=True)


def main():
    total = len(SLIDES)
    for i in range(1, total + 1):
        make_clip(i)
    concatenate(total)
    print("Final video: Just_Enough_Calculus_for_ML.mp4")


if __name__ == "__main__":
    main()
