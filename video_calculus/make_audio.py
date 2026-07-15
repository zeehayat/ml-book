"""Generate narration audio (gTTS) for each slide."""
import time
from gtts import gTTS
from script import SLIDES


def main():
    for i, slide in enumerate(SLIDES, start=1):
        out_path = f"audio/narration_{i:02d}.mp3"
        text = slide["narration"]
        for attempt in range(3):
            try:
                tts = gTTS(text, lang="en", tld="co.uk", slow=False)
                tts.save(out_path)
                print(f"saved {out_path} ({len(text)} chars)")
                break
            except Exception as e:
                print(f"attempt {attempt+1} failed for slide {i}: {e}")
                time.sleep(2)
        else:
            raise RuntimeError(f"failed to generate audio for slide {i}")


if __name__ == "__main__":
    main()
