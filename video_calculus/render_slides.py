"""Render each slide in script.py to a 1920x1080 PNG image."""
import textwrap
from PIL import Image, ImageDraw, ImageFont

from script import SLIDES

W, H = 1920, 1080
NAVY = (27, 38, 79)
TEAL = (13, 148, 136)
AMBER = (184, 106, 0)
LIGHT_BG = (247, 249, 252)
TEXT_DARK = (34, 34, 42)
WHITE = (255, 255, 255)

FONT_DIR = "/usr/share/fonts/truetype/dejavu/"
f_title = ImageFont.truetype(FONT_DIR + "DejaVuSans-Bold.ttf", 58)
f_title_small = ImageFont.truetype(FONT_DIR + "DejaVuSans-Bold.ttf", 46)
f_bullet = ImageFont.truetype(FONT_DIR + "DejaVuSans.ttf", 34)
f_subtitle = ImageFont.truetype(FONT_DIR + "DejaVuSans-Oblique.ttf", 34)
f_pagenum = ImageFont.truetype(FONT_DIR + "DejaVuSans.ttf", 22)


def wrap_text(text, font, max_width, draw):
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def render_slide(idx, total, slide, out_path):
    img = Image.new("RGB", (W, H), LIGHT_BG)
    draw = ImageDraw.Draw(img)

    # Header band
    header_h = 160
    draw.rectangle([0, 0, W, header_h], fill=NAVY)
    title = slide["title"]
    title_font = f_title if len(title) < 45 else f_title_small
    tw = draw.textlength(title, font=title_font)
    # If title too wide even for the small font, wrap it
    if tw > W - 100:
        lines = wrap_text(title, title_font, W - 100, draw)
        y = header_h / 2 - (len(lines) * 56) / 2
        for ln in lines:
            lw = draw.textlength(ln, font=title_font)
            draw.text(((W - lw) / 2, y), ln, font=title_font, fill=WHITE)
            y += 56
    else:
        draw.text(((W - tw) / 2, (header_h - 58) / 2 - 6), title, font=title_font, fill=WHITE)

    content_top = header_h + 60

    if slide["diagram"]:
        # Diagram on the right half, bullets on the left half
        diagram_path = f"diagrams/{slide['diagram']}.png"
        dimg = Image.open(diagram_path)
        max_dw, max_dh = 980, 760
        scale = min(max_dw / dimg.width, max_dh / dimg.height)
        dimg = dimg.resize((int(dimg.width * scale), int(dimg.height * scale)))
        dx = W - dimg.width - 90
        dy = content_top + (max_dh - dimg.height) // 2
        img.paste(dimg, (dx, dy))

        text_max_width = dx - 140
        y = content_top + 20
        for bullet in slide["bullets"]:
            lines = wrap_text("•  " + bullet, f_bullet, text_max_width, draw)
            for ln in lines:
                draw.text((90, y), ln, font=f_bullet, fill=TEXT_DARK)
                y += 46
            y += 20
    else:
        # Full-width bullets, vertically centered-ish
        text_max_width = W - 220
        lines_all = []
        for bullet in slide["bullets"]:
            wrapped = wrap_text("•  " + bullet, f_bullet, text_max_width, draw)
            lines_all.append(wrapped)

        total_lines = sum(len(w) for w in lines_all) + (len(lines_all) - 1)
        y = content_top + max(0, (H - content_top - 160 - total_lines * 50) / 2)
        for wrapped in lines_all:
            for ln in wrapped:
                draw.text((110, y), ln, font=f_bullet, fill=TEXT_DARK)
                y += 50
            y += 26

    # Footer accent line + page number
    draw.rectangle([0, H - 8, W, H], fill=TEAL)
    pg = f"{idx} / {total}"
    pw = draw.textlength(pg, font=f_pagenum)
    draw.text((W - pw - 40, H - 46), pg, font=f_pagenum, fill=(120, 120, 120))

    img.save(out_path)


def main():
    total = len(SLIDES)
    for i, slide in enumerate(SLIDES, start=1):
        out_path = f"slides/slide_{i:02d}.png"
        render_slide(i, total, slide, out_path)
        print(f"rendered {out_path}")


if __name__ == "__main__":
    main()
