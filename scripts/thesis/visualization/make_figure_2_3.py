#!/usr/bin/env python3
"""
Composes Figure 2.3 (continuum solvation) from two source panels.

  panel a : cavity construction schematic (VdW / SAS / SES)
  panel b : cavity surface around a molecular solute

Usage:  python3 make_figure_2_3.py <panel_a.png> <panel_b.png> [outdir]
"""

import sys
import os
from PIL import Image, ImageDraw, ImageFont

PANEL_A = sys.argv[1]
PANEL_B = sys.argv[2]
OUTDIR = sys.argv[3] if len(sys.argv) > 3 else "."
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------- layout ---
SCALE = 3            # supersampling factor; final image is downsampled
TARGET_H = 420 * SCALE   # common panel height before padding
GAP = 34 * SCALE         # horizontal space between panels
MARGIN = 16 * SCALE      # outer margin
LABEL_PAD = 30 * SCALE   # vertical room reserved for the a) / b) labels
INK = (26, 26, 26)


def load_on_white(path):
    """Flatten RGBA onto white, then trim uniform white borders."""
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    im = Image.alpha_composite(bg, im).convert("RGB")

    # Trim: find the bounding box of everything that is not near-white.
    px = im.load()
    w, h = im.size
    left, right, top, bottom = w, 0, h, 0
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            if r < 248 or g < 248 or b < 248:
                left = min(left, x)
                right = max(right, x)
                top = min(top, y)
                bottom = max(bottom, y)
    if right <= left or bottom <= top:
        return im
    return im.crop((left, top, right + 1, bottom + 1))


def scale_to_height(im, h):
    w = round(im.width * h / im.height)
    return im.resize((w, h), Image.LANCZOS)


def get_font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


# Panel a carries a soft halo of its own, so its drawn content reads smaller
# than panel b at equal image height. Scale it up slightly to balance them.
a = scale_to_height(load_on_white(PANEL_A), round(TARGET_H * 1.16))
b = scale_to_height(load_on_white(PANEL_B), TARGET_H)

band = max(a.height, b.height)
W = MARGIN * 2 + a.width + GAP + b.width
H = MARGIN * 2 + LABEL_PAD + band

canvas = Image.new("RGB", (W, H), "white")
ax = MARGIN
bx = MARGIN + a.width + GAP
top = MARGIN + LABEL_PAD
canvas.paste(a, (ax, top + (band - a.height) // 2))
canvas.paste(b, (bx, top + (band - b.height) // 2))

draw = ImageDraw.Draw(canvas)
font = get_font(21 * SCALE)
draw.text((ax, MARGIN), "a)", fill=INK, font=font)
draw.text((bx, MARGIN), "b)", fill=INK, font=font)

final = canvas.resize((W // SCALE, H // SCALE), Image.LANCZOS)
final.save(os.path.join(OUTDIR, "figure_2_3_cpcm.png"), dpi=(600, 600))
final.save(os.path.join(OUTDIR, "figure_2_3_cpcm.pdf"), "PDF",
           resolution=600.0)
print("wrote figure_2_3_cpcm.png / .pdf", final.size)
