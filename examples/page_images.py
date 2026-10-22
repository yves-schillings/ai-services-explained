"""Fictional page images for the deep-learning examples.

Author: Yves Schillings.

Each image is drawn by this file: no real document, no download. A page is drawn
at 128 x 128 pixels, degraded like a scan (scale, shift, rotation, blur, noise) and
reduced to 32 x 32 grey pixels. Ink is 1, paper is 0.
"""
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIZE = 32
PAGE_TYPES = ["meeting_minutes", "incident_report", "item_inventory"]
GENERIC_PIECES = ["text_block", "table", "stamps", "check_boxes", "dialogue_lines", "header_and_signature"]


def _place(img, rng, hard):
    s = rng.uniform(0.65, 1.0) if hard else rng.uniform(0.75, 1.0)
    w = int(128 * s)
    canvas = Image.new("L", (128, 128), 255)
    canvas.paste(img.resize((w, w), Image.BILINEAR), (rng.randint(0, 128 - w), rng.randint(0, 128 - w)))
    return canvas


def scan(img, rng, hard=False):
    """Make a clean drawing look like a scan, then reduce it to SIZE x SIZE values in [0, 1]."""
    img = _place(img, rng, hard)
    angle = rng.uniform(-14, 14) if hard else rng.uniform(-5, 5)
    shift = (rng.randint(-6, 6), rng.randint(-6, 6)) if hard else (rng.randint(-3, 3), rng.randint(-3, 3))
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=255, translate=shift)
    if hard or rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(1.0, 2.2) if hard else rng.uniform(0.3, 0.9)))
    a = 1.0 - np.asarray(img.resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    a = a * rng.uniform(0.6, 1.0) + rng.uniform(0.0, 0.15)
    a = a + np.random.default_rng(rng.randint(0, 10 ** 9)).normal(0, 0.12 if hard else 0.05, a.shape)
    return np.clip(a, 0, 1).astype(np.float32)


def _lines(d, rng, x0, y0, x1, y1, n, indent=False):
    for i, y in enumerate(np.linspace(y0, y1, n)):
        xs = x0 + (14 if indent and i % 2 else 0)
        d.line([(xs, y), (rng.uniform(x0 + 30, x1), y)], fill=0, width=2)


def draw_page(kind, rng):
    """Three fictional layouts: dialogue lines, a table with lines, a checklist with a stamp."""
    img = Image.new("L", (128, 128), 255)
    d = ImageDraw.Draw(img)
    d.rectangle([8, 6, 120, 16], fill=rng.randint(0, 90))          # header bar on every page
    if kind == 0:                                                   # meeting minutes: alternating lines
        _lines(d, rng, 12, 26, 116, 118, rng.randint(9, 12), indent=True)
    elif kind == 1:                                                 # incident report: boxes, then lines
        rows, cols = rng.randint(3, 4), rng.randint(2, 3)
        for r in range(rows):
            for c in range(cols):
                x, y = 12 + c * (104 // cols), 24 + r * 14
                d.rectangle([x, y, x + 104 // cols - 4, y + 10], outline=0, width=2)
        _lines(d, rng, 12, 24 + rows * 14 + 6, 116, 118, rng.randint(3, 5))
    else:                                                           # item inventory: check boxes, stamp
        for i in range(rng.randint(4, 6)):
            y = 26 + i * 15
            d.rectangle([12, y, 22, y + 9], outline=0, width=2)
            d.line([(28, y + 5), (rng.uniform(60, 100), y + 5)], fill=0, width=2)
        cx, cy, r = rng.randint(88, 104), rng.randint(96, 110), rng.randint(10, 14)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=0, width=3)
    return img


def draw_generic(kind, rng):
    """Generic pieces for pretraining. The network never sees the three page types here."""
    img = Image.new("L", (128, 128), 255)
    d = ImageDraw.Draw(img)
    y0 = rng.randint(10, 40)
    if kind == 0:
        _lines(d, rng, 12, y0, 116, min(118, y0 + 70), rng.randint(5, 9))
    elif kind == 1:
        rows, cols = rng.randint(2, 5), rng.randint(2, 4)
        for r in range(rows):
            for c in range(cols):
                x, y = 12 + c * (104 // cols), y0 + r * 14
                d.rectangle([x, y, x + 104 // cols - 4, y + 10], outline=0, width=2)
    elif kind == 2:
        for _ in range(rng.randint(1, 3)):
            cx, cy, r = rng.randint(25, 103), rng.randint(25, 103), rng.randint(9, 16)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=0, width=3)
    elif kind == 3:
        for i in range(rng.randint(3, 6)):
            y = y0 + i * 14
            d.rectangle([12, y, 22, y + 9], outline=0, width=2)
    elif kind == 4:
        _lines(d, rng, 12, y0, 116, min(118, y0 + 80), rng.randint(6, 10), indent=True)
    else:
        d.rectangle([8, 6, 120, 16], fill=rng.randint(0, 90))
        d.line([(70, 110), (118, 110)], fill=0, width=2)
    return img


def draw_unusual(rng):
    """A page that matches none of the known layouts: a photo-like patch of shapes."""
    img = Image.new("L", (128, 128), 255)
    d = ImageDraw.Draw(img)
    for _ in range(rng.randint(15, 30)):
        x, y = rng.randint(0, 120), rng.randint(0, 120)
        d.rectangle([x, y, x + rng.randint(4, 30), y + rng.randint(4, 30)], fill=rng.randint(0, 200))
    return img


def make_set(drawer, n_classes, per_class, seed, hard=False):
    """Return images (N, SIZE, SIZE) and labels (N,), shuffled."""
    rng = random.Random(seed)
    xs, ys = [], []
    for k in range(n_classes):
        for _ in range(per_class):
            xs.append(scan(drawer(k, rng), rng, hard))
            ys.append(k)
    order = list(range(len(ys)))
    rng.shuffle(order)
    return np.stack([xs[i] for i in order]), np.array([ys[i] for i in order])
