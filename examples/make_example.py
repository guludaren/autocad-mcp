"""Generate the example drawing used in the docs (and by the CLI demo).

Run from the repository root::

    python examples/make_example.py

It writes ``examples/sample_part.png``: a deliberately *simple* drawing that
still exercises everything the tracer has to get right — a thick outline, a
hidden edge, a centre line, a circular hole and a fillet arc.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 1000, 760
OUTPUT = Path(__file__).resolve().parent / "sample_part.png"


def dashed_line(image, start, end, pattern, thickness: int) -> None:
    """Draw a line with an explicit [(ink, gap), ...] pixel pattern."""
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    position = 0.0
    index = 0
    while position < length:
        ink, gap = pattern[index % len(pattern)]
        segment = min(ink, length - position)
        cv2.line(
            image,
            (round(x1 + ux * position), round(y1 + uy * position)),
            (round(x1 + ux * (position + segment)), round(y1 + uy * (position + segment))),
            0,
            thickness,
        )
        position += ink + gap
        index += 1


def build() -> np.ndarray:
    image = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)

    # Thick outline of a plate.
    cv2.rectangle(image, (120, 120), (880, 640), 0, 7)

    # Thin circular hole, with a centre line through it.
    cv2.circle(image, (330, 380), 110, 0, 3)
    dashed_line(image, (170, 380), (490, 380), [(60, 14), (18, 14)], 2)
    dashed_line(image, (330, 220), (330, 540), [(60, 14), (18, 14)], 2)

    # Fillet arc in the right half.
    cv2.ellipse(image, (700, 380), (140, 140), 0, 0, 120, 0, 3)

    # Hidden edge showing an internal step.
    dashed_line(image, (180, 560), (820, 560), [(24, 16)], 3)

    # Dashed reference line near the top.
    dashed_line(image, (180, 190), (820, 190), [(52, 26)], 4)
    return image


def main() -> int:
    image = build()
    cv2.imwrite(str(OUTPUT), image)
    print(f"wrote {OUTPUT} ({image.shape[1]}x{image.shape[0]} px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
