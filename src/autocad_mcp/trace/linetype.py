"""Linetype and lineweight classification from pixel runs.

This is the answer to "the model cannot tell a solid line from a dashed one".
A language model asked to *look* at a drawing guesses; measuring the ink along
the line decides. For every traced segment we sample the binarised image along
its length and turn the resulting on/off pattern into:

* a **linetype** — CONTINUOUS / DASHED / HIDDEN / CENTER / PHANTOM,
* a **lineweight** — from the stroke thickness measured with a distance
  transform, mapped onto the standard ISO lineweight ladder,
* a **layer + role** — the conventional drafting layer for that combination.

Two honest caveats are encoded in the API rather than hidden:

1. ``CONTINUOUS`` vs a *dotted* pattern is a sampling question: runs shorter
   than ``min_run_px`` are anti-aliasing noise, not ink gaps.
2. ``DASHED`` vs ``HIDDEN`` is a **scale** question — they share the same
   dash:gap ratio (2:1) and differ mostly by period. When the drawing scale is
   known we separate them by period in drawing units; otherwise we return
   ``DASHED`` and report low confidence so the caller can correct it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Reference patterns (AutoCAD standard .lin definitions, in drawing units)
# ---------------------------------------------------------------------------

#: DXF linetype pattern definitions, used when a document lacks them.
LINETYPE_PATTERNS: dict[str, str] = {
    "CONTINUOUS": "",
    "DASHED": "12.7,-6.35",
    "HIDDEN": "6.35,-3.175",
    "CENTER": "31.75,-6.35,6.35,-6.35",
    "PHANTOM": "31.75,-6.35,6.35,-6.35,6.35,-6.35",
}

#: Standard ISO lineweights in hundredths of a millimetre (DXF codes 370).
LINEWEIGHT_LADDER = (13, 18, 25, 35, 50, 70, 100, 140, 200)

#: Above this period (drawing units) a 2:1 dash pattern reads as DASHED, below
#: it as HIDDEN — the two standard patterns differ only by scale.
HIDDEN_PERIOD_MAX = 12.0

#: Duty cycle (ink fraction) above which the line is simply solid.
SOLID_DUTY = 0.92

#: Layer conventions used by this project (see README pitfalls #1).
LAYER_THICK = "Thick"
LAYER_THIN = "Thin"
LAYER_HIDDEN = "Hidden"
LAYER_CENTER = "Center"

ROLE_OUTLINE = "outline"
ROLE_THIN = "thin"
ROLE_HIDDEN = "hidden"
ROLE_CENTER = "center"


@dataclass
class LinetypeVerdict:
    """Classification result for one sampled line."""

    linetype: str = "CONTINUOUS"
    confidence: float = 1.0
    duty: float = 1.0
    dash_mean: float = 0.0
    gap_mean: float = 0.0
    period: float = 0.0
    dashes: int = 0
    pattern: str = "solid"
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "linetype": self.linetype,
            "confidence": round(self.confidence, 3),
            "duty": round(self.duty, 3),
            "dash_mean": round(self.dash_mean, 2),
            "gap_mean": round(self.gap_mean, 2),
            "period": round(self.period, 2),
            "dashes": self.dashes,
            "pattern": self.pattern,
        }


# ---------------------------------------------------------------------------
# Run-length analysis
# ---------------------------------------------------------------------------


def run_lengths(occupancy: list[bool], min_run_px: int = 1) -> tuple[list[float], list[float]]:
    """Split an on/off pattern into (ink runs, gap runs).

    Leading and trailing gaps are discarded — they are outside the traced
    segment and carry no linetype information. Runs shorter than
    ``min_run_px`` are absorbed into their neighbour (anti-aliasing noise).
    """
    if not occupancy:
        return [], []

    runs: list[tuple[bool, int]] = []
    current = occupancy[0]
    length = 1
    for value in occupancy[1:]:
        if value == current:
            length += 1
        else:
            runs.append((current, length))
            current = value
            length = 1
    runs.append((current, length))

    if min_run_px > 1:
        merged: list[tuple[bool, int]] = []
        for value, length in runs:
            if merged and length < min_run_px:
                prev_value, prev_len = merged[-1]
                merged[-1] = (prev_value, prev_len + length)
            else:
                merged.append((value, length))
        runs = merged

    # Trim the outside-in gaps.
    if runs and not runs[0][0]:
        runs = runs[1:]
    if runs and not runs[-1][0]:
        runs = runs[:-1]

    dashes = [float(length) for value, length in runs if value]
    gaps = [float(length) for value, length in runs if not value]
    return dashes, gaps


def _pattern_signature(dashes: list[float]) -> str:
    """Classify the *shape* of a dash sequence: solid, uniform, long-short…

    The repeating unit is recovered by scoring every candidate period against
    the whole sequence, tolerating a truncated final unit ("LsLsLss" is still
    "Ls"). An exact-repeat test is not enough: the last dash of a traced line is
    routinely cut short by the segment end, and failing to reduce there is what
    turns a centre line into a "uniform" pattern.
    """
    if len(dashes) < 2:
        return "solid"
    lo, hi = min(dashes), max(dashes)
    if lo <= 0 or hi / max(lo, 1e-6) < 2.5:
        return "uniform"
    threshold = math.sqrt(lo * hi)  # geometric mean splits long/short well
    letters = "".join("L" if d >= threshold else "s" for d in dashes)

    best_unit: str | None = None
    for unit in range(1, len(letters) // 2 + 1):
        mismatches = sum(1 for index, ch in enumerate(letters) if ch != letters[index % unit])
        if mismatches <= 1:  # one mismatch == one truncated trailing unit
            best_unit = letters[:unit]
            break
    return best_unit if best_unit else letters


def classify_linetype(
    occupancy: list[bool],
    min_run_px: int = 2,
    scale: float = 1.0,
    hidden_period_max: float = HIDDEN_PERIOD_MAX,
) -> LinetypeVerdict:
    """Classify a sampled line's on/off pattern into a CAD linetype.

    ``scale`` converts pixel runs into drawing units so the DASHED/HIDDEN
    decision can use the real period.
    """
    dashes, gaps = run_lengths(occupancy, min_run_px=min_run_px)
    total = len(occupancy)
    ink = sum(dashes)

    if not gaps or len(dashes) < 2:
        return LinetypeVerdict(
            linetype="CONTINUOUS",
            confidence=1.0 if dashes else 0.0,
            duty=round(ink / total, 3) if total else 1.0,
            dash_mean=(ink / len(dashes)) if dashes else 0.0,
            dashes=len(dashes),
            pattern="solid",
            metrics={"reason": "no interior gaps"},
        )

    dash_mean = ink / len(dashes)
    gap_mean = sum(gaps) / len(gaps)
    period = dash_mean + gap_mean
    duty = dash_mean / period if period else 1.0
    period_units = period * scale
    signature = _pattern_signature(dashes)
    metrics = {
        "dash_cv": round(_coefficient_of_variation(dashes), 3),
        "period_px": round(period, 2),
        "period_units": round(period_units, 2),
    }

    # Solid-but-noisy: duty very high, or the "gaps" are sub-pixel noise.
    if duty >= SOLID_DUTY or gap_mean < max(1.0, 0.08 * dash_mean):
        return LinetypeVerdict(
            linetype="CONTINUOUS",
            confidence=0.9,
            duty=round(duty, 3),
            dash_mean=dash_mean,
            gap_mean=gap_mean,
            period=period,
            dashes=len(dashes),
            pattern="solid",
            metrics=metrics,
        )

    # Alternating long/short patterns: CENTER (L-s) and PHANTOM (L-s-s).
    if signature in ("Ls", "sL"):
        return LinetypeVerdict(
            "CENTER", 0.8, round(duty, 3), dash_mean, gap_mean, period, len(dashes), signature, metrics
        )
    if signature in ("Lss", "ssL", "sLs"):
        return LinetypeVerdict(
            "PHANTOM", 0.75, round(duty, 3), dash_mean, gap_mean, period, len(dashes), signature, metrics
        )

    # Uniform 2:1-ish dash pattern: DASHED and HIDDEN are the same shape at
    # different scales, so the period decides which one this is.
    if 0.15 <= duty <= 0.85:
        ratio = gap_mean / dash_mean if dash_mean else 0.0
        if period_units <= hidden_period_max:
            confidence = 0.7 if 0.3 <= ratio <= 2.0 else 0.5
            return LinetypeVerdict(
                "HIDDEN", confidence, round(duty, 3), dash_mean, gap_mean, period, len(dashes), signature, metrics
            )
        confidence = 0.85 if 0.3 <= ratio <= 2.0 else 0.55
        return LinetypeVerdict(
            "DASHED", confidence, round(duty, 3), dash_mean, gap_mean, period, len(dashes), signature, metrics
        )

    # Very low duty: sparse, dot-like marks. Closest standard is a dashed line.
    return LinetypeVerdict(
        "DASHED", 0.4, round(duty, 3), dash_mean, gap_mean, period, len(dashes), signature, metrics
    )


def _coefficient_of_variation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean <= 0:
        return 0.0
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var) / mean


# ---------------------------------------------------------------------------
# Lineweight
# ---------------------------------------------------------------------------


def snap_lineweight(width_mm: float) -> int:
    """Nearest standard lineweight in hundredths of a millimetre."""
    return min(LINEWEIGHT_LADDER, key=lambda candidate: abs(candidate / 100.0 - width_mm))


def classify_lineweight(width_px: float, scale: float = 1.0) -> tuple[int, float, bool]:
    """(DXF lineweight, width in drawing units, is_thick) for a stroke width."""
    width_units = max(0.0, width_px) * scale
    # A traced outline is usually the drawn lineweight, so use it directly.
    lineweight = snap_lineweight(width_units)
    return lineweight, width_units, width_units >= 0.35


# ---------------------------------------------------------------------------
# Layer / role convention
# ---------------------------------------------------------------------------


def layer_for(linetype: str, is_thick: bool) -> tuple[str, str, int]:
    """Map (linetype, thickness) → (layer name, role, ACI colour)."""
    if linetype in ("CENTER", "PHANTOM"):
        return LAYER_CENTER, ROLE_CENTER, 4
    if linetype in ("DASHED", "HIDDEN"):
        return LAYER_HIDDEN, ROLE_HIDDEN, 8
    if is_thick:
        return LAYER_THICK, ROLE_OUTLINE, 7
    return LAYER_THIN, ROLE_THIN, 7
