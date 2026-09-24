"""Raster → CAD IR vectorisation.

The core design decision of this module: **the model never looks at pixels.**
Asking a vision model (or worse, a text-only model) to "read" a drawing is what
produced the two failures this project set out to fix — incomplete geometry and
mis-read line styles. Extraction is therefore done with deterministic image
processing, and the LLM is only handed the *resulting geometry* to label.

Pipeline
--------
1. grayscale + binarise (Otsu or adaptive), polarity auto-detected,
2. optional auto-upscale for low-resolution / hairline drawings,
3. straight segments via probabilistic Hough, chained across dash gaps and
   merged when collinear (so a dashed line becomes **one** entity, not N),
4. circles and arcs via circle-fitting every contour, with the angular span
   recovered from the largest gap in the contour's angle coverage,
5. per-entity measurements: ink occupancy → linetype, distance transform →
   stroke width → lineweight,
6. endpoint welding and arc-endpoint snapping so arcs actually meet their
   neighbouring lines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from autocad_mcp.trace import geometry, linetype
from autocad_mcp.trace.ir import Arc, Circle, Drawing, Layer, Line

Point = tuple[float, float]


@dataclass
class VectorizeOptions:
    """Knobs for :func:`vectorize`. Pixel values are in *original* image pixels."""

    threshold: str = "otsu"          # otsu | adaptive
    block_size: int = 35             # adaptive-threshold window (odd, >1)
    c_constant: int = 9
    min_line_px: float = 22.0        # shortest straight segment to keep
    max_gap_px: float = 9.0          # gap chained by Hough → one dashed segment
    merge_angle_deg: float = 3.0     # collinear merge tolerance
    merge_dist_px: float = 3.0
    # Dash gaps are chained during collinear merging so a dashed line becomes
    # ONE entity; its ink pattern is then measured and classified. Without this
    # the dashes survive as separate "solid" lines — the exact bug being fixed.
    merge_max_gap_px: float = 26.0
    hough_threshold: int = 40
    arc_min_px: float = 26.0
    circle_fit_tol: float = 0.04     # rms / radius
    arc_min_span_deg: float = 12.0   # shorter spans are noise, not arcs
    blob_max_area_frac: float = 0.0025  # of image area → treated as text/noise
    blob_max_perimeter_px: float = 60.0
    sample_step_px: float = 1.0
    min_run_px: int = 2
    weld_tol_px: float = 3.5
    snap_arc_tol_px: float = 6.0
    auto_upscale: bool = True
    max_upscale: int = 3
    scale: float = 1.0               # drawing units per pixel
    units: str = "mm"
    origin: Point = (0.0, 0.0)


@dataclass
class VectorizeResult:
    """Vectorisation output: the IR plus everything needed to explain it."""

    drawing: Drawing
    binary: np.ndarray | None = None
    upscale: float = 1.0
    stroke_width_px: float = 1.0
    counters: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Binarisation
# ---------------------------------------------------------------------------


def load_gray(image) -> np.ndarray:
    """Load a path or pass through an array as single-channel grayscale."""
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return image.astype(np.uint8, copy=False)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    path = Path(image)
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"cannot decode image: {path}")
    return gray


def binarize(gray: np.ndarray, options: VectorizeOptions) -> np.ndarray:
    """Return ink as 255 on a 0 background, polarity auto-detected.

    Ink is assumed to be the *minority* class, which holds for every normal
    drawing regardless of whether it is dark-on-light or light-on-dark.
    """
    if options.threshold == "adaptive":
        block = options.block_size if options.block_size % 2 else options.block_size + 1
        block = max(3, block)
        forward = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, options.c_constant
        )
        inverse = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, options.c_constant
        )
    else:
        _, forward = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, inverse = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Pick whichever interpretation makes ink the minority.
    if _ink_fraction(forward) <= _ink_fraction(inverse):
        binary = forward
    else:
        binary = inverse
    return _despeckle(binary)


def _ink_fraction(binary: np.ndarray) -> float:
    return float(np.count_nonzero(binary)) / float(binary.size) if binary.size else 0.0


def _despeckle(binary: np.ndarray, min_pixels: int = 4) -> np.ndarray:
    """Drop 1–3 pixel specks without eroding real (possibly 1px) strokes."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_pixels
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def estimate_stroke_width(binary: np.ndarray) -> float:
    """Median full stroke width in pixels, from the distance transform."""
    if not np.count_nonzero(binary):
        return 1.0
    dt = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    values = dt[binary > 0]
    if values.size == 0:
        return 1.0
    # The ridge of a stroke sits at ~half width; the 85th percentile avoids
    # junctions and blobs, which inflate the maximum.
    return float(max(1.0, 2.0 * float(np.percentile(values, 85))))


# ---------------------------------------------------------------------------
# Straight segments
# ---------------------------------------------------------------------------


@dataclass
class _Segment:
    p1: Point
    p2: Point
    length: float
    width: float = 1.0

    def direction(self) -> Point:
        dx, dy = self.p2[0] - self.p1[0], self.p2[1] - self.p1[1]
        norm = math.hypot(dx, dy) or 1.0
        return (dx / norm, dy / norm)


@dataclass
class _Group:
    """A collinear run of segments being merged into one entity."""

    origin: Point          # a point on the line
    direction: Point       # unit vector
    t_min: float
    t_max: float
    angle_mod: float       # direction angle in [0, 180)
    offset_total: float = 0.0   # signed perpendicular offsets of merged members
    offset_count: int = 0

    def perpendicular_distance(self, point: Point) -> float:
        return abs(self.signed_offset(point))

    def signed_offset(self, point: Point) -> float:
        """Signed perpendicular distance along the left normal ``(-dy, dx)``.

        The sign convention matters: :meth:`recentre` shifts the origin along
        that same normal, so a flipped sign here walks the line *away* from the
        stroke midline — measured at roughly a full lineweight of error.
        """
        vx = point[0] - self.origin[0]
        vy = point[1] - self.origin[1]
        return vy * self.direction[0] - vx * self.direction[1]

    def project(self, point: Point) -> float:
        return (point[0] - self.origin[0]) * self.direction[0] + (point[1] - self.origin[1]) * self.direction[1]

    def recentre(self) -> None:
        """Shift onto the midline of the strokes that were merged into it.

        A thick stroke produces an edge pair; without this the merged entity
        would sit on whichever edge happened to be detected first, biasing every
        coordinate by up to half a lineweight.
        """
        if self.offset_count:
            shift = self.offset_total / self.offset_count
            self.origin = (
                self.origin[0] - self.direction[1] * shift,
                self.origin[1] + self.direction[0] * shift,
            )
            self.offset_total = 0.0
            self.offset_count = 0

    def endpoints(self) -> tuple[Point, Point]:
        return (
            (self.origin[0] + self.direction[0] * self.t_min, self.origin[1] + self.direction[1] * self.t_min),
            (self.origin[0] + self.direction[0] * self.t_max, self.origin[1] + self.direction[1] * self.t_max),
        )


def _angle_mod(direction: Point) -> float:
    return math.degrees(math.atan2(direction[1], direction[0])) % 180.0


def _angle_difference(a: float, b: float) -> float:
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


def detect_segments(
    binary: np.ndarray, options: VectorizeOptions, up: float, stroke_px: float = 1.0
) -> list[_Segment]:
    """Detect straight segments, chaining dash gaps and merging collinear runs."""
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360.0,
        threshold=max(10, int(options.hough_threshold * up)),
        minLineLength=max(4, int(options.min_line_px * up)),
        maxLineGap=max(1, int(options.max_gap_px * up)),
    )
    if raw is None:
        return []

    segments: list[_Segment] = []
    for x1, y1, x2, y2 in raw.reshape(-1, 4):
        p1 = (float(x1), float(y1))
        p2 = (float(x2), float(y2))
        length = geometry.dist(p1, p2)
        if length < options.min_line_px * up:
            continue
        segments.append(_Segment(p1, p2, length))
    segments.sort(key=lambda s: s.length, reverse=True)

    # A thick stroke yields an edge pair one lineweight apart; the tolerance has
    # to absorb that or every thick line is emitted twice.
    dist_tol = max(options.merge_dist_px * up, stroke_px * 1.2)
    max_gap = options.merge_max_gap_px * up
    groups: list[_Group] = []
    for segment in segments:
        direction = segment.direction()
        angle = _angle_mod(direction)
        merged = False
        for group in groups:
            if _angle_difference(group.angle_mod, angle) > options.merge_angle_deg:
                continue
            if group.perpendicular_distance(segment.p1) > dist_tol:
                continue
            if group.perpendicular_distance(segment.p2) > dist_tol:
                continue
            gt1 = group.project(segment.p1)
            gt2 = group.project(segment.p2)
            lo, hi = min(gt1, gt2), max(gt1, gt2)
            if lo > group.t_max + max_gap or hi < group.t_min - max_gap:
                continue  # collinear but far away: a different wall, not a dash
            group.t_min = min(group.t_min, lo)
            group.t_max = max(group.t_max, hi)
            group.offset_total += group.signed_offset(segment.p1)
            group.offset_count += 1
            merged = True
            break
        if not merged:
            groups.append(
                _Group(
                    origin=segment.p1,
                    direction=direction,
                    t_min=0.0,
                    t_max=segment.length,
                    angle_mod=angle,
                )
            )

    merged_segments: list[_Segment] = []
    for group in groups:
        group.recentre()
        p1, p2 = group.endpoints()
        length = geometry.dist(p1, p2)
        if length >= options.min_line_px * up:
            merged_segments.append(_Segment(p1, p2, length))
    return merged_segments


def recentre_on_ink(
    binary: np.ndarray, p1: Point, p2: Point, max_shift: int, step_px: float = 1.0
) -> tuple[Point, Point, float]:
    """Slide a segment perpendicular until it sits on the ink it was detected from.

    Hough lines land on a Canny *edge*, i.e. half a lineweight away from the
    stroke centreline, and a merged edge pair keeps whichever edge came first.
    Sampling a 1 px-wide probe across a small perpendicular window and keeping
    the offset with the most ink fixes the coordinates and, just as importantly,
    the linetype measurement that follows.

    Returns ``(p1, p2, best_ink_fraction)``.
    """
    length = geometry.dist(p1, p2)
    if length <= 0:
        return p1, p2, 0.0
    dx = (p2[0] - p1[0]) / length
    dy = (p2[1] - p1[1]) / length
    nx, ny = -dy, dx

    best_shift = 0.0
    best_fraction = -1.0
    for shift in range(-max_shift, max_shift + 1):
        moved = (
            (p1[0] + nx * shift, p1[1] + ny * shift),
            (p2[0] + nx * shift, p2[1] + ny * shift),
        )
        pattern = sample_occupancy(binary, moved[0], moved[1], step_px, half_window=0)
        fraction = (sum(pattern) / len(pattern)) if pattern else 0.0
        # Prefer the strongest ink; on a tie keep the smaller correction.
        if fraction > best_fraction + 1e-9 or (
            abs(fraction - best_fraction) <= 1e-9 and abs(shift) < abs(best_shift)
        ):
            best_fraction = fraction
            best_shift = float(shift)
    return (
        (p1[0] + nx * best_shift, p1[1] + ny * best_shift),
        (p2[0] + nx * best_shift, p2[1] + ny * best_shift),
        max(0.0, best_fraction),
    )


def chain_segments(
    segments: list[_Segment], options: VectorizeOptions, stroke_px: float = 1.0, up: float = 1.0
) -> tuple[list[_Segment], int]:
    """Join collinear fragments across dash gaps, **before** any linetype call.

    Order matters here and it is the crux of the whole feature. Hough splits a
    dashed line into per-dash fragments, and a single dash has no interior gap —
    measured alone it is indistinguishable from a solid line. So the fragments
    are chained first and the ink pattern is measured once over the full run,
    which is what makes "solid vs dashed" decidable at all.

    Coordinates here are still in working-resolution **pixels**, so the
    tolerances are pixel quantities scaled by the upscale factor — not by the
    drawing scale, which would shrink them by the units-per-pixel factor and
    silently disable chaining.
    """
    tolerance = max(options.merge_dist_px, stroke_px * 1.2) * up
    max_gap = options.merge_max_gap_px * up
    angle_tol = max(options.merge_angle_deg, 12.0)

    def project(direction: Point, origin: Point, point: Point) -> float:
        return (point[0] - origin[0]) * direction[0] + (point[1] - origin[1]) * direction[1]

    def perpendicular(direction: Point, origin: Point, point: Point) -> float:
        return abs((point[0] - origin[0]) * direction[1] - (point[1] - origin[1]) * direction[0])

    def try_merge(group: _Segment, other: _Segment, gap_limit: float | None = None) -> bool:
        """Extend ``group`` to cover ``other`` when they are the same line."""
        limit = max_gap if gap_limit is None else gap_limit
        group_direction = group.direction()
        other_direction = other.direction()
        if _angle_difference(_angle_mod(group_direction), _angle_mod(other_direction)) > angle_tol:
            return False
        if perpendicular(group_direction, group.p1, other.p1) > tolerance:
            return False
        if perpendicular(group_direction, group.p1, other.p2) > tolerance:
            return False
        group_ts = sorted(
            (project(group_direction, group.p1, group.p1), project(group_direction, group.p1, group.p2))
        )
        lo, hi = sorted(
            (project(group_direction, group.p1, other.p1), project(group_direction, group.p1, other.p2))
        )
        if lo > group_ts[1] + limit or hi < group_ts[0] - limit:
            return False
        low = min(group_ts[0], lo)
        high = max(group_ts[1], hi)
        group.p1 = (
            group.p1[0] + group_direction[0] * low,
            group.p1[1] + group_direction[1] * low,
        )
        group.p2 = (
            group.p1[0] + group_direction[0] * (high - low),
            group.p1[1] + group_direction[1] * (high - low),
        )
        group.length = geometry.dist(group.p1, group.p2)
        group.width = max(group.width, other.width)
        return True

    chained: list[_Segment] = []
    for segment in sorted(segments, key=lambda item: item.length, reverse=True):
        if not any(try_merge(group, segment) for group in chained):
            chained.append(segment)

    # A greedy pass only ever folds a new fragment into an *existing* group, so
    # two adjacency chains can grow towards each other and never meet. Fold the
    # groups into each other until nothing changes.
    for _pass in range(8):
        changed = False
        for index in range(len(chained)):
            for other_index in range(index + 1, len(chained)):
                if try_merge(chained[index], chained[other_index]):
                    del chained[other_index]
                    changed = True
                    break
            if changed:
                break
        if not changed:
            break

    # Long-dash centre lines are separated by more than one dash period, so each
    # long dash on its own looks solid. Short fragments are therefore chained
    # again with a relaxed gap — long runs are left alone, which keeps two
    # genuinely separate collinear lines from being welded together.
    short_limit = max_gap * 2.5
    short_length = max_gap * 6.0
    for _pass in range(8):
        changed = False
        for index in range(len(chained)):
            for other_index in range(index + 1, len(chained)):
                first, second = chained[index], chained[other_index]
                if min(first.length, second.length) > short_length:
                    continue
                if try_merge(first, second, gap_limit=short_limit):
                    del chained[other_index]
                    changed = True
                    break
            if changed:
                break
        if not changed:
            break

    return chained, len(segments) - len(chained)


def sample_occupancy(
    binary: np.ndarray,
    p1: Point,
    p2: Point,
    step_px: float,
    half_window: int = 2,
) -> list[bool]:
    """Ink/blank pattern sampled along a segment.

    Each sample looks at a small perpendicular window so a 1 px centre-line
    offset does not turn a solid line into a dashed one.
    """
    height, width = binary.shape[:2]
    length = geometry.dist(p1, p2)
    count = max(2, int(length / max(step_px, 0.5)))
    dx = (p2[0] - p1[0]) / count
    dy = (p2[1] - p1[1]) / count
    # Unit normal for the perpendicular window.
    norm = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / norm, dx / norm

    pattern: list[bool] = []
    for i in range(count + 1):
        x = p1[0] + dx * i
        y = p1[1] + dy * i
        ink = False
        for offset in range(-half_window, half_window + 1):
            sx = int(round(x + nx * offset))
            sy = int(round(y + ny * offset))
            if 0 <= sx < width and 0 <= sy < height and binary[sy, sx]:
                ink = True
                break
        pattern.append(ink)
    return pattern


def measure_width(distance: np.ndarray, p1: Point, p2: Point, step_px: float, window: int = 2) -> float:
    """Full stroke width (px) along a segment, from distance-transform peaks."""
    height, width = distance.shape[:2]
    length = geometry.dist(p1, p2)
    count = max(2, int(length / max(step_px, 0.5)))
    dx = (p2[0] - p1[0]) / count
    dy = (p2[1] - p1[1]) / count
    norm = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / norm, dx / norm

    peaks: list[float] = []
    for i in range(count + 1):
        x = p1[0] + dx * i
        y = p1[1] + dy * i
        best = 0.0
        for offset in range(-window, window + 1):
            sx = int(round(x + nx * offset))
            sy = int(round(y + ny * offset))
            if 0 <= sx < width and 0 <= sy < height:
                best = max(best, float(distance[sy, sx]))
        if best > 0:
            peaks.append(best)
    if not peaks:
        return 1.0
    return float(max(1.0, 2.0 * float(np.median(peaks))))


# ---------------------------------------------------------------------------
# Circles and arcs
# ---------------------------------------------------------------------------


@dataclass
class _Round:
    """A fitted circle or arc, in working-resolution pixels."""

    center: Point
    radius: float
    start_angle: float
    end_angle: float
    span: float
    fit_rms: float
    closed: bool


def _angular_span(points: np.ndarray, center: Point) -> tuple[float, float, float]:
    """Recover (start, end, span) from the largest gap in angle coverage."""
    angles = np.sort(np.array([geometry.angle_of(center, (float(x), float(y))) for x, y in points]))
    if angles.size < 3:
        return 0.0, 0.0, 0.0
    gaps = np.diff(np.concatenate([angles, angles[:1] + 360.0]))
    largest = int(np.argmax(gaps))
    gap = float(gaps[largest])
    span = 360.0 - gap
    start = float(angles[(largest + 1) % angles.size])
    end = float(angles[largest])
    return start % 360.0, end % 360.0, span


def detect_rounds(binary: np.ndarray, options: VectorizeOptions, up: float) -> tuple[list[_Round], dict]:
    """Fit circles/arcs to every contour; report text-like blobs that were dropped."""
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    image_area = float(binary.shape[0] * binary.shape[1])
    rounds: list[_Round] = []
    counters = {"contours": len(contours), "blobs_dropped": 0, "fit_rejected": 0}

    for contour in contours:
        points = contour.reshape(-1, 2).astype(float)
        if points.shape[0] < 8:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        area = float(cv2.contourArea(contour))
        if perimeter < options.arc_min_px * up:
            # Small closed blob: text, arrowhead, dimension dot.
            if options.blob_max_area_frac > 0 and 0 < area < options.blob_max_area_frac * image_area:
                counters["blobs_dropped"] += 1
            continue
        if (
            options.blob_max_area_frac > 0
            and 0 < area < options.blob_max_area_frac * image_area
            and perimeter < options.blob_max_perimeter_px * up
        ):
            counters["blobs_dropped"] += 1
            continue

        fit = geometry.fit_circle([(p[0], p[1]) for p in points])
        if fit is None:
            counters["fit_rejected"] += 1
            continue
        center, radius, rms = fit
        if radius <= 1.0 or rms > options.circle_fit_tol * radius:
            counters["fit_rejected"] += 1
            continue

        start, end, span = _angular_span(points, center)
        closed = span >= 360.0 - max(options.arc_min_span_deg, 8.0)
        if not closed and span < max(options.arc_min_span_deg, 1.0):
            counters["fit_rejected"] += 1
            continue
        rounds.append(
            _Round(
                center=center,
                radius=radius,
                start_angle=start,
                end_angle=end,
                span=360.0 if closed else span,
                fit_rms=rms,
                closed=closed,
            )
        )
    return rounds, counters


def detect_rounds_hough(
    gray: np.ndarray, binary: np.ndarray, options: VectorizeOptions, up: float
) -> list[_Round]:
    """Second circle detector, for circles that other ink runs across.

    Contour fitting fails on a circle crossed by centre lines or hidden edges,
    because the crossing merges several strokes into one connected component.
    :func:`cv2.HoughCircles` is not confused by that; every proposal is then
    accepted only if the ink actually covers the proposed circle, which keeps
    the false-positive rate of the accumulator under control.
    """
    height, width = binary.shape[:2]
    min_radius = max(4, int(options.arc_min_px * up / 2))
    max_radius = int(min(height, width) / 2)
    if max_radius <= min_radius:
        return []
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    found = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=max(8.0, min_radius * 1.5),
        param1=120,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if found is None:
        return []

    proposals: list[_Round] = []
    window = max(1, int(round(up)))
    for cx, cy, radius in found[0][:64]:
        center = (float(cx), float(cy))
        if not (0 <= center[0] < width and 0 <= center[1] < height):
            continue
        pattern = sample_occupancy_arc(
            binary, center, float(radius), 0.0, 360.0, step_px=max(1.0, up), radial_window=window
        )
        support = (sum(pattern) / len(pattern)) if pattern else 0.0
        if support < 0.45:
            continue  # the accumulator guessed; the ink disagrees
        proposals.append(
            _Round(
                center=center,
                radius=float(radius),
                start_angle=0.0,
                end_angle=0.0,
                span=360.0,
                fit_rms=0.0,
                closed=True,
            )
        )
    return proposals


def merge_rounds(
    rounds: list[_Round], options: VectorizeOptions, up: float, stroke_px: float = 1.0
) -> tuple[list[_Round], dict]:
    """Merge duplicate strokes (outer/inner contours) and dashed arcs.

    A thick circle stroke has two contours; a dashed circle produces several
    short arcs. Both are collapsed into one entity here.
    """
    counters = {"merged_duplicates": 0, "merged_arc_chains": 0}
    kept: list[_Round] = []
    # A thick stroke's outer and inner contour differ by roughly the stroke
    # width, so the merge tolerance has to allow for it.
    tol_radius = max(1.5 * up, 1.2 * stroke_px)
    for item in sorted(rounds, key=lambda r: (-r.closed, -r.span, r.fit_rms)):
        merged = False
        for index, existing in enumerate(kept):
            if geometry.dist(existing.center, item.center) > max(tol_radius, 0.04 * existing.radius):
                continue
            if abs(existing.radius - item.radius) > max(tol_radius, 0.04 * existing.radius):
                continue
            if existing.closed and item.closed:
                if item.fit_rms < existing.fit_rms:
                    kept[index] = item
                counters["merged_duplicates"] += 1
                merged = True
                break
            if existing.closed or item.closed:
                counters["merged_duplicates"] += 1
                merged = True
                break
            # Two open arcs on the same circle: join when the angular gap is
            # small (a dash gap), otherwise keep both.
            gap = _angular_gap(existing.start_angle, existing.end_angle, item.start_angle, item.end_angle)
            max_gap_deg = math.degrees(max(options.max_gap_px * up, 1.0) / max(existing.radius, 1.0))
            if gap <= max_gap_deg:
                combined = _union_span(existing, item)
                kept[index] = combined
                counters["merged_arc_chains"] += 1
                merged = True
                break
        if not merged:
            kept.append(item)
    return kept, counters


def _angular_gap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Smallest angular gap between two arcs on the same circle."""
    candidates = [
        geometry.span_ccw(a_end, b_start),
        geometry.span_ccw(b_end, a_start),
        geometry.span_ccw(a_end, b_start) - 360.0,
        geometry.span_ccw(b_end, a_start) - 360.0,
    ]
    return min(abs(value) for value in candidates)


def _union_span(a: _Round, b: _Round) -> _Round:
    """Union of two adjacent arcs on the same circle."""
    best: _Round | None = None
    for first, second in ((a, b), (b, a)):
        span = geometry.span_ccw(first.start_angle, second.end_angle)
        candidate = _Round(
            center=first.center,
            radius=(first.radius + second.radius) / 2.0,
            start_angle=first.start_angle,
            end_angle=second.end_angle,
            span=span,
            fit_rms=max(first.fit_rms, second.fit_rms),
            closed=span >= 360.0 - 1e-6,
        )
        if best is None or candidate.span > best.span:
            best = candidate
    assert best is not None
    return best


def drop_chords_of_rounds(
    lines: list[Line],
    rounds: list[_Round],
    up: float,
    height: int,
    options: VectorizeOptions,
) -> tuple[list[Line], int]:
    """Remove short straight chords that are really part of a detected circle.

    Hough happily reports a chord across a circular stroke. Once the circle has
    been fitted, any short line whose two endpoints both sit on that circle is
    an artefact, not a drawn edge.
    """
    if not rounds:
        return lines, 0
    circles = [
        (
            _to_drawing(item.center, up, height, options),
            item.radius / up * options.scale,
        )
        for item in rounds
    ]
    kept: list[Line] = []
    dropped = 0
    for line in lines:
        length = geometry.dist(line.start, line.end)
        artefact = False
        for center, radius in circles:
            tolerance = max(2.0 * options.scale, radius * 0.06)
            on_circle = (
                abs(geometry.dist(center, line.start) - radius) <= tolerance
                and abs(geometry.dist(center, line.end) - radius) <= tolerance
            )
            if on_circle and length < max(radius, 1e-6) * 1.2:
                artefact = True
                break
        if artefact:
            dropped += 1
        else:
            kept.append(line)
    return kept, dropped


def sample_occupancy_arc(
    binary: np.ndarray,
    center: Point,
    radius: float,
    start_angle: float,
    span: float,
    step_px: float,
    radial_window: int = 2,
) -> list[bool]:
    """Ink pattern sampled *along the arc*, radially tolerant."""
    height, width = binary.shape[:2]
    arc_length = math.radians(max(span, 0.0)) * max(radius, 1e-6)
    count = max(4, int(arc_length / max(step_px, 0.5)))
    pattern: list[bool] = []
    for i in range(count + 1):
        angle = start_angle + span * (i / count)
        rad = math.radians(angle)
        cx, cy = math.cos(rad), math.sin(rad)
        ink = False
        for offset in range(-radial_window, radial_window + 1):
            r = radius + offset
            sx = int(round(center[0] + cx * r))
            sy = int(round(center[1] + cy * r))
            if 0 <= sx < width and 0 <= sy < height and binary[sy, sx]:
                ink = True
                break
        pattern.append(ink)
    return pattern


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def vectorize(image, options: VectorizeOptions | None = None) -> VectorizeResult:
    """Trace an image into a CAD IR drawing."""
    options = options or VectorizeOptions()
    gray = load_gray(image)

    up = 1.0
    binary = binarize(gray, options)
    stroke_px = estimate_stroke_width(binary)
    if options.auto_upscale and stroke_px < 2.5:
        up = float(min(options.max_upscale, max(1.0, math.ceil(3.0 / max(stroke_px, 0.5)))))
        if up > 1.0:
            gray = cv2.resize(gray, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
            binary = binarize(gray, options)
            stroke_px = estimate_stroke_width(binary)

    height, width = binary.shape[:2]
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    drawing = Drawing(units=options.units, scale=options.scale)
    drawing.width = width / up * options.scale
    drawing.height = height / up * options.scale
    drawing.diagnose("image", f"working resolution {width}x{height} px, upscale x{up:g}")

    half_window = max(1, int(round(stroke_px * 0.75)))
    step = max(options.sample_step_px * up, 1.0)

    # --- straight segments -------------------------------------------------
    # Measure first, chain second, classify last: a single dash of a dashed
    # line has no interior gap and would otherwise be read as a solid line.
    segments = detect_segments(binary, options, up, stroke_px)
    search = max(3, int(round(stroke_px * 1.5)))
    measured: list[_Segment] = []
    dropped_low_ink = 0
    for segment in segments:
        p1, p2, ink_fraction = recentre_on_ink(binary, segment.p1, segment.p2, search, step)
        if ink_fraction < 0.12:
            # An edge with no ink under it is not a line.
            dropped_low_ink += 1
            continue
        width_px = measure_width(distance, p1, p2, step, window=1)
        measured.append(_Segment(p1, p2, geometry.dist(p1, p2), width_px))

    chained, chained_fragments = chain_segments(measured, options, stroke_px, up)

    line_entities: list[Line] = []
    linetype_counts: dict[str, int] = {}
    for segment in chained:
        # The sampling window comes from *this* stroke, not the global estimate:
        # a fine dashed pattern chained next to a thick outline must not have its
        # gaps filled by an over-wide probe.
        width_px = segment.width
        half_window_entity = int(min(4, max(1, round(width_px * 0.6))))
        pattern = sample_occupancy(binary, segment.p1, segment.p2, step, half_window_entity)
        verdict = linetype.classify_linetype(
            pattern,
            min_run_px=max(1, int(options.min_run_px * up)),
            scale=options.scale / up,
        )
        lineweight, width_units, is_thick = linetype.classify_lineweight(width_px / up, options.scale)
        layer_name, role, _color = linetype.layer_for(verdict.linetype, is_thick)
        linetype_counts[verdict.linetype] = linetype_counts.get(verdict.linetype, 0) + 1

        start = _to_drawing(segment.p1, up, height, options)
        end = _to_drawing(segment.p2, up, height, options)
        entity = Line(start=start, end=end, layer=layer_name, role=role, confidence=verdict.confidence)
        entity.verdict = verdict  # type: ignore[attr-defined]
        entity.width_units = width_units  # type: ignore[attr-defined]
        entity.lineweight = lineweight  # type: ignore[attr-defined]
        line_entities.append(entity)

    # --- circles and arcs --------------------------------------------------
    rounds, round_counters = detect_rounds(binary, options, up)
    hough_rounds = detect_rounds_hough(gray, binary, options, up)
    round_counters["hough_circles"] = len(hough_rounds)
    rounds.extend(hough_rounds)
    rounds, merge_counters = merge_rounds(rounds, options, up, stroke_px)
    line_entities, chords_dropped = drop_chords_of_rounds(
        line_entities, rounds, up, height, options
    )
    for item in rounds:
        pattern = sample_occupancy_arc(
            binary, item.center, item.radius, item.start_angle, item.span, step, max(2, half_window)
        )
        verdict = linetype.classify_linetype(
            pattern,
            min_run_px=max(1, int(options.min_run_px * up)),
            scale=options.scale / up,
        )
        radial_width = _round_width(distance, item)
        lineweight, width_units, is_thick = linetype.classify_lineweight(radial_width / up, options.scale)
        layer_name, role, _color = linetype.layer_for(verdict.linetype, is_thick)
        linetype_counts[verdict.linetype] = linetype_counts.get(verdict.linetype, 0) + 1

        center = _to_drawing(item.center, up, height, options)
        if item.closed:
            entity: Arc | Circle = Circle(
                center=center,
                radius=item.radius / up * options.scale,
                layer=layer_name,
                role=role,
                confidence=verdict.confidence,
            )
            entity.fit_rms = item.fit_rms / up * options.scale
        else:
            # Image Y points down and drawing Y points up, so an angle θ in the
            # image becomes -θ in the drawing. The image arc sweeps CCW from
            # start_angle to end_angle, which after mirroring runs clockwise —
            # so the DXF (CCW) arc starts at the mirrored *end* angle.
            start = (-item.end_angle) % 360.0
            end = (-item.start_angle) % 360.0
            entity = Arc(
                center=center,
                radius=item.radius / up * options.scale,
                start_angle=start,
                end_angle=end,
                fit_rms=item.fit_rms / up * options.scale,
                layer=layer_name,
                role=role,
                confidence=verdict.confidence,
            )
        entity.verdict = verdict  # type: ignore[attr-defined]
        entity.width_units = width_units  # type: ignore[attr-defined]
        entity.lineweight = lineweight  # type: ignore[attr-defined]
        drawing.add(entity)

    # --- welding: make arcs meet the lines they belong to ------------------
    weld_counters = weld_and_snap(drawing, line_entities, options)

    for entity in line_entities:
        drawing.add(entity)

    _build_layers(drawing)
    drawing.notes["vectorize"] = {
        "upscale": up,
        "stroke_width_px": round(stroke_px, 2),
        "segments": len(line_entities),
        "rounds": len(rounds),
        "linetypes": linetype_counts,
    }
    drawing.diagnose("vectorize", f"{len(line_entities)} lines, {len(rounds)} circles/arcs detected")

    counters = {
        "upscale": up,
        "stroke_width_px": round(stroke_px, 2),
        "segments_raw_merged": len(segments),
        "dropped_low_ink": dropped_low_ink,
        "chained_fragments": chained_fragments,
        "chords_dropped": chords_dropped,
        "linetypes": linetype_counts,
        **round_counters,
        **merge_counters,
        **weld_counters,
    }
    warnings: list[str] = []
    if stroke_px / up < 1.5:
        warnings.append("very thin strokes: results improve at higher resolution")
    return VectorizeResult(
        drawing=drawing,
        binary=binary,
        upscale=up,
        stroke_width_px=stroke_px,
        counters=counters,
        warnings=warnings,
    )


def _round_width(distance: np.ndarray, item: _Round) -> float:
    """Stroke width of a circular stroke, sampled radially at several angles."""
    height, width = distance.shape[:2]
    samples: list[float] = []
    for i in range(12):
        angle = math.radians(item.start_angle + item.span * (i / 11.0 if item.span else 0.0))
        for offset in range(-3, 4):
            sx = int(round(item.center[0] + math.cos(angle) * (item.radius + offset)))
            sy = int(round(item.center[1] + math.sin(angle) * (item.radius + offset)))
            if 0 <= sx < width and 0 <= sy < height:
                value = float(distance[sy, sx])
                if value > 0:
                    samples.append(value)
    if not samples:
        return 1.0
    return float(max(1.0, 2.0 * float(np.median(samples))))


def _to_drawing(point: Point, up: float, height: int, options: VectorizeOptions) -> Point:
    """Pixel (Y down) → drawing units (Y up), with optional origin offset."""
    x = point[0] / up * options.scale + options.origin[0]
    y = (height - point[1]) / up * options.scale + options.origin[1]
    return (x, y)


def _build_layers(drawing: Drawing) -> None:
    """Materialise the layers implied by the entity set."""
    seen: dict[str, Layer] = {}
    for entity in drawing.entities:
        linetype_name = getattr(getattr(entity, "verdict", None), "linetype", "CONTINUOUS")
        lineweight = int(getattr(entity, "lineweight", 25))
        layer = seen.get(entity.layer)
        if layer is None:
            _name, _role, color = linetype.layer_for(linetype_name, lineweight >= 35)
            seen[entity.layer] = Layer(
                name=entity.layer, color=color, linetype=linetype_name, lineweight=lineweight
            )
        else:
            layer.lineweight = max(layer.lineweight, lineweight)
            if layer.linetype == "CONTINUOUS" and linetype_name != "CONTINUOUS":
                layer.linetype = linetype_name
    for layer in sorted(seen.values(), key=lambda item: item.name):
        drawing.ensure_layer(layer)


def weld_and_snap(drawing: Drawing, lines: list[Line], options: VectorizeOptions) -> dict:
    """Weld near-coincident line endpoints, then snap arcs onto those vertices.

    This is the fix for "the arc does not match its two points": line endpoints
    are clustered under a tolerance, and every arc endpoint within
    ``snap_arc_tol_px`` of a vertex has its angle recomputed *from the centre to
    that vertex*, so the drawn arc terminates exactly on the junction.
    """
    tolerance = options.weld_tol_px * options.scale
    endpoints: list[Point] = []
    for line in lines:
        endpoints.append(line.start)
        endpoints.append(line.end)

    clusters: list[list[int]] = []
    for index, point in enumerate(endpoints):
        for cluster in clusters:
            if geometry.dist(endpoints[cluster[0]], point) <= tolerance:
                cluster.append(index)
                break
        else:
            clusters.append([index])

    welded = 0
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        mx = sum(endpoints[i][0] for i in cluster) / len(cluster)
        my = sum(endpoints[i][1] for i in cluster) / len(cluster)
        for i in cluster:
            if geometry.dist(endpoints[i], (mx, my)) > 1e-9:
                welded += 1
            endpoints[i] = (mx, my)
    vertices = sorted({endpoints[i] for i in range(len(endpoints))})
    for index, line in enumerate(lines):
        line.start = endpoints[index * 2]
        line.end = endpoints[index * 2 + 1]

    snap_tolerance = options.snap_arc_tol_px * options.scale
    snapped = 0
    for entity in drawing.entities:
        if not isinstance(entity, Arc):
            continue
        for attribute in ("start_angle", "end_angle"):
            angle = getattr(entity, attribute)
            point = geometry.point_on_circle(entity.center, entity.radius, angle)
            nearest = _nearest_vertex(point, vertices, snap_tolerance)
            if nearest is None:
                continue
            new_angle = geometry.angle_of(entity.center, nearest)
            if abs(new_angle - angle) > 1e-6:
                setattr(entity, attribute, new_angle)
                snapped += 1

    dangling = _count_dangling(endpoints, tolerance)
    drawing.notes["welding"] = {
        "tolerance": round(tolerance, 4),
        "welded_endpoints": welded,
        "arc_endpoints_snapped": snapped,
        "dangling_endpoints": dangling,
    }
    if snapped:
        drawing.diagnose("arc-snap", f"{snapped} arc endpoint(s) snapped onto line vertices")
    if welded:
        drawing.diagnose("weld", f"{welded} near-coincident endpoint(s) welded")
    return {"welded_endpoints": welded, "arc_endpoints_snapped": snapped, "dangling_endpoints": dangling}


def _nearest_vertex(point: Point, vertices: list[Point], tolerance: float) -> Point | None:
    best: Point | None = None
    best_distance = tolerance
    for vertex in vertices:
        distance = geometry.dist(point, vertex)
        if distance <= best_distance:
            best = vertex
            best_distance = distance
    return best


def _count_dangling(endpoints: list[Point], tolerance: float) -> int:
    """Endpoints that share their position with no other endpoint."""
    dangling = 0
    for index, point in enumerate(endpoints):
        if not any(
            other != index and geometry.dist(point, candidate) <= tolerance
            for other, candidate in enumerate(endpoints)
        ):
            dangling += 1
    return dangling
