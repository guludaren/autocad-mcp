"""Arc / circle geometry: the part an LLM gets wrong and we do deterministically.

Why this module exists
----------------------
Asking a language model to emit CAD *commands* for arcs produces arcs whose
endpoints do not meet the neighbouring lines: a model that only "computes
coordinates" will happily write an ARC whose start/end angles are unrelated to
the two points it was asked to connect. Every conversion here is therefore
closed-form and testable:

* three points on a circle  → centre, radius, start/end angle
* two endpoints + bulge     → the same canonical form (and back)
* point cloud               → algebraic (Kasa) circle fit with an RMS residual
* near-coincident endpoints → welded under a tolerance

Angles are degrees, counter-clockwise (CCW) from +X, matching DXF group codes
50/51 and ``ezdxf``.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

Point = tuple[float, float]

EPS = 1e-9


# ---------------------------------------------------------------------------
# Basic vector helpers
# ---------------------------------------------------------------------------


def dist(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def angle_of(center: Point, point: Point) -> float:
    """Degrees CCW from +X of the vector centre → point, in [0, 360)."""
    return math.degrees(math.atan2(point[1] - center[1], point[0] - center[0])) % 360.0


def norm360(angle: float) -> float:
    return angle % 360.0


def span_ccw(start_angle: float, end_angle: float) -> float:
    """CCW sweep from start to end, in [0, 360)."""
    return (end_angle - start_angle) % 360.0


def point_on_circle(center: Point, radius: float, angle_deg: float) -> Point:
    rad = math.radians(angle_deg)
    return (center[0] + radius * math.cos(rad), center[1] + radius * math.sin(rad))


def arc_endpoints(center: Point, radius: float, start_angle: float, end_angle: float) -> tuple[Point, Point]:
    """The two endpoints an ARC entity actually draws."""
    return (
        point_on_circle(center, radius, start_angle),
        point_on_circle(center, radius, end_angle),
    )


# ---------------------------------------------------------------------------
# Three points → canonical arc
# ---------------------------------------------------------------------------


def arc_from_three_points(p1: Point, p2: Point, p3: Point) -> tuple[Point, float, float, float] | None:
    """Circle through three points as (centre, radius, start_angle, end_angle).

    The arc runs from ``p1`` to ``p3`` **through** ``p2``; the sweep direction is
    derived from the sign of the cross product so the middle point always lies
    on the returned arc. Returns ``None`` for collinear/degenerate input.
    """
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < EPS:
        return None  # collinear: no unique circle
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    center = (ux, uy)
    radius = dist(center, p1)

    start_angle = angle_of(center, p1)
    end_angle = angle_of(center, p3)

    # Orientation: is p2 on the CCW path p1 → p3?
    cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    if cross < 0:
        # Clockwise through p2 == CCW from p3 to p1: swap the roles.
        start_angle, end_angle = end_angle, start_angle
    return center, radius, norm360(start_angle), norm360(end_angle)


# ---------------------------------------------------------------------------
# Bulge ↔ arc  (bulge = tan(sweep/4), the DXF polyline arc encoding)
# ---------------------------------------------------------------------------


def bulge_from_endpoints(start: Point, end: Point, center: Point, ccw: bool = True) -> float:
    """Bulge value for the arc from ``start`` to ``end`` around ``center``."""
    radius = dist(center, start)
    if radius < EPS:
        return 0.0
    start_angle = angle_of(center, start)
    end_angle = angle_of(center, end)
    sweep = span_ccw(start_angle, end_angle) if ccw else -span_ccw(end_angle, start_angle)
    if abs(sweep) < EPS:
        sweep = 360.0 if ccw else -360.0
    return math.tan(math.radians(sweep) / 4.0)


def arc_from_bulge(start: Point, end: Point, bulge: float) -> tuple[Point, float, float, float] | None:
    """Inverse of :func:`bulge_from_endpoints` — (centre, radius, start°, end°)."""
    if abs(bulge) < EPS:
        return None
    chord = dist(start, end)
    if chord < EPS:
        return None
    sweep = 4.0 * math.atan(bulge)  # radians, signed
    radius = abs(chord / (2.0 * math.sin(sweep / 2.0)))
    # Sagitta from the chord midpoint towards the centre.
    mid = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
    half = chord / 2.0
    height = math.sqrt(max(radius * radius - half * half, 0.0))
    # Direction: left normal of start→end, flipped for negative (CW) bulges.
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    nx, ny = -dy / length, dx / length
    sign = 1.0 if bulge > 0 else -1.0
    if abs(sweep) > math.pi:  # major arc: centre sits on the other side
        sign = -sign
    center = (mid[0] + sign * nx * height, mid[1] + sign * ny * height)

    start_angle = angle_of(center, start)
    end_angle = angle_of(center, end)
    if bulge < 0:
        start_angle, end_angle = end_angle, start_angle
    return center, radius, norm360(start_angle), norm360(end_angle)


# ---------------------------------------------------------------------------
# Least-squares circle fit (Kasa) — no scipy dependency
# ---------------------------------------------------------------------------


def fit_circle(points: Sequence[Point]) -> tuple[Point, float, float] | None:
    """Algebraic circle fit → (centre, radius, rms residual).

    ``None`` when there are fewer than three distinct points or the system is
    singular (collinear input).
    """
    pts = [(float(x), float(y)) for x, y in points]
    n = len(pts)
    if n < 3:
        return None
    mean_x = sum(p[0] for p in pts) / n
    mean_y = sum(p[1] for p in pts) / n

    suu = suv = svv = suuu = svvv = suvv = svuu = 0.0
    for x, y in pts:
        u = x - mean_x
        v = y - mean_y
        suu += u * u
        svv += v * v
        suv += u * v
        suuu += u * u * u
        svvv += v * v * v
        suvv += u * v * v
        svuu += v * u * u

    det = suu * svv - suv * suv
    if abs(det) < EPS:
        return None
    # Solve for the centre offset (uc, vc).
    uc = (svv * (suuu + suvv) - suv * (svvv + svuu)) / (2.0 * det)
    vc = (suu * (svvv + svuu) - suv * (suuu + suvv)) / (2.0 * det)
    center = (uc + mean_x, vc + mean_y)
    radius = math.sqrt(uc * uc + vc * vc + (suu + svv) / n)

    residuals = [dist(center, p) - radius for p in pts]
    rms = math.sqrt(sum(r * r for r in residuals) / n)
    return center, radius, rms


# ---------------------------------------------------------------------------
# Endpoint welding — "the arc does not meet its two points"
# ---------------------------------------------------------------------------


def arc_endpoint_closure(
    center: Point, radius: float, start_angle: float, end_angle: float,
    tolerance: float = 0.05,
) -> bool:
    """True when an ARC's drawn endpoints actually coincide with its angles.

    A guard against the classic failure mode: angles and radius that do not
    reproduce the two points the arc was supposed to join.
    """
    p_start, p_end = arc_endpoints(center, radius, start_angle, end_angle)
    for point in (p_start, p_end):
        if abs(dist(center, point) - radius) > tolerance:
            return False
    return True


def weld_endpoints(
    arcs: Iterable[tuple[Point, Point]],
    tolerance: float,
) -> list[tuple[Point, Point]]:
    """Snap near-coincident endpoints in a chain so arcs and lines share vertices.

    Endpoints within ``tolerance`` of each other are replaced by their midpoint,
    which is what turns "almost touching" traced geometry into a closed CAD
    profile. Returns the welded list in input order.
    """
    items = [(tuple(a), tuple(b)) for a, b in arcs]  # type: ignore[misc]
    flat: list[Point] = []
    for a, b in items:
        flat.append(a)
        flat.append(b)

    clusters: list[list[int]] = []
    for i, point in enumerate(flat):
        placed = False
        for cluster in clusters:
            anchor = flat[cluster[0]]
            if dist(anchor, point) <= tolerance:
                cluster.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])

    welded = list(flat)
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        mx = sum(flat[i][0] for i in cluster) / len(cluster)
        my = sum(flat[i][1] for i in cluster) / len(cluster)
        for i in cluster:
            welded[i] = (mx, my)

    out: list[tuple[Point, Point]] = []
    for i in range(0, len(welded), 2):
        out.append((welded[i], welded[i + 1]))
    return out
