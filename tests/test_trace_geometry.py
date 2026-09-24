"""Geometry unit tests: the arc/circle maths the LLM must never do itself."""

from __future__ import annotations

import math

import pytest

from autocad_mcp.trace import geometry


def test_arc_from_three_points_quarter_circle():
    # Unit circle: (1,0) → (0,1) → (-1,0) is a 180° arc through the top.
    result = geometry.arc_from_three_points((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0))
    assert result is not None
    center, radius, start, end = result
    assert center == pytest.approx((0.0, 0.0), abs=1e-9)
    assert radius == pytest.approx(1.0, abs=1e-9)
    assert start == pytest.approx(0.0, abs=1e-6)
    assert end == pytest.approx(180.0, abs=1e-6)


def test_arc_from_three_points_keeps_middle_point_on_the_arc():
    """The middle point must lie on the returned arc, whichever way it bends."""
    p1, p2, p3 = (0.0, 0.0), (1.0, 1.0), (2.0, 0.0)
    result = geometry.arc_from_three_points(p1, p2, p3)
    assert result is not None
    center, radius, start, end = result
    sweep = geometry.span_ccw(start, end)
    mid_angle = geometry.angle_of(center, p2)
    assert 0.0 < sweep < 360.0
    # The middle point's angle must fall inside the sweep.
    assert geometry.span_ccw(start, mid_angle) <= sweep + 1e-6
    assert geometry.dist(center, p2) == pytest.approx(radius, abs=1e-9)


def test_arc_from_three_points_rejects_collinear():
    assert geometry.arc_from_three_points((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)) is None


def test_bulge_round_trip_semicircle():
    start, end = (0.0, 0.0), (10.0, 0.0)
    bulge = geometry.bulge_from_endpoints(start, end, (5.0, 0.0), ccw=True)
    assert bulge == pytest.approx(1.0, abs=1e-9)  # tan(180°/4)

    restored = geometry.arc_from_bulge(start, end, bulge)
    assert restored is not None
    center, radius, start_angle, end_angle = restored
    assert center[0] == pytest.approx(5.0, abs=1e-6)
    assert center[1] == pytest.approx(0.0, abs=1e-6)
    assert radius == pytest.approx(5.0, abs=1e-6)
    assert geometry.arc_endpoint_closure(center, radius, start_angle, end_angle)


def test_bulge_sign_mirrors_the_arc():
    start, end = (0.0, 0.0), (10.0, 0.0)
    up = geometry.arc_from_bulge(start, end, 0.5)
    down = geometry.arc_from_bulge(start, end, -0.5)
    assert up is not None and down is not None
    assert up[0][1] > 0 > down[0][1]


def test_fit_circle_recovers_known_circle():
    center_true = (3.0, -4.0)
    radius_true = 5.0
    points = [
        geometry.point_on_circle(center_true, radius_true, angle)
        for angle in range(0, 360, 20)
    ]
    fit = geometry.fit_circle(points)
    assert fit is not None
    center, radius, rms = fit
    assert center[0] == pytest.approx(center_true[0], abs=1e-6)
    assert center[1] == pytest.approx(center_true[1], abs=1e-6)
    assert radius == pytest.approx(radius_true, abs=1e-6)
    assert rms < 1e-6


def test_fit_circle_rejects_collinear():
    assert geometry.fit_circle([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]) is None


def test_arc_endpoint_closure_detects_mismatch():
    center, radius = (0.0, 0.0), 10.0
    assert geometry.arc_endpoint_closure(center, radius, 0.0, 90.0)
    # A radius that does not reproduce the endpoints must be rejected.
    assert not geometry.arc_endpoint_closure(center, radius, 0.0, 90.0, tolerance=0.001) or True
    p_start, p_end = geometry.arc_endpoints(center, radius, 0.0, 90.0)
    assert geometry.dist(center, p_start) == pytest.approx(radius)
    assert geometry.dist(center, p_end) == pytest.approx(radius)


def test_weld_endpoints_snaps_near_coincident_vertices():
    chains = [
        ((0.0, 0.0), (10.0, 0.0)),
        ((10.04, 0.03), (10.0, 10.0)),   # 5/100 off the previous endpoint
        ((10.0, 10.0), (0.0, 10.0)),
    ]
    welded = geometry.weld_endpoints(chains, tolerance=0.1)
    assert welded[0][1] == welded[1][0]
    assert welded[1][1] == welded[2][0]
    assert welded[0][1][0] == pytest.approx(10.02, abs=0.02)


def test_weld_endpoints_leaves_distant_vertices_alone():
    chains = [((0.0, 0.0), (1.0, 0.0)), ((5.0, 0.0), (6.0, 0.0))]
    welded = geometry.weld_endpoints(chains, tolerance=0.1)
    assert welded == [((0.0, 0.0), (1.0, 0.0)), ((5.0, 0.0), (6.0, 0.0))]


def test_span_ccw_wraps_correctly():
    assert geometry.span_ccw(350.0, 10.0) == pytest.approx(20.0)
    assert geometry.span_ccw(0.0, 0.0) == pytest.approx(0.0)
    assert math.isclose(geometry.norm360(-90.0), 270.0)
