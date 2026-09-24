"""End-to-end tracing tests: synthetic drawing → IR → DXF.

The image is generated here rather than committed as a fixture so the expected
geometry is exact, and so the test documents what the vectoriser can and cannot
do: solid outlines, dashed/hidden/centre lines, a circle and an arc.
"""

from __future__ import annotations

import math

import cv2
import ezdxf
import numpy as np
import pytest

from autocad_mcp.trace import geometry
from autocad_mcp.trace.ir import Arc, Drawing, Line
from autocad_mcp.trace.pipeline import TraceOptions, trace_image
from autocad_mcp.trace.vectorize import VectorizeOptions, weld_and_snap

SCALE = 0.25  # drawing units (mm) per pixel


def _draw_dashed_line(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    pattern: list[tuple[int, int]],
    thickness: int,
) -> None:
    """Draw a line with an explicit (ink, gap) pixel pattern."""
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    position = 0.0
    index = 0
    while position < length:
        ink, gap = pattern[index % len(pattern)]
        segment = min(ink, length - position)
        ax, ay = x1 + ux * position, y1 + uy * position
        bx, by = x1 + ux * (position + segment), y1 + uy * (position + segment)
        cv2.line(image, (int(round(ax)), int(round(ay))), (int(round(bx)), int(round(by))), 0, thickness)
        position += ink + gap
        index += 1


def build_drawing(path) -> dict:
    """A small drawing with every line style this project classifies."""
    image = np.full((700, 900), 255, dtype=np.uint8)

    # Thick solid outline (a rectangle).
    cv2.rectangle(image, (80, 80), (820, 620), 0, 6)
    # Dashed line: 40 px ink / 20 px gap → 15 drawing units of period.
    _draw_dashed_line(image, (120, 560), (780, 560), [(40, 20)], 3)
    # Hidden line: 14 / 10 → 6 drawing units of period. The gaps have to clear
    # OpenCV's round line caps, which would otherwise fill a 4 px gap and make
    # the line read as solid.
    _draw_dashed_line(image, (120, 508), (780, 508), [(14, 10)], 2)
    # Centre line: long-short alternation.
    _draw_dashed_line(image, (120, 466), (780, 466), [(60, 10), (16, 10)], 2)
    # Thin circle and a thin quarter arc.
    cv2.circle(image, (250, 250), 90, 0, 2)
    cv2.ellipse(image, (650, 250), (100, 100), 0, 0, 90, 0, 2)

    cv2.imwrite(str(path), image)
    return {
        "path": str(path),
        "rect": (80, 80, 820, 620),
        "circle_center_px": (250, 250),
        "circle_radius_px": 90,
        "arc_center_px": (650, 250),
        "arc_radius_px": 100,
    }


@pytest.fixture(scope="module")
def traced(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("trace")
    image = tmp / "drawing.png"
    build_drawing(image)
    options = TraceOptions(
        vectorize=VectorizeOptions(
            scale=SCALE,
            units="mm",
            auto_upscale=False,  # deterministic pixel counts for assertions
            min_line_px=20.0,
        ),
        use_llm=False,  # offline: geometry must stand on its own
        dxf_path=str(tmp / "drawing.dxf"),
        preview_path=str(tmp / "drawing_preview.png"),
    )
    result = trace_image(image, options)
    return result


def test_trace_succeeds_and_writes_files(traced):
    assert traced.ok, traced.error
    assert traced.dxf_path and traced.dxf_path.endswith(".dxf")
    assert traced.preview_path
    from pathlib import Path

    assert Path(traced.dxf_path).exists()
    assert Path(traced.preview_path).exists()


def test_entities_are_extracted_completely(traced):
    """The failure mode being fixed: an incomplete read of the drawing."""
    kinds = traced.ir.stats()["by_kind"]
    assert kinds.get("LINE", 0) >= 5, kinds
    assert kinds.get("CIRCLE", 0) >= 1, kinds
    assert kinds.get("ARC", 0) >= 1, kinds


def test_every_linetype_is_distinguished(traced):
    """A model that 'only computes coordinates' cannot do this part.

    Asserted on the *measured* linetype of each entity, because a dashed line
    deliberately shares the conventional ``Hidden`` layer with hidden lines and
    carries its own linetype override in the DXF.
    """
    measured = set(traced.counters["linetypes"])
    assert {"CONTINUOUS", "DASHED", "HIDDEN", "CENTER"} <= measured, measured


def test_detected_geometry_lands_on_conventional_layers(traced):
    layers = {layer.name for layer in traced.ir.layers}
    assert {"Thick", "Hidden", "Center"} <= layers, layers


def test_circle_geometry_is_close_to_truth(traced):
    circle_entities = [e for e in traced.ir.entities if e.kind == "CIRCLE"]
    assert circle_entities
    # Image Y is flipped into drawing space: y_d = (height - y_px) * scale.
    expected_x = 250 * SCALE
    expected_y = (700 - 250) * SCALE
    expected_radius = 90 * SCALE
    best = min(
        circle_entities,
        key=lambda e: abs(e.center[0] - expected_x) + abs(e.center[1] - expected_y),
    )
    assert best.center[0] == pytest.approx(expected_x, abs=6 * SCALE)
    assert best.center[1] == pytest.approx(expected_y, abs=6 * SCALE)
    assert best.radius == pytest.approx(expected_radius, rel=0.15)


def test_arc_endpoints_close_on_their_circle(traced):
    """The other classic failure: arc angles that do not match two points."""
    arcs = [entity for entity in traced.ir.entities if isinstance(entity, Arc)]
    assert arcs
    for arc in arcs:
        assert geometry.arc_endpoint_closure(
            arc.center, arc.radius, arc.start_angle, arc.end_angle, tolerance=max(0.05, SCALE * 3)
        ), arc.to_dict()


def test_dxf_is_readable_and_carries_linetypes(traced):
    doc = ezdxf.readfile(traced.dxf_path)
    modelspace = doc.modelspace()
    kinds = {}
    for entity in modelspace:
        kinds[entity.dxftype()] = kinds.get(entity.dxftype(), 0) + 1
    assert kinds, "DXF has no entities"
    assert kinds.get("LINE", 0) >= 5, kinds
    assert kinds.get("CIRCLE", 0) >= 1, kinds

    for layer in traced.ir.layers:
        assert layer.name in doc.layers
        assert doc.layers.get(layer.name).dxf.linetype == layer.linetype


def test_dashed_line_survives_as_an_entity_level_linetype(traced):
    """The dashed line shares the Hidden layer, so its style rides the entity."""
    doc = ezdxf.readfile(traced.dxf_path)
    entity_linetypes = {
        entity.dxf.get("linetype", None)
        for entity in doc.modelspace()
        if entity.dxftype() in ("LINE", "ARC", "CIRCLE")
    }
    assert "DASHED" in entity_linetypes, entity_linetypes


def test_no_single_dash_is_left_behind_as_a_solid_line(traced):
    """A lone dash would be classified CONTINUOUS — chaining must prevent that."""
    solid = [e for e in traced.ir.entities if e.kind == "LINE" and e.role == "outline"]
    dashed_ish = [e for e in traced.ir.entities if e.kind == "LINE" and e.role == "hidden"]
    # the three patterned lines are grouped, not shattered into their dashes
    assert len(dashed_ish) <= 3, [e.to_dict() for e in dashed_ish]
    assert len(solid) <= 6, [e.to_dict() for e in solid]


def test_report_is_serialisable(traced):
    payload = traced.to_dict(include_ir=False, include_entities=False)
    assert payload["ok"] is True
    assert payload["stats"]["entities"] == len(traced.ir.entities)
    assert isinstance(payload["counters"], dict)
    import json

    json.dumps(payload)  # must not raise


# ---------------------------------------------------------------------------
# The targeted fix: arcs must meet the points they were asked to join
# ---------------------------------------------------------------------------


def test_weld_and_snap_pulls_an_arc_endpoint_onto_the_vertex():
    drawing = Drawing(units="mm", scale=1.0)
    line = Line(start=(0.0, 0.0), end=(10.0, 0.0), layer="Thin")
    # Arc centred (10,10) r=10: the true junction is (10,0), i.e. 270°.
    # Start the arc 5° off so its endpoint misses the line by ~0.87 units.
    arc = Arc(
        center=(10.0, 10.0),
        radius=10.0,
        start_angle=275.0,
        end_angle=90.0,
        layer="Thin",
    )
    drawing.add(arc)

    counters = weld_and_snap(drawing, [line], VectorizeOptions(scale=1.0, snap_arc_tol_px=6.0))

    assert counters["arc_endpoints_snapped"] == 1
    assert arc.start_angle == pytest.approx(270.0, abs=0.5)
    endpoint, _other = geometry.arc_endpoints(arc.center, arc.radius, arc.start_angle, arc.end_angle)
    assert geometry.dist(endpoint, (10.0, 0.0)) < 1e-6


def test_weld_and_snap_welds_near_coincident_line_ends():
    drawing = Drawing(units="mm", scale=1.0)
    first = Line(start=(0.0, 0.0), end=(10.0, 0.0), layer="Thin")
    second = Line(start=(10.05, 0.04), end=(10.0, 10.0), layer="Thin")

    counters = weld_and_snap(drawing, [first, second], VectorizeOptions(scale=1.0, weld_tol_px=3.5))

    assert counters["welded_endpoints"] >= 2
    assert geometry.dist(first.end, second.start) < 1e-9


def test_weld_and_snap_leaves_a_genuine_gap_alone():
    drawing = Drawing(units="mm", scale=1.0)
    first = Line(start=(0.0, 0.0), end=(10.0, 0.0), layer="Thin")
    second = Line(start=(25.0, 0.0), end=(35.0, 0.0), layer="Thin")

    counters = weld_and_snap(drawing, [first, second], VectorizeOptions(scale=1.0, weld_tol_px=3.5))

    assert counters["welded_endpoints"] == 0
    assert first.end == (10.0, 0.0)
    assert second.start == (25.0, 0.0)
