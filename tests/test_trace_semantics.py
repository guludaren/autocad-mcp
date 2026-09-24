"""Semantic-layer tests: what the LLM may and may not decide.

The governing rule is that geometry is a measured fact and the model only
labels it. These tests pin that down: the digest contains no pixel data, the
validator drops invented entity indices, and the pipeline still produces a
labelled drawing when there is no API key at all.
"""

from __future__ import annotations

from autocad_mcp.trace.deepseek import DeepSeekClient, _extract_json
from autocad_mcp.trace.ir import Arc, Drawing, Line
from autocad_mcp.trace.semantics import (
    apply_semantics,
    build_digest,
    parse_semantic_response,
    semantic_pass,
)


def _sample_drawing() -> Drawing:
    drawing = Drawing(units="mm", scale=0.25, width=225.0, height=175.0)
    drawing.add(Line(start=(0.0, 0.0), end=(200.0, 0.0), layer="Thick", role="outline"))
    drawing.add(Line(start=(200.0, 0.0), end=(200.0, 100.0), layer="Hidden", role="hidden"))
    drawing.add(
        Arc(
            center=(50.0, 50.0),
            radius=20.0,
            start_angle=0.0,
            end_angle=90.0,
            layer="Center",
            role="center",
        )
    )
    return drawing


def test_client_reports_unavailable_without_a_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("AUTOCAD_MCP_DEEPSEEK_API_KEY", raising=False)
    client = DeepSeekClient(api_key="")
    assert client.available is False
    assert client.base_url == "https://api.deepseek.com"
    assert client.model == "deepseek-chat"


def test_digest_carries_geometry_but_no_pixels():
    drawing = _sample_drawing()
    digest = build_digest(drawing)
    assert digest["units"] == "mm"
    assert digest["stats"]["by_kind"] == {"LINE": 2, "ARC": 1}
    assert len(digest["entities"]) == 3
    assert digest["entities"][0]["kind"] == "LINE"
    assert "start" in digest["entities"][0]
    # No raster data may ever leak into the prompt.
    assert "image" not in digest
    assert "binary" not in digest


def test_semantic_pass_degrades_to_the_deterministic_labelling(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("AUTOCAD_MCP_DEEPSEEK_API_KEY", raising=False)
    drawing = _sample_drawing()

    semantic = semantic_pass(drawing, DeepSeekClient(api_key=""))

    assert semantic["source"] == "deterministic"
    assert semantic["layers"], "the deterministic grouping must still be offered"
    names = {layer["name"] for layer in semantic["layers"]}
    assert names == {"Thick", "Hidden", "Center"}
    assert drawing.notes["semantic"]["source"] == "deterministic"


def test_parse_drops_invented_and_duplicate_indices():
    drawing = _sample_drawing()
    payload = {
        "drawing_type": "mechanical part",
        "summary": "a bracket outline",
        "layers": [
            {"name": "outline", "entities": [0, 1, 99, "x"], "reason": "outer boundary"},
            {"name": "centre lines", "entities": [2, 1], "reason": "symmetry"},
            {"name": "bad layer", "entities": [99], "reason": "nothing valid"},
        ],
        "warnings": ["maybe a missing fillet"],
    }

    parsed = parse_semantic_response(payload, drawing)

    assert parsed["layers"][0]["entities"] == [0, 1]
    assert parsed["layers"][1]["entities"] == [2]  # 1 was already assigned
    assert len(parsed["layers"]) == 2  # the all-invalid layer was dropped
    assert parsed["rejected"]["unknown_index"] == 3  # 99, "x", and 99 again
    assert parsed["rejected"]["duplicate_index"] == 1
    assert parsed["rejected"]["empty_layer"] == 1
    assert parsed["unassigned"] == 0


def test_layer_names_are_sanitised():
    drawing = _sample_drawing()
    payload = {"layers": [{"name": "walls/doors:*?", "entities": [0]}]}
    parsed = parse_semantic_response(payload, drawing)
    assert parsed["layers"][0]["name"] == "wallsdoors"


def test_long_layer_names_are_truncated():
    drawing = _sample_drawing()
    payload = {"layers": [{"name": "x" * 80, "entities": [0]}]}
    parsed = parse_semantic_response(payload, drawing)
    assert len(parsed["layers"][0]["name"]) <= 32


def test_apply_semantics_renames_layers_and_keeps_linetypes():
    drawing = _sample_drawing()
    semantic = parse_semantic_response(
        {"layers": [{"name": "outer walls", "entities": [0, 1], "reason": "boundary"}]},
        drawing,
    )
    apply_semantics(drawing, semantic)

    assert drawing.entities[0].layer == "outer walls"
    assert drawing.entities[1].layer == "outer walls"
    assert semantic["renamed_entities"] == 2
    # A semantic rename must not rewrite measured geometry facts.
    assert drawing.entities[0].role == "outline"
    assert drawing.entities[0].kind == "LINE"
    assert drawing.entities[2].role == "center"  # untouched entity keeps its role
    assert any(layer.name == "outer walls" for layer in drawing.layers)


def test_extract_json_tolerates_fences_and_prose():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('Sure! {"a": 2} hope that helps') == {"a": 2}
