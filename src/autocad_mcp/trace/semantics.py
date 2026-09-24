"""The CAD semantic layer: what the LLM is actually good for.

The division of labour in this module is the whole point of the design:

* **geometry** — extracted by OpenCV, never by the model (no hallucinated
  lines, no missed segments, no guessing whether a line is dashed),
* **semantics** — naming, grouping and describing, done by DeepSeek over a
  *text digest* of that geometry.

The digest contains entity indices, kinds, rounded coordinates, the
deterministically detected linetype and the vectoriser's own confidence
numbers. It deliberately contains **no pixels**, because DeepSeek has no vision
endpoint: asking it to "look at the drawing" is what produced incomplete
traces in the first place.

Everything degrades gracefully. With no API key, or on any model/network
failure, the deterministic classifier's own layer/role assignment stands and
the result is still a valid, layered DXF.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from autocad_mcp.trace.deepseek import DeepSeekClient, LLMError
from autocad_mcp.trace.ir import Drawing, Entity

log = structlog.get_logger()

#: Entities included verbatim in the digest; the rest are summarised.
MAX_ENTITIES_IN_DIGEST = 240

#: Longest accepted semantic layer name.
MAX_LAYER_NAME = 32

SYSTEM_PROMPT = """\
You are a CAD drafting assistant. You receive a JSON digest describing geometry \
that has ALREADY been extracted from a raster drawing by a deterministic image \
pipeline. You never see the image, and you must not speculate about pixels.

Your job is purely semantic: name layers the way a drafter would, group the \
given entities into those layers, and point out anything suspicious.

Constraints:
- Use ONLY the entity indices present in the digest. Never invent indices.
- Geometry, linetypes (CONTINUOUS/DASHED/HIDDEN/CENTER/PHANTOM) and coordinates \
are facts measured from the image. Do not contradict them and do not re-derive \
them; you may only quote them.
- Prefer the drafting conventions: outlines/walls, hidden edges, centre lines, \
thin/detail lines, dimensions and text.
- Report missing or doubtful geometry in "warnings" rather than silently fixing it.
- Answer with a single JSON object and nothing else.
"""

RESPONSE_SCHEMA = """\
{
  "drawing_type": "short label, e.g. mechanical part / floor plan / P&ID",
  "summary": "one or two sentences describing what the drawing shows",
  "layers": [
    {"name": "layer name (<=32 chars, ASCII letters/digits/underscore/space)",
     "entities": [0, 1, 2],
     "reason": "why these belong together"}
  ],
  "warnings": ["anything the extractor probably missed or mis-read"]
}
"""


def build_digest(drawing: Drawing, max_entities: int = MAX_ENTITIES_IN_DIGEST) -> dict:
    """Compact, token-efficient description of the extracted geometry."""
    entities: list[dict] = []
    for entity in drawing.entities[:max_entities]:
        entry: dict[str, Any] = {
            "i": entity.index,
            "kind": entity.kind,
            "role": entity.role,
            "conf": round(float(entity.confidence), 2),
        }
        digest_geometry = entity.to_dict()
        for key in ("start", "end", "center", "radius", "start_angle", "end_angle", "points", "closed"):
            if key in digest_geometry:
                entry[key] = digest_geometry[key]
        verdict = getattr(entity, "verdict", None)
        if verdict is not None:
            entry["linetype"] = verdict.linetype
            entry["duty"] = round(float(verdict.duty), 2)
        entities.append(entry)

    digest: dict[str, Any] = {
        "units": drawing.units,
        "size": {"width": round(drawing.width, 3), "height": round(drawing.height, 3)},
        "scale_units_per_pixel": drawing.scale,
        "extraction": drawing.notes.get("vectorize", {}),
        "welding": drawing.notes.get("welding", {}),
        "stats": drawing.stats(),
        "current_layers": [layer.to_dict() for layer in drawing.layers],
        "entities": entities,
    }
    if len(drawing.entities) > max_entities:
        digest["entities_truncated"] = len(drawing.entities) - max_entities
    return digest


def build_user_prompt(digest: dict) -> str:
    import json

    return (
        "Geometry digest of the traced drawing (facts, already verified):\n"
        f"{json.dumps(digest, separators=(',', ':'), ensure_ascii=False)}\n\n"
        "Group these entities into semantically named layers and describe the "
        f"drawing. Respond with JSON matching this schema exactly:\n{RESPONSE_SCHEMA}"
    )


def _sanitize_layer_name(name: Any, fallback: str) -> str:
    """Keep model-provided layer names safe for DXF and for humans."""
    text = re.sub(r"[^0-9A-Za-z_ \-]", "", str(name or "")).strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return fallback
    return text[:MAX_LAYER_NAME]


def parse_semantic_response(payload: dict, drawing: Drawing) -> dict:
    """Validate a model reply against the drawing; drop anything invented.

    Returns ``{"drawing_type", "summary", "layers": [...], "warnings": [...],
    "rejected": {...}}`` where every entity index is guaranteed to exist and
    each index is assigned at most once.
    """
    valid_indices = {entity.index for entity in drawing.entities}
    assigned: set[int] = set()
    layers: list[dict] = []
    rejected = {"unknown_index": 0, "duplicate_index": 0, "empty_layer": 0}

    raw_layers = payload.get("layers")
    if isinstance(raw_layers, list):
        for position, raw in enumerate(raw_layers):
            if not isinstance(raw, dict):
                continue
            indices: list[int] = []
            for value in raw.get("entities", []) or []:
                if not isinstance(value, int) or isinstance(value, bool):
                    rejected["unknown_index"] += 1
                    continue
                if value not in valid_indices:
                    rejected["unknown_index"] += 1
                    continue
                if value in assigned:
                    rejected["duplicate_index"] += 1
                    continue
                assigned.add(value)
                indices.append(value)
            if not indices:
                rejected["empty_layer"] += 1
                continue
            layers.append(
                {
                    "name": _sanitize_layer_name(raw.get("name"), f"Layer {position + 1}"),
                    "entities": sorted(indices),
                    "reason": str(raw.get("reason", ""))[:200],
                }
            )

    warnings = payload.get("warnings")
    return {
        "drawing_type": str(payload.get("drawing_type", ""))[:80],
        "summary": str(payload.get("summary", ""))[:600],
        "layers": layers,
        "warnings": [str(item)[:300] for item in warnings] if isinstance(warnings, list) else [],
        "assigned": len(assigned),
        "unassigned": len(valid_indices - assigned),
        "rejected": rejected,
    }


def apply_semantics(drawing: Drawing, semantic: dict) -> dict:
    """Move entities onto their semantic layers, preserving detected linetypes.

    Linetype and lineweight stay per-entity when the model groups lines of
    different styles onto one layer, so a semantic rename can never destroy the
    measured line style.
    """
    by_index: dict[int, Entity] = {entity.index: entity for entity in drawing.entities}
    renamed = 0
    for layer in semantic.get("layers", []):
        for index in layer["entities"]:
            entity = by_index.get(index)
            if entity is None:
                continue
            if entity.layer != layer["name"]:
                renamed += 1
            entity.layer = layer["name"]
            entity.semantic_reason = layer.get("reason", "")  # type: ignore[attr-defined]

    # Rebuild the layer table from the (possibly renamed) entities.
    for layer in list(drawing.layers):
        if layer.name not in {entity.layer for entity in drawing.entities}:
            drawing.layers.remove(layer)
    _rebuild_layers(drawing)
    semantic["renamed_entities"] = renamed
    return semantic


def _rebuild_layers(drawing: Drawing) -> None:
    """Recreate layer entries for every layer an entity currently uses.

    A semantic rename must not rewrite geometric facts, so entity ``role`` is
    left untouched; the new layer inherits the line style its entities were
    measured with.
    """
    from autocad_mcp.trace import linetype
    from autocad_mcp.trace.ir import Layer

    existing = {layer.name for layer in drawing.layers}
    for entity in drawing.entities:
        if entity.layer in existing:
            continue
        verdict = getattr(entity, "verdict", None)
        linetype_name = getattr(verdict, "linetype", "CONTINUOUS")
        lineweight = int(getattr(entity, "lineweight", 25))
        _name, _role, color = linetype.layer_for(linetype_name, lineweight >= 35)
        drawing.ensure_layer(
            Layer(name=entity.layer, color=color, linetype=linetype_name, lineweight=lineweight)
        )
        existing.add(entity.layer)


def semantic_pass(drawing: Drawing, client: DeepSeekClient | None = None) -> dict:
    """Run the semantic pass, falling back to the deterministic labelling.

    Never raises: any failure is reported inside the returned dict, because a
    missing LLM must not cost the user a drawing.
    """
    fallback = {
        "source": "deterministic",
        "drawing_type": "",
        "summary": "",
        "layers": [],
        "warnings": [],
        "assigned": 0,
        "unassigned": len(drawing.entities),
        "rejected": {},
    }
    client = client or DeepSeekClient()
    if not client.available:
        fallback["reason"] = "no API key configured (set DEEPSEEK_API_KEY)"
        fallback["layers"] = _deterministic_layers(drawing)
        return _store(drawing, fallback)

    digest = build_digest(drawing)
    try:
        payload = client.chat_json(SYSTEM_PROMPT, build_user_prompt(digest))
    except LLMError as exc:
        log.warning("semantic_pass_failed", error=str(exc))
        fallback["reason"] = str(exc)
        fallback["layers"] = _deterministic_layers(drawing)
        if drawing.diagnostics is not None:
            drawing.diagnose("semantic", f"LLM semantic pass skipped: {exc}")
        return _store(drawing, fallback)

    semantic = parse_semantic_response(payload, drawing)
    semantic["source"] = f"deepseek:{client.model}"
    if not semantic["layers"]:
        semantic["reason"] = "model returned no usable layer grouping"
        semantic["layers"] = _deterministic_layers(drawing)
        return _store(drawing, semantic)

    apply_semantics(drawing, semantic)
    if drawing.diagnostics is not None:
        drawing.diagnose(
            "semantic",
            f"{semantic['assigned']} entities grouped into {len(semantic['layers'])} semantic layers",
        )
    return _store(drawing, semantic)


def _deterministic_layers(drawing: Drawing) -> list[dict]:
    """The classifier's own grouping, shaped like a model reply."""
    grouped: dict[str, list[int]] = {}
    for entity in drawing.entities:
        grouped.setdefault(entity.layer, []).append(entity.index)
    return [
        {"name": name, "entities": indices, "reason": "deterministic linetype/lineweight classification"}
        for name, indices in sorted(grouped.items())
    ]


def _store(drawing: Drawing, semantic: dict) -> dict:
    drawing.notes["semantic"] = semantic
    return semantic
