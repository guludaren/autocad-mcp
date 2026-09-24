"""IR → DXF emission with ``ezdxf``.

The emitter is the only place that knows about DXF group codes. It writes
canonical entities — ``LINE``, ``ARC`` (centre/radius/angles), ``CIRCLE`` and
``LWPOLYLINE`` with arc bulges — so an arc traced from a drawing keeps its
exact geometry instead of being flattened into a polyline.

Line style is emitted at two levels, mirroring CAD practice:

* the **layer** carries the linetype and lineweight inferred for it,
* an **entity** overrides them only when it differs from its layer — which is
  what happens after the semantic pass groups lines of different styles onto
  one semantically named layer.
"""

from __future__ import annotations

from pathlib import Path

import ezdxf
import structlog
from ezdxf import units as dxf_units

from autocad_mcp.trace.ir import Arc, Circle, Drawing, Line, Polyline
from autocad_mcp.trace.linetype import LINETYPE_PATTERNS

log = structlog.get_logger()

#: Drawing-units name → ezdxf unit code (0 = unitless, per the DXF spec).
UNIT_CODES = {
    "mm": dxf_units.MM,
    "cm": dxf_units.CM,
    "m": dxf_units.M,
    "in": dxf_units.IN,
    "inch": dxf_units.IN,
    "ft": dxf_units.FT,
    "px": 0,
    "unitless": 0,
}


def ensure_linetypes(doc) -> list[str]:
    """Make sure every linetype this project classifies exists in the document."""
    added: list[str] = []
    for name, pattern in LINETYPE_PATTERNS.items():
        if name in doc.linetypes:
            continue
        if not pattern:
            continue
        try:
            doc.linetypes.add(name, pattern=pattern.split(","))
            added.append(name)
        except Exception as exc:  # pragma: no cover - defensive, ezdxf raises on odd patterns
            log.warning("linetype_add_failed", linetype=name, error=str(exc))
    return added


def ensure_layers(doc, drawing: Drawing) -> list[str]:
    """Create the drawing's layers, carrying linetype and lineweight."""
    created: list[str] = []
    for layer in drawing.layers:
        if layer.name in doc.layers:
            dxf_layer = doc.layers.get(layer.name)
            dxf_layer.color = layer.color
            dxf_layer.dxf.linetype = layer.linetype
            dxf_layer.dxf.lineweight = layer.lineweight
            continue
        doc.layers.add(
            layer.name,
            color=layer.color,
            linetype=layer.linetype,
            lineweight=layer.lineweight,
        )
        created.append(layer.name)
    return created


def _entity_attribs(entity, layer_lookup: dict) -> dict:
    """dxfattribs that override the layer only when the entity differs."""
    attribs: dict = {"layer": entity.layer}
    layer = layer_lookup.get(entity.layer)
    verdict = getattr(entity, "verdict", None)
    linetype_name = getattr(verdict, "linetype", None)
    if linetype_name and layer is not None and linetype_name != layer.linetype:
        attribs["linetype"] = linetype_name
    lineweight = getattr(entity, "lineweight", None)
    if lineweight is not None and layer is not None and int(lineweight) != layer.lineweight:
        attribs["lineweight"] = int(lineweight)
    return attribs


def emit_dxf(drawing: Drawing, path: str | Path) -> dict:
    """Write the IR to a DXF file; returns emission statistics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    doc = ezdxf.new("R2010", setup=True)
    doc.units = UNIT_CODES.get(drawing.units.lower(), dxf_units.MM)
    added_linetypes = ensure_linetypes(doc)
    created_layers = ensure_layers(doc, drawing)
    layer_lookup = {layer.name: layer for layer in drawing.layers}
    msp = doc.modelspace()

    written: dict[str, int] = {}
    skipped: list[dict] = []
    for entity in drawing.entities:
        attribs = _entity_attribs(entity, layer_lookup)
        try:
            if isinstance(entity, Line):
                msp.add_line(entity.start, entity.end, dxfattribs=attribs)
            elif isinstance(entity, Circle):
                msp.add_circle(entity.center, entity.radius, dxfattribs=attribs)
            elif isinstance(entity, Arc):
                msp.add_arc(
                    entity.center,
                    entity.radius,
                    entity.start_angle,
                    entity.end_angle,
                    dxfattribs=attribs,
                )
            elif isinstance(entity, Polyline):
                points = [
                    (x, y, 0.0, 0.0, bulge)
                    for (x, y), bulge in zip(
                        entity.points,
                        entity.bulges or [0.0] * len(entity.points),
                    )
                ]
                msp.add_lwpolyline(points, format="xyseb", close=entity.closed, dxfattribs=attribs)
            else:
                skipped.append({"index": entity.index, "kind": entity.kind, "reason": "unsupported kind"})
                continue
        except Exception as exc:
            log.warning("emit_entity_failed", kind=entity.kind, error=str(exc))
            skipped.append({"index": entity.index, "kind": entity.kind, "reason": str(exc)})
            continue
        written[entity.kind] = written.get(entity.kind, 0) + 1

    doc.saveas(path)
    stats = {
        "path": str(path),
        "entities": sum(written.values()),
        "by_kind": written,
        "layers": [layer.name for layer in drawing.layers],
        "layers_created": created_layers,
        "linetypes_added": added_linetypes,
        "skipped": skipped,
        "units": drawing.units,
    }
    log.info("dxf_written", path=str(path), entities=stats["entities"])
    return stats
