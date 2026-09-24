"""CAD intermediate representation (IR) for image tracing.

The IR is deliberately **not** a command log and **not** raw pixel data. It is
the minimal set of geometric facts a drawing is made of, expressed in canonical
CAD entity forms (LINE / ARC / CIRCLE / LWPOLYLINE-with-bulge), so that every
downstream consumer works from the same verified geometry:

* the validator can check endpoint continuity and arc fit error,
* the DXF emitter can write exact entities (including arc bulges),
* the LLM semantic pass can reason about *labelled* geometry without ever
  being asked to look at pixels.

All coordinates are in drawing units, already converted from pixels and with
the Y axis pointing up (image Y grows downwards, CAD Y grows upwards).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Standard AutoCAD linetypes this project can classify and emit.
LINETYPES = ("CONTINUOUS", "DASHED", "HIDDEN", "CENTER", "PHANTOM")

# Role names the deterministic classifier assigns before any LLM pass runs.
ROLE_OUTLINE = "outline"
ROLE_HIDDEN = "hidden"
ROLE_CENTER = "center"
ROLE_THIN = "thin"
ROLE_UNKNOWN = "unknown"


def _round(value: Any, digits: int = 4) -> Any:
    """Round floats for compact, diff-friendly JSON."""
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, (list, tuple)):
        return [_round(v, digits) for v in value]
    return value


@dataclass
class Layer:
    """A drawing layer: name plus the properties tracing inferred for it."""

    name: str
    color: int = 7  # AutoCAD Color Index (7 = black/white)
    linetype: str = "CONTINUOUS"
    lineweight: int = 25  # hundredths of a millimetre, as DXF stores it

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "color": self.color,
            "linetype": self.linetype,
            "lineweight": self.lineweight,
        }


@dataclass
class Entity:
    """Base entity: a kind tag, its layer, and provenance/confidence."""

    kind: str = "UNKNOWN"
    layer: str = "0"
    confidence: float = 1.0
    role: str = ROLE_UNKNOWN
    index: int = -1  # stable id used by the semantic pass

    def geometry(self) -> dict:
        return {}

    def to_dict(self) -> dict:
        data = {
            "index": self.index,
            "kind": self.kind,
            "layer": self.layer,
            "role": self.role,
            "confidence": round(float(self.confidence), 3),
        }
        data.update(_round(self.geometry()))
        return data


@dataclass
class Line(Entity):
    start: tuple[float, float] = (0.0, 0.0)
    end: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        self.kind = "LINE"

    def geometry(self) -> dict:
        return {"start": list(self.start), "end": list(self.end)}


@dataclass
class Arc(Entity):
    """Circular arc in canonical CAD form: centre + radius + CCW angles."""

    center: tuple[float, float] = (0.0, 0.0)
    radius: float = 0.0
    start_angle: float = 0.0  # degrees, CCW from +X
    end_angle: float = 0.0
    fit_rms: float = 0.0  # circle-fit residual in drawing units

    def __post_init__(self) -> None:
        self.kind = "ARC"

    def geometry(self) -> dict:
        return {
            "center": list(self.center),
            "radius": self.radius,
            "start_angle": self.start_angle,
            "end_angle": self.end_angle,
            "fit_rms": self.fit_rms,
        }


@dataclass
class Circle(Entity):
    center: tuple[float, float] = (0.0, 0.0)
    radius: float = 0.0
    fit_rms: float = 0.0

    def __post_init__(self) -> None:
        self.kind = "CIRCLE"

    def geometry(self) -> dict:
        return {
            "center": list(self.center),
            "radius": self.radius,
            "fit_rms": self.fit_rms,
        }


@dataclass
class Polyline(Entity):
    """Lightweight polyline; each vertex carries an optional arc bulge."""

    points: list[tuple[float, float]] = field(default_factory=list)
    bulges: list[float] = field(default_factory=list)
    closed: bool = False

    def __post_init__(self) -> None:
        self.kind = "POLYLINE"

    def geometry(self) -> dict:
        return {
            "points": [list(p) for p in self.points],
            "bulges": list(self.bulges),
            "closed": self.closed,
        }


@dataclass
class Drawing:
    """The traced drawing: layers, entities, and how they were derived."""

    units: str = "mm"
    scale: float = 1.0  # drawing units per pixel
    width: float = 0.0  # drawing size in drawing units
    height: float = 0.0
    layers: list[Layer] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    diagnostics: list[dict] = field(default_factory=list)
    notes: dict = field(default_factory=dict)

    def next_index(self) -> int:
        return len(self.entities)

    def add(self, entity: Entity) -> Entity:
        entity.index = self.next_index()
        self.entities.append(entity)
        return entity

    def layer_names(self) -> list[str]:
        return [layer.name for layer in self.layers]

    def ensure_layer(self, layer: Layer) -> None:
        if layer.name not in self.layer_names():
            self.layers.append(layer)

    def stats(self) -> dict:
        by_kind: dict[str, int] = {}
        by_linetype: dict[str, int] = {}
        by_layer: dict[str, int] = {}
        linetype_of = {layer.name: layer.linetype for layer in self.layers}
        for entity in self.entities:
            by_kind[entity.kind] = by_kind.get(entity.kind, 0) + 1
            by_layer[entity.layer] = by_layer.get(entity.layer, 0) + 1
            lt = linetype_of.get(entity.layer, "CONTINUOUS")
            by_linetype[lt] = by_linetype.get(lt, 0) + 1
        return {
            "entities": len(self.entities),
            "by_kind": by_kind,
            "by_layer": by_layer,
            "by_linetype": by_linetype,
        }

    def diagnose(self, code: str, message: str, **extra: Any) -> None:
        entry = {"code": code, "message": message}
        entry.update(extra)
        self.diagnostics.append(entry)

    def to_dict(self) -> dict:
        return {
            "units": self.units,
            "scale": self.scale,
            "width": round(self.width, 4),
            "height": round(self.height, 4),
            "layers": [layer.to_dict() for layer in self.layers],
            "entities": [entity.to_dict() for entity in self.entities],
            "stats": self.stats(),
            "diagnostics": self.diagnostics,
            "notes": self.notes,
        }
