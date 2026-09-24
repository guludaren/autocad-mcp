"""End-to-end pipeline: image → CAD IR → (semantic pass) → DXF.

    image ──► vectorize ──► IR ──► semantic_pass ──► emit_dxf ──► .dxf
                 │            │          │
                 │            │          └── DeepSeek (text only, optional)
                 │            └── verified geometry + measured linetypes
                 └── OpenCV, deterministic, no model involved

The LLM stage is optional by construction: pass ``use_llm=False`` (or simply
have no API key) and the pipeline still emits a fully layered DXF.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from autocad_mcp.trace.emit import emit_dxf
from autocad_mcp.trace.ir import Arc, Circle, Drawing, Line
from autocad_mcp.trace.semantics import semantic_pass
from autocad_mcp.trace.vectorize import VectorizeOptions, VectorizeResult, load_gray, vectorize

log = structlog.get_logger()

#: Preview dash patterns, mirroring the DXF linetype definitions.
PREVIEW_DASHES = {
    "CONTINUOUS": None,
    "DASHED": (6.0, 3.0),
    "HIDDEN": (3.0, 2.0),
    "CENTER": (12.0, 3.0, 3.0, 3.0),
    "PHANTOM": (12.0, 3.0, 3.0, 3.0, 3.0, 3.0),
}

#: Minimal ACI → RGB map for previews (the colours this project assigns).
PREVIEW_COLORS = {1: "#d62728", 4: "#17becf", 7: "#111111", 8: "#7f7f7f"}


@dataclass
class TraceOptions:
    """Everything the pipeline can be tuned with."""

    vectorize: VectorizeOptions = field(default_factory=VectorizeOptions)
    use_llm: bool = True
    write_dxf: bool = True
    dxf_path: str | None = None
    json_path: str | None = None
    preview_path: str | None = None
    preview_dpi: int = 110


@dataclass
class TraceResult:
    """Pipeline outcome, serialisable for MCP responses and CLI output."""

    ok: bool
    dxf_path: str | None = None
    image_path: str | None = None
    ir: Drawing | None = None
    semantic: dict = field(default_factory=dict)
    emission: dict = field(default_factory=dict)
    counters: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    preview_path: str | None = None
    json_path: str | None = None
    error: str | None = None

    def to_dict(self, include_ir: bool = False, include_entities: bool = True) -> dict:
        payload: dict = {
            "ok": self.ok,
            "image": self.image_path,
            "dxf": self.dxf_path,
            "preview": self.preview_path,
            "counters": self.counters,
            "warnings": self.warnings,
            "semantic": self.semantic,
            "emission": self.emission,
        }
        if self.error:
            payload["error"] = self.error
        if self.ir is not None:
            payload["stats"] = self.ir.stats()
            payload["layers"] = [layer.to_dict() for layer in self.ir.layers]
            payload["diagnostics"] = self.ir.diagnostics
            if include_ir:
                payload["drawing"] = self.ir.to_dict()
            elif include_entities:
                payload["entities"] = [entity.to_dict() for entity in self.ir.entities]
        return payload


def default_dxf_path(image_path: str | Path) -> Path:
    image = Path(image_path)
    return image.with_suffix(".dxf")


def resolve_scale(image_path: str | Path, scale: float = 1.0, width: float | None = None) -> float:
    """Drawing units per pixel, from an explicit scale or a real-world width.

    ``width`` is the real width of the drawing (in ``units``); it is the honest
    way to get a DXF at true size, because it also fixes the DASHED/HIDDEN
    decision, which is a question of scale rather than of shape.
    """
    if not width:
        return float(scale)
    gray = load_gray(image_path)
    if not gray.shape[1]:
        return float(scale)
    return float(width) / float(gray.shape[1])


def trace_image(image_path: str | Path, options: TraceOptions | None = None) -> TraceResult:
    """Trace an image into a DXF file plus a full report.

    Failures are returned as ``ok=False`` results rather than raised, so the MCP
    tool and the CLI can both report them uniformly.
    """
    options = options or TraceOptions()
    image_path = Path(image_path)
    try:
        result: VectorizeResult = vectorize(image_path, options.vectorize)
    except Exception as exc:  # decoding / IO problems
        log.error("trace_failed", image=str(image_path), error=str(exc))
        return TraceResult(ok=False, image_path=str(image_path), error=f"vectorize failed: {exc}")

    drawing = result.drawing
    semantic: dict = {}
    if options.use_llm:
        semantic = semantic_pass(drawing)
    else:
        semantic = {
            "source": "disabled",
            "reason": "semantic pass disabled by caller",
            "layers": [],
        }
        drawing.notes["semantic"] = semantic

    dxf_path = None
    emission: dict = {}
    if options.write_dxf:
        dxf_path = Path(options.dxf_path) if options.dxf_path else default_dxf_path(image_path)
        try:
            emission = emit_dxf(drawing, dxf_path)
        except Exception as exc:
            log.error("emit_failed", error=str(exc))
            return TraceResult(
                ok=False,
                image_path=str(image_path),
                ir=drawing,
                semantic=semantic,
                counters=result.counters,
                warnings=result.warnings,
                error=f"DXF emission failed: {exc}",
            )

    preview_path = None
    if options.preview_path:
        try:
            preview_path = render_preview(drawing, options.preview_path, options.preview_dpi)
        except Exception as exc:  # previews are a convenience, never fatal
            log.warning("preview_failed", error=str(exc))
            result.warnings.append(f"preview rendering failed: {exc}")

    json_path = None
    if options.json_path:
        json_path = str(options.json_path)
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(json_path).open("w", encoding="utf-8") as handle:
            json.dump(drawing.to_dict(), handle, indent=2, ensure_ascii=False)

    return TraceResult(
        ok=True,
        dxf_path=str(dxf_path) if dxf_path else None,
        image_path=str(image_path),
        ir=drawing,
        semantic=semantic,
        emission=emission,
        counters=result.counters,
        warnings=result.warnings,
        preview_path=preview_path,
        json_path=json_path,
    )


def render_preview(drawing: Drawing, path: str | Path, dpi: int = 110) -> str:
    """Render the IR to a PNG so a trace can be eyeballed (and diffed) fast."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc as MplArc
    from matplotlib.patches import Circle as MplCircle
    from matplotlib.collections import LineCollection

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    linetype_of = {layer.name: layer.linetype for layer in drawing.layers}
    color_of = {layer.name: layer.color for layer in drawing.layers}

    figure, axes = plt.subplots(figsize=(9, 9 * max(drawing.height, 1) / max(drawing.width, 1) or 9))
    segments_by_style: dict[tuple[str, str], list] = {}
    for entity in drawing.entities:
        linetype_name = linetype_of.get(entity.layer, "CONTINUOUS")
        color = PREVIEW_COLORS.get(color_of.get(entity.layer, 7), "#111111")
        key = (linetype_name, color)
        if isinstance(entity, Line):
            segments_by_style.setdefault(key, []).append([entity.start, entity.end])
        elif isinstance(entity, Circle):
            axes.add_patch(
                MplCircle(
                    entity.center,
                    entity.radius,
                    fill=False,
                    edgecolor=color,
                    linestyle=(0, PREVIEW_DASHES.get(linetype_name) or (1, 0)),
                    linewidth=1.2,
                )
            )
        elif isinstance(entity, Arc):
            axes.add_patch(
                MplArc(
                    entity.center,
                    entity.radius * 2,
                    entity.radius * 2,
                    theta1=entity.start_angle,
                    theta2=entity.end_angle,
                    edgecolor=color,
                    linestyle=(0, PREVIEW_DASHES.get(linetype_name) or (1, 0)),
                    linewidth=1.2,
                )
            )

    for (linetype_name, color), segments in segments_by_style.items():
        collection = LineCollection(
            segments,
            colors=color,
            linewidths=1.2,
            linestyles=(0, PREVIEW_DASHES.get(linetype_name) or (1, 0)),
        )
        axes.add_collection(collection)

    axes.set_aspect("equal", adjustable="box")
    axes.autoscale_view()
    axes.margins(0.05)
    axes.set_title(f"traced IR — {len(drawing.entities)} entities", fontsize=9)
    axes.grid(True, linewidth=0.2, alpha=0.4)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return str(path)


__all__ = [
    "TraceOptions",
    "TraceResult",
    "default_dxf_path",
    "load_gray",
    "render_preview",
    "resolve_scale",
    "trace_image",
    "vectorize",
]
