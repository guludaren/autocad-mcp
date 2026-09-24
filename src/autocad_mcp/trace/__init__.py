"""Image → DXF tracing for AutoCAD MCP.

Public entry points::

    from autocad_mcp.trace import trace_image, TraceOptions, VectorizeOptions

    result = trace_image("drawing.png", TraceOptions(dxf_path="out.dxf"))
    print(result.dxf_path, result.counters)

Design in one line: **geometry from OpenCV, semantics from DeepSeek** — the
model is never asked to look at pixels, and every stage works without an API
key.
"""

from autocad_mcp.trace.deepseek import DeepSeekClient, LLMError
from autocad_mcp.trace.emit import emit_dxf
from autocad_mcp.trace.ir import Arc, Circle, Drawing, Layer, Line, Polyline
from autocad_mcp.trace.pipeline import (
    TraceOptions,
    TraceResult,
    default_dxf_path,
    render_preview,
    resolve_scale,
    trace_image,
)
from autocad_mcp.trace.semantics import semantic_pass
from autocad_mcp.trace.vectorize import VectorizeOptions, VectorizeResult, vectorize

__all__ = [
    "Arc",
    "Circle",
    "DeepSeekClient",
    "Drawing",
    "LLMError",
    "Layer",
    "Line",
    "Polyline",
    "TraceOptions",
    "TraceResult",
    "VectorizeOptions",
    "VectorizeResult",
    "default_dxf_path",
    "emit_dxf",
    "render_preview",
    "resolve_scale",
    "semantic_pass",
    "trace_image",
    "vectorize",
]
