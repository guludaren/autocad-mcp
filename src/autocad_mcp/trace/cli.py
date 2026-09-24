"""Command-line interface: ``python -m autocad_mcp.trace.cli drawing.png -o out.dxf``.

The CLI exists so the image→DXF feature is usable **without** an MCP client and
without any API key: geometry extraction is local and deterministic, and the
DeepSeek semantic pass is opt-in via ``DEEPSEEK_API_KEY`` (disable explicitly
with ``--no-llm``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autocad_mcp.trace.pipeline import TraceOptions, resolve_scale, trace_image
from autocad_mcp.trace.vectorize import VectorizeOptions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m autocad_mcp.trace.cli",
        description="Trace a raster drawing (PNG/JPG) into a layered DXF file.",
    )
    parser.add_argument("image", help="input image path")
    parser.add_argument("-o", "--out", help="output DXF path (default: alongside the image)")
    parser.add_argument("--json", dest="json_path", help="also write the CAD IR as JSON")
    parser.add_argument("--preview", help="also render the traced IR to a PNG")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="drawing units per pixel (default: 1.0)",
    )
    parser.add_argument(
        "--width",
        type=float,
        help="real width of the drawing; overrides --scale (e.g. --width 420 --units mm)",
    )
    parser.add_argument("--units", default="mm", help="drawing units for the DXF (default: mm)")
    parser.add_argument(
        "--threshold",
        choices=("otsu", "adaptive"),
        default="otsu",
        help="binarisation method (use adaptive for uneven lighting/photos)",
    )
    parser.add_argument("--min-line", type=float, default=22.0, help="shortest segment to keep, px")
    parser.add_argument("--max-gap", type=float, default=9.0, help="dash gap chained into one line, px")
    parser.add_argument("--no-llm", action="store_true", help="skip the DeepSeek semantic pass")
    parser.add_argument("--quiet", action="store_true", help="only print the DXF path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        scale = resolve_scale(args.image, args.scale, args.width)
    except Exception as exc:
        print(f"warning: could not read image size for --width ({exc}); using --scale", file=sys.stderr)
        scale = args.scale

    options = TraceOptions(
        vectorize=VectorizeOptions(
            threshold=args.threshold,
            min_line_px=args.min_line,
            max_gap_px=args.max_gap,
            scale=scale,
            units=args.units,
        ),
        use_llm=not args.no_llm,
        dxf_path=args.out,
        json_path=args.json_path,
        preview_path=args.preview,
    )
    result = trace_image(args.image, options)

    if args.quiet:
        if result.ok and result.dxf_path:
            print(result.dxf_path)
            return 0
        print(result.error or "trace failed", file=sys.stderr)
        return 1

    report = result.to_dict(include_ir=False, include_entities=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not result.ok:
        return 1
    semantic = result.semantic or {}
    if semantic.get("summary"):
        print(f"\nsummary: {semantic['summary']}", file=sys.stderr)
    if semantic.get("reason") and semantic.get("source") == "deterministic":
        print(f"semantic pass: {semantic['reason']}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover - thin wrapper
    raise SystemExit(main())
