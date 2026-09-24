# Image → DXF tracing

How this project turns a raster drawing (a screenshot, a scan, a photo of a
printout) into a layered DXF file — and why the obvious approach does not work.

---

## The short version

```
 image ──► binarise ──► vectorise ──► CAD IR ──► (semantic pass) ──► DXF
             │             │            │              │
             │             │            │              └── DeepSeek, TEXT ONLY
             │             │            └── verified geometry + measured linetypes
             │             └── OpenCV: segments, circles, arcs, dash patterns
             └── Otsu / adaptive threshold, polarity auto-detected
```

**The model never looks at the image.** Geometry is measured; the model only
labels what was measured. That single decision is what fixes the three failures
below, and it also means the tracer works with a text-only API key — or with no
API key at all.

---

## Why not "send the picture to a vision model and ask for CAD commands"

That was the first attempt, and it failed in three specific ways worth naming:

| Failure | What actually happens | What this project does instead |
|---|---|---|
| **Incomplete read** | DeepSeek's public API has no vision endpoint at all; weaker vision models drop thin lines, small arcs and most dash detail. You get a partial drawing and no way to tell what is missing. | OpenCV extracts *every* stroke deterministically. Nothing is hallucinated, nothing is silently dropped, and `diagnostics`/`counters` report exactly what was found. |
| **"It cannot tell a solid line from a dashed one"** | A model that only computes coordinates has no notion of line style. It will emit `LINE` entities for dashes and call them solid. | Each line's ink pattern is *measured* along its length (run-length analysis) and classified: `CONTINUOUS`, `DASHED`, `HIDDEN`, `CENTER`, `PHANTOM`, with a confidence value. |
| **"Arc commands do not match the two points"** | Asked for an arc between two points, a coordinate-computing model produces `ARC` centre/radius/angles that do not pass through them. The result looks plausible and is geometrically wrong. | Arcs are fitted from the pixels and stored in canonical form, then **snapped**: every arc endpoint within tolerance of a line vertex has its angle recomputed from the centre to that vertex, so the arc terminates exactly on the junction. |

A useful mental model: the LLM is good at *naming things* and bad at *measuring
things*. So the pipeline hands it only what has already been measured.

---

## Stage by stage

### 1. Binarise (`vectorize.binarize`)

* Otsu by default, adaptive threshold (`--threshold adaptive`) for uneven
  lighting, photos and scans.
* **Polarity is auto-detected**: ink is assumed to be the minority class, which
  holds whether the drawing is dark-on-light or light-on-dark.
* Specks of 1–3 px are removed with connected-component filtering — *not*
  morphological opening, which would erase hairline strokes.
* If the measured stroke width is under ~2.5 px the image is auto-upscaled
  (up to 3×) before binarising, because dash gaps of a 1 px pattern are not
  recoverable at that resolution.

### 2. Straight segments (`vectorize.detect_segments`)

* Probabilistic Hough on a Canny edge map.
* Collinear runs are **merged**, absorbing the edge pair a thick stroke
  produces (otherwise every thick line is emitted twice) and recentring the
  result on the stroke's midline.
* Dash gaps are chained here so that a dashed line arrives as one candidate run
  rather than fifteen dashes.

### 3. Measure, then chain, then classify

This ordering is the crux of the whole feature, and it is easy to get wrong:

1. **Measure** each candidate: slide it perpendicular onto the ink
   (`recentre_on_ink`), measure its own stroke width from the distance
   transform, and pick a sampling window from *that* width.
2. **Chain** collinear fragments (`chain_segments`) — across dash gaps, and with
   a relaxed gap for short fragments, because long-dash patterns are separated
   by more than one period.
3. **Classify** the ink pattern of the *whole chained run*.

Why the order matters: **a single dash has no interior gap.** Measure a dash on
its own and it is indistinguishable from a solid line. Chain first, and the
pattern becomes readable — that is what makes "solid vs dashed" decidable at
all. (Skipping this step was the single biggest source of wrong linetypes during
development.)

### 4. Circles and arcs (`vectorize.detect_rounds`, `detect_rounds_hough`)

* Every contour is fitted with an algebraic (Kåsa) circle fit; the arc's angular
  span comes from the **largest gap in the contour's angle coverage**, which
  recovers arcs of any sweep (including > 180°) instead of relying on
  first/middle/last points.
* Circles crossed by other ink merge into one connected component and break
  contour fitting, so `cv2.HoughCircles` runs as a second detector. Every
  proposal is accepted only if the ink actually covers the proposed circle
  (support ≥ 45%), which keeps accumulator false positives out.
* Duplicate strokes (outer/inner contour of a thick circle) and dashed arcs on
  the same circle are merged.
* Short straight chords that Hough reports across a circular stroke are dropped
  once the circle is known.

### 5. Linetype and lineweight (`linetype.classify_linetype`)

Run-length analysis of the sampled pattern gives dash/gap statistics, then:

| Pattern | Classification |
|---|---|
| No interior gaps, or gaps below the noise floor | `CONTINUOUS` |
| Uniform dashes | `DASHED` if the period is large, `HIDDEN` if it is small |
| Alternating long/short (`Ls`) | `CENTER` |
| Long + two shorts (`Lss`) | `PHANTOM` |
| Sparse dot-like ink | `DASHED` (low confidence) |

Two honest caveats, encoded in the API rather than hidden:

* **`DASHED` vs `HIDDEN` is a scale question.** Both are 2:1 patterns and differ
  mainly by period, so the decision is made in *drawing units*. Pass the real
  size of the drawing (`--width 420 --units mm`) and the choice becomes
  meaningful; otherwise the tracer reports lower confidence.
* **`CONTINUOUS` vs dotted is a sampling question.** Runs shorter than
  `min_run_px` are treated as anti-aliasing noise, not as ink gaps.

Stroke width (from the distance transform) is snapped onto the ISO lineweight
ladder and mapped to conventional layers: `Thick` / `Thin` / `Hidden` / `Center`.

### 6. Welding and arc snapping (`vectorize.weld_and_snap`)

* Line endpoints within tolerance are clustered and replaced by their centroid,
  turning "almost touching" into a closed profile.
* Each arc endpoint near a vertex gets its angle recomputed *from the centre to
  that vertex*, keeping the radius — so the arc lands exactly on the junction.
* The number of welds, snaps and remaining dangling endpoints is reported.

### 7. The semantic pass (`semantics.py`) — optional, text-only

A compact JSON **digest** is sent to DeepSeek: units, size, extraction counters,
per-entity kind/geometry/measured linetype/confidence, current layers. **No
pixels, ever.** The model returns:

```json
{
  "drawing_type": "mechanical part",
  "summary": "A rectangular plate with a circular boss, a centre-line cross …",
  "layers": [{"name": "Outline", "entities": [0, 1, 2], "reason": "outer boundary"}],
  "warnings": ["the lower fillet may be missing"]
}
```

The reply is **validated, not trusted**: unknown entity indices are dropped,
each index may be assigned once, layer names are sanitised and truncated, and
anything invalid is counted in `rejected`. Linetype and lineweight stay
per-entity, so a rename can never destroy a measured line style.

With no API key the deterministic classifier's own grouping stands, and the DXF
is still complete. That is a requirement, not a fallback: the tool has to be
useful with nothing but `pip install`.

---

## Usage

### CLI (no MCP client needed)

```powershell
# Install (any of these)
pip install autocad-mcp            # published package
uv sync                            # from a checkout

# Trace
autocad-trace drawing.png -o drawing.dxf --width 200 --units mm --preview traced.png
python -m autocad_mcp.trace.cli drawing.png -o drawing.dxf --no-llm
```

| Flag | Meaning |
|---|---|
| `-o/--out` | output DXF (default: alongside the image) |
| `--width` + `--units` | real width of the drawing → true scale, and a meaningful DASHED/HIDDEN call |
| `--scale` | drawing units per pixel, when the real size is unknown (default 1.0) |
| `--threshold` | `otsu` (default) or `adaptive` for photos/scans |
| `--min-line` | shortest straight segment to keep, px (default 22) |
| `--max-gap` | gap chained into one dashed line, px (default 9) |
| `--json` / `--preview` | write the CAD IR as JSON / render the traced IR as PNG |
| `--no-llm` | skip the semantic pass (offline, deterministic) |

### MCP tool

```
trace(operation="image_to_dxf", data={"image": "C:/drawings/part.png", "dxf": "C:/out/part.dxf"})
trace(operation="vectorize",    data={"image": "C:/drawings/part.png"})   # IR only
trace(operation="describe",     data={"image": "C:/drawings/part.png"})   # + semantics, no file
```

Works on any backend and without AutoCAD installed — the tracer never touches
AutoCAD.

### Environment

| Variable | Default | Purpose |
|---|---|---|
| `DEEPSEEK_API_KEY` | – | enables the semantic pass (optional) |
| `AUTOCAD_MCP_DEEPSEEK_MODEL` | `deepseek-chat` | model for the semantic pass |
| `AUTOCAD_MCP_DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | any OpenAI-compatible endpoint |

---

## What it reads well, and what it does not

**Good fit** — clean, simple drawings: mechanical part outlines, plates with
holes, brackets, simple plans, diagrams, single-line schematics. Straight lines,
circles and arcs, all four standard line styles, uniform stroke widths, plain
backgrounds.

**Known limitations** (reported in `diagnostics`, not hidden):

* **Text is not read.** Small closed blobs are dropped as annotation; there is
  no OCR. Dimensions and notes are not recovered.
* **Hatching and dense detail** are not modelled; heavy fill can hide the edges
  underneath.
* **Collinear lines separated by a small gap may be chained** into one entity:
  the chaining step deliberately bridges dash gaps, so a wall broken by a door
  gap can close up. Raise `--min-line`, or lower `merge_max_gap_px`.
* **Curvature that is not circular** (splines, ellipses, involutes) is fitted as
  arcs or polylines at best; the residual (`fit_rms`) is reported so you can see
  where the fit is poor.
* **Very low-resolution images** lose dash detail; the auto-upscale helps but
  cannot invent information.
* **The semantic pass cannot verify geometry** — by design it never sees the
  image, so treat its `warnings` as questions, not findings.

---

## Tuning

```python
from autocad_mcp.trace import TraceOptions, VectorizeOptions, trace_image

result = trace_image(
    "part.png",
    TraceOptions(
        vectorize=VectorizeOptions(
            scale=0.25,            # mm per pixel
            threshold="adaptive",  # photos and scans
            min_line_px=16.0,      # keep shorter segments
            merge_max_gap_px=36.0, # longer dashes chain into one line
            auto_upscale=True,
        ),
        use_llm=True,
        dxf_path="part.dxf",
        preview_path="part.traced.png",
        json_path="part.ir.json",
    ),
)
print(result.dxf_path, result.counters, result.warnings)
```

`result.counters` reports what happened at every stage (`dropped_low_ink`,
`chained_fragments`, `chords_dropped`, `blobs_dropped`, `welded_endpoints`,
`arc_endpoints_snapped`, `dangling_endpoints`, per-linetype counts), which is
usually enough to tell whether a knob needs turning.

---

## Tests

```powershell
pytest tests/test_trace_geometry.py tests/test_trace_linetype.py `
       tests/test_trace_pipeline.py tests/test_trace_semantics.py -v
```

* `test_trace_geometry.py` — arc from three points, bulge round-trips, circle
  fits, endpoint welding.
* `test_trace_linetype.py` — every linetype, including "the same pixel pattern
  at a different scale is HIDDEN, not DASHED".
* `test_trace_pipeline.py` — a generated drawing with all four line styles, a
  circle and an arc, traced end to end and read back with `ezdxf`.
* `test_trace_semantics.py` — digest contents, index validation, sanitisation,
  and full offline degradation.
