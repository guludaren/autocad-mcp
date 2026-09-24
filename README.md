# AutoCAD MCP Server

MCP server for AutoCAD LT automation, headless DXF generation, and **image → DXF tracing**.

Three capabilities, one server:

| Capability | Runtime | Requires AutoCAD? | Notes |
|---------|---------|-------------------|------------|
| **File IPC** backend | Windows Python | Yes — AutoCAD LT 2024+ (Windows) | Win32 PrintWindow screenshots |
| **ezdxf** backend | Any platform | No (headless) | in-memory DXF, matplotlib render |
| **Image → DXF tracing** | Any platform | No | OpenCV vectorisation + optional DeepSeek semantic pass — see [docs/image-to-dxf.md](docs/image-to-dxf.md) |

The server exposes **9 consolidated tools** (`drawing`, `entity`, `layer`, `block`, `annotation`, `pid`, `view`, `system`, `trace`) over the MCP stdio transport. An MCP client (Claude Desktop, Claude Code, etc.) connects and drives AutoCAD through natural-language requests.

## Prerequisites (File IPC backend)

- **Windows 10/11** (the File IPC backend uses Win32 APIs for focus-free window messaging)
- **AutoCAD LT 2024 or newer** — AutoLISP support was added in LT 2024 for Windows. AutoCAD LT for Mac exists but does **not** support AutoLISP.
- **Python 3.10+** (Windows native — not WSL Python)
- **uv** package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))

> The ezdxf headless backend works on any platform (Linux, macOS, WSL) for offline DXF generation without AutoCAD installed.

## Quick Start

### 1. Clone and install

```powershell
git clone https://github.com/puran-water/autocad-mcp.git
cd autocad-mcp
uv sync
```

### 2. Load the LISP dispatcher in AutoCAD LT

Open AutoCAD LT and load `mcp_dispatch.lsp` using **APPLOAD**:

1. Type `APPLOAD` in the AutoCAD command line
2. Browse to `<repo>/lisp-code/mcp_dispatch.lsp`
3. Click **Load**
4. You should see: `=== MCP Dispatch v3.1 loaded ===` and `Ready for commands via (c:mcp-dispatch)`

> **Tip:** Add the file to your AutoCAD Startup Suite (in the APPLOAD dialog) so it loads automatically with every drawing.

### 3. Configure your MCP client

Add to your MCP client configuration (e.g. Claude Desktop `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "autocad-mcp": {
      "command": "C:\\path\\to\\autocad-mcp\\.venv\\Scripts\\python.exe",
      "args": ["-m", "autocad_mcp"],
      "env": { "AUTOCAD_MCP_BACKEND": "auto" }
    }
  }
}
```

**Key points:**

- The `command` must point to the **Windows Python** inside the project venv (not WSL python).
- `AUTOCAD_MCP_BACKEND` can be `auto` (default — tries File IPC, falls back to ezdxf), `file_ipc` (requires AutoCAD), or `ezdxf` (headless only).

#### Running from WSL

If your MCP client runs in WSL (e.g. Claude Code), launch the server through `cmd.exe` so it runs as a native Windows process:

```json
{
  "mcpServers": {
    "autocad-mcp": {
      "type": "stdio",
      "command": "cmd.exe",
      "args": ["/d", "/s", "/c", "cd /d C:\\path\\to\\autocad-mcp && .venv\\Scripts\\python.exe -m autocad_mcp"],
      "env": { "AUTOCAD_MCP_BACKEND": "auto" }
    }
  }
}
```

### 4. Verify

From your MCP client, call:

```
system(operation="status")
```

You should see `backend: "file_ipc"` if AutoCAD is running, or `backend: "ezdxf"` for headless mode.

## Tools

### `drawing` — File/drawing management

| Operation | Description | File IPC | ezdxf |
|-----------|-------------|----------|-------|
| `create` | Reset to clean drawing (erase all + purge) | Yes | Yes |
| `open` | Open an existing drawing | Yes | Yes (DXF) |
| `info` | Get entity count and layers | Yes | Yes |
| `save` | Save current drawing (to path if given) | Yes | Yes |
| `save_as_dxf` | Export as DXF | Yes | Yes |
| `plot_pdf` | Plot to PDF | Yes | No |
| `purge` | Purge unused objects | Yes | Yes |
| `get_variables` | Get system variables by name | Yes | Yes |
| `undo` | Undo last operation | Yes | No |
| `redo` | Redo last undone operation | Yes | No |

### `entity` — Entity CRUD + modification

**Create:** `create_line`, `create_circle`, `create_polyline`, `create_rectangle`, `create_arc`, `create_ellipse`, `create_mtext`, `create_hatch`

**Read:** `list`, `count`, `get`

**Modify:** `copy`, `move`, `rotate`, `scale`, `mirror`, `offset`\*, `array`, `fillet`\*, `chamfer`\*, `erase`

> \* `offset`, `fillet`, `chamfer` are File IPC only (not supported in ezdxf headless backend).

### `layer` — Layer management

`list`, `create`, `set_current`, `set_properties`, `freeze`, `thaw`, `lock`, `unlock`

### `block` — Block operations

| Operation | File IPC | ezdxf |
|-----------|----------|-------|
| `list` | Yes | Yes |
| `insert` | Yes | Yes |
| `insert_with_attributes` | Yes | Yes |
| `get_attributes` | Yes | Yes |
| `update_attribute` | Yes | Yes |
| `define` | No | Yes |

### `annotation` — Text, dimensions, leaders

`create_text`, `create_dimension_linear`, `create_dimension_aligned`, `create_dimension_angular`, `create_dimension_radius`, `create_leader`

### `pid` — P&ID operations (CTO symbol library)

`setup_layers`, `insert_symbol`, `list_symbols`, `draw_process_line`, `connect_equipment`, `add_flow_arrow`, `add_equipment_tag`, `add_line_number`, `insert_valve`, `insert_instrument`, `insert_pump`, `insert_tank`

> P&ID symbol insertion requires the [CAD Tools Online](https://www.cadtoolsonline.com/) (CTO) P&ID Symbol Library installed at `C:\PIDv4-CTO\`. The ezdxf backend has built-in CTO library support. For the File IPC backend, some P&ID operations require additional LISP helpers — see the P&ID section in the wiki for setup details.

### `view` — Viewport and screenshot

| Operation | Description |
|-----------|-------------|
| `zoom_extents` | Zoom to show all entities |
| `zoom_window` | Zoom to a specified window |
| `get_screenshot` | Capture current AutoCAD view as PNG |

Screenshots use `PrintWindow` (Win32) for the File IPC backend — works even when AutoCAD is minimized or in the background. The ezdxf backend renders via matplotlib.

### `system` — Server management

`status`, `health`, `get_backend`, `runtime`, `init`, `execute_lisp`

> `execute_lisp` runs arbitrary AutoLISP code (File IPC only). Pass `data: {code: "(+ 1 2)"}`. This turns the server into an extensible automation platform — any valid AutoLISP expression can be executed.

### `trace` — Image → DXF (no vision model required)

Turn a screenshot, scan or photo of a **simple** drawing into a layered DXF file. Geometry is extracted with OpenCV — deterministically, locally, with no vision model and no API key — and an optional DeepSeek (text-only) pass names and groups what was found.

| Operation | Description |
|-----------|-------------|
| `image_to_dxf` | Trace and write a DXF. `data: {image, dxf?, scale?, width?, units?, use_llm?, preview?}` |
| `vectorize` | Extract the CAD IR only, no file written |
| `describe` | Trace in memory and return the semantic layer |

```powershell
# Same feature from the command line, no MCP client needed
autocad-trace drawing.png -o drawing.dxf --width 200 --units mm --preview traced.png
```

What it actually solves, measured rather than guessed:

- **Complete geometry** — every stroke is extracted from the pixels; nothing is hallucinated and nothing is silently dropped (`counters` reports what was found).
- **Line styles are measured, not guessed** — ink run-length analysis classifies `CONTINUOUS` / `DASHED` / `HIDDEN` / `CENTER` / `PHANTOM` per entity, and stroke width maps to ISO lineweights and conventional layers.
- **Arcs land on their two points** — arcs are fitted from pixels into canonical centre/radius/angles form, then snapped onto neighbouring line vertices.

Full design notes, tuning knobs and known limitations: **[docs/image-to-dxf.md](docs/image-to-dxf.md)**. A generated example lives in [`examples/`](examples/) (`python examples/make_example.py`).

## Architecture

```
MCP Client (Claude)
    │  stdio (JSON-RPC)
    ▼
Python MCP Server (autocad_mcp)
    │
    ├── File IPC Backend ──► C:/temp/*.json ──► mcp_dispatch.lsp (AutoCAD LT)
    │   PostMessageW(WM_CHAR) to MDIClient — no focus steal
    │
    ├── ezdxf Backend ──► in-memory DXF (headless, no AutoCAD needed)
    │
    └── trace (image → DXF) ──► OpenCV vectorisation ──► CAD IR ──► ezdxf
                                    │                      │
                                    │                      └── optional DeepSeek (text only)
                                    └── deterministic: complete geometry, measured linetypes
```

The File IPC backend sends keystrokes to AutoCAD's MDIClient window via `PostMessageW(WM_CHAR)`, triggering the `(c:mcp-dispatch)` AutoLISP command. This approach does **not** steal window focus — you can continue working in other applications while automation runs.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTOCAD_MCP_BACKEND` | `auto` | Backend selection: `auto`, `file_ipc`, `ezdxf` |
| `AUTOCAD_MCP_IPC_DIR` | `C:/temp` | Directory for IPC command/result JSON files (must match on both Python and LISP sides) |
| `AUTOCAD_MCP_IPC_TIMEOUT` | `10.0` | IPC command timeout in seconds (1-300) |
| `AUTOCAD_MCP_ONLY_TEXT` | `false` | Disable screenshot capture (text feedback only) |

> **Note:** If you change `AUTOCAD_MCP_IPC_DIR`, you must also update the `*mcp-ipc-dir*` variable in `mcp_dispatch.lsp` to match.

## Development

```powershell
uv sync
uv run pytest tests/ -v
```

## AutoCAD LT AutoLISP Compatibility

AutoLISP was added to AutoCAD LT in the **2024 release (Windows only)**. AutoCAD LT for Mac does not support AutoLISP.

| Supported (LT 2024+ Windows) | Not Supported |
|-------------------------------|---------------|
| `.lsp` / `.fas` / `.vlx` / `.dcl` | VLIDE (Visual LISP IDE) |
| All `vl-*` utility functions | `vlax-*` (ActiveX/COM) |
| File I/O (`open`, `read-line`, etc.) | Express Tools |
| Entity access (`entget`, `entmod`, etc.) | 3D operations |
| Selection sets | AutoLISP on Mac |

The `mcp_dispatch.lsp` dispatcher is fully compatible with LT 2024+.

## What's New in v3.2

- **Image → DXF tracing** (`trace` tool + `autocad-trace` CLI) — the `image_to_dxf`, `vectorize` and `describe` operations described above, with a full write-up in [docs/image-to-dxf.md](docs/image-to-dxf.md).
- **Deterministic geometry extraction** — no vision model, no API key, works offline: binarisation with polarity auto-detection, auto-upscale for hairline art, collinear merging with midline recentring, contour + Hough circle detection validated against the ink.
- **Measured linetypes** — `CONTINUOUS` / `DASHED` / `HIDDEN` / `CENTER` / `PHANTOM` from ink run-length analysis, plus ISO lineweights and conventional layer names.
- **Arcs that close on their endpoints** — canonical arc fitting, endpoint welding, and arc-angle snapping onto line vertices.
- **Optional DeepSeek semantic pass** — the model receives a JSON digest of verified geometry (never pixels) and returns validated layer names and a drawing summary; with no key, the deterministic labelling stands.
- **New modules**: `autocad_mcp.trace.{ir,geometry,linetype,vectorize,deepseek,semantics,emit,pipeline,cli}` and `examples/make_example.py`.
- **40 new tests** covering arc geometry, linetype classification, the end-to-end trace, and semantic-response validation.

## What's New in v3.1

- **`execute_lisp`** — Run arbitrary AutoLISP code via temp file pattern. Turns the server from a fixed command set into an extensible automation platform.
- **Undo / Redo** — Single-step undo and redo via `drawing` tool.
- **Drawing open** — Open existing `.dwg` files programmatically (FILEDIA suppressed).
- **Drawing create** — Now resets current drawing (erase all + purge) instead of `_.NEW`, preserving the LISP dispatcher namespace.
- **Drawing save with path** — `save` with a `path` parameter uses SAVEAS; without path uses QSAVE.
- **`get_variables` fix** — Respects the `names` parameter; returns requested variables with proper type handling.
- **Polyline/leader fix** — Point arrays properly encoded via semicolon-delimited format.
- **ESC prefix** — Sends 2x ESC before each dispatch to cancel stale pending commands from prior timeouts.
- **UTF-8/cp1252 fallback** — Handles non-ASCII characters in LISP result files (AutoCAD writes Windows-1252).
- **Configurable IPC timeout** — `AUTOCAD_MCP_IPC_TIMEOUT` env var (1–300 seconds, default 10).
- **Thread-safe backend init** — `asyncio.Lock` prevents parallel initialization races.

## 踩坑记录 — 开发中遇到的实际问题

> Engineering internship project — exploring AI-driven CAD automation feasibility. 以下如实记录开发过程中遇到的坑，既是自我复盘，也供后来者参考。

### 1. 中文图层名导致 IPC 超时（已解决）

**现象**：Python 端发送 `setvar CLAYER` 命令后，AutoCAD 端无响应，MCP 调用超时。

**根因**：Python `json.dumps` 默认 `ensure_ascii=True`，把中文字符（如「粗实线」）转成 `\uXXXX` 编码。AutoLISP 端的 `mcp-json-get-string` 只做简单字符串提取，不解码 `\uXXXX` 转义序列，导致 AutoCAD 收到的 layer name 是乱码，`setvar` 命令失败。

**解决**：将图层名全部改为英文（Thick / Thin / Center / Dim / Text / Hatch），或修改 Python 端使用 `ensure_ascii=False`。

**教训**：跨语言（Python ↔ AutoLISP）IPC 通信时，编码一致性问题是最容易忽略的坑。

### 2. execute_lisp 超时但实体已创建

**现象**：通过 `execute_lisp` 发送 `(command ...)` 类命令时，Python 端报超时错误，但打开 AutoCAD 发现实体其实已经画上去了。

**根因**：`(command ...)` 在 AutoLISP 中是同步阻塞的，执行时间取决于命令复杂度。MCP 的 IPC 超时设置（默认 10 秒）在复杂绘图命令下不够用，但 LISP 侧的 `(command)` 调用已经完成了。

**当前策略**：接受超时 → 通过 `get_screenshot` 截图验证实体是否实际创建成功。这是权宜之计，更好的方案是实现异步回调机制。

### 3. LISP 调度器需每次手动加载

每次打开 AutoCAD 后需手动 `APPLOAD` 加载 `mcp_dispatch.lsp`，或使用 Python 端 `WM_CHAR` 自动发送 `(load "...")` 命令。理想方案是加入 AutoCAD Startup Suite，但目前未自动化此步骤。

### 4. COM 接口对复杂曲线的限制

AutoCAD COM 接口（ActiveX）对渐开线齿廓、样条曲线等复杂几何的原生支持有限。对于齿轮工程图等场景，需要结合 AutoLISP 脚本或 ezdxf 后端在 Python 层面生成曲线数据再导入。

### 5. ezdxf 后端与 File IPC 后端的功能差异

ezdxf（headless）后端无需 AutoCAD 即可运行，但不支持 `offset`、`fillet`、`chamfer`、`plot_pdf`、`execute_lisp` 等依赖 AutoCAD 运行时 API 的操作。两个后端的行为差异容易让调用方困惑——同一段 MCP 调用在 File IPC 下成功、在 ezdxf 下静默失败。

### 6. 「把图丢给大模型生成 DXF」这条路走不通（v3.2 的根本动机）

**现象**：最初的思路是「一张图 → 交给 AI → AI 按 CAD 命令生成 DXF」。实际结果是图纸读不全、线型分不出来、圆弧对不上点。

**根因**（三个独立的坑，不是模型不够聪明）：

1. **DeepSeek 的公开 API 根本没有视觉能力**，而换用识图能力有限的模型时，细线、小圆弧、虚线的间隙大量丢失——更麻烦的是**丢了你也看不出来**，因为它会照样给你一份看起来合理的输出。
2. **线型不是"算"出来的，是"量"出来的**。让一个只会算坐标的模型判断实线还是虚线，它只能猜；而猜错在图纸上是实质错误（实线/虚线在制图里是不同语义：轮廓 vs 不可见边）。
3. **圆弧需要的是拟合，不是端点坐标**。只给两个端点，模型给出的圆心/半径/起止角往往不经过这两个点——数值看着正常，几何是错的。

**解决**（v3.2 的架构）：**"看见"这件事不交给大模型。** 几何用 OpenCV 确定性提取（读全、不幻觉、不静默丢失），线型用沿线墨迹游程分析量出来，圆弧用像素拟合 + 端点吸附对齐；大模型只拿一份**几何 JSON 摘要**（不含任何像素）去做它真正擅长的事——命名图层、分组、写图纸摘要。没有 API key 时，确定性分类照常给出可用的分层 DXF。

**教训**：判断一个任务该不该交给 LLM，先看它需要的是**测量**还是**命名**。测量类任务（坐标、长度、线型、几何一致性）交给确定性算法；命名与归纳类任务才交给模型。

### 7. 虚线被拆成十几条"实线"（开发中最耗时的坑）

**现象**：一张图上的虚线，识别结果是十几段独立的实线，图层也乱了。

**根因**：单段虚线（一条 dash）自身**没有任何内部间隙**——单独量它，它和实线完全无法区分。Hough 检测天然会把虚线按 dash 拆开，于是每一段都被判成 CONTINUOUS。

**解决**：把顺序改成 **先量、再链、最后判**——先测每段的位置与笔画宽度，再把共线片段跨虚线间隙链接成整条线（短片段用更宽的间隙阈值，因为长划中心线的长划之间隔着一整个周期），最后用**整条线**的墨迹图案判一次线型。

**教训**：抽样的**顺序**会决定结论的对错。同一个像素集，切成碎片判断和合并后判断，得到的是两种答案。

### 8. 「中线校正」把线推得更偏（符号写反）

**现象**：矩形四条边识别出来位置整体偏了约 8–10px（差不多一个线宽的两倍）。

**根因**：粗线经 Canny 会得到一对边缘，合并时要把结果拉回笔画中线。校正量的符号写反了——`signed_offset` 与校正方向用了相反的符号约定，于是"校正"变成了"加速偏离"，偏移量正好是 2× 半线宽。

**解决**：统一符号约定并写进文档字符串；再加一步"按实际墨迹重新居中"（把线段沿法向滑动，取墨迹最多的一档），把 Canny/Hough 残留的半个线宽偏差一并消掉。

**教训**：涉及方向的几何代码，符号约定必须显式写出来并配单元测试。这类 bug 不会崩、不会报错，只会让所有坐标静默偏移。

---

**总结**：这个项目让我真正理解了「AI 落地的最后一公里」——不是模型不够聪明，而是工程环境（编码、IPC、超时、不同后端的一致性）里藏着一堆需要逐个解决的小问题。这些问题在教科书和 API 文档里都不会告诉你。

## 关于本项目

本项目 fork 自 [puran-water/autocad-mcp](https://github.com/puran-water/autocad-mcp)，感谢原作者的开源工作。

## License

MIT
