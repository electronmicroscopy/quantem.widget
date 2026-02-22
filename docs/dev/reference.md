# Reference


## Glossary

### Python terms

**`traitlets`** — The library that defines typed attributes on Python objects.
Every widget property that syncs to JavaScript is a traitlet.

```python
pos_row = traitlets.Int(0).tag(sync=True)
```

Common types: `Int`, `Float`, `Bool`, `Unicode` (string), `Bytes` (binary),
`List`, `Dict`.

**`.tag(sync=True)`** — Marks a traitlet for synchronization with JavaScript.
Without this tag, the trait exists only in Python.

**`.observe(callback, names=[...])`** — Registers a function that runs whenever
the named traits change:

```python
self.observe(self._update_frame, names=["pos_row", "pos_col"])
```

**`anywidget.AnyWidget`** — The base class every widget extends. Handles trait
synchronization, binary transport, and widget lifecycle.

**`@property`** — Turns a method into a read-only attribute. Used for computed
values like `widget.profile_values`.

**`Self` return type** — Type hint for method chaining (`from typing import Self`).

**`.tobytes()`** — Converts a NumPy array to raw bytes for sending to JS:

```python
self.frame_bytes = frame.astype(np.float32).tobytes()
```

**`hasattr` (duck typing)** — Auto-detect `quantem` Dataset objects without
importing `quantem`:

```python
if hasattr(data, "array"):
    title = getattr(data, "name", "")
```


### JavaScript / TypeScript terms

**JSX** — HTML-like syntax inside JavaScript. `<Box sx={{ color: "red" }}>Hello</Box>`
compiles to `React.createElement(...)`.

**Hooks** — Functions that manage state and side effects:

- **`useState(initial)`** — state variable, triggers re-render on change
- **`useEffect(() => { ... }, [deps])`** — runs code when dependencies change (canvas rendering, data processing)
- **`useRef(initial)`** — persistent reference that survives re-renders (canvas elements, drag state)
- **`useMemo(() => value, [deps])`** — cached computed value
- **`useCallback(() => fn, [deps])`** — cached function reference

**`useModelState`** — The anywidget hook bridging React to Python traitlets:

```tsx
const [cmap, setCmap] = useModelState<string>("cmap");
```

The string must exactly match the Python trait name. A typo silently returns
`undefined`.

**`DataView` → `Float32Array`** — `Bytes` traitlets arrive as `DataView`.
Convert with `extractFloat32()` from `format.ts`.

**Canvas API** — `getContext("2d")`, `clearRect`, `drawImage`, `save`/`restore`.
Widgets render to offscreen canvas first, then `drawImage()` with zoom/pan
transforms.

**DPR** — `window.devicePixelRatio`. UI canvas uses `width={cssW * DPR}`,
`style={{ width: cssW }}`, `ctx.scale(DPR, DPR)` for crisp text on HiDPI.

**`ref` / `.current`** — Persistent DOM references via `useRef()`. Don't trigger
re-renders.


### MUI (Material UI)

- **`Box`** — styled `<div>` with `sx` prop
- **`Stack`** — flexbox layout (`direction="row"` for horizontal)
- **`Typography`** — text with consistent styling
- **`sx`** — inline CSS as JS object (`fontSize: 10`, `mb: "4px"`, `bgcolor: "#fff"`)
- **`Switch`** — toggle (FFT, log scale); **`Select`** — dropdown (colormap); **`Button`** — action; **`Slider`** — range


### Build tools

**esbuild** — JS bundler. `npm run build` (one-shot, ~130ms) or `npm run dev` (watch mode).

**anywidget HMR** — Hot Module Replacement. With `npm run dev` + `ANYWIDGET_HMR=1`,
JS changes appear without kernel restart.


## Troubleshooting

**Trait exists in Python but JS reads `undefined`** — You forgot `.tag(sync=True)`.

**Python changes don't take effect** — Restart the kernel. Python code is loaded
at import time.

**Trait isn't syncing** — Check spelling. `useModelState("pos_row")` must exactly
match the Python trait name.

**Theme colors** — Use `colors` from `useTheme()` for all UI chrome:

```tsx
const { colors } = useTheme();
<Box sx={{ bgcolor: colors.bg, color: colors.text, border: `1px solid ${colors.border}` }}>
```

Available: `bg`, `bgAlt`, `text`, `textMuted`, `border`, `controlBg`, `accent`.

**Colormaps** — Applied entirely in JS. Python sends raw float32, JS maps via LUT:

```tsx
import { COLORMAPS, applyColormap } from "../colormaps";
applyColormap(floatData, rgbaBuffer, COLORMAPS["inferno"], vmin, vmax);
```


## Testing

### Unit tests

```bash
python -m pytest tests/ -v --ignore=tests/test_e2e_smoke.py   # all
python -m pytest tests/test_widget_show4dstem.py -v            # one widget
```

### End-to-end smoke tests

Requires Playwright + JupyterLab. Renders widgets in a real browser, captures
screenshots (light + dark theme), tests interactions. ~4 minutes:

```bash
python -m pytest tests/test_e2e_smoke.py -v                   # all
python -m pytest tests/test_e2e_smoke.py -v -k show2d          # one widget
```

### Screenshot verification

After modifying widget UI: `npm run build` → run smoke tests → visually verify
`tests/screenshots/smoke/`.

| File | Purpose |
|------|---------|
| `test_widget_*.py` | Unit tests (traits, shapes, data, state, ROI, display) |
| `capture_*.py` | Screenshot capture scripts (run manually) |
| `test_e2e_smoke.py` | Full E2E via Playwright |


## Publishing

### Docs

```bash
pip install -e ".[docs]"

# One-shot build
sphinx-build docs docs/_build/html

# Live reload (rebuilds on file change, opens browser)
sphinx-autobuild docs docs/_build/html --open-browser --port 8322
```

### TestPyPI

Currently published on [TestPyPI](https://test.pypi.org/project/quantem-widget/) (not yet on PyPI).

1. Bump version in `pyproject.toml`
2. Tag and push: `git tag vX.Y.Z && git push origin main && git push origin vX.Y.Z`
3. CI builds and uploads to TestPyPI. Verify: `./scripts/verify_testpypi.sh X.Y.Z`

Install from TestPyPI:

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ quantem-widget
```
