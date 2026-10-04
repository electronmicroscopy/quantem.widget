import { useTheme } from "../theme";
import * as React from "react";
import {
  Slider,
  Popover,
  FormControlLabel,
  Switch,
  IconButton,
} from "@mui/material";
import SettingsOutlined from "@mui/icons-material/SettingsOutlined";
import { CompactSelect, compactSlider, CifControlTheme } from "./controls";
import { createRender, useModelState } from "@anywidget/react";
import {
  temporalAverageFrameIndices,
  normalizedAverageWindow,
} from "../show3d/frameTransform";
import { extractFloat32 } from "../format";
import { ScaleBar } from "./ScaleBar";
import { PotentialPanel } from "./PotentialPanel";
import { AtomRenderer } from "./render";
import {
  projectionBasis,
  specimenTiltBasis,
  cameraBasis,
  cellCorners,
  dot,
  type V3,
  repeatedAtom,
  repeatCount,
  calibratedFov,
  orthogonalBases,
} from "./geometry";

type Species = {
  symbol: string;
  number: number;
  color: number[];
  count: number;
};
function App() {
  const { colors } = useTheme();
  const [settingsAnchor, setSettingsAnchor] =
    React.useState<HTMLElement | null>(null);
  const settingsId = React.useId();
  const [showProjection, setShowProjection] = React.useState(true);
  const [showSlices, setShowSlices] = React.useState(false);
  const [showPotential, setShowPotential] = React.useState(false);
  const [title] = useModelState<string>("title"),
    [bytes] = useModelState<DataView>("unit_atom_bytes");
  const [unit] = useModelState<number[][]>("unit_cell");
  const [savedSpecies] = useModelState<Species[]>("species"),
    [repeats, setRepeats] = useModelState<number[]>("repeats");
  const [zone, setZone] = useModelState<number[]>("zone_axis"),
    [visible, setVisible] = useModelState<boolean[]>("visible_species");
  const [savedTilt, setSavedTilt] =
    useModelState<number[]>("specimen_tilt_mrad");
  const [tilt, setTilt] = React.useState<number[]>(savedTilt ?? [0, 0]);
  const tiltDraft = React.useRef(tilt);
  const tiltFrame = React.useRef(0);
  React.useEffect(() => {
    tiltDraft.current = savedTilt ?? [0, 0];
    setTilt(tiltDraft.current);
  }, [savedTilt]);
  React.useEffect(() => () => cancelAnimationFrame(tiltFrame.current), []);
  const changeTilt = (axis: number, value: number) => {
    tiltDraft.current = tiltDraft.current.map((v, i) =>
      i === axis ? value : v,
    );
    if (!tiltFrame.current)
      tiltFrame.current = requestAnimationFrame(() => {
        tiltFrame.current = 0;
        setTilt([...tiltDraft.current]);
      });
  };
  const commitTilt = () => setSavedTilt([...tiltDraft.current]);
  const [potentialBytes] = useModelState<DataView>("potential_bytes");
  const [potentialPixels] = useModelState<number>("potential_pixels");
  const [potentialSigma] = useModelState<number>("potential_sigma_A");
  const [sliceCount, setSliceCount] = useModelState<number>("num_slices");
  const [viewMode, setViewMode] = useModelState<string>("view_mode");
  const [fov, setFov] = useModelState<number>("field_of_view_A");
  const [calibration, setCalibration] = useModelState<number[]>(
    "magnification_calibration",
  );
  const [orthogonal, setOrthogonal] =
    useModelState<boolean>("orthogonal_views");
  const [fovUnit, setFovUnit] = React.useState("nm");
  const [calDraft, setCalDraft] = React.useState(["", ""]);
  const [calError, setCalError] = React.useState("");
  const microscope = viewMode === "microscope";
  const unitFactor = fovUnit === "nm" ? 10 : 1;
  const hasCalibration = calibration?.length === 2;
  const [gpuDevice, setGpuDevice] = React.useState<GPUDevice>();
  const potentialTable = React.useMemo(
    () => extractFloat32(potentialBytes) ?? new Float32Array(),
    [potentialBytes],
  );
  const [summary] = useModelState<string>("source_summary");
  const [draft, setDraft] = React.useState((zone || [0, 0, 1]).join(" "));
  const [status, setStatus] = React.useState("Starting WebGPU…"),
    [pick, setPick] = React.useState(
      "Click an atom or projected column to inspect it.",
    );
  const [sliceSelection, setSliceSelection] = React.useState(false);
  const [sliceIndex, setSliceIndex] = React.useState(0);
  const [averageWidth, setAverageWidth] = React.useState(1);
  const [previewSlice, setPreviewSlice] = React.useState<number | null>(null);
  const previewFrame = React.useRef(0);
  const preview = (i: number | null) => {
    cancelAnimationFrame(previewFrame.current);
    previewFrame.current = requestAnimationFrame(() => setPreviewSlice(i));
  };
  React.useEffect(() => () => cancelAnimationFrame(previewFrame.current), []);
  const selectedIndex = Math.min(sliceCount - 1, previewSlice ?? sliceIndex);
  const selectedFrames =
    sliceSelection || previewSlice !== null
      ? temporalAverageFrameIndices(selectedIndex, sliceCount, averageWidth)
      : undefined;
  const slab = selectedFrames
    ? [
        selectedFrames[0] / sliceCount,
        (selectedFrames[selectedFrames.length - 1] + 1) / sliceCount,
      ]
    : [0, 1];
  const [radius, setRadius] = React.useState(5),
    [zoom, setZoom] = React.useState(1);
  const refs = [
    React.useRef<HTMLCanvasElement>(null),
    React.useRef<HTMLCanvasElement>(null),
    React.useRef<HTMLCanvasElement>(null),
    React.useRef<HTMLCanvasElement>(null),
  ];
  const lines = [
    React.useRef<SVGSVGElement>(null),
    React.useRef<SVGSVGElement>(null),
    React.useRef<SVGSVGElement>(null),
    React.useRef<SVGSVGElement>(null),
  ];
  const renderers = React.useRef<AtomRenderer[]>([]),
    angles = React.useRef([0.55, 0.35]);
  const pending = React.useRef(0),
    drawRef = React.useRef(() => {}),
    drag = React.useRef<{
      x: number;
      y: number;
      active: boolean;
      moved: boolean;
    }>({ x: 0, y: 0, active: false, moved: false });
  const atoms = React.useMemo(
    () => extractFloat32(bytes) ?? new Float32Array(),
    [bytes],
  );
  const cell = React.useMemo(
    () => unit.map((v, i) => v.map((x) => x * repeats[i])),
    [unit, repeats],
  );
  const copies = repeats.reduce((n, x) => n * x, 1);
  const atomCount = repeatCount(atoms.length / 4, repeats);
  const species = React.useMemo(
    () =>
      savedSpecies.map((s, i) => ({
        ...s,
        count:
          Array.from({ length: atoms.length / 4 }, (_, k) =>
            atoms[k * 4 + 3] === i ? 1 : 0,
          ).reduce((a: number, b: number) => a + b, 0) * copies,
      })),
    [savedSpecies, atoms, copies],
  );
  const [repeatDraft, setRepeatDraft] = React.useState(repeats.map(String));
  const [repeatError, setRepeatError] = React.useState("");
  React.useEffect(() => setRepeatDraft(repeats.map(String)), [repeats]);
  const changeRepeat = (index: number, text: string) => {
    const draft = [...repeatDraft];
    draft[index] = text;
    setRepeatDraft(draft);
    const values = draft.map(Number);
    try {
      repeatCount(atoms.length / 4, values);
    } catch (e) {
      setRepeatError(String((e as Error).message));
      return;
    }
    setRepeatError("");
    setRepeats(values);
    setPick(
      "Inspection region changed; CIF unit cell and simulation are unchanged.",
    );
  };
  const corners = React.useMemo(() => cellCorners(cell), [cell]);
  const basis = React.useMemo(
    () => specimenTiltBasis(projectionBasis(unit, zone), tilt),
    [unit, zone, tilt],
  );
  const center = corners[7].map((v) => v / 2) as V3;
  const depth = corners.map((p) => dot(p, basis.beam));
  const zmin = Math.min(...depth),
    zmax = Math.max(...depth);
  const lateral =
    Math.max(...corners.map((p) => dot(p, basis.right))) -
    Math.min(...corners.map((p) => dot(p, basis.right)));
  const vertical =
    Math.max(...corners.map((p) => dot(p, basis.up))) -
    Math.min(...corners.map((p) => dot(p, basis.up)));
  const size =
    Math.max(
      ...corners.map((p) => Math.hypot(...p.map((v, i) => v - center[i]))),
    ) * 2;
  const spans = [
    (size * 1.15) / zoom,
    microscope ? fov : (Math.max(lateral, vertical) * 1.15) / zoom,
    microscope
      ? Math.max(fov, (zmax - zmin) * 1.15)
      : (Math.max(lateral, zmax - zmin) * 1.15) / zoom,
    microscope
      ? Math.max(fov, (zmax - zmin) * 1.15)
      : (Math.max(vertical, zmax - zmin) * 1.15) / zoom,
  ];
  const viewBases = () => [
    cameraBasis(basis, angles.current[0], angles.current[1]),
    ...orthogonalBases(basis),
  ];
  const limits = [
    zmin + slab[0] * (zmax - zmin),
    zmin + slab[1] * (zmax - zmin),
  ];
  const schedule = () => {
    if (!pending.current)
      pending.current = requestAnimationFrame(() => {
        pending.current = 0;
        drawRef.current();
      });
  };
  const edges = React.useMemo(
    () =>
      Array.from({ length: 8 }, (_, k) =>
        [0, 1, 2].filter((i) => !(k & (1 << i))).map((i) => [k, k | (1 << i)]),
      ).flat(),
    [],
  );
  drawRef.current = () => {
    viewBases().forEach((b, i) => {
      if ((i === 1 && !showProjection) || (i > 1 && !orthogonal)) return;
      renderers.current[i]?.draw(
        atomCount,
        b,
        center,
        spans[i],
        basis.beam,
        limits,
        species.map((s) => s.color),
        visible,
        radius,
        unit,
        repeats,
        atoms.length / 4,
        limits[1] >= zmax,
        ((zmax - zmin) / sliceCount) * 1e-5,
      );
      const box = refs[i].current?.getBoundingClientRect();
      if (!box) return;
      const svg = lines[i].current;
      if (!svg) return;
      const pos = corners.map((p) => {
        const q = p.map((v, j) => v - center[j]);
        return [
          50 + (dot(q, b.right) * 100) / spans[i],
          50 - (((dot(q, b.up) * 100) / spans[i]) * box.width) / box.height,
        ];
      });
      [...svg.querySelectorAll("line")].forEach((line, j) => {
        const [a, c] = edges[j];
        for (const [name, v] of Object.entries({
          x1: pos[a][0],
          y1: pos[a][1],
          x2: pos[c][0],
          y2: pos[c][1],
        }))
          line.setAttribute(name, String(v));
      });
    });
  };
  React.useEffect(() => {
    let cancelled = false;
    let device: GPUDevice | undefined;
    const observer = new ResizeObserver(schedule);
    (async () => {
      if (!navigator.gpu)
        throw Error(
          "WebGPU unavailable. Open this widget over HTTPS or localhost in a WebGPU browser.",
        );
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw Error("No WebGPU adapter available.");
      device = await adapter.requestDevice();
      if (cancelled) {
        device.destroy();
        return;
      }
      device.addEventListener("uncapturederror", (e) =>
        setStatus("WebGPU error: " + e.error.message),
      );
      renderers.current = refs.map(
        (r) =>
          new AtomRenderer(
            r.current!,
            device!,
            atoms,
            species.map((s) => s.color),
          ),
      );
      refs.forEach((r) => observer.observe(r.current!));
      setGpuDevice(device);
      setStatus("WebGPU · exact atomic positions · display filters only");
      schedule();
    })().catch((e) => setStatus(String(e)));
    return () => {
      cancelled = true;
      observer.disconnect();
      cancelAnimationFrame(pending.current);
      renderers.current.forEach((r) => r.destroy());
      renderers.current = [];
      setGpuDevice(undefined);
      device?.destroy();
    };
  }, [atoms]);
  React.useEffect(schedule, [
    tilt,
    zone,
    visible,
    slab,
    radius,
    zoom,
    repeats,
    viewMode,
    fov,
    orthogonal,
    showProjection,
  ]);
  const selectZone = (v: number[]) => {
    if (
      v.length !== 3 ||
      !v.some((x) => x !== 0) ||
      v.some((x) => !Number.isInteger(x))
    ) {
      setPick("Enter three integers [u v w], not all zero.");
      return;
    }
    setZone(v);
    setDraft(v.join(" "));
    setPick("Projection direction changed; 3D camera rotation is independent.");
  };
  const inspect = (e: React.PointerEvent<HTMLCanvasElement>, panel: number) => {
    const box = e.currentTarget.getBoundingClientRect(),
      x = ((e.clientX - box.left - box.width / 2) * spans[panel]) / box.width,
      y = (-(e.clientY - box.top - box.height / 2) * spans[panel]) / box.width;
    const b = viewBases()[panel];
    const tolerance = ((radius + 3) * spans[panel]) / box.width;
    const hits: {
      id: number;
      s: number;
      z: number;
      cameraZ: number;
      pos: number[];
    }[] = [];
    for (let i = 0; i < atomCount; i++) {
      const atom = repeatedAtom(atoms, unit, repeats, i);
      const s = atom[3];
      if (!visible[s]) continue;
      const p = atom.slice(0, 3),
        z = dot(p, basis.beam);
      if (
        z < limits[0] - ((zmax - zmin) / sliceCount) * 1e-5 ||
        (limits[1] < zmax
          ? z >= limits[1] - ((zmax - zmin) / sliceCount) * 1e-5
          : z > limits[1] + ((zmax - zmin) / sliceCount) * 1e-5)
      )
        continue;
      const q = p.map((v, j) => v - center[j]);
      if (Math.hypot(dot(q, b.right) - x, dot(q, b.up) - y) < tolerance)
        hits.push({ id: i, s, z, cameraZ: dot(p, b.beam), pos: p });
    }
    hits.sort((a, b) => b.cameraZ - a.cameraZ);
    const counts = species
      .map((s, i) => `${s.symbol}: ${hits.filter((h) => h.s === i).length}`)
      .join(" · ");
    setPick(
      hits.length
        ? `${hits.length} atoms in selection radius · ${counts}. Beam depths ${hits.reduce((m, h) => Math.min(m, h.z), Infinity).toFixed(2)}–${hits.reduce((m, h) => Math.max(m, h.z), -Infinity).toFixed(2)} Å. Front atom #${hits[0].id}: ${hits[0].pos.map((v) => v.toFixed(3)).join(", ")} Å (Cartesian x,y,z).`
        : "No visible atoms at this position.",
    );
  };
  const chooseSlice = (i: number) => {
    preview(null);
    setSliceIndex(i);
    setShowSlices(true);
    setSliceSelection(true);
  };
  return (
    <CifControlTheme.Provider value={colors}>
      <div
        className="qcif"
        style={
          {
            "--cif-bg": colors.bg,
            "--cif-control": colors.controlBg,
            "--cif-text": colors.text,
            "--cif-muted": colors.textMuted,
            "--cif-border": colors.border,
            "--cif-accent": colors.accent,
          } as React.CSSProperties
        }
      >
        <style>{`.qcif{font:13px system-ui;background:var(--cif-bg);color:var(--cif-text);padding:10px;border:1px solid var(--cif-border);border-radius:0}.qcif *{box-sizing:border-box}.qcif h3{margin:0 0 5px;font-size:16px}.qcif p{margin:6px 0;line-height:1.45}.qcif .row{display:flex;align-items:center;flex-wrap:wrap;gap:7px;margin:5px 0}.qcif button:not(.MuiButtonBase-root),.qcif input[type=text],.qcif input[type=number]{background:var(--cif-control);color:var(--cif-text);border:1px solid var(--cif-border);border-radius:0;padding:4px 7px;font:inherit}.qcif button[aria-pressed=true]{background:var(--cif-control);border-color:var(--cif-accent);color:var(--cif-accent)}.qcif button{cursor:pointer}.qcif .panels{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.qcif figure{margin:0;min-width:0}.qcif .scene{position:relative;aspect-ratio:1;overflow:hidden}.qcif canvas{display:block;width:100%;height:100%;touch-action:none}.qcif .scene>svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.qcif figcaption{margin:4px 0}.qcif label{display:inline-flex;align-items:center;gap:5px}.qcif input[type=range]{width:120px}.qcif .notes{margin:5px 0;color:var(--cif-muted);font-size:12px}.qcif summary{cursor:pointer;width:fit-content;padding:4px 0}.qcif .hint{color:var(--cif-muted);font-size:12px}.qcif output{font-variant-numeric:tabular-nums}.qcif section:not(.potential-panel){border:0;padding:0;margin:0;background:transparent}.qcif .potential-panel{background:transparent;border-radius:0;margin-top:12px;border-top:1px solid var(--cif-border);padding-top:10px}.qcif .potential-gallery,.qcif .potential-maps{display:grid;grid-template-columns:repeat(var(--map-cols,2),minmax(0,1fr));gap:8px}.qcif .slice-image{display:block;padding:0;width:100%;aspect-ratio:1;overflow:hidden}.qcif .slice-image canvas{width:100%;height:100%;pointer-events:none}@media(max-width:700px){.qcif .potential-gallery,.qcif .potential-maps{grid-template-columns:repeat(min(var(--map-cols,2),2),minmax(0,1fr))}}@media(max-width:600px){.qcif .panels{grid-template-columns:1fr}}@media(max-width:450px){.qcif .potential-gallery,.qcif .potential-maps{grid-template-columns:1fr}}`}</style>
        <div
          className="row"
          style={{ justifyContent: "space-between", marginTop: 0 }}
        >
          <h3 style={{ margin: 0, flex: 1 }}>{title}</h3>
          <IconButton
            aria-label="View settings"
            title="View settings"
            size="small"
            aria-expanded={Boolean(settingsAnchor)}
            aria-controls={settingsAnchor ? settingsId : undefined}
            onClick={(e) => setSettingsAnchor(e.currentTarget)}
            sx={{
              color: colors.textMuted,
              width: 28,
              height: 28,
              borderRadius: 0,
            }}
          >
            <SettingsOutlined fontSize="small" />
          </IconButton>
        </div>
        <Popover
          open={Boolean(settingsAnchor)}
          anchorEl={settingsAnchor}
          onClose={() => setSettingsAnchor(null)}
          anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
          transformOrigin={{ vertical: "top", horizontal: "right" }}
          slotProps={{
            paper: {
              sx: {
                backgroundColor: colors.controlBg,
                color: colors.text,
                border: `1px solid ${colors.border}`,
                borderRadius: 0,
                p: 1.5,
                maxWidth: "calc(100vw - 32px)",
              },
            },
          }}
        >
          <div
            id={settingsId}
            role="group"
            aria-label="Visible views"
            style={{ display: "flex", flexDirection: "column" }}
          >
            <strong style={{ fontSize: 13, marginBottom: 4 }}>
              Show views
            </strong>
            <FormControlLabel
              sx={{ m: 0 }}
              control={
                <Switch
                  size="small"
                  checked={showProjection}
                  onChange={(_, v) => setShowProjection(v)}
                />
              }
              label="Column projection"
            />
            <FormControlLabel
              sx={{ m: 0 }}
              control={
                <Switch
                  size="small"
                  checked={orthogonal}
                  onChange={(_, v) => setOrthogonal(v)}
                />
              }
              label="Orthogonal projections"
            />
            <FormControlLabel
              sx={{ m: 0 }}
              control={
                <Switch
                  size="small"
                  checked={showSlices}
                  onChange={(_, v) => {
                    setShowSlices(v);
                    if (v) setSliceSelection(true);
                  }}
                />
              }
              label="Slice controls"
            />
            <FormControlLabel
              sx={{ m: 0 }}
              disabled={!potentialTable.length}
              control={
                <Switch
                  size="small"
                  checked={showPotential}
                  onChange={(_, v) => setShowPotential(v)}
                />
              }
              label="Potential / phase maps"
            />
          </div>
        </Popover>
        <details className="notes">
          <summary>Structure details</summary>
          {summary}
        </details>
        <div className="row" aria-label="Structure view mode">
          <label>
            View{" "}
            <CompactSelect
              label="Structure view mode"
              value={viewMode}
              options={[
                ["unit_cells", "Unit cells"],
                ["microscope", "Microscope"],
              ]}
              onChange={setViewMode}
            />
          </label>
        </div>
        {microscope && (
          <section aria-label="Microscope field of view">
            <div className="row">
              <label>
                FOV
                <input
                  type="number"
                  aria-label="Microscope field of view"
                  min="0.001"
                  step="0.1"
                  value={Number((fov / unitFactor).toPrecision(8))}
                  style={{ width: 100 }}
                  onChange={(e) => {
                    const value = Number(e.target.value) * unitFactor;
                    if (Number.isFinite(value) && value > 0) setFov(value);
                  }}
                />
              </label>
              <CompactSelect
                label="Field of view unit"
                value={fovUnit}
                options={[
                  ["nm", "nm"],
                  ["Å", "Å"],
                ]}
                onChange={setFovUnit}
              />
            </div>
            <details className="notes">
              <summary>Magnification calibration</summary>
              {hasCalibration ? (
                <div className="row">
                  <label>
                    Magnification
                    <input
                      type="number"
                      aria-label="Microscope magnification in million times"
                      min="0.000001"
                      step="0.1"
                      style={{ width: 95 }}
                      value={Number(
                        (
                          (calibration[0] * calibration[1]) /
                          fov /
                          1e6
                        ).toPrecision(8),
                      )}
                      onChange={(e) => {
                        try {
                          setFov(
                            calibratedFov(
                              Number(e.target.value) * 1e6,
                              calibration,
                            ),
                          );
                        } catch {}
                      }}
                    />
                    M×
                  </label>
                  <span className="hint">
                    User calibration · {(calibration[0] / 1e6).toPrecision(4)}{" "}
                    M× = {(calibration[1] / 10).toPrecision(4)} nm · same camera
                    and acquisition geometry
                  </span>
                  <button onClick={() => setCalibration([])}>
                    Clear Calibration
                  </button>
                </div>
              ) : (
                <>
                  <p className="hint">
                    Magnification alone does not determine field of view. Enter
                    it directly above, or supply a measured reference below.
                  </p>
                  <div className="row">
                    <label>
                      Reference Mag{" "}
                      <input
                        type="number"
                        aria-label="Reference magnification in million times"
                        min="0.000001"
                        step="0.1"
                        style={{ width: 85 }}
                        value={calDraft[0]}
                        onChange={(e) =>
                          setCalDraft([e.target.value, calDraft[1]])
                        }
                      />{" "}
                      M×
                    </label>
                    <label>
                      Reference FOV{" "}
                      <input
                        type="number"
                        aria-label="Reference field of view in nm"
                        min="0.001"
                        step="0.1"
                        style={{ width: 85 }}
                        value={calDraft[1]}
                        onChange={(e) =>
                          setCalDraft([calDraft[0], e.target.value])
                        }
                      />{" "}
                      nm
                    </label>
                    <button
                      onClick={() => {
                        const c = [
                          Number(calDraft[0]) * 1e6,
                          Number(calDraft[1]) * 10,
                        ];
                        try {
                          calibratedFov(c[0], c);
                          setCalibration(c);
                          setCalError("");
                        } catch {
                          setCalError(
                            "Enter positive reference magnification and FOV from the same acquisition geometry.",
                          );
                        }
                      }}
                    >
                      Apply Calibration
                    </button>
                  </div>
                  {calError && <p role="alert">{calError}</p>}
                </>
              )}
              <p className="hint">
                FOV changes the view, not the number of atoms or specimen depth.
                Adjust Unit Cells below to cover a larger field. Open boundaries
                remain visible; no periodic atoms are silently added.
              </p>
            </details>
            {fov > Math.min(lateral, vertical) && (
              <p role="status">
                Field exceeds the inspection cell's projected extent (
                {lateral.toFixed(2)} × {vertical.toFixed(2)} Å). Increase
                repeats to inspect a larger crystal patch.
              </p>
            )}
          </section>
        )}
        <div className="row">
          <strong>Unit Cells</strong>
          {["a", "b", "c"].map((axis, i) => (
            <label key={axis}>
              {axis}
              <input
                type="number"
                aria-label={`Unit cells along ${axis}`}
                min="1"
                step="1"
                value={repeatDraft[i]}
                onChange={(e) => changeRepeat(i, e.target.value)}
                onBlur={() => {
                  setRepeatDraft(repeats.map(String));
                  setRepeatError("");
                }}
                style={{ width: 72 }}
              />
            </label>
          ))}
          <output>
            {copies.toLocaleString()} cells · {atomCount.toLocaleString()} atoms
          </output>
          <span className="hint">
            {cell.map((v) => Math.hypot(...v).toFixed(2)).join(" × ")} Å along
            a, b, c
          </span>
        </div>
        {repeatError && <p role="alert">{repeatError}</p>}
        <div className="row">
          <strong>Show atoms</strong>
          {species.map((s, i) => (
            <button
              key={s.symbol}
              aria-pressed={visible[i]}
              onClick={() =>
                setVisible(visible.map((v, j) => (i === j ? !v : v)))
              }
            >
              <span
                style={{
                  color: `rgb(${s.color.map((c) => Math.round(c * 255)).join(",")})`,
                }}
              >
                ●
              </span>{" "}
              {s.symbol} ({s.count})
            </button>
          ))}
          <button onClick={() => setVisible(species.map(() => true))}>
            Show All
          </button>
        </div>
        <div className="row">
          <label>
            Beam{" "}
            <CompactSelect
              label="Beam direction"
              value={
                [
                  [0, 0, 1],
                  [1, 0, 0],
                  [1, 1, 0],
                  [1, 1, 1],
                ].some((v) => v.join() === zone.join())
                  ? zone.join()
                  : "custom"
              }
              options={[
                ["0,0,1", "[001]"],
                ["1,0,0", "[100]"],
                ["1,1,0", "[110]"],
                ["1,1,1", "[111]"],
                ["custom", "Custom"],
              ]}
              onChange={(v) => {
                if (v !== "custom") selectZone(v.split(",").map(Number));
              }}
            />
          </label>
          <details className="notes">
            <summary>Custom direction</summary>
            <div className="row">
              <label>
                Custom{" "}
                <input
                  type="text"
                  aria-label="Custom direct lattice direction"
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  style={{ width: 85 }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter")
                      selectZone(draft.trim().split(/[ ,]+/).map(Number));
                  }}
                />
              </label>
              <button
                onClick={() =>
                  selectZone(draft.trim().split(/[ ,]+/).map(Number))
                }
              >
                Apply
              </button>
            </div>
          </details>
        </div>
        <div
          className="row"
          aria-label="Specimen tilt"
          title="Rigid specimen tilt about the cell center. Positive row/column leans down/right with increasing beam depth. Independent of camera rotation."
        >
          <strong>Tilt</strong>
          {["Row", "Col"].map((label, axis) => (
            <label key={label} style={{ gap: 8 }}>
              {label}
              <Slider
                size="small"
                aria-label={`Specimen ${axis === 0 ? "row" : "column"} tilt in mrad`}
                min={-15}
                max={15}
                step={0.1}
                marks={[{ value: 0 }]}
                value={tilt[axis]}
                onChange={(_, v) => changeTilt(axis, v as number)}
                onChangeCommitted={commitTilt}
                sx={{ ...compactSlider, width: 140, mx: 1 }}
              />
              <output style={{ width: 40, textAlign: "right" }}>
                {tilt[axis].toFixed(1)}
              </output>
            </label>
          ))}
          <span className="hint">mrad</span>
          <button
            aria-label="Reset specimen tilt"
            onClick={() => {
              cancelAnimationFrame(tiltFrame.current);
              tiltFrame.current = 0;
              tiltDraft.current = [0, 0];
              setTilt([0, 0]);
              setSavedTilt([0, 0]);
            }}
          >
            Zero
          </button>
        </div>
        <div hidden={!showSlices}>
          <div className="row" aria-label="Beam slice controls">
            <span>Slice</span>
            <Slider
              size="small"
              aria-label="Beam slice"
              min={0}
              max={Math.max(1, sliceCount - 1)}
              disabled={sliceCount === 1}
              step={1}
              value={selectedIndex}
              onChange={(_, v) => chooseSlice(v as number)}
              sx={{ ...compactSlider, flex: 1, maxWidth: 260, mx: 1 }}
            />
            <output>
              {selectedIndex} / {sliceCount - 1}
            </output>
            <label>
              Avg{" "}
              <CompactSelect
                label="Moving average slices"
                value={Math.min(sliceCount, averageWidth)}
                options={Array.from(
                  { length: Math.min(15, sliceCount) },
                  (_, i) => [i + 1, String(i + 1)] as const,
                )}
                onChange={(v) => setAverageWidth(normalizedAverageWindow(v))}
              />
            </label>
            <span className="hint" role="status">
              {selectedFrames
                ? `${selectedFrames[0]}–${selectedFrames[selectedFrames.length - 1]}`
                : "All"}{" "}
              · {limits.map((v) => v.toFixed(2)).join("–")} Å
            </span>
          </div>
          <details className="notes">
            <summary>Slice settings</summary>
            <div className="row">
              <label>
                Slices{" "}
                <input
                  aria-label="Number of beam slices"
                  type="number"
                  min="1"
                  max="64"
                  step="1"
                  value={sliceCount}
                  style={{ width: 64 }}
                  onChange={(e) => {
                    const n = Number(e.target.value);
                    if (Number.isInteger(n) && n >= 1 && n <= 64) {
                      preview(null);
                      setSliceCount(n);
                      setSliceIndex(Math.min(sliceIndex, n - 1));
                      setAverageWidth(Math.min(averageWidth, n));
                    }
                  }}
                />
              </label>
              <span>{((zmax - zmin) / sliceCount).toFixed(3)} Å / slice</span>
            </div>
            <p>
              Avg is the arithmetic mean of adjacent planes, using Show3D's
              centered window. At the ends it shifts inward to keep its width.
              Atom views show the included slab. Potential maps below have
              independent depth and averaging controls.
            </p>
          </details>
        </div>
        <div
          className="panels"
          style={{
            gridTemplateColumns:
              !showProjection && !orthogonal ? "minmax(0,1fr)" : undefined,
          }}
        >
          {[
            "3D structure · drag to rotate",
            "Projected columns · click to inspect",
            "Column–Depth · side projection",
            "Row–Depth · side projection",
          ].map((label, i) => (
            <figure
              key={i}
              style={{
                display:
                  (i === 1 && !showProjection) || (i > 1 && !orthogonal)
                    ? "none"
                    : undefined,
              }}
            >
              <figcaption>{label}</figcaption>
              <div className="scene">
                <canvas
                  ref={refs[i]}
                  aria-label={
                    [
                      "3D atomic structure",
                      "Atomic column projection",
                      "Column depth projection",
                      "Row depth projection",
                    ][i]
                  }
                  onPointerDown={(e) => {
                    e.currentTarget.setPointerCapture(e.pointerId);
                    drag.current = {
                      x: e.clientX,
                      y: e.clientY,
                      active: true,
                      moved: false,
                    };
                  }}
                  onPointerMove={(e) => {
                    const d = drag.current;
                    if (i || !d.active) return;
                    const dx = e.clientX - d.x,
                      dy = e.clientY - d.y;
                    d.moved ||= Math.abs(dx) + Math.abs(dy) > 2;
                    angles.current[0] += dx * 0.008;
                    angles.current[1] = Math.max(
                      -1.5,
                      Math.min(1.5, angles.current[1] + dy * 0.008),
                    );
                    d.x = e.clientX;
                    d.y = e.clientY;
                    schedule();
                  }}
                  onPointerUp={(e) => {
                    if (!drag.current.moved) inspect(e, i);
                    drag.current.active = false;
                  }}
                  onPointerCancel={() => {
                    drag.current.active = false;
                  }}
                />
                <svg
                  ref={lines[i]}
                  viewBox="0 0 100 100"
                  preserveAspectRatio="none"
                  aria-hidden="true"
                >
                  {edges.map((_, j) => (
                    <line key={j} stroke="#6a839e" strokeWidth=".2" />
                  ))}
                </svg>
                <ScaleBar span={spans[i]} />
              </div>
              <figcaption className="hint">
                {spans[i].toFixed(2)} Å view width ·{" "}
                {i === 0
                  ? "orthographic camera"
                  : i > 1
                    ? "orthogonal beam-frame view · depth increases upward"
                    : (microscope ? "microscope field · " : "") +
                      "direct-lattice [" +
                      zone.join(" ") +
                      "]"}
              </figcaption>
            </figure>
          ))}
        </div>
        <div className="row">
          <label>
            Atom radius{" "}
            <Slider
              size="small"
              aria-label="Display atom radius"
              min={2}
              max={14}
              step={1}
              value={radius}
              onChange={(_, v) => setRadius(v as number)}
              sx={{ ...compactSlider, width: 100, mx: 1 }}
            />
          </label>
          <label>
            {microscope ? "3D Zoom" : "Zoom"}{" "}
            <Slider
              size="small"
              aria-label="Structure zoom"
              min={0.5}
              max={5}
              step={0.05}
              value={zoom}
              onChange={(_, v) => setZoom(v as number)}
              sx={{ ...compactSlider, width: 100, mx: 1 }}
            />
          </label>
          <button
            onClick={() => {
              angles.current = [0.55, 0.35];
              setZoom(1);
              setSliceIndex(0);
              setAverageWidth(1);
              schedule();
            }}
          >
            Reset View
          </button>
        </div>
        <p role="status">{pick}</p>
        <PotentialPanel
          active={showPotential}
          device={gpuDevice}
          atoms={atoms}
          table={potentialTable}
          pixels={potentialPixels}
          sigma={potentialSigma}
          geometry={{
            ...basis,
            center,
            span: microscope ? fov : Math.max(lateral, vertical) + 16,
            zmin,
            zmax,
            limits,
            slices: sliceCount,
            unit,
            repeats,
            visible,
          }}
        />
        <details className="notes">
          <summary>Rendering notes</summary>
          <p className="hint">
            {status}. Atom colors and radii are schematic; this is neither
            potential nor scattering intensity. Hiding atoms does not remove
            them from the CIF or simulation.
          </p>
        </details>
      </div>
    </CifControlTheme.Provider>
  );
}
export default { render: createRender(App) };
