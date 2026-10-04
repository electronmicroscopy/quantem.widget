import * as React from "react";
import { useModelState } from "@anywidget/react";
import { PlayPauseButton } from "../PlayPauseButton";
import {
  temporalAverageFrameIndices,
  normalizedAverageWindow,
} from "../show3d/frameTransform";
import { ScaleBar } from "./ScaleBar";
import { interactionConstant } from "./phase";
import { Slider } from "@mui/material";
import { CompactSelect, compactSlider, CifControlTheme } from "./controls";
import { COLORMAP_NAMES, COLORMAP_POINTS } from "../colormaps";
import { PotentialGPU, type PotentialGeometry } from "./potential";

type Props = {
  active: boolean;
  device?: GPUDevice;
  atoms: Float32Array;
  table: Float32Array;
  geometry: PotentialGeometry;
  pixels: number;
  sigma: number;
};
export function PotentialPanel({
  active,
  device,
  atoms,
  table,
  geometry,
  pixels,
  sigma,
}: Props) {
  const colors = React.useContext(CifControlTheme);
  const [potentialSlices, setPotentialSlices] = React.useState(geometry.slices);
  const [index, setIndex] = React.useState(0);
  const [averageWidth, setAverageWidth] = React.useState(1);
  const [playing, setPlaying] = React.useState(false);
  const [fps, setFps] = React.useState(5);
  const [preview, setPreview] = React.useState<number | null>(null);
  const previewFrame = React.useRef(0);
  const g = { ...geometry, slices: potentialSlices };
  const selectedIndex = Math.min(g.slices - 1, preview ?? index);
  const averageFrames = temporalAverageFrameIndices(
    selectedIndex,
    g.slices,
    averageWidth,
  );
  const onSlice = (i: number) => {
    setPlaying(false);
    cancelAnimationFrame(previewFrame.current);
    setPreview(null);
    setIndex(i);
  };
  const onPreview = (i: number | null) => {
    if (i !== null) setPlaying(false);
    cancelAnimationFrame(previewFrame.current);
    previewFrame.current = requestAnimationFrame(() => setPreview(i));
  };
  React.useEffect(() => () => cancelAnimationFrame(previewFrame.current), []);
  React.useEffect(() => {
    if (!playing || !active || potentialSlices < 2) return;
    let frame = 0,
      last = performance.now();
    const tick = (now: number) => {
      if (now - last >= 1000 / fps) {
        setIndex((i) => (i + 1) % potentialSlices);
        last = now;
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, active, fps, potentialSlices]);
  React.useEffect(() => {
    const pauseHidden = () => {
      if (document.hidden) setPlaying(false);
    };
    document.addEventListener("visibilitychange", pauseHidden);
    return () => document.removeEventListener("visibilitychange", pauseHidden);
  }, []);
  const [gallery, setGallery] = React.useState(false);
  const [customEnergy, setCustomEnergy] = React.useState(false);
  const [columns, setColumns] = React.useState(2);
  const [blur, setBlur] = React.useState(0);
  const gpu = React.useRef<PotentialGPU | undefined>(undefined);
  const computedSignature = React.useRef("");
  const canvases = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const [quantity, setQuantity] = useModelState<string>("potential_quantity");
  const [energy, setEnergy] = useModelState<number>("energy_keV");
  const [colormap, setColormap] = useModelState<string>("potential_colormap");
  const average = quantity === "average",
    phase = quantity === "phase";
  const interaction = interactionConstant(energy);
  const phaseInvalid = phase && interaction === null;
  const [max, setMax] = React.useState(
      quantity === "phase" ? 1 : quantity === "integrated" ? 1000 : 100,
    ),
    [status, setStatus] = React.useState("Preparing potential preview…");
  const [ready, setReady] = React.useState(0);
  const count = g.repeats.reduce((a, b) => a * b, atoms.length / 4),
    blocked = count > 8192;
  React.useEffect(() => {
    if (!active || blocked || phaseInvalid || potentialSlices < 2)
      setPlaying(false);
  }, [active, blocked, phaseInvalid, potentialSlices]);
  // All planes are computed once per physical geometry. Scrubbing only reduces
  // already-resident planes with the same averaging engine used by Show3D.
  const computeGeometry = { ...g, limits: [g.zmin, g.zmax] };
  const signature = JSON.stringify(computeGeometry);
  const averageKey = averageFrames?.join(",") ?? "all";
  const draw = () => {
    if (!active || !gpu.current || blocked) return;
    if (averageFrames) gpu.current.average(averageFrames, g.slices);
    canvases.current.forEach((c, i) => {
      if (!c || (i > 1 && !gallery)) return;
      const slot =
        i === 0
          ? averageFrames
            ? g.slices + 1
            : g.slices
          : i === 1
            ? g.slices
            : i - 2;
      const dz =
        i === 0
          ? (g.zmax - g.zmin) / (averageFrames ? g.slices : 1)
          : i === 1
            ? g.zmax - g.zmin
            : (g.zmax - g.zmin) / g.slices;
      gpu.current!.draw(
        c,
        slot,
        phase ? (interaction ?? 0) : average ? (dz > 0 ? 1 / dz : 0) : 1,
        max,
        colormap,
      );
    });
  };
  React.useEffect(() => {
    if (!device || !table.length) return;
    const instance = new PotentialGPU(device, atoms, table, pixels);
    gpu.current = instance;
    computedSignature.current = "";
    setReady((x) => x + 1);
    return () => {
      instance.destroy();
      gpu.current = undefined;
    };
  }, [device, atoms, table, pixels]);
  React.useEffect(() => {
    if (!active || !gpu.current) return;
    const frame = requestAnimationFrame(() => {
      if (blocked) {
        setStatus(
          "Potential preview supports 8,192 atoms. Reduce Unit Cells to see the potential; no atoms are silently dropped.",
        );
        return;
      }
      try {
        if (computedSignature.current !== signature) {
          gpu.current!.compute(computeGeometry, atoms.length / 4);
          computedSignature.current = signature;
        }
        gpu.current!.filter(blur / (g.span / pixels), g.slices + 2);
        draw();
        setStatus(
          "WebGPU potential · all visible species · finite inspection patch",
        );
      } catch (e) {
        setStatus(String(e));
      }
    });
    return () => cancelAnimationFrame(frame);
  }, [ready, signature, blur, active]);
  React.useEffect(() => {
    const frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, [ready, quantity, colormap, max, energy, averageKey, gallery, active]);
  React.useEffect(() => () => gpu.current?.clearViews(), [g.slices]);
  if (!table.length) return null;
  const units = phase ? "rad" : average ? "V" : "V Å";
  return (
    <section className="potential-panel" hidden={!active}>
      <div className="row" aria-label="Potential slice playback">
        <PlayPauseButton
          playing={playing}
          color={colors.accent}
          disabled={g.slices < 2 || blocked || phaseInvalid}
          label={playing ? "Pause potential slices" : "Play potential slices"}
          onToggle={() => {
            cancelAnimationFrame(previewFrame.current);
            setPreview(null);
            setPlaying((v) => !v);
          }}
        />
        <span>Depth</span>
        <Slider
          size="small"
          aria-label="Potential depth slice"
          min={0}
          max={Math.max(1, g.slices - 1)}
          step={1}
          disabled={g.slices < 2}
          value={selectedIndex}
          onChange={(_, v) => onSlice(v as number)}
          sx={{ ...compactSlider, flex: 1, maxWidth: 260, mx: 1 }}
        />
        <output>
          {selectedIndex} / {g.slices - 1}
        </output>
        <label>
          Avg{" "}
          <CompactSelect
            label="Potential moving average slices"
            value={Math.min(g.slices, averageWidth)}
            options={Array.from(
              { length: Math.min(15, g.slices) },
              (_, i) => [i + 1, String(i + 1)] as const,
            )}
            onChange={(v) => setAverageWidth(normalizedAverageWindow(v))}
          />
        </label>
        <span className="hint">
          {averageFrames[0]}–{averageFrames[averageFrames.length - 1]} ·{" "}
          {((averageFrames[0] / g.slices) * (g.zmax - g.zmin) + g.zmin).toFixed(
            2,
          )}
          –
          {(
            ((averageFrames[averageFrames.length - 1] + 1) / g.slices) *
              (g.zmax - g.zmin) +
            g.zmin
          ).toFixed(2)}{" "}
          Å
        </span>
      </div>
      <details className="notes">
        <summary>Potential slicing & playback</summary>
        <div className="row">
          <label>
            Slices{" "}
            <input
              type="number"
              aria-label="Number of potential slices"
              min="1"
              max="64"
              value={potentialSlices}
              style={{ width: 64 }}
              onChange={(e) => {
                const n = Number(e.target.value);
                if (Number.isInteger(n) && n >= 1 && n <= 64) {
                  onSlice(Math.min(index, n - 1));
                  setPotentialSlices(n);
                  setAverageWidth(Math.min(averageWidth, n));
                }
              }}
            />
          </label>
          <span>{((g.zmax - g.zmin) / g.slices).toFixed(3)} Å / slice</span>
          <label>
            fps{" "}
            <CompactSelect
              label="Potential playback frames per second"
              value={fps}
              options={[1, 2, 5, 10, 15].map((n) => [n, String(n)] as const)}
              onChange={(v) => setFps(Number(v))}
            />
          </label>
        </div>
        <p>
          Independent of atom-view slicing. Play loops through every depth; Avg
          uses Show3D's centered moving mean. Full Projection stays fixed. This
          is a model-potential preview, not reconstructed data or finite-z
          multislice propagation.
        </p>
      </details>
      <div className="row">
        <strong>Potential</strong>
        <CompactSelect
          label="Potential quantity"
          value={quantity}
          options={[
            ["integrated", "Integrated · V Å"],
            ["average", "Average · V"],
            ["phase", "Phase · rad"],
          ]}
          onChange={(v) => {
            setQuantity(v);
            setMax(v === "phase" ? 1 : v === "average" ? 100 : 1000);
          }}
        />
        <label>
          Color{" "}
          <CompactSelect
            label="Potential colormap"
            value={colormap}
            options={COLORMAP_NAMES.map(
              (n) => [n, n.charAt(0).toUpperCase() + n.slice(1)] as const,
            )}
            onChange={setColormap}
          />
        </label>
        <label>
          Cols{" "}
          <CompactSelect
            label="Potential panel columns"
            value={columns}
            options={[1, 2, 3, 4].map((n) => [n, String(n)] as const)}
            onChange={(v) => setColumns(Number(v))}
          />
        </label>
        {phase && (
          <label>
            Energy{" "}
            <CompactSelect
              label="Electron energy preset"
              value={
                customEnergy || ![60, 80, 100, 120, 200, 300].includes(energy)
                  ? "custom"
                  : energy
              }
              options={[
                ...[60, 80, 100, 120, 200, 300].map(
                  (n) => [n, `${n} keV`] as const,
                ),
                ["custom", "Custom…"],
              ]}
              onChange={(v) => {
                setCustomEnergy(v === "custom");
                if (v !== "custom") setEnergy(Number(v));
              }}
            />
          </label>
        )}
        {phase &&
          (customEnergy || ![60, 80, 100, 120, 200, 300].includes(energy)) && (
            <label>
              <input
                type="number"
                aria-label="Electron energy in keV"
                min="0"
                max="300"
                value={energy}
                style={{ width: 72 }}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  if (Number.isFinite(v) && v >= 0 && v <= 300) setEnergy(v);
                }}
              />{" "}
              keV
            </label>
          )}
        <button
          aria-pressed={gallery}
          onClick={() => setGallery(!gallery)}
          title="Show every slice; hover to preview, click to keep"
        >
          Gallery
        </button>
      </div>
      <div className="row">
        <label>
          Max{" "}
          <input
            aria-label="Potential color scale maximum"
            type="number"
            min="0.001"
            step={phase ? 0.1 : 10}
            value={max}
            style={{ width: 75 }}
            onChange={(e) => {
              const v = Number(e.target.value);
              if (Number.isFinite(v) && v > 0) setMax(v);
            }}
          />{" "}
          {units}
        </label>
        <span title="Gaussian display blur only; raw potential is unchanged">
          Blur σ
        </span>
        <Slider
          size="small"
          aria-label="Potential display blur in angstrom"
          min={0}
          max={Math.min(1, (32 * g.span) / pixels)}
          step={0.01}
          value={blur}
          onChange={(_, v) => setBlur(v as number)}
          sx={{ ...compactSlider, width: 120 }}
        />
        <output>{blur.toFixed(2)} Å</output>
        <button
          onClick={() => setBlur(0)}
          title="Reset display blur"
          aria-label="Reset display blur"
        >
          ↺
        </button>
        <span className="hint">
          {phase ? "Projected phase · no propagation" : "Atomic potential"}
          {blur > 0 ? " · display blur" : ""}
        </span>
      </div>
      {phaseInvalid && (
        <p role="status">
          Phase is undefined at 0 keV. Choose a positive energy.
        </p>
      )}
      <div
        className="potential-colorbar"
        style={{
          height: 5,
          background: `linear-gradient(to right,${COLORMAP_POINTS[colormap].map((rgb) => `rgb(${rgb.join(",")})`).join(",")})`,
        }}
      />
      <div className="row" style={{ justifyContent: "space-between" }}>
        <span>0 {units}</span>
        <span>
          {max} {units}
        </span>
      </div>
      {average && g.limits[1] <= g.limits[0] && (
        <p className="hint">
          Zero-thickness selection: the average is undefined; an empty map is
          shown.
        </p>
      )}
      {status !==
        "WebGPU potential · all visible species · finite inspection patch" && (
        <p role="status">{status}</p>
      )}
      <div style={{ display: blocked || phaseInvalid ? "none" : undefined }}>
        <div
          className="potential-maps"
          style={{ "--map-cols": columns } as React.CSSProperties}
        >
          {[
            averageFrames
              ? `Slice ${averageFrames[0]}${averageFrames.length > 1 ? `–${averageFrames[averageFrames.length - 1]} · mean of ${averageFrames.length}` : ""}`
              : "Full Depth",
            "Full Projection",
          ].map((label, i) => (
            <figure key={i}>
              <figcaption>
                {label} · {units}
              </figcaption>
              <div className="scene">
                <canvas
                  aria-label={`${label} potential`}
                  ref={(el) => {
                    canvases.current[i] = el;
                  }}
                />
                <ScaleBar span={g.span} />
              </div>
            </figure>
          ))}
        </div>

        <div
          className="potential-gallery"
          style={
            {
              "--map-cols": columns,
              display: gallery ? undefined : "none",
            } as React.CSSProperties
          }
        >
          {Array.from({ length: g.slices }, (_, i) => (
            <figure key={i}>
              <button
                className="slice-image"
                style={{ position: "relative" }}
                aria-label={`Inspect potential slice ${i}`}
                onPointerEnter={() => onPreview(i)}
                onPointerLeave={() => onPreview(null)}
                onFocus={() => onPreview(i)}
                onBlur={() => onPreview(null)}
                onClick={() => onSlice(i)}
              >
                <canvas
                  aria-label={`Potential slice ${i}`}
                  ref={(el) => {
                    canvases.current[i + 2] = el;
                  }}
                />
                <ScaleBar span={g.span} />
              </button>
              <figcaption>
                #{i} ·{" "}
                {(g.zmin + (i * (g.zmax - g.zmin)) / g.slices).toFixed(2)}–
                {(g.zmin + ((i + 1) * (g.zmax - g.zmin)) / g.slices).toFixed(2)}{" "}
                Å
              </figcaption>
            </figure>
          ))}
        </div>
      </div>
      <details className="notes">
        <summary>Model & display notes</summary>
        <p className="hint">
          {phase
            ? "φ = σ(E) ∫V dz. This is unwrapped projected-potential phase, not the multislice exit-wave phase of a thick specimen."
            : average
              ? "Integrated potential divided by slab thickness. The full view is thickness-weighted, not a mean of displayed images."
              : "Slabs sum to the full projected potential."}
          {interaction !== null &&
            ` σ(${energy} keV) = ${interaction.toExponential(6)} rad/(V Å).`}{" "}
          Shared color limits; values above Max saturate without changing data.
        </p>
        <p className="hint">
          Lobato neutral independent atoms · infinite atomic projection assigned
          by site depth (half-open slabs). Not finite-z integration,
          reconstructed phase, bonding charge density, or a diffraction
          simulation. Hidden species are excluded from this preview only.
        </p>
        <p className="hint">
          Explicit transverse Gaussian preview filter σ {sigma} Å; not a thermal
          model. {pixels} × {pixels} grid · {(g.span / pixels).toFixed(4)} Å/px
          · {g.span.toFixed(2)} Å view width · 2 × 2 pixel quadrature · radial
          cutoff 8 Å. Inspection-cell boundaries are open, with no periodic
          atoms added outside the chosen repeats.
        </p>
        {g.span / pixels > sigma && (
          <p className="hint">
            Preview pixels exceed the Gaussian width. Use fewer repeats or a
            larger potential_pixels grid for sharper inspection.
          </p>
        )}
      </details>
    </section>
  );
}
