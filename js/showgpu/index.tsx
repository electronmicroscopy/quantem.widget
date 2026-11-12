import * as React from "react";
import { createRender, useModel, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import StopRounded from "@mui/icons-material/StopRounded";
import { compactButton } from "../controlStyles";
import { useTheme, WIDGET_UI_FONT } from "../theme";

type GPUReading = {
  index: number;
  uuid: string;
  name: string;
  pci_bus_id: string;
  total_bytes: number | null;
  used_bytes: number | null;
  free_bytes: number | null;
  utilization: number | null;
  temperature: number | null;
  power: number | null;
  users?: {
    name: string | null;
    used_bytes: number | null;
    processes: number;
    process_details?: { pid: number; name: string | null; used_bytes: number | null;
      created_at?: number; can_terminate?: boolean }[];
  }[];
};

const TEXT_SIZES = { small: 13, medium: 15, large: 17, xlarge: 20 } as const;
type DisplaySize = keyof typeof TEXT_SIZES;
const SIZE_LABELS: Record<DisplaySize, string> = { small: "S", medium: "M", large: "L", xlarge: "XL" };

type ScanReading = {
  id: string; name: string; device: string; shape: number[];
  received: number; total: number; resident_bytes: number; index_bytes: number;
  staging_bytes: number; state: string; saved?: boolean; can_unload: boolean; unload_reason: string;
  encoding?: string; dtype?: string; source_path?: string; encoded?: number;
};

type Snapshot = {
  sequence?: number;
  timestamp?: number;
  query_ms?: number;
  error?: string;
  process_error?: string;
  cuda_visible_devices?: string | null;
  gpus?: GPUReading[];
  terminating?: number[];
  activity?: { timestamp: number; message: string }[];
  system?: {
    total_bytes?: number;
    used_bytes?: number;
    available_bytes?: number;
    kernel_rss_bytes?: number;
  };
  recording?: {
    watch_id: number;
    receiving?: boolean;
    error?: string | null;
    can_select?: boolean;
    scans: ScanReading[];
    releasing?: ScanReading[];
  };
};

export function gib(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? "unavailable" : `${(value / 2 ** 30).toFixed(1)} GiB`;
}

function reading(value: number | null, unit: string): string {
  return value == null ? "unavailable" : `${Math.round(value)}${unit}`;
}

function processMemory(value: number | null): string {
  return value != null && value < 2 ** 30 ? `${(value / 2 ** 20).toFixed(0)} MiB` : gib(value);
}

function Sparkline({ values, total, color }: { values: number[]; total: number | null; color: string }) {
  if (!total || values.length < 2) return <div style={{ height: 24 }} />;
  const points = values.map((value, index) => `${index * 200 / Math.max(1, values.length - 1)},${23 - 22 * Math.min(1, value / total)}`).join(" ");
  return <svg viewBox="0 0 200 24" preserveAspectRatio="none" role="img" aria-label="Recent VRAM usage" style={{ width: "100%", height: 24 }}>
    <polyline points={points} fill="none" stroke={color} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
  </svg>;
}

function Meter({ used, total, color, background, label }: {
  used: number | null | undefined; total: number | null | undefined; color: string; background: string; label: string;
}) {
  const percent = used != null && total ? Math.min(100, Math.max(0, 100 * used / total)) : undefined;
  return <div role="meter" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}
    style={{ background, height: 6, borderRadius: 3, overflow: "hidden", margin: "6px 0" }}>
    <div style={{ background: color, width: `${percent ?? 0}%`, height: "100%", transition: "width 180ms linear" }} />
  </div>;
}

function ShowGPUView() {
  const model = useModel();
  const [snapshot] = useModelState<Snapshot>("snapshot");
  const [running, setRunning] = useModelState<boolean>("running");
  const [interval] = useModelState<number>("interval");
  const [users, setUsers] = useModelState<boolean>("users");
  const [expand] = useModelState<boolean>("expand");
  const [size, setSize] = useModelState<DisplaySize>("size");
  const [actionStatus] = useModelState<{ message?: string; error?: boolean; busy?: boolean }>("action_status");
  const [confirmation, setConfirmation] = React.useState<{ message: string; command: Record<string, unknown> } | null>(null);
  const confirmationPanel = React.useRef<HTMLDivElement>(null);
  const { colors } = useTheme();
  const [now, setNow] = React.useState(Date.now());
  const [receivedAt, setReceivedAt] = React.useState<number | null>(null);
  const initialSequence = React.useRef(snapshot?.sequence);
  const [history, setHistory] = React.useState<Record<string, number[]>>({});
  const [scanHistory, setScanHistory] = React.useState<number[]>([]);
  const watched = React.useRef(snapshot?.recording?.watch_id);

  React.useEffect(() => {
    confirmationPanel.current?.scrollIntoView({ block: "nearest" });
  }, [confirmation]);

  React.useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  React.useEffect(() => {
    if (snapshot?.sequence !== initialSequence.current) setReceivedAt(Date.now());
    setHistory(previous => {
      const next: Record<string, number[]> = {};
      for (const gpu of snapshot?.gpus ?? []) {
        const values = previous[gpu.uuid] ?? [];
        next[gpu.uuid] = gpu.used_bytes == null ? values : [...values, gpu.used_bytes].slice(-120);
      }
      return next;
    });
    const current = snapshot?.recording;
    if (current) {
      const bytes = [...current.scans, ...(current.releasing ?? [])].reduce((sum, scan) => sum + scan.resident_bytes + scan.staging_bytes, 0);
      const changed = watched.current !== current.watch_id;
      setScanHistory(previous => [...(changed ? [] : previous), bytes].slice(-120));
    } else setScanHistory([]);
    watched.current = current?.watch_id;
  }, [snapshot?.sequence]);

  const stale = receivedAt != null && now - receivedAt > Math.max(5000, interval * 3000);
  const status = !running ? "Paused" : snapshot?.error ? "Telemetry unavailable" : receivedAt == null ? "Waiting for live sample" : stale ? "No recent update" : "Live";
  const system = snapshot?.system ?? {};
  const gpus = snapshot?.gpus ?? [];
  const card = { background: colors.bgAlt, padding: "10px 12px", borderRadius: 5, minWidth: 0 };
  const mask = snapshot?.cuda_visible_devices;
  const fontSize = TEXT_SIZES[size] ?? TEXT_SIZES.medium;
  const button = { ...compactButton, fontSize: "0.92em", lineHeight: 1.5 };
  const recording = snapshot?.recording;
  const scanRows = [...(recording?.scans ?? []), ...(recording?.releasing ?? [])];
  const scanBytes = scanRows.reduce((sum, scan) => sum + scan.resident_bytes + scan.staging_bytes, 0);

  return <Box data-testid="showgpu" data-size={size} sx={{ fontFamily: WIDGET_UI_FONT, fontSize, lineHeight: 1.5, color: colors.text,
    bgcolor: colors.bg, border: `1px solid ${colors.border}`, borderRadius: 1, p: 1,
    width: "100%", boxSizing: "border-box", fontVariantNumeric: "tabular-nums" }}>
    <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap", mb: 1 }}>
      <strong style={{ fontSize: "1.08em" }}>GPU monitor</strong>
      <span data-testid="telemetry-status" style={{ color: status === "Live" ? colors.accent : colors.textMuted }}>{status}</span>
      <span style={{ color: colors.textMuted }}>Notebook server · {interval}s interval</span>
      <Box sx={{ flex: 1 }} />
      <label style={{ display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
        <input type="checkbox" checked={users} onChange={event => setUsers(event.target.checked)} style={{ accentColor: colors.accent }} />
        Users
      </label>
      <Box role="group" aria-label="Display size" sx={{ display: "flex" }}>
        {(Object.keys(TEXT_SIZES) as DisplaySize[]).map(value =>
          <Button key={value} size="small" sx={button} aria-label={`${value} text`} aria-pressed={size === value}
            variant={size === value ? "contained" : "text"} onClick={() => setSize(value)}>{SIZE_LABELS[value]}</Button>
        )}
      </Box>
      <Button size="small" sx={button} onClick={() => setRunning(!running)}>{running ? "Pause" : "Resume"}</Button>
      <Button size="small" sx={button} onClick={() => model.send({ action: "refresh" })}>Refresh</Button>
    </Box>
    {confirmation && <Box ref={confirmationPanel} role="alert" sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap", mb: 1, p: 1, bgcolor: colors.bgAlt }}>
      <span style={{ flex: 1 }}>{confirmation.message}</span>
      <Button size="small" sx={button} disabled={!!actionStatus?.busy} onClick={() => {
        model.send(confirmation.command); setConfirmation(null);
      }}>Confirm</Button>
      <Button size="small" sx={button} onClick={() => setConfirmation(null)}>Cancel</Button>
    </Box>}
    {actionStatus?.message && <div role="status" style={{ marginBottom: 8, color: actionStatus.error ? "#d32f2f" : colors.textMuted }}>{actionStatus.message}</div>}
    {snapshot?.error && <div role="status" style={{ marginBottom: 8 }}>{snapshot.error}</div>}
    {users && snapshot?.process_error && <div role="status" style={{ marginBottom: 8 }}>{snapshot.process_error}</div>}
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 20em), 1fr))", gap: 8 }}>
      {gpus.map(gpu => <div key={gpu.uuid} style={card} data-testid={`gpu-${gpu.index}`}>
        <div style={{ display: "flex", gap: 8, alignItems: "baseline" }}>
          <strong style={{ whiteSpace: "nowrap" }}>GPU {gpu.index}</strong>
          <span title={`${gpu.name} · ${gpu.uuid} · ${gpu.pci_bus_id}`} style={{ color: colors.textMuted, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{gpu.name}</span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 8, marginTop: 8, flexWrap: "wrap" }}>
          <span><strong>{gib(gpu.used_bytes)}</strong> used</span>
          <span><strong style={{ color: colors.accent }}>{gib(gpu.free_bytes)}</strong> free</span>
        </div>
        <Meter used={gpu.used_bytes} total={gpu.total_bytes} color={colors.accent} background={colors.border} label={`GPU ${gpu.index} VRAM used`} />
        <div style={{ color: colors.textMuted, display: "flex", justifyContent: "space-between", gap: 8, flexWrap: "wrap" }}>
          <span>{gib(gpu.total_bytes)} total</span><span>Utilization {reading(gpu.utilization, "%")}</span>
        </div>
        <Sparkline values={history[gpu.uuid] ?? []} total={gpu.total_bytes} color={colors.accent} />
        <div style={{ color: colors.textMuted }}>{reading(gpu.temperature, " °C")} · {reading(gpu.power, " W")}</div>
        {users && !snapshot?.process_error && <div style={{ borderTop: `1px solid ${colors.border}`, marginTop: 8, paddingTop: 6 }}>
          <div style={{ color: colors.textMuted, fontSize: "0.92em" }}>Compute users</div>
          {gpu.users == null ? <div>Waiting for user sample</div> : gpu.users.length === 0 ? <div>No compute processes</div> :
            gpu.users.map(user => <details key={user.name ?? "\u0000"} open={expand ?? true}>
              <summary aria-label={`GPU ${gpu.index} processes for ${user.name ?? "unknown user"}`} style={{ cursor: "pointer", padding: "2px 0" }}>
                <span style={{ display: "inline-flex", width: "calc(100% - 1.2em)", justifyContent: "space-between", gap: 8, flexWrap: "wrap", verticalAlign: "top" }}>
                  <span style={{ overflowWrap: "anywhere" }}>{user.name ?? "Unknown user"}</span>
                  <span>{gib(user.used_bytes)} <span style={{ color: colors.textMuted }}>· {user.processes} {user.processes === 1 ? "process" : "processes"}</span></span>
                </span>
              </summary>
              {user.process_details == null ? <div style={{ color: colors.textMuted }}>Process details need a new sample from the updated monitor.</div> :
                <table aria-label={`GPU ${gpu.index} processes for ${user.name ?? "unknown user"}`} style={{ width: "100%", tableLayout: "fixed", borderCollapse: "collapse", fontSize: "0.92em", marginTop: 4 }}>
                  <thead style={{ color: colors.textMuted, textAlign: "left" }}><tr>
                    <th scope="col" style={{ width: "5.5em", fontWeight: 400 }}>PID</th>
                    <th scope="col" style={{ fontWeight: 400 }}>Process</th>
                    <th scope="col" style={{ width: "6em", fontWeight: 400, textAlign: "right" }}>VRAM</th>
                    <th scope="col" style={{ width: "6em", fontWeight: 400 }} aria-label="Process actions" />
                  </tr></thead>
                  <tbody>{user.process_details.map(process => <tr key={process.pid}>
                    <td style={{ verticalAlign: "top", padding: "3px 4px 3px 0" }}>{process.pid}</td>
                    <td style={{ verticalAlign: "top", padding: "3px 4px 3px 0", overflowWrap: "anywhere" }}>{process.name ?? "Unavailable"}</td>
                    <td style={{ verticalAlign: "top", padding: "3px 0", textAlign: "right" }}>{processMemory(process.used_bytes)}</td>
                    <td style={{ textAlign: "right", verticalAlign: "top" }}>
                      {snapshot?.terminating?.includes(process.pid) ? <span style={{ color: colors.textMuted }}>Termination requested</span> : process.can_terminate && <Button size="small" sx={button} aria-label={`Terminate PID ${process.pid}`} disabled={!!actionStatus?.busy}
                        onClick={() => setConfirmation({
                          message: `Terminate PID ${process.pid} (${process.name ?? "process"})? This stops all its GPU work and discards unsaved data.`,
                          command: { action: "terminate", pid: process.pid, created_at: process.created_at },
                        })}>Terminate</Button>}
                    </td>
                  </tr>)}</tbody>
                </table>}
            </details>)}
        </div>}
      </div>)}
      <div style={card} data-testid="system-memory">
        <strong>System RAM</strong>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 8, marginTop: 8, flexWrap: "wrap" }}>
          <span><strong>{gib(system.used_bytes)}</strong> used</span>
          <span><strong style={{ color: colors.accent }}>{gib(system.available_bytes)}</strong> available</span>
        </div>
        <Meter used={system.used_bytes} total={system.total_bytes} color={colors.accent} background={colors.border} label="System RAM used" />
        <div style={{ color: colors.textMuted }}>{gib(system.total_bytes)} total</div>
        <div style={{ marginTop: 14 }}>This kernel: {gib(system.kernel_rss_bytes)} RAM</div>
        <div style={{ color: colors.textMuted, marginTop: 4 }}>Includes reclaimable RAM in available.</div>
      </div>
    </div>
    {recording && <section aria-label="Resident scans" style={{ marginTop: 12, borderTop: `1px solid ${colors.border}`, paddingTop: 8 }}>
      <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
        <strong>Resident scans {scanRows.length}</strong>
        {!!recording.releasing?.length && <span>{recording.releasing.length} awaiting release</span>}
        <span style={{ color: colors.accent }}>{recording.receiving ? "Acquiring / processing" : "Stopped"}</span>
        <span>{processMemory(scanBytes)} in tracked buffers</span>
        <span style={{ color: colors.textMuted }}>Peak {processMemory(Math.max(0, ...scanHistory))} · last {scanHistory.length} samples</span>
        {recording.receiving && <Button size="small" sx={button} startIcon={<StopRounded />}
          aria-label="Stop acquisition" title="Stop acquisition; keep retained scans"
          disabled={!!actionStatus?.busy} onClick={() => setConfirmation({
          message: "Stop this acquisition and drain pending work? Retained scans stay in memory.",
          command: { action: "stop_acquisition", watch_id: recording.watch_id },
        })}>Stop</Button>}
      </div>
      <Sparkline values={scanHistory} total={Math.max(1, ...scanHistory)} color={colors.accent} />
      {recording.error && <div role="status" style={{ color: "#d32f2f" }}>{recording.error}</div>}
      {!scanRows.length && <div style={{ color: colors.textMuted }}>No retained raw scans.</div>}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 24em), 1fr))", gap: 8 }}>
        {scanRows.map(scan => <div key={scan.id} style={card} data-testid={`scan-${scan.id}`}>
          <div style={{ display: "flex", gap: 8, alignItems: "baseline", justifyContent: "space-between" }}>
            {recording.can_select && scan.received > 0 && scan.state !== "Waiting for viewers" ? <Button size="small" sx={{ ...button, textTransform: "none", maxWidth: "70%", overflowWrap: "anywhere" }}
              aria-label={`View scan ${scan.name}`} onClick={() => model.send({ action: "select_scan", scan_id: scan.id, watch_id: recording.watch_id })}>{scan.name}</Button> :
              <strong style={{ overflowWrap: "anywhere" }}>{scan.name}</strong>}
            <span style={{ color: colors.accent }}>{scan.state === "Receiving" ? "Receiving / encoding" : scan.state}</span>
          </div>
          <div title={scan.source_path || "No source filename was reported by the live detector."} style={{ overflowWrap: "anywhere", color: colors.textMuted }}>
            {scan.source_path ? scan.source_path.split(/[\\/]/).pop() : "Live detector"}
            {scan.encoding && ` · ${scan.encoding}`}{scan.dtype && ` · ${scan.dtype}`}
          </div>
          <div>{scan.received.toLocaleString()} / {scan.total.toLocaleString()} DPs received</div>
          <Meter used={scan.received} total={scan.total} color={colors.accent} background={colors.border} label={`${scan.name} received patterns`} />
          {scan.encoded != null && <div>{scan.encoded.toLocaleString()} ANS encoded · {Math.max(0, scan.received - scan.encoded).toLocaleString()} staged</div>}
          <div style={{ color: colors.textMuted }}>{scan.shape.slice(0, 2).join(" × ")} scan · {scan.shape.slice(2).join(" × ")} DP · {scan.device} (logical)</div>
          <div>{processMemory(Math.max(0, scan.resident_bytes - scan.index_bytes))} ANS + metadata · {processMemory(scan.index_bytes)} indexes · {processMemory(scan.staging_bytes)} staging</div>
          <div style={{ display: "flex", gap: 8, alignItems: "center", justifyContent: "space-between" }}>
            <span style={{ color: colors.textMuted }}>{scan.state === "Waiting for viewers" ? "Close other viewers using this scan to finish release." : scan.saved === true ? "Saved" : scan.saved === false ? "Not saved" : "Save status unknown"}</span>
            <span title={scan.unload_reason}><Button size="small" sx={button} disabled={!scan.can_unload || !!actionStatus?.busy}
              aria-label={`Unload scan ${scan.name}`} onClick={() => setConfirmation({
                message: `Unload scan ${scan.name}? Unsaved counts are discarded; saved files remain. Finish other analyses using this scan first.`,
                command: { action: "unload", scan_id: scan.id, watch_id: recording.watch_id },
              })}>Unload</Button></span>
          </div>
        </div>)}
      </div>
      <div style={{ color: colors.textMuted, fontSize: "0.85em", marginTop: 6 }}>Tracked buffers exclude allocator reserves, shared codec tables, receiver queues and separate analysis sessions. Device VRAM above includes them.</div>
    </section>}
    {!!snapshot?.activity?.length && <details open style={{ marginTop: 8 }}>
      <summary style={{ cursor: "pointer", color: colors.textMuted }}>Recent activity</summary>
      <div role="log" aria-live="polite" style={{ maxHeight: "9em", overflowY: "auto" }}>
        {[...snapshot.activity].reverse().map((event, index) => <div key={`${event.timestamp}-${index}`}>
          <span style={{ color: colors.textMuted }}>{new Date(event.timestamp * 1000).toLocaleTimeString()} </span>{event.message}
        </div>)}
      </div>
    </details>}
    <div style={{ marginTop: 8, color: colors.textMuted, fontSize: "0.85em", display: "flex", flexWrap: "wrap", gap: "4px 12px" }}>
      <span>Physical driver IDs · all processes · driver reserve may make used + free less than total.</span>
      {users && <span>User totals cover reported compute processes; graphics and driver memory are excluded.</span>}
      {mask != null && <span style={{ overflowWrap: "anywhere" }}>Compute mask: CUDA_VISIBLE_DEVICES={mask === "" ? '"" (none)' : mask}</span>}
      <span data-testid="sample-time">{snapshot?.timestamp ? `Sample ${snapshot.sequence} · ${new Date(snapshot.timestamp * 1000).toLocaleTimeString()}` : "No sample"}
        {snapshot?.query_ms != null ? ` · query ${snapshot.query_ms.toFixed(0)} ms` : ""}</span>
    </div>
  </Box>;
}

export default { render: createRender(ShowGPUView) };
