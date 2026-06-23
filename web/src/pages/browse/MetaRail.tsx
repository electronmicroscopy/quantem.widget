import { useState } from "react";
import { useNavigate } from "react-router-dom";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import { colors, fontSizes, radii } from "../../theme";
import { type BrowseDtype, type MasterFile, type Session } from "./types";

interface Props {
  session: Session;
  file: MasterFile;
  browseDtype?: BrowseDtype;
  onToggleDtype?: () => void;
}

const MONO = "ui-monospace, SFMono-Regular, Menlo, monospace";

function MetaRow({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <Box sx={{ display: "flex", gap: 1, fontSize: fontSizes.xs, py: 0.25 }}>
      <Box sx={{ minWidth: 70, color: colors.text.muted, fontFamily: MONO }}>{k}</Box>
      <Box sx={{ flex: 1, fontFamily: MONO, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{v}</Box>
    </Box>
  );
}

/** Persist collapsed state per-panel in localStorage so the user's layout
 *  preference survives reloads. The hook returns [collapsed, toggle]. */
function useCollapsedPanel(name: string, defaultCollapsed: boolean) {
  const key = `browse.panel.${name}.collapsed`;
  const [collapsed, setCollapsed] = useState<boolean>(() => {
    try {
      const v = localStorage.getItem(key);
      if (v === null) return defaultCollapsed;
      return v === "1";
    } catch { return defaultCollapsed; }
  });
  const toggle = () => {
    setCollapsed((c) => {
      const next = !c;
      try { localStorage.setItem(key, next ? "1" : "0"); } catch { /* ignore */ }
      return next;
    });
  };
  return [collapsed, toggle] as const;
}

/** Generic collapsible Panel for the right rail. Header always visible;
 *  body hidden when collapsed. The chevron toggle persists per-panel. */
function Panel({
  name, title, sub, defaultCollapsed = false, headerOverride, children,
}: {
  name: string;
  title: string;
  sub?: React.ReactNode;
  defaultCollapsed?: boolean;
  /** When provided AND the panel is collapsed, render this in the header
   *  row instead of the title/sub pair. Used by the GPU panel to fit the
   *  whole status into one line. */
  headerOverride?: React.ReactNode;
  children: React.ReactNode;
}) {
  const [collapsed, toggle] = useCollapsedPanel(name, defaultCollapsed);
  return (
    <Box>
      <Box
        onClick={toggle}
        sx={{ display: "flex", alignItems: "center", justifyContent: "space-between",
              px: 0.5, pb: collapsed ? 0 : 0.75, cursor: "pointer", gap: 1,
              "&:hover .browse-panel-chev": { opacity: 1 } }}
      >
        {collapsed && headerOverride ? (
          <Box sx={{ flex: 1, minWidth: 0, overflow: "hidden" }}>{headerOverride}</Box>
        ) : (
          <>
            <Typography sx={{ fontSize: fontSizes.lg, fontWeight: 700 }}>{title}</Typography>
            {sub && (
              <Typography sx={{ fontSize: fontSizes.xs, color: colors.text.muted,
                                overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                                flex: 1, textAlign: "right", minWidth: 0 }}>
                {sub}
              </Typography>
            )}
          </>
        )}
        <Box className="browse-panel-chev"
             sx={{ fontSize: fontSizes.xs, color: colors.text.muted,
                   transform: collapsed ? "rotate(0deg)" : "rotate(90deg)",
                   transition: "transform 120ms", opacity: 0.6, flex: "0 0 auto",
                   userSelect: "none", lineHeight: 1 }}>
          ▸
        </Box>
      </Box>
      {!collapsed && <Box sx={{ px: 0.5 }}>{children}</Box>}
    </Box>
  );
}

function CalBadge({ cal }: { cal: MasterFile["cal"] }) {
  if (cal === "ok") {
    return (
      <Box component="span"
        sx={{ display: "inline-block", px: 0.75, py: 0.125, fontSize: fontSizes.xs, fontFamily: MONO,
                 bgcolor: colors.success.bg, color: colors.success.text, borderRadius: radii.sm }}
           title="This master is listed in the session's dataset.yaml">
        registered ✓ dataset.yaml
      </Box>
    );
  }
  return (
    <Box component="span"
         sx={{ display: "inline-block", px: 0.75, py: 0.125, fontSize: fontSizes.xs, fontFamily: MONO,
               bgcolor: colors.warning.bg, color: colors.warning.text, borderRadius: radii.sm }}>
      {cal === "warn" ? "DRAFT" : "UNCALIBRATED"}
    </Box>
  );
}

function ActionButton({
  primary, label, desc, onClick,
}: {
  primary?: boolean; label: string; desc: string; onClick?: () => void;
}) {
  return (
    <Box>
      <Box
        component="button"
        onClick={onClick}
        sx={{ display: "flex", alignItems: "center", justifyContent: "flex-start",
              width: "100%", px: 1.25, py: 0.75, cursor: "pointer",
              border: `1px solid ${primary ? colors.interactive.text : colors.border.default}`,
              bgcolor: primary ? colors.interactive.text : colors.bg.page,
              color: primary ? colors.text.white : colors.text.primary,
              borderRadius: radii.md, fontSize: fontSizes.md, fontWeight: 600,
              "&:hover": { bgcolor: primary ? colors.interactive.hover : colors.bg.hover } }}
      >
        <span>{label}</span>
      </Box>
      <Typography sx={{ fontSize: fontSizes.xs, color: colors.text.muted, mt: 0.5, mb: 1, lineHeight: 1.3 }}>
        {desc}
      </Typography>
    </Box>
  );
}

export default function MetaRail({ session, file, browseDtype = "uint8", onToggleDtype }: Props) {
  const navigate = useNavigate();
  const stem = file.name.replace(/_master\.h5$/, "").replace(/\.h5$/, "");
  const folderHint = `${session.source}/${session.date}`;
  const sessionBasename = session.date.split("/").filter(Boolean).at(-1) ?? session.date;
  const goScreening = () => navigate(
    `/screening?stem=${encodeURIComponent(stem)}&folder=${encodeURIComponent(folderHint)}`,
  );
  const goDrift = () => navigate("/drift");
  const goDenoise = () => navigate("/denoise");

  const [wx, wy, kx, ky] = file.shape;
  const haveScan = wx > 0 && wy > 0;
  const haveDet = kx > 0 && ky > 0;

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5, minHeight: 0,
               overflow: "auto", pr: 0.5 }}>
      <Panel name="file" title="File" sub={`${file.size} · HDF5`}>
        <MetaRow k="name" v={<span title={file.name}>{file.name}</span>} />
        <MetaRow k="format" v={<>HDF5 · <span style={{ opacity: 0.7 }}>master.h5</span></>} />
        <MetaRow k="dtype" v={
          <span
            onClick={onToggleDtype}
            title="Click to toggle browse precision. uint8 halves VRAM (~2× more masters fit) and is lossless when counts ≤ 255; uint16 keeps raw counts."
            style={{ cursor: onToggleDtype ? "pointer" : "default",
                     textDecoration: onToggleDtype ? "underline dotted" : "none" }}
          >
            {browseDtype === "uint8" ? "uint8 (browse · ½ mem)" : "uint16 (raw counts)"}
            {onToggleDtype && (
              <span style={{ opacity: 0.6 }}>
                {browseDtype === "uint8" ? "  → uint16" : "  → uint8"}
              </span>
            )}
          </span>
        } />
        <MetaRow k="layout" v="bitshuffle+lz4 · chunked" />
        <MetaRow k="session" v={<span title={folderHint}>{sessionBasename}</span>} />
        <MetaRow k="scan" v={haveScan ? `${wx}×${wy}` : "(unknown — no dataset.yaml)"} />
        <MetaRow k="detector" v={haveDet ? `${kx}×${ky}` : "(unknown)"} />
        <MetaRow k="cal" v={<CalBadge cal={file.cal} />} />
      </Panel>

      <Panel name="promote" title="Promote">
        <ActionButton primary label="→ Screening" onClick={goScreening}
                      desc="Run BF / DF / iCoM / SSB previews for this master" />
        <ActionButton label="→ Drift Correction" onClick={goDrift}
                      desc="Align HAADF 0°/90° pairs and refine non-rigid drift" />
        <ActionButton label="→ Denoise this folder" onClick={goDenoise}
                      desc="N2V / BM3D / NLM batch over the parent directory" />
      </Panel>
    </Box>
  );
}
