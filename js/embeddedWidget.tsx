import * as React from "react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";

type WidgetView = {
  el?: HTMLElement;
  remove?: () => void;
};

type WidgetModel = {
  get?: (key: string) => unknown;
  set?: (key: string, value: unknown) => void;
  on?: (eventName: string, callback: () => void) => void;
  off?: (eventName?: string | null, callback?: (() => void) | null) => void;
  save_changes?: () => void;
};

type WidgetManagerWithViews = {
  get_model?: (id: string) => Promise<unknown>;
  create_view?: (model: unknown, options?: Record<string, unknown>) => Promise<WidgetView>;
};

type HostModelWithManager = {
  widget_manager?: WidgetManagerWithViews;
};

function widgetModelId(value: unknown): string | null {
  if (typeof value !== "string") return null;
  return value.startsWith("IPY_MODEL_") ? value.slice("IPY_MODEL_".length) : value;
}

type LinkedTrait = {
  source: string;
  target?: string;
};

async function resolveWidgetModel(hostModel: unknown, value: unknown): Promise<unknown | null> {
  if (!value) return null;
  if (typeof value !== "string") return value;
  const manager = (hostModel as HostModelWithManager | null)?.widget_manager;
  const id = widgetModelId(value);
  if (!manager?.get_model || !id) return null;
  return manager.get_model(id);
}

function syncLinkedTraits(hostModel: unknown, childModel: unknown, traits: LinkedTrait[]): () => void {
  const sourceModel = hostModel as WidgetModel | null;
  const targetModel = childModel as WidgetModel | null;
  if (!sourceModel?.get || !sourceModel?.set || !sourceModel?.on || !sourceModel?.off) return () => {};
  if (!targetModel?.get || !targetModel?.set || !targetModel?.on || !targetModel?.off) return () => {};

  let syncing = false;
  const cleanups: Array<() => void> = [];

  const copy = (from: WidgetModel, to: WidgetModel, fromKey: string, toKey: string) => {
    if (syncing || !from.get || !to.get || !to.set) return;
    const value = from.get(fromKey);
    if (value === undefined || to.get(toKey) === undefined) return;
    syncing = true;
    try {
      to.set(toKey, value);
      to.save_changes?.();
    } finally {
      syncing = false;
    }
  };

  for (const trait of traits) {
    const source = trait.source;
    const target = trait.target || trait.source;
    copy(sourceModel, targetModel, source, target);

    const sourceHandler = () => copy(sourceModel, targetModel, source, target);
    const targetHandler = () => copy(targetModel, sourceModel, target, source);
    sourceModel.on(`change:${source}`, sourceHandler);
    targetModel.on(`change:${target}`, targetHandler);
    cleanups.push(() => sourceModel.off?.(`change:${source}`, sourceHandler));
    cleanups.push(() => targetModel.off?.(`change:${target}`, targetHandler));
  }

  return () => {
    for (const cleanup of cleanups) cleanup();
  };
}

export function EmbeddedWidgetView({
  hostModel,
  widgetModel,
  title,
  onClose,
  themeColors,
  linkedTraits = [],
}: {
  hostModel: unknown;
  widgetModel: unknown;
  title: string;
  onClose: () => void;
  linkedTraits?: LinkedTrait[];
  themeColors: {
    bg: string;
    border: string;
    text: string;
    textMuted: string;
    accent: string;
  };
}) {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const [error, setError] = React.useState("");

  React.useEffect(() => {
    let cancelled = false;
    let view: WidgetView | null = null;
    let unlinkTraits: (() => void) | null = null;
    const container = containerRef.current;
    if (!container) return undefined;
    container.replaceChildren();
    setError("");

    const mount = async () => {
      try {
        const manager = (hostModel as HostModelWithManager | null)?.widget_manager;
        if (!manager?.create_view) {
          throw new Error("this frontend cannot create child widget views");
        }
        const childModel = await resolveWidgetModel(hostModel, widgetModel);
        if (!childModel) return;
        unlinkTraits = syncLinkedTraits(hostModel, childModel, linkedTraits);
        view = await manager.create_view(childModel);
        if (cancelled) {
          view?.remove?.();
          return;
        }
        if (view?.el) {
          container.replaceChildren(view.el);
        }
      } catch (exc) {
        if (!cancelled) setError(exc instanceof Error ? exc.message : String(exc));
      }
    };

    void mount();
    return () => {
      cancelled = true;
      unlinkTraits?.();
      view?.remove?.();
      container.replaceChildren();
    };
  }, [hostModel, linkedTraits, widgetModel]);

  if (!widgetModel) return null;

  return (
    <Box
      sx={{
        mt: 1,
        pt: 1,
        borderTop: `1px solid ${themeColors.border}`,
        maxWidth: "100%",
        boxSizing: "border-box",
      }}
    >
      <Box
        role="region"
        aria-label={title}
        sx={{
          width: "fit-content",
          maxWidth: "100%",
          overflowX: "auto",
          bgcolor: themeColors.bg,
          color: themeColors.text,
          border: `1px solid ${themeColors.border}`,
          boxSizing: "border-box",
          p: 1,
        }}
      >
        <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 1, mb: 0.75 }}>
          <Typography sx={{ fontSize: 12, color: themeColors.textMuted, fontWeight: 600 }}>
            {title}
          </Typography>
          <Button
            size="small"
            onClick={onClose}
            sx={{
              fontSize: 10,
              textTransform: "none",
              letterSpacing: 0,
              minWidth: 0,
              px: 0.75,
              py: 0,
              color: themeColors.accent,
            }}
          >
            Hide
          </Button>
        </Box>
        {error ? (
          <Typography sx={{ fontSize: 11, color: "#d32f2f" }}>
            View failed: {error}
          </Typography>
        ) : (
          <Box ref={containerRef} sx={{ maxWidth: "100%" }} />
        )}
      </Box>
    </Box>
  );
}
