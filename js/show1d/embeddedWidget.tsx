import * as React from "react";
import type { useModel } from "@anywidget/react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";

type WidgetModel = ReturnType<typeof useModel>;

// The ipywidgets manager behind anywidget models (JupyterLab, VS Code, Colab)
// also builds views of other widgets; the anywidget type declares only get_model.
type ViewManager = WidgetModel["widget_manager"] & {
  create_view(model: WidgetModel): Promise<{ el: HTMLElement; remove(): void }>;
};

type LinkedTrait = {
  source: string;
  target?: string;
};

// Mirror each linked trait both ways between the host and the embedded view.
// A trait only one of them has stays unlinked.
function syncLinkedTraits(hostModel: WidgetModel, childModel: WidgetModel, traits: LinkedTrait[]): () => void {
  let syncing = false;
  const cleanups: Array<() => void> = [];

  const copy = (from: WidgetModel, to: WidgetModel, fromKey: string, toKey: string) => {
    if (syncing) return;
    const value = from.get(fromKey);
    if (value === undefined || to.get(toKey) === undefined) return;
    // The set fires the other side's change handler, which must not copy the value back.
    syncing = true;
    to.set(toKey, value);
    to.save_changes();
    syncing = false;
  };

  for (const trait of traits) {
    const source = trait.source;
    const target = trait.target || trait.source;
    copy(hostModel, childModel, source, target);

    const sourceHandler = () => copy(hostModel, childModel, source, target);
    const targetHandler = () => copy(childModel, hostModel, target, source);
    hostModel.on(`change:${source}`, sourceHandler);
    childModel.on(`change:${target}`, targetHandler);
    cleanups.push(() => hostModel.off(`change:${source}`, sourceHandler));
    cleanups.push(() => childModel.off(`change:${target}`, targetHandler));
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
  hostModel: WidgetModel;
  // The serialized reference of the widget to embed, "IPY_MODEL_<model id>".
  widgetModel: string;
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
    let view: { el: HTMLElement; remove(): void } | null = null;
    let unlinkTraits: (() => void) | null = null;
    const container = containerRef.current;
    if (!container) return undefined;
    container.replaceChildren();
    setError("");

    const mount = async () => {
      try {
        const manager = hostModel.widget_manager as ViewManager;
        const childModel = await manager.get_model(widgetModel.replace(/^IPY_MODEL_/, ""));
        unlinkTraits = syncLinkedTraits(hostModel, childModel, linkedTraits);
        view = await manager.create_view(childModel);
        if (cancelled) {
          view.remove();
          return;
        }
        container.replaceChildren(view.el);
      } catch (exc) {
        if (!cancelled) setError(exc instanceof Error ? exc.message : String(exc));
      }
    };

    void mount();
    return () => {
      cancelled = true;
      unlinkTraits?.();
      view?.remove();
      container.replaceChildren();
    };
  }, [hostModel, linkedTraits, widgetModel]);

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
