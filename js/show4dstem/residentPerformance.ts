import * as React from "react";
import { useModel } from "@anywidget/react";

const samples: Record<string, number[]> = {};
function record(kind: string, duration: number) {
  const values = samples[kind] ?? (samples[kind] = []);
  values.push(duration);
  if (values.length > 4096) values.splice(0, values.length - 4096);
}

/** Bounded wall-time evidence for the optional resident scientific stream. */
export function useResidentRenderTiming(enabled: boolean, kind: string) {
  const started = performance.now();
  React.useLayoutEffect(() => {
    if (enabled) record(kind, performance.now() - started);
  });
}

export function useResidentChanges(enabled: boolean, kind: string, values: Record<string, unknown>) {
  const previous = React.useRef(values);
  React.useLayoutEffect(() => {
    if (enabled) for (const key of Object.keys(values)) {
      if (!Object.is(previous.current[key], values[key])) record(`${kind}.${key}`, 1);
    }
    previous.current = values;
  });
}

export function useResidentPerformance(model: ReturnType<typeof useModel>, enabled: boolean) {
  useResidentRenderTiming(enabled, "root_render_to_commit_ms");
  React.useEffect(() => {
    if (!enabled) return;
    const observer = new PerformanceObserver(list => {
      for (const entry of list.getEntries()) record("main_thread_long_task_ms", entry.duration);
    });
    if (PerformanceObserver.supportedEntryTypes.includes("longtask"))
      observer.observe({entryTypes:["longtask"]});
    const command = (content: {type?:string;reset?:boolean}) => {
      if (content.type !== "resident_browser_profile_request") return;
      model.send({type:"resident_browser_profile",samples,epoch_ms:Date.now()});
      if (content.reset) for (const key of Object.keys(samples)) samples[key] = [];
    };
    model.on("msg:custom", command);
    return () => { observer.disconnect();model.off("msg:custom", command); };
  }, [model,enabled]);
}
