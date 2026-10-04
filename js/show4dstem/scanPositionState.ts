import * as React from "react";
import type { useModel } from "@anywidget/react";

/** Model positions stay live; expensive React consumers settle during GPU drags. */
export function useScanPositionState(model: ReturnType<typeof useModel>, live: React.RefObject<boolean>) {
  const [position, setPosition] = React.useState<[number, number]>(() => [model.get("pos_row"), model.get("pos_col")]);
  React.useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const publish = () => setPosition([model.get("pos_row"), model.get("pos_col")]);
    const changed = () => {
      clearTimeout(timer);
      if (live.current) timer = setTimeout(publish, 80);
      else publish();
    };
    model.on("change:pos_row", changed); model.on("change:pos_col", changed);
    return () => { clearTimeout(timer); model.off("change:pos_row", changed); model.off("change:pos_col", changed); };
  }, [model, live]);
  const setRow = React.useCallback((value: React.SetStateAction<number>) => {
    model.set("pos_row", typeof value === "function" ? value(model.get("pos_row")) : value); model.save_changes();
  }, [model]);
  const setCol = React.useCallback((value: React.SetStateAction<number>) => {
    model.set("pos_col", typeof value === "function" ? value(model.get("pos_col")) : value); model.save_changes();
  }, [model]);
  return [position, setRow, setCol] as const;
}
