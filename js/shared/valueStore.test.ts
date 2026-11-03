import * as React from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { expect, it } from "vitest";
import { Subscribe, ValueStore } from "./valueStore";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

it("re-renders only the subscriber when a frame's values change", async () => {
  const store = new ValueStore({ min: 0, max: 1, bins: [0] });
  let parentRenders = 0;
  let childRenders = 0;
  function Parent() {
    parentRenders++;
    return React.createElement(Subscribe<{ min: number; max: number; bins: number[] }>, {
      store,
      children: (value) => { childRenders++; return React.createElement("span", null, `${value.min}..${value.max}`); },
    });
  }
  const host = document.createElement("div");
  const root = createRoot(host);
  await act(async () => root.render(React.createElement(Parent)));
  expect(host.textContent).toBe("0..1");
  await act(async () => { store.update({ min: -2, max: 3 }); });
  expect(host.textContent).toBe("-2..3");
  expect([parentRenders, childRenders]).toEqual([1, 2]);
  // An update that changes nothing notifies nobody.
  await act(async () => { store.update({ min: -2 }); });
  expect(childRenders).toBe(2);
  await act(async () => root.unmount());
});
