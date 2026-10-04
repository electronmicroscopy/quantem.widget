import { describe, expect, it } from "vitest";
import { createDpPointerOwner } from "./dpPointerOwner";

const mouse = { pointerId: 1, pointerType: "mouse", isPrimary: true, button: 0, buttons: 1 };
const pen = { pointerId: 2, pointerType: "pen", isPrimary: true, button: 0, buttons: 0 };
const pressedPen = { ...pen, buttons: 1 };
const releasedMouse = { ...mouse, buttons: 0 };
const rightMouse = { ...mouse, button: 2, buttons: 2 };

describe("DP pointer ownership", () => {
  it("keeps a mouse detector drag through unrelated primary pen hover and release", () => {
    const owner = createDpPointerOwner();
    let col = 95.5;
    expect(owner.start(mouse)).toBe(true);
    const move = (event: typeof mouse, nextCol: number) => {
      if (owner.move(event) === "drag") col = nextCol;
    };
    move(mouse, 96);
    // Both inputs are primary: isPrimary alone cannot prevent this real case.
    move(pen, 190); move(pen, 189);
    expect(owner.start(pressedPen)).toBe(false);
    expect(owner.release(pen)).toBe(false);
    expect(owner.active).toBe(true);
    expect(col).toBe(96);
    move(mouse, 99.5);
    expect(col).toBe(99.5);
    expect(owner.release(releasedMouse)).toBe(true);
    expect(owner.move(mouse)).toBe("hover");
    expect(owner.release(mouse)).toBe(false); // Bubbling window up is not a second finalize.
  });

  it("continues outer or inner resize outside the canvas after capture loss", () => {
    for (const initialRadius of [28, 40]) {
      const owner = createDpPointerOwner();
      let radius = initialRadius;
      owner.start(mouse);
      // A leave/capture transition does not change the owner. Window events
      // resize with the same identity and foreign up/cancel cannot end it.
      expect(owner.active).toBe(true);
      expect(owner.owns(pen)).toBe(false);
      expect(owner.release(pen)).toBe(false);
      if (owner.owns(mouse) && owner.move(mouse) === "drag") radius += 5;
      expect(radius).toBe(initialRadius + 5);
      expect(owner.release(mouse)).toBe(true);
      expect(owner.active).toBe(false);
    }
  });

  it("finishes a touch gesture only with its own up or cancel and accepts the next contact", () => {
    const owner = createDpPointerOwner();
    const touch = { ...mouse, pointerId: 5, pointerType: "touch" };
    const second = { ...touch, pointerId: 6, isPrimary: false };
    expect(owner.start(touch)).toBe(true);
    expect(owner.start(second)).toBe(false);
    expect(owner.move(second)).toBe("ignore");
    expect(owner.release(second)).toBe(false);
    expect(owner.move({ ...touch, buttons: 0 })).toBe("drag");
    expect(owner.release(touch)).toBe(true);
    expect(owner.start({ ...touch, pointerId: 7 })).toBe(true);
    owner.reset();
    expect(owner.active).toBe(false);
  });

  it("ends a lost mouse release instead of resuming on hover and ignores secondary buttons", () => {
    const owner = createDpPointerOwner();
    expect(owner.start(rightMouse)).toBe(false);
    expect(owner.start(mouse)).toBe(true);
    expect(owner.move(pen)).toBe("ignore");
    const released = { ...mouse, buttons: 0 };
    expect(owner.move(released)).toBe("release");
    expect(owner.release(released)).toBe(true);
    expect(owner.move(released)).toBe("hover");
    expect(owner.start(pressedPen)).toBe(true);
    expect(owner.move({ ...pen, buttons: 1 })).toBe("drag");
  });
});
