/** One DP gesture belongs to its initiating pointer until up/cancel.
 * Pointer identity, not isPrimary, separates a pen from a concurrent mouse:
 * both may be primary for their respective device types.
 */
export function createDpPointerOwner() {
  let pointerId: number | null = null;
  return {
    get active() { return pointerId !== null; },
    start(event: { pointerId: number; button: number; isPrimary: boolean }) {
      if (pointerId !== null || event.button !== 0 || !event.isPrimary) return false;
      pointerId = event.pointerId;
      return true;
    },
    owns(event: { pointerId: number }) { return pointerId === event.pointerId; },
    move(event: { pointerId: number; pointerType: string; buttons: number }) {
      if (pointerId === null) return "hover" as const;
      if (pointerId !== event.pointerId) return "ignore" as const;
      // A mouse/pen returning after release outside the window must not resume
      // a stale drag. Touch completion remains governed by up/cancel.
      if (event.buttons === 0 && event.pointerType !== "touch") return "release" as const;
      return "drag" as const;
    },
    release(event: { pointerId: number }) {
      if (pointerId !== event.pointerId) return false;
      pointerId = null;
      return true;
    },
    reset() { pointerId = null; },
  };
}
