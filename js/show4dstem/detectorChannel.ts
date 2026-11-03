type Request = { view: string; epoch: number; seq: number; center: number[] };

export function detectorViewId() {
  return Array.from(crypto.getRandomValues(new Uint32Array(4))).join("-");
}

/** One unaccepted request and one replaceable position, scoped to this view. */
export function detectorChannel(view: string, send: (request: Request) => void, settled: () => void) {
  let epoch = 0, serial = 0, waiting = 0, completed = 0;
  let stopped = false;
  let pending: Request | null = null;
  const flush = () => {
    if (waiting || !pending) return;
    const request = pending; pending = null; waiting = request.seq;
    send(request);
  };
  return {
    request(row: number, col: number) {
      if (stopped) return false;
      pending = { view, epoch, seq: ++serial, center: [row, col] }; flush();
      return true;
    },
    receive(reply: Request & { type: string; status: string }) {
      if (reply.type !== "show4dstem_detector_receipt" || reply.view !== view) return;
      if (!["accepted", "complete", "rejected"].includes(reply.status)) return;
      if (reply.seq === waiting) { waiting = 0; flush(); }
      if (reply.status !== "accepted" && reply.epoch === epoch && reply.seq === serial) {
        completed = serial; settled();
      }
    },
    clear() { epoch++; pending = null; completed = serial; },
    stop() { stopped = true; epoch++; pending = null; completed = serial; },
    isPending() { return completed !== serial; },
  };
}
