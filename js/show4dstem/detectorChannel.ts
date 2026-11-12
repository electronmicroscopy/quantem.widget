type Request = { view: string; epoch: number; seq: number; center: number[] };

export function detectorViewId() {
  return Array.from(crypto.getRandomValues(new Uint32Array(4))).join("-");
}

/** A bounded request window and one replaceable position, scoped to this view. */
export function detectorChannel(view: string, send: (request: Request) => void, settled: (reply: Request & { status: string }) => void,
  receiptType = "show4dstem_detector_receipt", pendingLimit = 1) {
  let epoch = 0, serial = 0, completed = 0;
  const waiting = new Set<number>();
  let stopped = false;
  let pending: Request | null = null;
  const flush = () => {
    if (waiting.size >= pendingLimit || !pending) return;
    const request = pending; pending = null; waiting.add(request.seq);
    send(request);
  };
  return {
    request(row: number, col: number) {
      if (stopped) return false;
      pending = { view, epoch, seq: ++serial, center: [row, col] }; flush();
      return true;
    },
    receive(reply: Request & { type: string; status: string }) {
      if (reply.type !== receiptType || reply.view !== view) return;
      if (!["accepted", "complete", "rejected"].includes(reply.status)) return;
      if (waiting.delete(reply.seq)) flush();
      if (reply.status !== "accepted" && reply.epoch === epoch && reply.seq === serial) {
        completed = serial; settled(reply);
      }
    },
    clear() { epoch++; pending = null; completed = serial; },
    stop() { stopped = true; epoch++; pending = null; completed = serial; },
    isPending() { return completed !== serial; },
  };
}
