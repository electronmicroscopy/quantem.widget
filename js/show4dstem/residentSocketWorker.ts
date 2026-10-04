/** Self-contained worker entrypoint, bundled into the widget's existing ESM. */
export function residentSocketWorker() {
  const scope = self as unknown as {
    postMessage: (message: unknown, transfer?: Transferable[]) => void;
    onmessage: ((event: MessageEvent) => void) | null;
    close: () => void;
  };
  let socket: WebSocket | null = null;
  let config: {url:string;generation:string} | null = null;
  let disposed = false;
  let retry: ReturnType<typeof setTimeout> | undefined;
  let header: Record<string, unknown> | null = null;
  let latest = -1;
  let outstanding: number | null = null;
  let pending: {bytes:ArrayBuffer;info:Record<string,unknown>} | null = null;
  let superseded = 0;
  const deliver = () => {
    if (!pending || outstanding !== null) return;
    const frame = pending;
    pending = null;
    outstanding = Number(frame.info.request_id);
    scope.postMessage({type:"frame",...frame,superseded},[frame.bytes]);
  };
  const connect = () => {
    if (disposed || !config) return;
    const current = new WebSocket(config.url);
    socket = current;
    current.binaryType = "arraybuffer";
    current.onopen = () => current.send(JSON.stringify({type:"ready"}));
    current.onmessage = event => {
      if (current !== socket) return;
      try {
        if (typeof event.data === "string") {
          const metadata = JSON.parse(event.data);
          if (metadata.type !== "resident_batch" || metadata.generation !== config?.generation || header)
            throw new Error("Unexpected scientific batch metadata or missing binary payload");
          header = metadata;
          return;
        }
        const info = header;
        header = null;
        const bytes = event.data;
        if (!info || !(bytes instanceof ArrayBuffer) || bytes.byteLength !== info.bytes)
          throw new Error("Incomplete scientific batch; retained the last complete image set");
        const request = Number(info.request_id);
        if (!Number.isSafeInteger(request) || request <= latest)
          throw new Error("Scientific batch request IDs must increase");
        latest = request;
        const receivedEpoch = performance.timeOrigin + performance.now();
        current.send(JSON.stringify({type:"received",request_id:request,generation:info.generation,
          byte_length:bytes.byteLength,received_epoch_ms:receivedEpoch,worker:true}));
        if (pending) superseded++;
        pending = {bytes,info:{...info,received_epoch_ms:receivedEpoch}};
        deliver();
      } catch(error) {
        scope.postMessage({type:"error",error:String(error)});
        current.close(1002,"Invalid scientific batch");
      }
    };
    current.onclose = () => {
      if (disposed || current !== socket) return;
      header = null;
      pending = null;
      outstanding = null;
      scope.postMessage({type:"disconnected"});
      retry = setTimeout(connect,1000);
    };
    current.onerror = () => scope.postMessage({type:"error",error:"Scientific batch socket connection failed"});
  };
  scope.onmessage = event => {
    const message = event.data;
    if (message.type === "connect") { config=message.config;connect(); }
    else if (message.type === "adopted" && message.request_id === outstanding) {
      outstanding=null;deliver();
    } else if (message.type === "validation" && socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message));
    } else if (message.type === "dispose") {
      disposed=true;clearTimeout(retry);socket?.close();scope.close();
    }
  };
}
