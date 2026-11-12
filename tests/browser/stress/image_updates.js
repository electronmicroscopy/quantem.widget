// Observe new native scalar buffers, not repeated draws of an old image.
(() => {
  const state = window.scientificStress = {
    active: false, frameBytes: 0, stackBytes: 0, count: 0, revision: 0,
    uploads: [], presentations: [], frames: [], inputs: [], errors: [], pageErrors: [],
  };
  // Playwright disconnects while timing: retain uncaught errors in the page.
  addEventListener('error', event => state.pageErrors.push(String(event.message).slice(0, 300)));
  addEventListener('unhandledrejection', event => state.pageErrors.push(String(event.reason?.stack || event.reason).slice(0, 1000)));
  const buffers = new WeakMap();
  let latest = null;
  // Hash outside the timed interval; retain a view of only the newest stack.
  window.scientificStressUploadHash = async () => {
    if (!latest) return null;
    const bytes = new Uint8Array(latest.source, latest.start, state.stackBytes);
    const digest = await crypto.subtle.digest('SHA-256', bytes);
    return {bytes: bytes.byteLength, sha256: [...new Uint8Array(digest)]
      .map(value => value.toString(16).padStart(2, '0')).join('')};
  };
  const targets = new Set();
  const submitted = new Map();
  const write = GPUQueue.prototype.writeBuffer;
  GPUQueue.prototype.writeBuffer = function(buffer, offset, data, dataOffset, size) {
    const result = write.apply(this, arguments);
    const source = ArrayBuffer.isView(data) ? data.buffer : data;
    const bytes = size === undefined ? data.byteLength : size;
    if (state.active && bytes === state.frameBytes && source.byteLength >= state.stackBytes) {
      if (!buffers.has(source)) buffers.set(source, ++state.revision);
      state.current = buffers.get(source);
      const unit = ArrayBuffer.isView(data) && 'BYTES_PER_ELEMENT' in data ? data.BYTES_PER_ELEMENT : 1;
      const start = (ArrayBuffer.isView(data) ? data.byteOffset : 0) + (dataOffset || 0) * unit;
      if (!latest || latest.source !== source) latest = {source, start};
      else latest.start = Math.min(latest.start, start);
      state.uploads.push({at: performance.now(), revision: state.current, bytes});
    }
    return result;
  };
  const texture = GPUCanvasContext.prototype.getCurrentTexture;
  GPUCanvasContext.prototype.getCurrentTexture = function() {
    const result = texture.call(this);
    const name = this.canvas.dataset?.quantemScientificOutput;
    if (state.active && name?.startsWith('show4dstem-compare-')) targets.add(name);
    return result;
  };
  const submit = GPUQueue.prototype.submit;
  GPUQueue.prototype.submit = function() {
    const result = submit.apply(this, arguments);
    if (state.active && targets.size && state.current) {
      let frame = submitted.get(state.current);
      if (!frame) {
        frame = {targets: new Set(), scheduled: false};
        submitted.set(state.current, frame);
      }
      for (const name of targets) frame.targets.add(name);
      targets.clear();
      if (!frame.scheduled && frame.targets.size === state.count) {
        frame.scheduled = true;
        const item = {queued: performance.now(), revision: state.current, targets: [...frame.targets]};
        this.onSubmittedWorkDone().then(() => requestAnimationFrame(now => {
          item.ready = now;
          state.presentations.push(item);
        })).catch(error => state.errors.push(String(error)));
      }
    }
    return result;
  };
  const requestDevice = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function() {
    const device = await requestDevice.apply(this, arguments);
    device.lost.then(info => state.errors.push(`Device lost: ${info.reason}: ${info.message}`));
    device.addEventListener('uncapturederror', event => state.errors.push(event.error.message));
    return device;
  };
  addEventListener('pointermove', event => {
    if (state.active && event.buttons) state.inputs.push({at: performance.now(), trusted: event.isTrusted});
  }, true);
  function frame(now) {
    if (state.active) state.frames.push(now);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
