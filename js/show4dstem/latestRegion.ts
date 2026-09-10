/** Bound live kernel work to one request plus the most recent pointer position. */
export function latestRegion(send: (row: number, col: number) => void, initial: number[]) {
  let busy = false;
  let last = initial;
  let pending: number[] | null = null;
  const flush = () => {
    if (busy || !pending) return;
    const position = pending;
    pending = null;
    if (position[0] === last[0] && position[1] === last[1]) return;
    busy = true;
    last = position;
    send(position[0], position[1]);
  };
  return {
    request(row: number, col: number) { pending = [row, col]; flush(); },
    acknowledge() { busy = false; flush(); },
    clear() { pending = null; },
    isPending() { return busy || pending !== null; },
  };
}
