import { afterEach, beforeEach, expect, it, vi } from 'vitest';
const fake = vi.hoisted(() => ({engines: [] as any[]}));
vi.mock('../colormaps', () => ({COLORMAPS: {inferno: []}, GPUColormapEngine: class {
  slots = new Map(); reads = vi.fn(async (..._args: any[]) => [new Float32Array([1,2,3,4])]);
  draw = vi.fn((..._args: any[]) => true); average = vi.fn((..._args: any[]) => true); destroy = vi.fn();
  context = {configure: vi.fn(), unconfigure: vi.fn()};
  constructor() { fake.engines.push(this); }
  configureCanvas() {return this.context;}
  borrowBuffer(...args: any[]) {this.slots.set(args[0], args[1]);}
  averageResidentSlotsInto(...args: any[]) {return this.average(...args);}
  readDataSlots(...args: any[]) {return this.reads(...args);}
  uploadLUT() {}
  renderSlotDirectWithGpuRangeToCanvas(...args: any[]) {return this.draw(...args);}
  renderPanelSlotsDirectToCanvas(...args: any[]) {return this.draw(...args);}
}}));
import { ResidentDpDisplay } from './residentDp';
const options = {width:2,height:2,colormap:'inferno',log:false,minPct:0,maxPct:100,min:null,max:null,zoom:1,panX:0,panY:0};
beforeEach(() => {vi.useFakeTimers(); fake.engines.length=0;
  vi.stubGlobal('GPUTextureUsage', {RENDER_ATTACHMENT:1,COPY_SRC:2});
  vi.stubGlobal('navigator', {gpu:{getPreferredCanvasFormat:()=> 'rgba8unorm'}});
});
afterEach(() => {vi.useRealTimers();vi.unstubAllGlobals();});
function fixture() {
  const device={} as GPUDevice, pending=vi.fn(), error=vi.fn(), settled=vi.fn();
  const canvas={dataset:{}} as HTMLCanvasElement;
  const display=new ResidentDpDisplay(canvas,device,options,pending,error);
  const sources=Array.from({length:3}, () => ({getDevice:()=>device,frameAtBuffer:vi.fn(()=>({buffer:{} as GPUBuffer,n:4}))}));
  return {display,sources,pending,error,settled,canvas,engine:fake.engines[0]};
}
it('selected and average drags submit immediately with no per-frame CPU readback', async () => {
  const f=fixture();
  for(let scan=0;scan<120;scan++) {f.display.show(f.sources,scan,f.settled);await vi.advanceTimersByTimeAsync(8);}
  expect(f.engine.draw).toHaveBeenCalledTimes(120);
  expect(f.engine.average).toHaveBeenLastCalledWith(0,[1,2,3]);
  expect(f.engine.reads).not.toHaveBeenCalled();
  expect(f.pending.mock.calls).toEqual([[true]]);
  await vi.advanceTimersByTimeAsync(80);
  expect(f.settled).toHaveBeenCalledTimes(1); expect(f.pending).toHaveBeenLastCalledWith(false);
  f.display.show([f.sources[0]],121,f.settled);
  expect(f.engine.draw.mock.calls.at(-1)[0]).toBe(1);
  f.display.destroy(); expect(f.engine.destroy).toHaveBeenCalledOnce();
});
it('a delayed old read cannot publish after a new position or mode', async () => {
  const f=fixture(); let finish!: (frames: Float32Array[]) => void;
  f.engine.reads.mockImplementationOnce(()=>new Promise(resolve=>{finish=resolve;}));
  f.display.show(f.sources,0,f.settled);await vi.advanceTimersByTimeAsync(80);
  f.display.show([f.sources[0]],1,f.settled);finish([new Float32Array([9,9,9,9])]);await Promise.resolve();
  expect(f.settled).not.toHaveBeenCalled();
  await vi.advanceTimersByTimeAsync(80);expect(f.settled).toHaveBeenCalledOnce();
  f.display.show(f.sources,2,f.settled);f.display.destroy();await vi.advanceTimersByTimeAsync(80);
  expect(f.engine.reads).toHaveBeenCalledTimes(2);
  expect(f.engine.context.unconfigure).toHaveBeenCalledOnce();
});
it('decoder-error sentinels fail instead of publishing plausible scientific values', async () => {
  const f=fixture();f.engine.reads.mockResolvedValueOnce([new Float32Array([-1,2,3,4])]);
  f.display.show(f.sources,0,f.settled);await vi.advanceTimersByTimeAsync(80);
  expect(f.settled).not.toHaveBeenCalled();expect(f.error).toHaveBeenCalledOnce();f.display.destroy();
});
