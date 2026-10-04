import {describe,expect,it} from 'vitest';
import {createLatestFrameQueue} from './latestFrameQueue';
const turn=()=>new Promise<void>(resolve=>setTimeout(resolve,0));
describe('scan point queries',()=>{
 it('coalesces paired coordinates, serializes work, and publishes the final dragged position',async()=>{
  let position=1,active=0,peak=0;const queried:number[]=[],published:number[]=[],release:Array<()=>void>=[];
  const queue=createLatestFrameQueue(async current=>{
   active++;peak=Math.max(peak,active);const point=position;queried.push(point);
   await new Promise<void>(resolve=>release.push(resolve));if(current())published.push(point);active--;
  },error=>{throw error});
  queue.request();queue.request();await turn();expect(queried).toEqual([1]);
  position=2;queue.request();position=3;queue.request();release.shift()!();await turn();
  expect(queried).toEqual([1,3]);expect(published).toEqual([1]);release.shift()!();await turn();
  expect(published).toEqual([1,3]);expect(peak).toBe(1);queue.close();
 });
 it('publishes during uninterrupted motion while retaining only the newest pending point',async()=>{
  let point=0;const published:number[]=[],release:Array<()=>void>=[];
  const queue=createLatestFrameQueue(async current=>{
   const selected=point;await new Promise<void>(resolve=>release.push(resolve));
   if(current())published.push(selected);
  },error=>{throw error});
  queue.request();await turn();
  for(let i=1;i<=5;i++){
   point=i;queue.request();release.shift()!();await turn();
   expect(published).toHaveLength(i);
  }
  release.shift()!();await turn();expect(published).toEqual([0,1,2,3,4,5]);queue.close();
 });
 it('rejects a previous acquisition while allowing its replacement to publish',async()=>{
  let source=0;const published:number[]=[],release:Array<()=>void>=[];
  const queue=createLatestFrameQueue(async current=>{
   const selected=source;await new Promise<void>(resolve=>release.push(resolve));
   if(current())published.push(selected);
  },error=>{throw error});
  queue.request();await turn();source=1;queue.invalidate();queue.request();
  release.shift()!();await turn();expect(published).toEqual([]);
  release.shift()!();await turn();expect(published).toEqual([1]);queue.close();
 });
 it('ignores old-source completions after replacement',async()=>{
  let release!:()=>void;const published:number[]=[];
  const queue=createLatestFrameQueue(async current=>{await new Promise<void>(resolve=>{release=resolve});if(current())published.push(1)},()=>{});
  queue.request();await turn();queue.close();release();queue.request();await turn();expect(published).toEqual([]);
 });
 it('reports the current failure and permits a subsequent valid point',async()=>{
  let failed=true;const errors:unknown[]=[],published:number[]=[];
  const queue=createLatestFrameQueue(async current=>{if(failed)throw Error('decode');if(current())published.push(1)},error=>errors.push(error));
  queue.request();await turn();expect(errors).toHaveLength(1);failed=false;queue.request();await turn();expect(published).toEqual([1]);queue.close();
 });
});
