// @vitest-environment jsdom
import * as React from 'react';
import {createRoot} from 'react-dom/client';
import {expect,it,vi} from 'vitest';
import {useScanPositionState} from './scanPositionState';
it('keeps model events live while settling React and preserves functional setters', async () => {
  vi.useFakeTimers();(globalThis as any).IS_REACT_ACT_ENVIRONMENT=true;
  const values:any={pos_row:0,pos_col:0},listeners=new Map<string,Set<()=>void>>();
  const model:any={get:(k:string)=>values[k],set(k:string,v:any){values[k]=v;listeners.get('change:'+k)?.forEach(fn=>fn());},
    save_changes:vi.fn(),on(k:string,fn:()=>void){if(!listeners.has(k))listeners.set(k,new Set());listeners.get(k)!.add(fn);},off(k:string,fn:()=>void){listeners.get(k)?.delete(fn);}};
  const live={current:true};let state:ReturnType<typeof useScanPositionState>;let renders=0;
  function View(){state=useScanPositionState(model,live);renders++;return null;}
  const root=createRoot(document.createElement('div'));
  try {
    await React.act(async()=>root.render(React.createElement(View)));
    const initial=renders;
    await React.act(async()=>{for(let i=1;i<=30;i++){model.set('pos_row',i);model.set('pos_col',i+1);await vi.advanceTimersByTimeAsync(8);}});
    expect(renders).toBe(initial);expect(model.get('pos_row')).toBe(30);
    await React.act(async()=>{await vi.advanceTimersByTimeAsync(80);});
    expect(state![0]).toEqual([30,31]);
    live.current=false;await React.act(async()=>{state![1](row=>row+1);});
    expect(state![0]).toEqual([31,31]);expect(model.save_changes).toHaveBeenCalledOnce();
    await React.act(async()=>root.unmount());expect([...listeners.values()].every(set=>set.size===0)).toBe(true);
  }finally{vi.useRealTimers();delete (globalThis as any).IS_REACT_ACT_ENVIRONMENT;}
});
