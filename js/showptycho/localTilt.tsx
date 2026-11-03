import * as React from 'react';
import { useModelState } from '@anywidget/react';
import { Box, Button, CircularProgress, IconButton, Switch, Select, MenuItem, Tooltip, Typography } from '@mui/material';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import { COLORMAPS, createGPUColormapEngine, renderToOffscreen } from '../display/colormaps';
import { percentileClip } from '../display/stats';
import { extractFloat32 } from '../format';

export type Region = { id: string; label: string; bounds: [number, number, number, number] };
type Zoom = { zoom: number; panX: number; panY: number };
const RESET: Zoom = { zoom: 1, panX: 0, panY: 0 };
const COLORS = ['#38bdf8', '#fb923c', '#c084fc', '#4ade80', '#facc15'];

export function makeGrid(extent: number[], size: number, overlap: number): Region[] {
  if (size < 32 || overlap < 0 || overlap >= 1) throw Error('Use size ≥32 and overlap below 100%.');
  const step = Math.max(1, Math.round(size * (1 - overlap)));
  const result: Region[] = [];
  for (let row = extent[0]; row + size <= extent[1]; row += step) {
    for (let col = extent[2]; col + size <= extent[3]; col += step) {
      result.push({ id: `window-${row}-${col}`, label: `(${row + size / 2}, ${col + size / 2})`,
        bounds: [row, row + size, col, col + size] });
    }
  }
  if (!result.length || result.length > 64) throw Error('Choose a window and overlap giving 1–64 windows within the selected extent.');
  return result;
}

function RegionOverlay({ regions, selected, setSelected, editMode, size, shape, zoom, commit, map, onDrawn, onDelete }: any) {
  const [draft, setDraft] = React.useState<Region[] | null>(null);
  const drag = React.useRef<any>(null);
  const pending = React.useRef<Region[] | null>(null);
  const frame = React.useRef(0);
  const svg = React.useRef<SVGSVGElement>(null);
  const scale = Math.min(size / shape[1], size / shape[0]) * zoom.zoom;
  const left = (size - shape[1] * scale) / 2 + zoom.panX;
  const top = (size - shape[0] * scale) / 2 + zoom.panY;
  const point = (event: React.PointerEvent) => {
    const rect = svg.current!.getBoundingClientRect();
    return [Math.max(0, Math.min(shape[0], Math.round((event.clientY - rect.top - top) / scale))),
      Math.max(0, Math.min(shape[1], Math.round((event.clientX - rect.left - left) / scale)))];
  };
  React.useEffect(() => () => cancelAnimationFrame(frame.current), []);
  function start(event: React.PointerEvent, index?: number, handle?: number) {
    if (editMode === 'pan' || event.button !== 0) return;
    event.preventDefault(); event.stopPropagation();
    svg.current!.focus({preventScroll:true});
    const p = point(event);
    let next = regions.map((r: Region) => ({ ...r, bounds: [...r.bounds] }));
    let target = index ?? selected;
    const drawing = !next.length || (editMode === 'draw' && handle === undefined);
    // Overview selects nothing: a click beside the rectangles moves none of them.
    if (!drawing && target < 0) return;
    if (drawing) {
      if (next.length >= 64) return;
      target = next.length;
      let ordinal = next.length + 1;
      while (next.some((r: Region) => r.label === `Region ${ordinal}`)) ordinal++;
      const row=Math.min(p[0],shape[0]-32), col=Math.min(p[1],shape[1]-32);
      next.push({ id: `region-${crypto.randomUUID()}`, label: `Region ${ordinal}`,
        bounds: [row, row + 32, col, col + 32] });
      handle = 3;
    } else if (index === undefined) {
      const b = next[target].bounds;
      const h = b[1] - b[0], w = b[3] - b[2];
      const row = Math.max(0, Math.min(shape[0] - h, p[0] - Math.round(h / 2)));
      const col = Math.max(0, Math.min(shape[1] - w, p[1] - Math.round(w / 2)));
      next[target].bounds = [row, row + h, col, col + w];
    }
    setSelected(target, next);
    drag.current = { point: p, regions: next, target, handle, drawing };
    pending.current = next; setDraft(next);
    svg.current!.setPointerCapture(event.pointerId);
  }
  function move(event: React.PointerEvent) {
    if (!drag.current) return;
    const d = drag.current, p = point(event);
    const next = d.regions.map((r: Region) => ({ ...r, bounds: [...r.bounds] }));
    let [r0, r1, c0, c1] = d.regions[d.target].bounds;
    const dr = p[0] - d.point[0], dc = p[1] - d.point[1];
    if (d.drawing) {
      r0=Math.min(Math.min(d.point[0],p[0]),shape[0]-32);
      c0=Math.min(Math.min(d.point[1],p[1]),shape[1]-32);
      r1=Math.min(shape[0],Math.max(d.point[0],p[0],r0+32));
      c1=Math.min(shape[1],Math.max(d.point[1],p[1],c0+32));
    } else if (d.handle === undefined) {
      const h = r1-r0, w = c1-c0;
      r0 = Math.max(0, Math.min(shape[0]-h, r0+dr)); r1=r0+h;
      c0 = Math.max(0, Math.min(shape[1]-w, c0+dc)); c1=c0+w;
    } else {
      if (d.handle < 2) r0=Math.max(0, Math.min(r1-32,r0+dr));
      else r1=Math.min(shape[0],Math.max(r0+32,r1+dr));
      if (d.handle % 2 === 0) c0=Math.max(0,Math.min(c1-32,c0+dc));
      else c1=Math.min(shape[1],Math.max(c0+32,c1+dc));
    }
    next[d.target].bounds=[r0,r1,c0,c1]; pending.current=next;
    if (!frame.current) frame.current=requestAnimationFrame(() => { frame.current=0; setDraft(pending.current); });
  }
  function finish(event: React.PointerEvent) {
    if (!drag.current) return;
    cancelAnimationFrame(frame.current); frame.current=0;
    if (event.type !== 'pointercancel' && pending.current) {
      commit(pending.current);
      if (drag.current.drawing) onDrawn();
    }
    drag.current=null; pending.current=null; setDraft(null);
  }
  return <svg ref={svg} width={size} height={size} aria-label="Local tilt region editor" tabIndex={0}
    style={{position:'absolute', inset:0, touchAction:'none', pointerEvents:editMode==='pan'?'none':'auto'}}
    onKeyDown={e=>{if(e.key==='Delete'||e.key==='Backspace'){e.preventDefault();e.stopPropagation();onDelete();}}}
    onPointerDown={e=>start(e)} onPointerMove={move} onPointerUp={finish} onPointerCancel={finish}>
    {map.map((m:any,i:number)=>{const row=(m.bounds[0]+m.bounds[1])/2,col=(m.bounds[2]+m.bounds[3])/2;
      const t=m.sample_tilt_mrad; if(!t)return null;
      const x=left+col*scale,y=top+row*scale;
      return <g key={`map-${i}`} pointerEvents="none"><circle cx={x} cy={y} r={2} fill="#f8fafc"/>
        <line x1={x} y1={y} x2={x+t[1]*scale*1.8} y2={y+t[0]*scale*1.8} stroke="#e2e8f0" strokeWidth={1}/></g>;})}
    {(draft||regions).map((r:Region,i:number)=>{const [r0,r1,c0,c1]=r.bounds; const color=COLORS[i%COLORS.length];
      return <g key={r.id}>
        <rect x={left+c0*scale} y={top+r0*scale} width={(c1-c0)*scale} height={(r1-r0)*scale}
          fill={i===selected?`${color}18`:'transparent'} stroke={color} strokeWidth={i===selected?2:1}
          onPointerDown={e=>start(e,i)} style={{cursor:editMode==='draw'?'crosshair':'move'}}/>
        {(regions.length<=4||i===selected)&&<text x={left+c0*scale+4} y={top+r0*scale+13} fill={color} fontSize={11} stroke="#0f172a" strokeWidth={2}
          paintOrder="stroke" pointerEvents="none">{r.label}</text>}
        {i===selected&&[[r0,c0],[r0,c1],[r1,c0],[r1,c1]].map(([row,col],h)=><rect key={h}
          x={left+col*scale-5} y={top+row*scale-5} width={10} height={10} rx={2} fill={color}
          onPointerDown={e=>start(e,i,h)} style={{cursor:h===0||h===3?'nwse-resize':'nesw-resize'}}/>)}
      </g>;})}
  </svg>;
}

function ResultPanel({ meta, values, size, Panel, tc, zoom, onZoomChange, limits, cmap, smooth }:any) {
  const [canvas,setCanvas]=React.useState<HTMLCanvasElement|null>(null);
  const [version,setVersion]=React.useState(0);
  const [path,setPath]=React.useState('Preparing display');
  const engine=React.useRef<ReturnType<typeof createGPUColormapEngine>|null>(null);
  const uploaded=React.useRef<Float32Array|null>(null);
  React.useEffect(()=>{
    engine.current=createGPUColormapEngine();
    return()=>{engine.current?.then(e=>e?.destroy());};
  },[]);
  React.useEffect(()=>{
    let gone=false;
    (async()=>{
      const gpu=await engine.current;
      if(gone)return;
      const [h,w]=meta.shape, lut=COLORMAPS[cmap as keyof typeof COLORMAPS]||COLORMAPS.magma;
      if(gpu){
        if(uploaded.current!==values){gpu.uploadData(0,values,w,h);uploaded.current=values;}
        const rgba=await gpu.applySingleWithLut(0,limits[0],limits[1],cmap,lut,false);
        if(!gone&&rgba){const next=document.createElement('canvas');next.width=w;next.height=h;
          next.getContext('2d')!.putImageData(new ImageData(new Uint8ClampedArray(rgba),w,h),0,0);
          setCanvas(next);setVersion(v=>v+1);setPath('WebGPU');}
      }else if(!gone){setCanvas(renderToOffscreen(values,w,h,lut,limits[0],limits[1]));setPath('Canvas · WebGPU unavailable');}
    })().catch(error=>{if(!gone)setPath(`Display error: ${String(error)}`);});
    return()=>{gone=true;};
  },[values,meta,limits,cmap]);
  return <Box sx={{minWidth:0}} title={`${limits[0].toFixed(3)} … ${limits[1].toFixed(3)} rad · ${meta.shape.join(' × ')} · ${path}`}>
    <Panel offscreen={canvas} offscreenVersion={version} label={meta.label} size={size} tc={tc} zoom={zoom} onZoomChange={onZoomChange}
      smooth={smooth} rawData={values} pixelSize={meta.sampling_A} imageWidth={meta.shape[1]}/>
  </Box>;
}

export function useLocalTiltWorkspace({ Panel, tc, ui, overviewSize, zoom, setZoom, onEnable, closeSettings, available = true, cmap, smooth, contrastRange, maxColumns=3, setMaxColumns }:any) {
  const [enabled,setEnabled]=useModelState<boolean>('local_tilt_enabled');
  const [activeRegion,setActiveRegion]=useModelState<string>('local_tilt_active_region');
  const [stale]=useModelState<boolean>('local_tilt_stale');
  const [configJson]=useModelState<string>('local_tilt_json');
  const [selectionJson,setSelectionJson]=useModelState<string>('local_tilt_selection_json');
  const [replyJson]=useModelState<string>('local_tilt_reply_json');
  const [,setRequest]=useModelState<string>('local_tilt_request_json');
  const [imagesJson]=useModelState<string>('local_tilt_images_json');
  const [imageBytes]=useModelState<DataView>('local_tilt_bytes');
  const [canFit]=useModelState<boolean>('local_tilt_can_fit');
  const config=React.useMemo(()=>JSON.parse(configJson || '{"regions":[]}'),[configJson]);
  const [regions,setRegions]=React.useState<Region[]>(()=>JSON.parse(selectionJson||'null')||config.regions);
  // -1 in Overview: no rectangle has handles, bounds inputs or click-to-place.
  const selected=regions.findIndex(region=>region.id===activeRegion);
  const [mode,setMode]=React.useState('move');
  const setSelected=(index:number,next=regions)=>setActiveRegion(next[index]?.id||'');
  const [auto,setAuto]=React.useState(true),[mapOn,setMapOn]=React.useState(true);
  const [windowSize,setWindowSize]=React.useState(128),[overlap,setOverlap]=React.useState(75);
  const [resultZoom,setResultZoom]=React.useState<Zoom>(RESET);
  const [status,setStatus]=React.useState('Ready');
  const [pending,setPending]=React.useState(false);
  const [elapsed,setElapsed]=React.useState(0);
  const [fittingSelection,setFittingSelection]=React.useState('');
  const [linked,setLinked]=React.useState(true),[individualZoom,setIndividualZoom]=React.useState<Record<string,Zoom>>({});
  const latest=React.useRef(regions); latest.current=regions;
  const timer=React.useRef(0);
  const busyRef=React.useRef(false);
  const autoRef=React.useRef(auto); autoRef.current=auto;
  React.useEffect(()=>()=>clearTimeout(timer.current),[]);
  React.useEffect(()=>{
    if(!pending)return;
    const started=performance.now();setElapsed(0);
    const tick=window.setInterval(()=>setElapsed((performance.now()-started)/1000),250);
    return()=>clearInterval(tick);
  },[pending]);
  const size=overviewSize;
  const fits=config.fits||[];
  const current=(r:Region)=>!stale && fits.find((f:any)=>f.id===r.id && JSON.stringify(f.bounds)===JSON.stringify(r.bounds));
  const fresh=regions.length>0 && regions.every(current);
  React.useEffect(()=>{
    const next=JSON.parse(selectionJson || 'null') || config.regions;
    if(JSON.stringify(next)!==JSON.stringify(latest.current)){
      latest.current=next;setRegions(next);
      setStatus('Selection updated from Python');
    }
  },[selectionJson]);
  React.useEffect(()=>{if(enabled)onEnable?.();else {clearTimeout(timer.current);setActiveRegion('');}},[enabled]);
  React.useEffect(()=>{if(!available)setEnabled(false);},[available]);
  function requestFit(next=latest.current,onlyChanged=false,forceRegion?:string){
    if(busyRef.current||!next.length)return;
    busyRef.current=true;
    const signature=JSON.stringify(next);
    setFittingSelection(signature);setPending(true);setStatus('Refitting regions…');
    setRequest(JSON.stringify({id:Date.now(),action:'fit',signature,regions:next,only_changed:onlyChanged,
      ...(forceRegion?{force_region:forceRegion}:{})}));
  }
  function commit(next:Region[],refit=true){
    latest.current=next;setRegions(next);setSelectionJson(JSON.stringify(next));
    setStatus(busyRef.current?'Selection changed · previous fit still running':'Selection changed · fit the edited regions');
    clearTimeout(timer.current);
    if(refit&&next.length&&enabled&&autoRef.current&&!busyRef.current)timer.current=window.setTimeout(()=>requestFit(next,true),350);
  }
  function deleteRegion(id:string){
    if(pending||!id)return;
    const region=regions.find(region=>region.id===id);
    if(!region)return;
    commit(regions.filter(region=>region.id!==id),false);
    if(activeRegion===id)setActiveRegion('');
    setStatus(`Deleted ${region.label}`);
  }
  const deleteSelected=()=>deleteRegion(activeRegion);
  React.useEffect(()=>{
    if(!replyJson)return;
    const reply=JSON.parse(replyJson);
    if(reply.status==='running'){
      setPending(true);busyRef.current=true;setFittingSelection(reply.signature);
      if(reply.stage==='joint_fit')setStatus(`Refitting ${reply.label} · repeat ${reply.repeat}/${reply.repeats} · ${reply.num_trials} trials + Newton`);
      else if(reply.stage==='preparing')setStatus(`Preparing ${reply.label} · ${reply.repeats} × ${reply.num_trials} trials + Newton`);
      else if(reply.stage==='reconstructing')setStatus(`Reconstructing ${reply.label} · full scan`);
      return;
    }
    setPending(false); busyRef.current=false;
    if(reply.signature!==JSON.stringify(latest.current)){
      setStatus(autoRef.current?'Fitting latest selection next…':'Earlier selection finished · current regions still need fitting');
      if(enabled&&autoRef.current){clearTimeout(timer.current);timer.current=window.setTimeout(()=>{if(autoRef.current)requestFit(latest.current,true);},350);}
    }
    else setStatus(reply.status==='error'?`Failed: ${reply.message}`:'Fits ready');
  },[replyJson]);
  const metas=React.useMemo(()=>JSON.parse(imagesJson||'[]'),[imagesJson]);
  const values=React.useMemo(()=>(imageBytes?.byteLength?extractFloat32(imageBytes):null)||new Float32Array(),[imageBytes]);
  const panels=React.useMemo(()=>metas.map((m:any)=>({meta:m,values:values.slice(m.offset,m.offset+m.shape[0]*m.shape[1])})),[metas,values]);
  const selectedRegion=selected>=0?regions[selected]:undefined;
  const control={...ui.button};
  const label=ui.typography.label;
  const value=ui.typography.value;
  const panelLimits=React.useMemo(()=>panels.map((p:any)=>{const q=percentileClip(p.values,contrastRange[0],contrastRange[1]);return [q.vmin,q.vmax];}),[panels,contrastRange]);
  const tiltText=(t:number[])=>`(${t.map(v=>v.toFixed(3)).join(', ')}) mrad`;
  function renderPanel(p:any){const i=panels.indexOf(p);return <ResultPanel key={p.meta.label} {...p} size={size} Panel={Panel} tc={tc}
    zoom={linked?resultZoom:(individualZoom[p.meta.label]||RESET)} onZoomChange={(z:Zoom)=>linked?setResultZoom(z):setIndividualZoom(prev=>({...prev,[p.meta.label]:z}))}
    limits={panelLimits[i]} cmap={cmap} smooth={smooth}/>;}
  function grid(){try{const extent=regions.length?regions.reduce((b,r)=>[Math.min(b[0],r.bounds[0]),Math.max(b[1],r.bounds[1]),Math.min(b[2],r.bounds[2]),Math.max(b[3],r.bounds[3])],[Infinity,0,Infinity,0]):[0,config.scan_shape[0],0,config.scan_shape[1]];
    const next=makeGrid(extent,windowSize,overlap/100);commit(next);setSelected(0,next);
  }catch(error){setStatus(`Failed: ${String(error)}`);}}
  const resetView=()=>{setActiveRegion('');setZoom(RESET);setResultZoom(RESET);setIndividualZoom({});};
  const canReset=!!activeRegion||[resultZoom,...Object.values(individualZoom)].some(view=>view.zoom!==1||view.panX!==0||view.panY!==0);
  if(!configJson)return null;
  const toggle=<Box component="label" sx={{...label,display:'inline-flex',alignItems:'center'}}
    title={available?'Show regional SSB alongside the full field':'Return to the full scan to edit local regions'}>
    Local Tilt<Switch size="small" inputProps={{'aria-label':'Local Tilt'}} checked={enabled&&available}
      disabled={!available} onChange={e=>setEnabled(e.target.checked)}/>
  </Box>;
  const overlay=<RegionOverlay regions={regions} selected={selected} setSelected={setSelected} editMode={mode}
    size={overviewSize} shape={config.scan_shape} zoom={zoom} commit={commit} onDrawn={()=>setMode('move')} onDelete={deleteSelected}
    map={mapOn&&!stale?(config.map||[]).filter((fit:any)=>regions.some(r=>r.id===fit.id)):[]}/>;
  const inputStyle={...value,width:58,bgcolor:tc.controlBg,color:tc.text,border:`1px solid ${tc.border}`,p:.25};
  const ready=`Fits ready${config.fitted_region_count===0?' · unchanged':config.last_fit_seconds!=null?` · ${config.last_fit_seconds.toFixed(2)} s`:''}`;
  const queued=pending&&fittingSelection!==JSON.stringify(regions);
  const message=pending?`${status}${queued?(auto?' · latest selection queued':' · selection changed'):''}`:!regions.length?'Draw a region':status.startsWith('Failed')?status:stale?'Optics changed · refit':!fresh?(auto?'Region changed · refit queued':'Region changed · refit'):status.startsWith('Deleted ')?status:ready;
  const toolbar=<Box data-local-tilt-controls sx={{display:'flex',gap:.5,flexWrap:'wrap',my:.5,alignItems:'center'}}>
    <Select size="small" displayEmpty value={regions.some(region=>region.id===activeRegion)?activeRegion:''} onChange={e=>setActiveRegion(e.target.value)}
      sx={{...ui.select,minWidth:90}} MenuProps={ui.menu} inputProps={{'aria-label':'Selected region'}}>
      <MenuItem value="" sx={label}>Overview</MenuItem>
      {regions.map(r=><MenuItem key={r.id} value={r.id} sx={label}>{r.label}</MenuItem>)}
    </Select>
    {['draw','move','pan'].map(m=><Tooltip key={m} title={m==='move'?'Click to place a region. Drag to move; corners resize.':m==='draw'?'Drag in any direction to draw a region':'Pan and zoom the overview'}>
      <Button sx={{...control,color:mode===m?tc.accent:tc.textMuted}} aria-pressed={mode===m}
        disabled={m==='draw'&&regions.length>=64}
        onClick={()=>setMode(m)}>{m==='move'?'Move':m==='draw'?'Draw':'Pan'}</Button></Tooltip>)}
    <Tooltip title="Delete the selected region. Delete or Backspace also works in the image."><span>
      <Button sx={control} aria-label="Delete selected region" disabled={!activeRegion||pending} onClick={deleteSelected}>Delete</Button>
    </span></Tooltip>
    <Box sx={{display:'inline-flex',alignItems:'center',gap:.5}}>
      <Typography sx={{...label,color:tc.text}}>Max Cols</Typography>
      <Select size="small" value={maxColumns} onChange={e=>setMaxColumns(Number(e.target.value))}
        sx={{...ui.select,minWidth:45}} MenuProps={ui.menu} inputProps={{'aria-label':'Maximum image columns'}}>
        {Array.from({length:8},(_,index)=>index+1).map(count=><MenuItem key={count} value={count} sx={label}>{count}</MenuItem>)}
      </Select>
    </Box>
    <Button sx={control} disabled={!canFit||pending||!regions.length} variant="outlined"
      onClick={()=>requestFit(latest.current,!!activeRegion,activeRegion||undefined)}>
      {activeRegion?'Fit Region':'Fit Regions'}
    </Button>
    <Tooltip title="Refit after releasing a moved or resized region"><Box component="label" sx={{...label,display:'inline-flex',alignItems:'center'}}>
      Auto<Switch size="small" checked={auto} disabled={!canFit} onChange={e=>setAuto(e.target.checked)} inputProps={{'aria-label':'Fit on release'}}/>
    </Box></Tooltip>
    <Box sx={{display:'inline-flex',alignItems:'center',gap:.5,ml:.5,minWidth:0,maxWidth:'100%'}}>
      <CircularProgress size={12} aria-label="Refitting regions" sx={{color:tc.accent,visibility:pending?'visible':'hidden',flexShrink:0}}/>
      <Typography role="status" aria-live="polite" title={message}
        sx={{...value,color:message.startsWith('Failed')?'#ef4444':pending?tc.accent:tc.textMuted,
          maxWidth:440,overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>{message}</Typography>
      {pending&&<Typography aria-hidden="true" sx={{...value,color:tc.textMuted,width:'7ch',flexShrink:0,fontVariantNumeric:'tabular-nums'}}>{elapsed.toFixed(1)} s</Typography>}
    </Box>
  </Box>;
  const settings=<Box sx={{pt:1}}>
        {selectedRegion&&<><Typography sx={{...label,mb:1}}>Region Bounds · Scan Pixels</Typography>
        <Box sx={{display:'flex',gap:.5}}>{['Row start','Row stop','Col start','Col stop'].map((name,i)=><Box component="label" key={name} sx={ui.typography.labelSmall}>
          {name}<br/><Box component="input" type="number" aria-label={name} value={selectedRegion.bounds[i]} disabled={pending}
            sx={inputStyle} onChange={(e:React.ChangeEvent<HTMLInputElement>)=>{
              const next=regions.map(r=>({...r,bounds:[...r.bounds] as Region['bounds']}));
              next[selected].bounds[i]=Number(e.target.value);commit(next);}}/>
        </Box>)}</Box></>}
        <Box sx={{display:'flex',gap:.5,my:1}}>
          <Button sx={control} disabled={pending} onClick={()=>{const next=config.anchors||config.regions;commit(next);setSelected(0,next);}}>Restore Regions</Button>
        </Box>
        <Typography sx={{...label,mb:.5}}>Overlapping Windows</Typography>
        <Box sx={{display:'flex',gap:1,alignItems:'center',mb:1}}>
          <Box component="label" sx={label}>Size <Box component="input" aria-label="Window size" type="number" min={32} step={32} value={windowSize} onChange={(e:any)=>setWindowSize(Number(e.target.value))} sx={inputStyle}/></Box>
          <Box component="label" sx={label}>Overlap % <Box component="input" aria-label="Window overlap" type="number" min={0} max={95} step={5} value={overlap} onChange={(e:any)=>setOverlap(Number(e.target.value))} sx={inputStyle}/></Box>
        </Box>
        <Button sx={control} disabled={pending} onClick={()=>{grid();closeSettings?.();}}>Build Grid</Button>
        <Box sx={{display:'flex',gap:1,mt:1}}>
          <Box component="label" sx={{...label,display:'inline-flex',alignItems:'center'}}>Tilt Map<Switch size="small" inputProps={{'aria-label':'Tilt Map'}} checked={mapOn} onChange={e=>setMapOn(e.target.checked)}/></Box>
          <Box component="label" sx={{...label,display:'inline-flex',alignItems:'center'}}>Link Zoom<Switch size="small" inputProps={{'aria-label':'Link Zoom'}} checked={linked} onChange={e=>{setLinked(e.target.checked);setIndividualZoom({});setResultZoom(RESET);}}/></Box>
        </Box>
        <Button sx={control} onClick={resetView}>Reset View</Button>
        <Typography sx={{...value,color:tc.textMuted,mt:1}}>Tilt (row, col), mrad. SD measures repeatability across optimizer seeds.</Typography>
      </Box>;
  const comparison=panels.map((p:any)=>{
    const fit=fits.find((f:any)=>`${f.label} · SSB`===p.meta.label);
    const region=regions.find(r=>r.id===fit?.id);
    if(fit&&!region)return null;
    const isCurrent=region&&current(region);
    return <Box key={p.meta.label} data-local-tilt-result onPointerDown={()=>{if(fit)setActiveRegion(fit.id);}}
      sx={{position:'relative',minWidth:0,maxWidth:size+2,outline:fit?.id===activeRegion?`1px solid ${tc.accent}`:'none',
        '&:hover .showptycho-region-delete, &:focus-within .showptycho-region-delete':{
          opacity:1,pointerEvents:'auto',transform:'translateY(0)',
        },
        '@media (hover: none), (pointer: coarse)':{
          '& .showptycho-region-delete':{opacity:1,pointerEvents:'auto',transform:'translateY(0)'},
        },
      }}>
      {renderPanel(p)}
      {region&&<IconButton className="showptycho-region-delete" disabled={pending}
        aria-label={`Delete ${region.label}`} title={`Delete ${region.label}`}
        onPointerDown={event=>event.stopPropagation()} onMouseDown={event=>event.stopPropagation()}
        onClick={event=>{event.stopPropagation();deleteRegion(region.id);}}
        sx={{position:'absolute',top:5,right:5,width:22,height:22,p:0,zIndex:3,
          opacity:0,pointerEvents:'none',transform:'translateY(-3px)',
          transition:'opacity 120ms ease, transform 120ms ease, background-color 120ms ease, color 120ms ease',
          color:'rgba(255,255,255,0.75)',bgcolor:'rgba(0,0,0,0.22)',
          '&:hover, &:focus-visible':{bgcolor:'rgba(0,0,0,0.42)',color:'rgba(255,255,255,0.95)'},
        }}>
        <DeleteOutlineIcon sx={{fontSize:15}}/>
      </IconButton>}
      <Typography title={fit?.aberrations?`Tilt ${tiltText(fit.sample_tilt_mrad)}, SD ${fit.seed_sd_mrad.toFixed(3)}. Joint fit: C10 ${fit.aberrations.C10_nm.toFixed(3)} nm, C12 ${fit.aberrations.C12_nm.toFixed(3)} nm, φ12 ${fit.aberrations.phi12_deg.toFixed(3)}°, depth ${fit.depth_spread_nm.toFixed(3)} nm`:undefined}
        sx={{...value,color:(!isCurrent||fit?.bound_hit?'#b7791f':tc.textMuted),mt:.25,whiteSpace:'nowrap',overflow:'hidden',textOverflow:'ellipsis'}}>
        {fit?`Tilt ${tiltText(fit.sample_tilt_mrad)} · SD ${fit.seed_sd_mrad.toFixed(3)}`:p.meta.label}{fit?.adjusted?' · adjusted':''}{fit?.bound_hit?' · fit bound':''}{!isCurrent?' · stale':''}
      </Typography>
    </Box>;
  });
  return {toggle,enabled,resetView,canReset,settings,overlay:enabled?overlay:null,toolbar:enabled?toolbar:null,comparison:enabled?comparison:null};
}
