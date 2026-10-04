import { CompareBatchWebGL } from "./batchCanvasWebGL";
import { residentLogTable, residentRawValues, residentDivisor, residentScalarType, type ResidentValues, type ResidentScalarType } from "./batchValues";
/** One upload, batched GPU display reductions, and one instanced grid draw.
 * Scientific values remain full resolution and unmodified in the source buffer.
 */
const shader = /* wgsl */ `
struct Settings { dims: vec4u, display: vec4f, view: vec4f, canvas: vec4f }
@group(0) @binding(0) var<storage, read> values: array<u32>;
@group(0) @binding(1) var<storage, read_write> partial: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> ranges: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> histogram: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> s: Settings;
fn read_value(p:u32)->f32 {
 if(s.canvas.w>0.0 && s.canvas.z==0.0){return 0.0;}
 if(s.canvas.w==2.0){return f32(values[p])/s.canvas.z;}
 if(s.canvas.w==1.0){let word=values[p/2u];let count=(word>>((p%2u)*16u))&65535u;return bitcast<f32>(values[s.dims.x*s.dims.y*s.dims.z/2u+count]);}
 return bitcast<f32>(values[p]);
}
var<workgroup> low: array<f32, 256>;
var<workgroup> high: array<f32, 256>;
fn raw_count(p:u32)->f32 { let word=values[p/2u];return f32((word>>((p%2u)*16u))&65535u); }
fn normalize_count(v:f32)->f32 { if(s.canvas.z==0.0){return 0.0;}return bitcast<f32>(values[s.dims.x*s.dims.y*s.dims.z/2u+u32(v)+select(0u,65536u,s.display.z>0.0)]); }
fn scaled(v:f32)->f32 { return select(v, log(1.0+max(v,0.0)), s.display.z>0.0); }
@compute @workgroup_size(256) fn partial_range(@builtin(workgroup_id) g:vec3u,@builtin(local_invocation_index) t:u32) {
 let size=s.dims.x*s.dims.y;
 var lo=3.402823e38; var hi=-3.402823e38;
 for(var p=g.x*256u+t;p<size;p+=s.dims.w*256u) { var v:f32;if(s.canvas.w==1.0){v=raw_count(g.y*size+p);}else{v=scaled(read_value(g.y*size+p));}lo=min(lo,v);hi=max(hi,v); }
 low[t]=lo; high[t]=hi; workgroupBarrier();
 for(var stride=128u;stride>0u;stride/=2u) { if(t<stride){low[t]=min(low[t],low[t+stride]);high[t]=max(high[t],high[t+stride]);}workgroupBarrier(); }
 if(t==0u){partial[g.y*s.dims.w+g.x]=vec2f(low[0],high[0]);}
}
@compute @workgroup_size(256) fn final_range(@builtin(workgroup_id) g:vec3u,@builtin(local_invocation_index) t:u32) {
 low[t]=3.402823e38;high[t]=-3.402823e38;
 if(t<s.dims.w){let v=partial[g.x*s.dims.w+t];low[t]=v.x;high[t]=v.y;}workgroupBarrier();
 for(var stride=128u;stride>0u;stride/=2u){if(t<stride){low[t]=min(low[t],low[t+stride]);high[t]=max(high[t],high[t+stride]);}workgroupBarrier();}
 if(t==0u){var lo=low[0];var hi=high[0];if(s.canvas.w==1.0){lo=normalize_count(lo);hi=normalize_count(hi);}let delta=hi-lo;ranges[g.x]=vec4f(lo,hi,lo+delta*s.display.x/100.0,lo+delta*s.display.y/100.0);}
}
@compute @workgroup_size(256) fn make_histogram(@builtin(workgroup_id) g:vec3u,@builtin(local_invocation_index) t:u32) {
 let size=s.dims.x*s.dims.y;let r=ranges[g.y];
 for(var p=g.x*256u+t;p<size;p+=s.dims.w*256u){var v:f32;if(s.canvas.w==1.0){v=normalize_count(raw_count(g.y*size+p));}else{v=scaled(read_value(g.y*size+p));}let bin=u32(clamp(floor((v-r.x)*1023.0/max(r.y-r.x,1e-30)),0.0,1023.0));atomicAdd(&histogram[g.y*1024u+bin],1u);}
}
@compute @workgroup_size(1) fn percentile(@builtin(workgroup_id) g:vec3u) {
 var r=ranges[g.x];let size=f32(s.dims.x*s.dims.y);var sum=0u;var found=false;
 for(var b=0u;b<1024u;b++){let n=atomicLoad(&histogram[g.x*1024u+b]);let prev=sum;sum+=n;
 if(!found && f32(sum)>=size*0.01){r.z=r.x+(f32(b)+(size*0.01-f32(prev))/f32(max(n,1u)))/1024.0*(r.y-r.x);found=true;}
 if(f32(sum)>=size*0.99){r.w=r.x+(f32(b)+(size*0.99-f32(prev))/f32(max(n,1u)))/1024.0*(r.y-r.x);break;}}
 ranges[g.x]=r;
}
`;
const drawing = /* wgsl */ `
struct Settings { dims: vec4u, display: vec4f, view: vec4f, canvas: vec4f }
struct Tile { rect: vec4f, source: vec4u }
@group(0) @binding(0) var<storage,read> values: array<u32>;
@group(0) @binding(1) var<storage,read> ranges: array<vec4f>;
@group(0) @binding(2) var<storage,read> tiles: array<Tile>;
@group(0) @binding(3) var<storage,read> lut: array<vec4f>;
@group(0) @binding(4) var<uniform> s: Settings;
fn read_value(p:u32)->f32 {
 if(s.canvas.w>0.0 && s.canvas.z==0.0){return 0.0;}
 if(s.canvas.w==2.0){return f32(values[p])/s.canvas.z;}
 if(s.canvas.w==1.0){let word=values[p/2u];let count=(word>>((p%2u)*16u))&65535u;return bitcast<f32>(values[s.dims.x*s.dims.y*s.dims.z/2u+count]);}
 return bitcast<f32>(values[p]);
}
struct Vertex { @builtin(position) position:vec4f,@location(0) uv:vec2f,@location(1) @interpolate(flat) source:u32 }
@vertex fn vertex(@builtin(vertex_index) v:u32,@builtin(instance_index) instance:u32)->Vertex {
 let corners=array<vec2f,6>(vec2f(0,0),vec2f(1,0),vec2f(0,1),vec2f(0,1),vec2f(1,0),vec2f(1,1));
 let uv=corners[v];let tile=tiles[instance];let pos=tile.rect.xy+uv*tile.rect.zw;
 var out:Vertex;out.position=vec4f(pos.x*2.0/s.canvas.x-1.0,1.0-pos.y*2.0/s.canvas.y,0,1);out.uv=uv;out.source=tile.source.x;return out;
}
fn color(source:u32,p:vec2i)->vec4f {
 let point=clamp(p,vec2i(0),vec2i(i32(s.dims.y)-1,i32(s.dims.x)-1));
 let at=source*s.dims.x*s.dims.y+u32(point.y)*s.dims.y+u32(point.x);
 let v=read_value(at);var z=v;
 if(s.canvas.w==1.0 && s.display.z>0.0 && s.canvas.z>0.0){let count=(values[at/2u]>>((at%2u)*16u))&65535u;z=bitcast<f32>(values[s.dims.x*s.dims.y*s.dims.z/2u+65536u+count]);}
 else if(s.canvas.w!=1.0){z=select(v,log(1.0+max(v,0.0)),s.display.z>0.0);}
 let r=ranges[source];let value=clamp((z-r.z)/max(r.w-r.z,1e-30),0.0,1.0);return lut[u32(select(0.5,value,r.w>r.z)*255.0)];
}
@fragment fn fragment(in:Vertex)->@location(0) vec4f {
 let p=(in.uv*vec2f(f32(s.dims.y),f32(s.dims.x))-s.view.yz)/s.view.x;
 if(any(p<vec2f(0))||p.x>=f32(s.dims.y)||p.y>=f32(s.dims.x)){return vec4f(0,0,0,1);}
 if(s.view.w<0.5){return color(in.source,vec2i(floor(p)));}
 let q=p-0.5;let base=vec2i(floor(q));let frac=fract(q);
 return mix(mix(color(in.source,base),color(in.source,base+vec2i(1,0)),frac.x),mix(color(in.source,base+vec2i(0,1)),color(in.source,base+vec2i(1,1)),frac.x),frac.y);
}
`;

export class CompareBatchCanvas {
  private buffers: GPUBuffer[] = [];
  private input!: GPUBuffer;
  private partial!: GPUBuffer;
  private ranges!: GPUBuffer;
  private histogram!: GPUBuffer;
  private tiles!: GPUBuffer;
  private lut!: GPUBuffer;
  private settings!: GPUBuffer;
  private compute!: GPUBindGroup;
  private drawGroup!: GPUBindGroup;
  private pipelines!: GPUComputePipeline[];
  private pipeline!: GPURenderPipeline;
  private context!: GPUCanvasContext;
  private previous: DataView | null = null;
  private sourceValues: ResidentValues | null = null;

  /** Borrow exact source values for scientific readout or binary export. */
  readSourceValues(): ResidentValues | null { return this.sourceValues; }
  private rangeKey = "";
  private logarithmsReady = false;
  readonly adapter: string;
  private constructor(
    private device: GPUDevice,
    readonly canvas: HTMLCanvasElement,
    private count: number,
    private rows: number,
    private cols: number,
    adapter: GPUAdapter,
    private scalarType: ResidentScalarType,
  ) {
    this.adapter = [
      adapter.info.vendor,
      adapter.info.architecture,
      adapter.info.description,
    ].join(" ");
  }
  static async create(
    canvas: HTMLCanvasElement,
    count: number,
    rows: number,
    cols: number,
    scalarType: ResidentScalarType = "<f4",
  ) {
    residentScalarType({dtype:scalarType});
    const adapter = await navigator.gpu?.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) return new CompareBatchWebGL(canvas, count, rows, cols, scalarType);
    const bytes = count * rows * cols * (scalarType === "<u2" ? 2 : 4) +
      (scalarType === "<u2" ? 2 * 65536 * 4 : 0);
    if (bytes > adapter.limits.maxStorageBufferBindingSize)
      throw new Error("Full batch exceeds WebGPU storage binding limit");
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: Math.max(128 * 1024 * 1024, bytes),
        maxBufferSize: Math.max(256 * 1024 * 1024, bytes),
      },
    });
    const result = new CompareBatchCanvas(
      device,
      canvas,
      count,
      rows,
      cols,
      adapter,
      scalarType,
    );
    await result.initialize();
    return result;
  }
  private async initialize() {
    const d = this.device;
    const buffer = (size: number, usage: number) => {
      const b = d.createBuffer({ size, usage });
      this.buffers.push(b);
      return b;
    };
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const inputBytes = this.count * this.rows * this.cols * (this.scalarType === "<u2" ? 2 : 4) +
      (this.scalarType === "<u2" ? 2 * 65536 * 4 : 0);
    this.input = buffer(Math.ceil(inputBytes / 4) * 4, storage);
    this.partial = buffer(this.count * 64 * 8, storage);
    this.ranges = buffer(this.count * 16, storage);
    this.histogram = buffer(this.count * 1024 * 4, storage);
    this.tiles = buffer(this.count * 32, storage);
    this.lut = buffer(256 * 16, storage);
    this.settings = buffer(
      64,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    );
    const layout = d.createBindGroupLayout({
      entries: [0, 1, 2, 3, 4].map((binding) => ({
        binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type:
            binding === 0
              ? "read-only-storage"
              : binding === 4
                ? "uniform"
                : "storage",
        },
      })),
    });
    const module = d.createShaderModule({ code: shader });
    this.pipelines = await Promise.all(
      ["partial_range", "final_range", "make_histogram", "percentile"].map(
        (entryPoint) =>
          d.createComputePipelineAsync({
            layout: d.createPipelineLayout({ bindGroupLayouts: [layout] }),
            compute: { module, entryPoint },
          }),
      ),
    );
    this.compute = d.createBindGroup({
      layout,
      entries: [
        this.input,
        this.partial,
        this.ranges,
        this.histogram,
        this.settings,
      ].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const draw = d.createShaderModule({ code: drawing });
    this.pipeline = await d.createRenderPipelineAsync({
      layout: "auto",
      vertex: { module: draw, entryPoint: "vertex" },
      fragment: {
        module: draw,
        entryPoint: "fragment",
        targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
      },
      primitive: { topology: "triangle-list" },
    });
    this.drawGroup = d.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        this.input,
        this.ranges,
        this.tiles,
        this.lut,
        this.settings,
      ].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    this.context = this.canvas.getContext("webgpu")!;
    this.context.configure({
      device: d,
      format: navigator.gpu.getPreferredCanvasFormat(),
      alphaMode: "premultiplied",
    });
  }
  async render(
    bytes: DataView,
    rectangles: Float32Array,
    sourceIndices: Uint32Array,
    lut: Uint8Array | number[],
    options: {
      log: boolean;
      auto: boolean;
      min: number;
      max: number;
      zoom: number;
      panX: number;
      panY: number;
      smooth: boolean;
      dtype?: ResidentScalarType;
      divisor?: number;
      normalizationOffset?: number;
    },
  ) {
    const started = performance.now();
    const d = this.device;
    if ((options.dtype ?? "<f4") !== this.scalarType) throw new Error("Batch scalar type changed without replacing its renderer.");
    const raw = residentRawValues(bytes, {dtype:this.scalarType,shape:[this.count,this.rows,this.cols],normalization_offset:options.normalizationOffset});
    residentDivisor({dtype:this.scalarType,mask_area:options.divisor});
    if (this.scalarType === "<u2" && options.normalizationOffset === undefined) throw new Error("GPU uint16 batches require a normalization table; use CPU display fallback.");
    this.sourceValues = raw;
    const fresh = this.previous !== bytes;
    if (fresh) {
      d.queue.writeBuffer(
        this.input,
        0,
        bytes.buffer,
        bytes.byteOffset,
        bytes.byteLength,
      );
      this.previous = bytes;
    }
    if (!options.log) this.logarithmsReady=false;
    if (this.scalarType === "<u2" && options.log && (fresh || !this.logarithmsReady)) {
      if (options.normalizationOffset === undefined) throw new Error("Exact logarithms require the normalization table.");
      d.queue.writeBuffer(this.input,options.normalizationOffset+65536*4,residentLogTable(bytes,options.normalizationOffset));
      this.logarithmsReady=true;
    }
    const tiles = new ArrayBuffer(sourceIndices.length * 32);
    const positions = new Float32Array(tiles);
    const indices = new Uint32Array(tiles);
    // Layout metadata only. No acquisition images are copied or colored here.
    sourceIndices.forEach((source, i) => {
      positions.set(rectangles.subarray(i * 4, i * 4 + 4), i * 8);
      indices[i * 8 + 4] = source;
    });
    d.queue.writeBuffer(this.tiles, 0, tiles);
    const colors = new Float32Array(256 * 4);
    for (let i = 0; i < 256; i++) {
      colors[i * 4] = lut[i * 3] / 255;
      colors[i * 4 + 1] = lut[i * 3 + 1] / 255;
      colors[i * 4 + 2] = lut[i * 3 + 2] / 255;
      colors[i * 4 + 3] = 1;
    }
    d.queue.writeBuffer(this.lut, 0, colors);
    const settings = new ArrayBuffer(64);
    new Uint32Array(settings).set([this.rows, this.cols, this.count, 64]);
    new Float32Array(settings).set(
      [
        options.min,
        options.max,
        +options.log,
        +options.auto,
        options.zoom,
        options.panX,
        options.panY,
        +options.smooth,
        this.canvas.width,
        this.canvas.height,
        options.divisor ?? 1,
        this.scalarType === "<u2" ? 1 : this.scalarType === "<u4" ? 2 : 0,
      ],
      4,
    );
    d.queue.writeBuffer(this.settings, 0, settings);
    const encoder = d.createCommandEncoder();
    const rangeKey = JSON.stringify([
      options.log,
      options.divisor,
      options.auto,
      options.min,
      options.max,
    ]);
    if (fresh || this.rangeKey !== rangeKey) {
      this.rangeKey = rangeKey;
      encoder.clearBuffer(this.histogram);
      const compute = encoder.beginComputePass();
      compute.setBindGroup(0, this.compute);
      compute.setPipeline(this.pipelines[0]);
      compute.dispatchWorkgroups(64, this.count);
      compute.setPipeline(this.pipelines[1]);
      compute.dispatchWorkgroups(this.count);
      if (options.auto) {
        compute.setPipeline(this.pipelines[2]);
        compute.dispatchWorkgroups(64, this.count);
        compute.setPipeline(this.pipelines[3]);
        compute.dispatchWorkgroups(this.count);
      }
      compute.end();
    }
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.drawGroup);
    pass.draw(6, sourceIndices.length);
    pass.end();
    d.queue.submit([encoder.finish()]);
    const submitted = performance.now();
    await d.queue.onSubmittedWorkDone();
    return {
      fresh,
      panels: sourceIndices.length,
      upload_submit_ms: submitted - started,
      completed_ms: performance.now() - started,
      adapter: this.adapter,
    };
  }
  destroy() {
    this.sourceValues = null;
    this.previous = null;
    this.buffers.forEach((buffer) => buffer.destroy());
    this.context?.unconfigure();
    this.device.destroy();
  }
}
