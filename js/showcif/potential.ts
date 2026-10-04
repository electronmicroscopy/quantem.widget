/** WebGPU accumulation of abTEM radial atomic projections, in V Å. */
import { GPUColormapEngine, COLORMAPS } from "../colormaps";
import { type V3 } from "./geometry";
export type PotentialGeometry = {
  right: V3;
  up: V3;
  beam: V3;
  center: V3;
  span: number;
  zmin: number;
  zmax: number;
  limits: number[];
  slices: number;
  unit: number[][];
  repeats: number[];
  visible: boolean[];
};
const computeShader = `
struct Params {right:vec4f,up:vec4f,beam:vec4f,center:vec4f,a:vec4f,b:vec4f,c:vec4f,repeat:vec4u,grid:vec4u,depth:vec4f};
@group(0) @binding(0) var<storage,read> atoms:array<vec4f>;
@group(0) @binding(1) var<storage,read> table:array<f32>;
@group(0) @binding(2) var<storage,read> visible:array<u32>;
@group(0) @binding(3) var<storage,read_write> volume:array<f32>;
@group(0) @binding(4) var<uniform> u:Params;
fn lookup(r:f32,s:u32)->f32{
 if(r>=8.){return 0.;}
 let v=r/0.005;let j=min(1599u,u32(v));
 return mix(table[s*1601u+j],table[s*1601u+j+1u],v-f32(j));
}
@compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) id:vec3u){
 let n=u.grid.x;if(id.x>=n || id.y>=n){return;}
 let pix=id.y*n+id.x;let plane=n*n;let ns=u.grid.y;
 let xy=vec2f((f32(id.x)+.5)/f32(n)-.5,.5-(f32(id.y)+.5)/f32(n))*u.right.w;
 let dx=u.right.w/f32(n);var values:array<f32,64>;
 for(var s=0u;s<ns;s++){values[s]=0.;}
 var total=0.;var selected=0.;
 for(var i=0u;i<u.grid.z;i++){
  let base=atoms[i%u.repeat.w];let species=u32(base.w);
  if(visible[species]==0u){continue;}
  let k=i/u.repeat.w;
  let p=base.xyz+f32(k/(u.repeat.y*u.repeat.z))*u.a.xyz+f32((k/u.repeat.z)%u.repeat.y)*u.b.xyz+f32(k%u.repeat.z)*u.c.xyz;
  let z=dot(p,u.beam.xyz);
  if(z<u.depth.x || z>u.depth.y){continue;}
  let q=p-u.center.xyz;let delta=xy-vec2f(dot(q,u.right.xyz),dot(q,u.up.xyz));
  if(length(delta)>8.+dx){continue;}
  // Explicit 2x2 midpoint quadrature of each preview pixel.
  var value=0.;
  for(var y=0u;y<2u;y++){for(var x=0u;x<2u;x++){
    let offset=(vec2f(f32(x),f32(y))-.5)*.5*dx;
    value+=lookup(length(delta+offset),species)*.25;
  }}
  var bin=(z-u.depth.x)/(u.depth.y-u.depth.x)*f32(ns);
  // Snap only floating-point boundary roundoff (1e-5 of a slab), not physical coordinates.
  if(abs(bin-round(bin))<1e-5){bin=round(bin);}
  let s=min(ns-1u,u32(floor(max(0.,bin))));
  values[s]+=value;total+=value;
  if(z>=u.depth.z && (z<u.depth.w || (u.depth.w>=u.depth.y && z<=u.depth.w))){selected+=value;}
 }
 for(var s=0u;s<ns;s++){volume[s*plane+pix]=values[s];}
 volume[ns*plane+pix]=total;volume[(ns+1u)*plane+pix]=select(selected,values[min(ns-1u,max(1u,u.grid.w)-1u)],u.grid.w>0u);
}`;
const drawShader = `
struct View {image:vec4u,scale:vec4f};
@group(0) @binding(0) var<storage,read> volume:array<f32>;
@group(0) @binding(1) var<uniform> u:View;
@group(0) @binding(2) var<storage,read> colors:array<vec4f>;
struct Out{@builtin(position) p:vec4f,@location(0) uv:vec2f};
@vertex fn vs(@builtin(vertex_index) i:u32)->Out{
 let p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));var o:Out;o.p=vec4f(p[i],0,1);o.uv=vec2f(p[i].x*.5+.5,.5-p[i].y*.5);return o;
}
@fragment fn fs(v:Out)->@location(0) vec4f{
 let n=u.image.x;let xy=min(vec2u(v.uv*f32(n)),vec2u(n-1u));
 let raw=volume[u.image.y*n*n+xy.y*n+xy.x]*u.scale.x;
 let x=clamp(raw/u.scale.y,0.,1.);
 return colors[u32(round(x*255.))];
}`;
const blurShader = `
struct Filter {n:u32, planes:u32, axis:u32, radius:u32, sigma:f32, a:f32, b:f32, c:f32};
@group(0) @binding(0) var<storage,read> source:array<f32>;
@group(0) @binding(1) var<storage,read_write> filteredValues:array<f32>;
@group(0) @binding(2) var<uniform> u:Filter;
@compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=u.n || id.y>=u.n || id.z>=u.planes){return;}
 let base=id.z*u.n*u.n;var sum=0.;var weight=0.;
 for(var k=-i32(u.radius);k<=i32(u.radius);k++){
  let x=i32(id.x)+select(0,k,u.axis==0u);let y=i32(id.y)+select(0,k,u.axis==1u);
  let w=exp(-.5*f32(k*k)/(u.sigma*u.sigma));weight+=w;
  // Zero extension of the finite inspection patch, without periodic wrapping.
  if(x>=0 && x<i32(u.n) && y>=0 && y<i32(u.n)){sum+=w*source[base+u32(y)*u.n+u32(x)];}
 }
 filteredValues[base+id.y*u.n+id.x]=sum/weight;
}`;
export class PotentialGPU {
  readonly volume: GPUBuffer;
  private averageEngine: GPUColormapEngine | undefined;
  private averageSources: GPUBuffer[] = [];
  private averageTarget: GPUBuffer | undefined;
  private averageKey = "";
  private filtered: GPUBuffer;
  private intermediate: GPUBuffer;
  private filterPipeline: GPUComputePipeline;
  private filterUniforms: GPUBuffer[];
  private filterBindings: GPUBindGroup[];
  private atoms: GPUBuffer;
  private colorLut: GPUBuffer;
  private currentColormap = "";
  private table: GPUBuffer;
  private visible: GPUBuffer;
  private uniform: GPUBuffer;
  private pipeline: GPUComputePipeline;
  private bind: GPUBindGroup;
  private renderPipeline: GPURenderPipeline;
  private views = new Map<
    HTMLCanvasElement,
    { context: GPUCanvasContext; uniform: GPUBuffer; bind: GPUBindGroup }
  >();
  constructor(
    readonly device: GPUDevice,
    atoms: Float32Array,
    table: Float32Array,
    readonly pixels: number,
  ) {
    const buffer = (data: Float32Array | number, usage: number) => {
      const b = device.createBuffer({
        size: typeof data === "number" ? data : data.byteLength,
        usage,
      });
      if (typeof data !== "number")
        device.queue.writeBuffer(b, 0, data as Float32Array<ArrayBuffer>);
      return b;
    };
    this.colorLut = buffer(
      256 * 16,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    );
    this.atoms = buffer(
      atoms,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    );
    this.table = buffer(
      table,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    );
    this.visible = buffer(
      Math.max(4, (table.length / 1601) * 4),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    );
    this.volume = buffer(
      pixels * pixels * 66 * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    );
    this.filtered = buffer(
      pixels * pixels * 66 * 4,
      GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    );
    this.intermediate = buffer(
      pixels * pixels * 66 * 4,
      GPUBufferUsage.STORAGE,
    );
    this.filterPipeline = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: blurShader }),
        entryPoint: "main",
      },
    });
    this.filterUniforms = [0, 1].map(() =>
      buffer(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
    );
    this.filterBindings = [0, 1].map((axis) =>
      device.createBindGroup({
        layout: this.filterPipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: { buffer: axis === 0 ? this.volume : this.intermediate },
          },
          {
            binding: 1,
            resource: {
              buffer: axis === 0 ? this.intermediate : this.filtered,
            },
          },
          { binding: 2, resource: { buffer: this.filterUniforms[axis] } },
        ],
      }),
    );
    this.uniform = buffer(
      160,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    );
    const module = device.createShaderModule({ code: computeShader });
    this.pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    this.bind = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        this.atoms,
        this.table,
        this.visible,
        this.volume,
        this.uniform,
      ].map((b, i) => ({ binding: i, resource: { buffer: b } })),
    });
    const draw = device.createShaderModule({ code: drawShader });
    this.renderPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: draw, entryPoint: "vs" },
      fragment: {
        module: draw,
        entryPoint: "fs",
        targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
      },
      primitive: { topology: "triangle-list" },
    });
  }
  compute(g: PotentialGeometry, atomsPerCell: number) {
    if (g.repeats.reduce((a, b) => a * b, atomsPerCell) > 8192)
      throw Error(
        "Potential preview supports 8,192 atoms. Reduce Unit Cells; the atom viewer supports larger structures.",
      );
    const u = new ArrayBuffer(160);
    const f = new Float32Array(u),
      i = new Uint32Array(u);
    f.set([
      ...g.right,
      g.span,
      ...g.up,
      0,
      ...g.beam,
      0,
      ...g.center,
      0,
      ...g.unit.flatMap((v) => [...v, 0]),
    ]);
    i.set([...g.repeats, atomsPerCell], 28);
    const start = ((g.limits[0] - g.zmin) / (g.zmax - g.zmin)) * g.slices;
    const end = ((g.limits[1] - g.zmin) / (g.zmax - g.zmin)) * g.slices;
    const exactSlice =
      Math.abs(start - Math.round(start)) < 1e-7 &&
      Math.abs(end - start - 1) < 1e-7;
    i.set(
      [
        this.pixels,
        g.slices,
        g.repeats.reduce((a, b) => a * b, atomsPerCell),
        exactSlice ? Math.round(start) + 1 : 0,
      ],
      32,
    );
    f.set([g.zmin, g.zmax, ...g.limits], 36);
    this.device.queue.writeBuffer(this.uniform, 0, u);
    this.device.queue.writeBuffer(
      this.visible,
      0,
      new Uint32Array(g.visible.map(Number)),
    );
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bind);
    pass.dispatchWorkgroups(
      Math.ceil(this.pixels / 8),
      Math.ceil(this.pixels / 8),
    );
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
  /** Separable Gaussian display filter; the physical volume remains immutable. */
  filter(sigmaPixels: number, planes: number) {
    this.averageKey = "";
    const enc = this.device.createCommandEncoder();
    if (sigmaPixels <= 0) {
      enc.copyBufferToBuffer(
        this.volume,
        0,
        this.filtered,
        0,
        planes * this.pixels * this.pixels * 4,
      );
    } else {
      const sigma = Math.min(32, sigmaPixels);
      for (let axis = 0; axis < 2; axis++) {
        const data = new ArrayBuffer(32);
        new Uint32Array(data).set([
          this.pixels,
          planes,
          axis,
          Math.ceil(3 * sigma),
        ]);
        new Float32Array(data)[4] = sigma;
        this.device.queue.writeBuffer(this.filterUniforms[axis], 0, data);
        const pass = enc.beginComputePass();
        pass.setPipeline(this.filterPipeline);
        pass.setBindGroup(0, this.filterBindings[axis]);
        pass.dispatchWorkgroups(
          Math.ceil(this.pixels / 8),
          Math.ceil(this.pixels / 8),
          planes,
        );
        pass.end();
      }
    }
    this.device.queue.submit([enc.finish()]);
  }
  /** Arithmetic mean via Show3D's resident GPU engine. No image readback. */
  average(indices: number[], slices: number) {
    if (
      !indices.length ||
      indices.some((i) => !Number.isInteger(i) || i < 0 || i >= slices)
    )
      throw Error("Select valid slice indices before averaging.");
    const key = `${slices}:${indices.join(",")}`;
    if (key === this.averageKey) return;
    const bytes = this.pixels * this.pixels * 4;
    const destination = (slices + 1) * bytes;
    this.averageEngine ??= new GPUColormapEngine(this.device);
    if (!this.averageTarget) {
      this.averageTarget = this.device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      this.averageEngine.borrowBuffer(
        0,
        this.averageTarget,
        this.pixels,
        this.pixels,
      );
    }
    while (this.averageSources.length < indices.length) {
      const source = this.device.createBuffer({
        size: bytes,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      });
      this.averageSources.push(source);
      this.averageEngine.borrowBuffer(
        this.averageSources.length,
        source,
        this.pixels,
        this.pixels,
      );
    }
    const copies = this.device.createCommandEncoder();
    indices.forEach((index, i) =>
      copies.copyBufferToBuffer(
        this.filtered,
        index * bytes,
        this.averageSources[i],
        0,
        bytes,
      ),
    );
    this.device.queue.submit([copies.finish()]);
    if (
      !this.averageEngine.averageResidentSlotsInto(
        0,
        indices.map((_, i) => i + 1),
      )
    )
      throw Error(
        "The resident slice average could not be computed. Reload the viewer.",
      );
    const output = this.device.createCommandEncoder();
    output.copyBufferToBuffer(
      this.averageTarget,
      0,
      this.filtered,
      destination,
      bytes,
    );
    this.device.queue.submit([output.finish()]);
    this.averageKey = key;
  }
  draw(
    canvas: HTMLCanvasElement,
    slot: number,
    multiplier: number,
    max: number,
    colormap: string,
  ) {
    if (this.currentColormap !== colormap) {
      const lut = COLORMAPS[colormap];
      if (!lut) throw Error(`Unknown colormap ${colormap}`);
      const rgba = new Float32Array(256 * 4);
      for (let i = 0; i < 256; i++)
        rgba.set(
          [lut[i * 3] / 255, lut[i * 3 + 1] / 255, lut[i * 3 + 2] / 255, 1],
          i * 4,
        );
      this.device.queue.writeBuffer(this.colorLut, 0, rgba);
      this.currentColormap = colormap;
    }
    let view = this.views.get(canvas);
    if (!view) {
      const context = canvas.getContext("webgpu")!;
      context.configure({
        device: this.device,
        format: navigator.gpu.getPreferredCanvasFormat(),
        alphaMode: "opaque",
      });
      const uniform = this.device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const bind = this.device.createBindGroup({
        layout: this.renderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.filtered } },
          { binding: 1, resource: { buffer: uniform } },
          { binding: 2, resource: { buffer: this.colorLut } },
        ],
      });
      view = { context, uniform, bind };
      this.views.set(canvas, view);
    }
    canvas.width = this.pixels;
    canvas.height = this.pixels;
    const u = new ArrayBuffer(32);
    new Uint32Array(u).set([this.pixels, slot, 0, 0]);
    new Float32Array(u).set([multiplier, max, 0, 0], 4);
    this.device.queue.writeBuffer(view.uniform, 0, u);
    const enc = this.device.createCommandEncoder();
    const p = enc.beginRenderPass({
      colorAttachments: [
        {
          view: view.context.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    p.setPipeline(this.renderPipeline);
    p.setBindGroup(0, view.bind);
    p.draw(3);
    p.end();
    this.device.queue.submit([enc.finish()]);
  }
  /** Explicit numerical readback for exports/tests, never the interactive render path. */
  async readback(slices: number, filtered = false): Promise<Float32Array> {
    const bytes = (slices + 2) * this.pixels * this.pixels * 4;
    const b = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const e = this.device.createCommandEncoder();
    e.copyBufferToBuffer(
      filtered ? this.filtered : this.volume,
      0,
      b,
      0,
      bytes,
    );
    this.device.queue.submit([e.finish()]);
    await b.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(b.getMappedRange().slice(0));
    b.unmap();
    b.destroy();
    return result;
  }
  clearViews() {
    for (const v of this.views.values()) {
      v.context.unconfigure();
      v.uniform.destroy();
    }
    this.views.clear();
  }
  destroy() {
    this.clearViews();
    this.averageEngine?.destroy();
    this.averageTarget?.destroy();
    this.averageSources.forEach((b) => b.destroy());
    for (const b of [
      this.colorLut,
      this.filtered,
      this.intermediate,
      ...this.filterUniforms,
      this.atoms,
      this.table,
      this.visible,
      this.volume,
      this.uniform,
    ])
      b.destroy();
  }
}
