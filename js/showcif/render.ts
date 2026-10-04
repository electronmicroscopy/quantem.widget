/** Instanced WebGPU atom spheres. Geometry and filtering remain on the GPU. */
import { type V3 } from "./geometry";
const shader = `
struct Params {right:vec4f, up:vec4f, beam:vec4f, center:vec4f, clip:vec4f, depth:vec4f, misc:vec4f, a:vec4f, b:vec4f, c:vec4f, repeats:vec4u};
struct Atom {p:vec4f}; struct Species {color:vec4f};
@group(0) @binding(0) var<storage,read> atoms:array<Atom>;
@group(0) @binding(1) var<storage,read> species:array<Species>;
@group(0) @binding(2) var<uniform> u:Params;
struct Out {@builtin(position) p:vec4f,@location(0) disk:vec2f,@location(1) color:vec3f,@location(2) valid:f32};
@vertex fn vs(@builtin(vertex_index) v:u32,@builtin(instance_index) i:u32)->Out{
 let corners=array<vec2f,6>(vec2f(-1,-1),vec2f(1,-1),vec2f(-1,1),vec2f(-1,1),vec2f(1,-1),vec2f(1,1));
 let n=u.repeats.w;let cellIndex=i/n;
 let iz=cellIndex%u.repeats.z;let iy=(cellIndex/u.repeats.z)%u.repeats.y;let ix=cellIndex/(u.repeats.y*u.repeats.z);
 let base=atoms[i%n].p;let atom=vec4f(base.xyz+f32(ix)*u.a.xyz+f32(iy)*u.b.xyz+f32(iz)*u.c.xyz,base.w);
 let p=atom.xyz-u.center.xyz;let c=species[u32(atom.w)].color;
 let z=dot(atom.xyz,u.clip.xyz);let valid=select(0.,1.,c.w>.5 && z>=u.clip.w-u.depth.z && (z<u.depth.x-u.depth.z || (u.depth.y>.5 && z<=u.depth.x+u.depth.z)));
 let d=corners[v];let offset=d*u.misc.xy;var o:Out;
 o.p=vec4f(dot(p,u.right.xyz)*u.right.w+offset.x,dot(p,u.up.xyz)*u.up.w+offset.y,clamp(.5-dot(p,u.beam.xyz)*u.beam.w,.01,.99),1.);
 o.disk=d;o.color=c.xyz;o.valid=valid;return o;
}
@fragment fn fs(i:Out)->@location(0) vec4f{
 let rr=dot(i.disk,i.disk);if(rr>1. || i.valid<.5){discard;}
 let nz=sqrt(max(0.,1.-rr));let light=.35+.65*max(0.,dot(vec3f(i.disk,nz),normalize(vec3f(-.4,.5,1.))));
 return vec4f(i.color*light,1.);
}`;
export class AtomRenderer {
  private pipeline: GPURenderPipeline;
  private bind: GPUBindGroup;
  private uniform: GPUBuffer;
  private colors: GPUBuffer;
  private atomBuffer: GPUBuffer;
  private depth?: GPUTexture;
  private context: GPUCanvasContext;
  private size = "";
  constructor(
    private canvas: HTMLCanvasElement,
    private device: GPUDevice,
    atoms: Float32Array,
    species: number[][],
  ) {
    this.context = canvas.getContext("webgpu")!;
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device, format, alphaMode: "opaque" });
    const module = device.createShaderModule({ code: shader });
    this.pipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: { module, entryPoint: "fs", targets: [{ format }] },
      primitive: { topology: "triangle-list" },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less-equal",
      },
    });
    const buf = (bytes: number, usage: number) =>
      device.createBuffer({ size: Math.max(16, bytes), usage });
    this.atomBuffer = buf(
      atoms.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    );
    device.queue.writeBuffer(
      this.atomBuffer,
      0,
      atoms as Float32Array<ArrayBuffer>,
    );
    this.colors = buf(
      species.length * 16,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    );
    this.uniform = buf(176, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.bind = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.atomBuffer } },
        { binding: 1, resource: { buffer: this.colors } },
        { binding: 2, resource: { buffer: this.uniform } },
      ],
    });
  }
  draw(
    count: number,
    basis: { right: V3; up: V3; beam: V3 },
    center: V3,
    span: number,
    clipBeam: V3,
    limits: number[],
    colors: number[][],
    visible: boolean[],
    radius: number,
    unit: number[][],
    repeats: number[],
    atomsPerCell: number,
    includeFar: boolean,
    boundaryEpsilon: number,
  ) {
    const box = this.canvas.getBoundingClientRect(),
      dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(box.width * dpr)),
      h = Math.max(1, Math.round(box.height * dpr));
    if (this.size !== `${w},${h}`) {
      this.canvas.width = w;
      this.canvas.height = h;
      this.depth?.destroy();
      this.depth = this.device.createTexture({
        size: [w, h],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.size = `${w},${h}`;
    }
    const c = new Float32Array(
      colors.flatMap((rgb, i) => [...rgb, visible[i] ? 1 : 0]),
    );
    this.device.queue.writeBuffer(this.colors, 0, c);
    const u = new Float32Array([
      ...basis.right,
      2 / span,
      ...basis.up,
      ((2 / span) * w) / h,
      ...basis.beam,
      0.45 / Math.max(span, limits[1] - limits[0]),
      ...center,
      0,
      ...clipBeam,
      limits[0],
      limits[1],
      includeFar ? 1 : 0,
      boundaryEpsilon,
      0,
      (2 * radius) / box.width,
      (2 * radius) / box.height,
      0,
      0,
    ]);
    this.device.queue.writeBuffer(this.uniform, 0, u);
    this.device.queue.writeBuffer(
      this.uniform,
      112,
      new Float32Array(unit.flatMap((v) => [...v, 0])),
    );
    this.device.queue.writeBuffer(
      this.uniform,
      160,
      new Uint32Array([...repeats, atomsPerCell]),
    );
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0.018, g: 0.024, b: 0.04, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: this.depth!.createView(),
        depthClearValue: 1,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bind);
    pass.draw(6, count);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }
  destroy() {
    this.depth?.destroy();
    this.atomBuffer.destroy();
    this.colors.destroy();
    this.uniform.destroy();
    this.context.unconfigure();
  }
}
