import { describe, expect, it } from "vitest";
import { CompareBatchWebGL } from "./batchCanvasWebGL";

/** CPU-only GL call recorder. It does not claim GLSL compilation or GPU parity. */
function recordingGL() {
  const calls: Array<{name:string;args:unknown[]}> = [];
  const constants = new Map<string,number>();
  const gl: WebGL2RenderingContext = new Proxy({}, {get(_target, key:string) {
    if (/^[A-Z][A-Z_0-9]*$/.test(key)) {
      if (!constants.has(key)) constants.set(key,key === "NO_ERROR" ? 0 : constants.size+1);
      return constants.get(key);
    }
    return (...args:unknown[]): unknown => {
      calls.push({name:key,args});
      if (key === "getExtension") return args[0] === "EXT_color_buffer_float" ? {} : null;
      if (key === "getParameter") return 512;
      if (key === "getShaderParameter" || key === "getProgramParameter") return true;
      if (key === "getUniformLocation") return args[1];
      if (key === "clientWaitSync") return gl.ALREADY_SIGNALED;
      if (key === "getError") return 0;
      if (key.startsWith("create") || key === "fenceSync") return {};
      return undefined;
    };
  }}) as WebGL2RenderingContext;
  const canvas = {width:64,height:64,getContext:() => gl} as unknown as HTMLCanvasElement;
  return {gl,canvas,calls};
}
const options = {log:false,auto:false,min:0,max:100,zoom:1,panX:0,panY:0,smooth:false,dtype:"<u4" as const,divisor:3};

describe("uint32 WebGL batch upload contract (mock GL, no GPU)", () => {
  it("uses R32UI/UNSIGNED_INT and retains the complete exact source through style changes", async () => {
    const {gl,canvas,calls} = recordingGL();
    const renderer = new CompareBatchWebGL(canvas,2,1,3,"<u4");
    const storage = new Uint32Array([777,0,65535,65536,16777217,4294967295,7,888]);
    const bytes = new DataView(storage.buffer,4,24);
    const rect = new Float32Array([0,0,32,32,32,0,32,32]);
    const sources = new Uint32Array([0,1]);
    await renderer.render(bytes,rect,sources,new Uint8Array(768),options);
    const allocation = calls.find(call => call.name === "texStorage3D")!;
    expect(allocation.args.slice(2)).toEqual([gl.R32UI,3,1,2]);
    const upload = calls.find(call => call.name === "texSubImage3D")!;
    expect(upload.args.slice(8,10)).toEqual([gl.RED_INTEGER,gl.UNSIGNED_INT]);
    const raw = upload.args[10] as Uint32Array;
    expect(raw).toBeInstanceOf(Uint32Array);
    expect(Array.from(raw)).toEqual([0,65535,65536,16777217,4294967295,7]);
    expect(renderer.readSourceValues()).toBe(raw);
    await renderer.render(bytes,rect,sources,new Uint8Array(768),{...options,log:true,divisor:0});
    expect(calls.filter(call => call.name === "texSubImage3D")).toHaveLength(1);
    expect(calls.filter(call => call.name === "uniform1f" && call.args[0] === "divisor").slice(-3)
      .every(call => call.args[1] === 0)).toBe(true);
    expect(Array.from(renderer.readSourceValues()!)).toEqual(Array.from(raw));
    const shaders = calls.filter(call => call.name === "shaderSource").map(call => String(call.args[1]));
    expect(shaders.filter(source => source.includes("uniform highp usampler2DArray images;") &&
      source.includes("if(divisor==0.0)return 0.0"))).toHaveLength(2);
    renderer.destroy();
    expect(renderer.readSourceValues()).toBeNull();
    expect(storage[4]).toBe(16777217);expect(storage[5]).toBe(4294967295);
  });
  it("rejects dtype or count extent changes before any texture upload", async () => {
    const {canvas,calls} = recordingGL();
    const renderer = new CompareBatchWebGL(canvas,2,1,3,"<u4");
    const bytes = new DataView(new ArrayBuffer(24));
    const args = [new Float32Array(8),new Uint32Array([0,1]),new Uint8Array(768)] as const;
    await expect(renderer.render(bytes,...args,{...options,dtype:"<f4"})).rejects.toThrow("scalar type changed");
    await expect(renderer.render(new DataView(new ArrayBuffer(20)),...args,options)).rejects.toThrow("shape");
    await expect(renderer.render(bytes,...args,{...options,divisor:-1})).rejects.toThrow("nonnegative");
    expect(calls.filter(call => call.name === "texSubImage3D")).toHaveLength(0);
    renderer.destroy();
  });
});
