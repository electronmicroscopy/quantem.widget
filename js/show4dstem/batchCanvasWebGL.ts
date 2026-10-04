/** WebGL2 batch presentation for notebook browsers without WebGPU. */
import { residentLogTable, residentRawValues, residentDivisor, residentScalarType, type ResidentValues, type ResidentScalarType } from "./batchValues";
const fullscreen = `#version 300 es
void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);gl_Position=vec4(p*2.0-1.0,0,1);}`;
const partial = `#version 300 es
precision highp float;precision highp sampler2DArray;
uniform sampler2DArray images;uniform ivec2 shape;uniform bool logarithmic;
out vec2 limits;
void main(){int chunk=int(gl_FragCoord.x);int frame=int(gl_FragCoord.y);float lo=3.402823e38;float hi=-3.402823e38;
int size=shape.x*shape.y;for(int p=chunk;p<size;p+=4096){float v=texelFetch(images,ivec3(p%shape.x,p/shape.x,frame),0).r;if(logarithmic)v=log(1.0+max(v,0.0));lo=min(lo,v);hi=max(hi,v);}limits=vec2(lo,hi);}`;
const finalRange = `#version 300 es
precision highp float;uniform sampler2D partial;uniform int stride;out vec2 limits;
void main(){int frame=int(gl_FragCoord.y);vec2 r=vec2(3.402823e38,-3.402823e38);for(int i=0;i<stride;i++){vec2 v=texelFetch(partial,ivec2(int(gl_FragCoord.x)*stride+i,frame),0).rg;r=vec2(min(r.x,v.x),max(r.y,v.y));}limits=r;}`;
const vertex = `#version 300 es
layout(location=0)in vec4 rect;layout(location=1)in float source;
uniform vec2 canvas;out vec2 uv;flat out int frame;
void main(){vec2 corners[6]=vec2[6](vec2(0,0),vec2(1,0),vec2(0,1),vec2(0,1),vec2(1,0),vec2(1,1));uv=corners[gl_VertexID];vec2 p=rect.xy+uv*rect.zw;gl_Position=vec4(p.x*2.0/canvas.x-1.0,1.0-p.y*2.0/canvas.y,0,1);frame=int(source);}`;
const fragment = `#version 300 es
precision highp float;precision highp sampler2DArray;
uniform sampler2DArray images;uniform sampler2D ranges;uniform sampler2D lut;
uniform ivec2 shape;uniform vec2 percent;uniform vec3 view;uniform bool logarithmic;uniform bool interpolatePixels;
in vec2 uv;flat in int frame;out vec4 color;
vec4 mapped(ivec2 p){p=clamp(p,ivec2(0),shape-1);float value=texelFetch(images,ivec3(p,frame),0).r;if(logarithmic)value=log(1.0+max(value,0.0));vec2 range=texelFetch(ranges,ivec2(0,frame),0).rg;vec2 bound=range.x+(range.y-range.x)*percent/100.0;float v=bound.y>bound.x?clamp((value-bound.x)/(bound.y-bound.x),0.0,1.0):0.5;return texelFetch(lut,ivec2(int(v*255.0),0),0);}
void main(){vec2 p=(uv*vec2(shape)-view.yz)/view.x;if(any(lessThan(p,vec2(0)))||any(greaterThanEqual(p,vec2(shape)))){color=vec4(0,0,0,1);return;}
if(!interpolatePixels){color=mapped(ivec2(floor(p)));return;}vec2 q=p-0.5;ivec2 base=ivec2(floor(q));vec2 f=fract(q);color=mix(mix(mapped(base),mapped(base+ivec2(1,0)),f.x),mix(mapped(base+ivec2(0,1)),mapped(base+ivec2(1,1)),f.x),f.y);}`;

export class CompareBatchWebGL {
  readonly adapter: string;
  private programs: WebGLProgram[] = [];
  private textures: WebGLTexture[] = [];
  private framebuffer: WebGLFramebuffer;
  private vertices: WebGLBuffer;
  private vao: WebGLVertexArrayObject;
  private previous: DataView | null = null;
  private sourceValues: ResidentValues | null = null;

  /** Borrow exact source values for scientific readout or binary export. */
  readSourceValues(): ResidentValues | null { return this.sourceValues; }
  private lastLog: boolean | null = null;
  private lastDivisor: number | undefined;
  private disposed = false;
  private packed: boolean;
  private wideCounts: boolean;
  private scientificReadback: {program:WebGLProgram;feedback:WebGLTransformFeedback;buffers:WebGLBuffer[]} | null = null;
  private timer: { TIME_ELAPSED_EXT: number; GPU_DISJOINT_EXT: number } | null;
  constructor(
    readonly canvas: HTMLCanvasElement,
    private count: number,
    private rows: number,
    private cols: number,
    private scalarType: ResidentScalarType = "<f4",
    reduceCountsFirst = true,
  ) {
    residentScalarType({dtype:scalarType});
    const packed = scalarType === "<u2";
    this.packed = packed;
    this.wideCounts = scalarType === "<u4";
    const gl = canvas.getContext("webgl2", {
      alpha: true,
      antialias: false,
      premultipliedAlpha: true,
      preserveDrawingBuffer: true,
    });
    if (!gl) throw new Error("Neither WebGPU nor WebGL2 is available");
    this.gl = gl;
    this.timer = gl.getExtension("EXT_disjoint_timer_query_webgl2");
    if (!gl.getExtension("EXT_color_buffer_float"))
      throw new Error("WebGL2 float render targets unavailable");
    if (count > gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS))
      throw new Error("Full batch exceeds texture-array limit");
    const debug = gl.getExtension("WEBGL_debug_renderer_info");
    this.adapter =
      "WebGL2 " +
      (debug
        ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL)
        : gl.getParameter(gl.RENDERER));
    const program = (vs: string, fs: string) => {
      const compile = (type: number, source: string) => {
        const shader = gl.createShader(type)!;
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
          throw new Error(
            gl.getShaderInfoLog(shader) || "Shader compilation failed",
          );
        return shader;
      };
      const p = gl.createProgram()!;
      const v = compile(gl.VERTEX_SHADER, vs),
        f = compile(gl.FRAGMENT_SHADER, fs);
      gl.attachShader(p, v);
      gl.attachShader(p, f);
      gl.linkProgram(p);
      gl.deleteShader(v);
      gl.deleteShader(f);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS))
        throw new Error(gl.getProgramInfoLog(p) || "Shader link failed");
      this.programs.push(p);
      return p;
    };
    const countShader = (source: string) => source
      .replace("precision highp sampler2DArray;", "precision highp usampler2DArray;")
      .replace("uniform sampler2DArray images;", `uniform highp usampler2DArray images;
uniform highp sampler2D normalization;
uniform float divisor;
float read_count(ivec3 point) { if(divisor==0.0)return 0.0;uint c=texelFetch(images,point,0).r;
return texelFetch(normalization,ivec2(int(c&255u),int(c>>8u)),0).r; }`)
      .replace("texelFetch(images,ivec3(p%shape.x,p/shape.x,frame),0).r", "read_count(ivec3(p%shape.x,p/shape.x,frame))")
      .replace("texelFetch(images,ivec3(p,frame),0).r", "read_count(ivec3(p,frame))");
    const exactLogCountShader = (source: string) => countShader(source)
      .replace("uniform bool logarithmic;", "")
      .replace("uniform highp sampler2D normalization;", "uniform highp sampler2D normalization;uniform highp sampler2D logarithms;uniform bool logarithmic;")
      .replace("return texelFetch(normalization,ivec2(int(c&255u),int(c>>8u)),0).r;", "return logarithmic?texelFetch(logarithms,ivec2(int(c&255u),int(c>>8u)),0).r:texelFetch(normalization,ivec2(int(c&255u),int(c>>8u)),0).r;")
      .replace("if(logarithmic)v=log(1.0+max(v,0.0));", "")
      .replace("if(logarithmic)value=log(1.0+max(value,0.0));", "");
    // Normalization is monotone over exact nonnegative counts. Reduce their
    // integer extrema first, then look up only the final66 pairs.
    const countPartial = partial
      .replace("precision highp sampler2DArray;", "precision highp usampler2DArray;")
      .replace("uniform sampler2DArray images;", "uniform highp usampler2DArray images;")
      .replace("float v=texelFetch(images,ivec3(p%shape.x,p/shape.x,frame),0).r;if(logarithmic)v=log(1.0+max(v,0.0));",
        "float v=float(texelFetch(images,ivec3(p%shape.x,p/shape.x,frame),0).r);");
    const countFinal = finalRange
      .replace("precision highp float;", `precision highp float;
uniform highp sampler2D normalization;uniform highp sampler2D logarithms;uniform bool convertRange;uniform bool logarithmic;uniform float divisor;
float normalized(float value){if(divisor==0.0)return 0.0;int count=int(value);ivec2 p=ivec2(count&255,count>>8);
return logarithmic?texelFetch(logarithms,p,0).r:texelFetch(normalization,p,0).r;}`)
      .replace("limits=r;", "limits=convertRange?vec2(normalized(r.x),normalized(r.y)):r;");
    const wideCountShader = (source: string) => source
      .replace("precision highp sampler2DArray;", "precision highp usampler2DArray;precision highp int;")
      .replace("uniform sampler2DArray images;", `uniform highp usampler2DArray images;
uniform float divisor;
float read_count(ivec3 point){if(divisor==0.0)return 0.0;return float(texelFetch(images,point,0).r)/divisor;}`)
      .replace("texelFetch(images,ivec3(p%shape.x,p/shape.x,frame),0).r", "read_count(ivec3(p%shape.x,p/shape.x,frame))")
      .replace("texelFetch(images,ivec3(p,frame),0).r", "read_count(ivec3(p,frame))");
    program(fullscreen, packed ? (reduceCountsFirst ? countPartial : exactLogCountShader(partial)) : this.wideCounts ? wideCountShader(partial) : partial);
    program(fullscreen, packed && reduceCountsFirst ? countFinal : finalRange);
    program(vertex, packed ? exactLogCountShader(fragment) : this.wideCounts ? wideCountShader(fragment) : fragment);
    const texture = (unit: number, target: number) => {
      const t = gl.createTexture()!;
      this.textures.push(t);
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(target, t);
      gl.texParameteri(target, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(target, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(target, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(target, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return t;
    };
    texture(0, gl.TEXTURE_2D_ARRAY);
    gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, packed ? gl.R16UI : this.wideCounts ? gl.R32UI : gl.R32F, cols, rows, count);
    texture(1, gl.TEXTURE_2D);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RG32F, 4096, count);
    texture(2, gl.TEXTURE_2D);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RG32F, 1, count);
    texture(3, gl.TEXTURE_2D);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGB8, 256, 1);
    texture(4, gl.TEXTURE_2D);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RG32F, 128, count);
    texture(5, gl.TEXTURE_2D);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R32F, 256, 256);
    texture(6, gl.TEXTURE_2D);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R32F, 256, 256);
    this.framebuffer = gl.createFramebuffer()!;
    this.vertices = gl.createBuffer()!;
    this.vao = gl.createVertexArray()!;
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vertices);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 20, 0);
    gl.vertexAttribDivisor(0, 1);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 20, 16);
    gl.vertexAttribDivisor(1, 1);
    gl.bindVertexArray(null);
  }
  private gl: WebGL2RenderingContext;
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
    if (options.auto)
      throw new Error(
        "WebGL batch renderer requires manual contrast; retain CPU percentile rendering for Auto contrast",
      );
    const start = performance.now(),
      gl = this.gl;
    if ((options.dtype ?? "<f4") !== this.scalarType) throw new Error("Batch scalar type changed without replacing its renderer.");
    const raw = residentRawValues(bytes, {dtype:this.scalarType,shape:[this.count,this.rows,this.cols],normalization_offset:options.normalizationOffset});
    residentDivisor({dtype:this.scalarType,mask_area:options.divisor});
    if (this.scalarType === "<u2" && options.normalizationOffset === undefined) throw new Error("GPU uint16 batches require a normalization table; use CPU display fallback.");
    this.sourceValues = raw;
    const fresh = bytes !== this.previous;
    const timer = this.timer;
    const query = timer ? gl.createQuery() : null;
    if (query && timer) gl.beginQuery(timer.TIME_ELAPSED_EXT, query);
    if (fresh) {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.textures[0]);
      gl.texSubImage3D(
        gl.TEXTURE_2D_ARRAY,
        0,
        0,
        0,
        0,
        this.cols,
        this.rows,
        this.count,
        this.packed || this.wideCounts ? gl.RED_INTEGER : gl.RED,
        this.packed ? gl.UNSIGNED_SHORT : this.wideCounts ? gl.UNSIGNED_INT : gl.FLOAT,
        raw,
      );
      if (this.packed) {
        if (options.normalizationOffset === undefined) throw new Error("Exact counts require their float32 normalization table.");
        gl.activeTexture(gl.TEXTURE5);
        gl.bindTexture(gl.TEXTURE_2D, this.textures[5]);
        gl.texSubImage2D(gl.TEXTURE_2D,0,0,0,256,256,gl.RED,gl.FLOAT,
          new Float32Array(bytes.buffer,bytes.byteOffset+options.normalizationOffset,65536));
      }
      this.previous = bytes;
    }
    if (this.packed && options.log && (fresh || this.lastLog !== true)) {
      if (options.normalizationOffset === undefined) throw new Error("Exact logarithms require the normalization table.");
      const logarithms=residentLogTable(bytes,options.normalizationOffset);
      gl.activeTexture(gl.TEXTURE6);gl.bindTexture(gl.TEXTURE_2D,this.textures[6]);
      gl.texSubImage2D(gl.TEXTURE_2D,0,0,0,256,256,gl.RED,gl.FLOAT,logarithms);
    }
    const uploaded = performance.now();
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.textures[3]);
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      0,
      0,
      256,
      1,
      gl.RGB,
      gl.UNSIGNED_BYTE,
      lut instanceof Uint8Array ? lut : new Uint8Array(lut),
    );
    const tiles = new Float32Array(sourceIndices.length * 5);
    sourceIndices.forEach((source, i) => {
      tiles.set(rectangles.subarray(i * 4, i * 4 + 4), i * 5);
      tiles[i * 5 + 4] = source;
    });
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vertices);
    gl.bufferData(gl.ARRAY_BUFFER, tiles, gl.DYNAMIC_DRAW);
    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.SCISSOR_TEST);
    gl.bindVertexArray(null);
    if (fresh || this.lastLog !== options.log || this.lastDivisor !== options.divisor) {
      this.lastLog = options.log;
      this.lastDivisor = options.divisor;
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
      const range = this.programs[0];
      gl.useProgram(range);
      gl.uniform1f(gl.getUniformLocation(range, "divisor"), options.divisor ?? 1);
      gl.uniform1i(gl.getUniformLocation(range, "normalization"), 5);
      gl.uniform1i(gl.getUniformLocation(range, "logarithms"), 6);
      gl.uniform1i(gl.getUniformLocation(range, "images"), 0);
      gl.uniform2i(gl.getUniformLocation(range, "shape"), this.cols, this.rows);
      gl.uniform1i(gl.getUniformLocation(range, "logarithmic"), +options.log);
      gl.framebufferTexture2D(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0,
        gl.TEXTURE_2D,
        this.textures[1],
        0,
      );
      gl.viewport(0, 0, 4096, this.count);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      const finish = this.programs[1];
      gl.useProgram(finish);
      gl.uniform1f(gl.getUniformLocation(finish, "divisor"), options.divisor ?? 1);
      gl.uniform1i(gl.getUniformLocation(finish, "normalization"), 5);
      gl.uniform1i(gl.getUniformLocation(finish, "logarithms"), 6);
      gl.uniform1i(gl.getUniformLocation(finish, "convertRange"), 0);
      gl.uniform1i(gl.getUniformLocation(finish, "logarithmic"), +options.log);
      gl.uniform1i(gl.getUniformLocation(finish, "partial"), 1);
      gl.uniform1i(gl.getUniformLocation(finish, "stride"), 32);
      gl.framebufferTexture2D(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0,
        gl.TEXTURE_2D,
        this.textures[4],
        0,
      );
      gl.viewport(0, 0, 128, this.count);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.uniform1i(gl.getUniformLocation(finish, "partial"), 4);
      gl.uniform1i(gl.getUniformLocation(finish, "stride"), 128);
      gl.uniform1i(gl.getUniformLocation(finish, "convertRange"), 1);
      gl.framebufferTexture2D(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0,
        gl.TEXTURE_2D,
        this.textures[2],
        0,
      );
      gl.viewport(0, 0, 1, this.count);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    const draw = this.programs[2];
    gl.useProgram(draw);
    gl.uniform1f(gl.getUniformLocation(draw, "divisor"), options.divisor ?? 1);
    gl.uniform1i(gl.getUniformLocation(draw, "normalization"), 5);
    gl.uniform1i(gl.getUniformLocation(draw, "logarithms"), 6);
    gl.uniform1i(gl.getUniformLocation(draw, "images"), 0);
    gl.uniform1i(gl.getUniformLocation(draw, "ranges"), 2);
    gl.uniform1i(gl.getUniformLocation(draw, "lut"), 3);
    gl.uniform2i(gl.getUniformLocation(draw, "shape"), this.cols, this.rows);
    gl.uniform2f(
      gl.getUniformLocation(draw, "canvas"),
      this.canvas.width,
      this.canvas.height,
    );
    gl.uniform2f(
      gl.getUniformLocation(draw, "percent"),
      options.min,
      options.max,
    );
    gl.uniform3f(
      gl.getUniformLocation(draw, "view"),
      options.zoom,
      options.panX,
      options.panY,
    );
    gl.uniform1i(gl.getUniformLocation(draw, "logarithmic"), +options.log);
    gl.uniform1i(
      gl.getUniformLocation(draw, "interpolatePixels"),
      +options.smooth,
    );
    gl.bindVertexArray(this.vao);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, sourceIndices.length);
    gl.bindVertexArray(null);
    if (query && timer) gl.endQuery(timer.TIME_ELAPSED_EXT);
    const submitted = performance.now();
    const fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0)!;
    gl.flush();
    await new Promise<void>((resolve, reject) => {
      const poll = () => {
        if (this.disposed) {
          resolve();
          return;
        }
        const status = gl.clientWaitSync(fence, 0, 0);
        if (status === gl.WAIT_FAILED) {
          reject(new Error("GPU fence failed"));
          return;
        }
        if (status === gl.TIMEOUT_EXPIRED) {
          setTimeout(poll, 0);
          return;
        }
        gl.deleteSync(fence);
        resolve();
      };
      poll();
    });
    let gpuElapsedMs: number | null = null;
    if (query && timer) {
      if (gl.getQueryParameter(query, gl.QUERY_RESULT_AVAILABLE) && !gl.getParameter(timer.GPU_DISJOINT_EXT))
        gpuElapsedMs = Number(gl.getQueryParameter(query, gl.QUERY_RESULT)) / 1e6;
      gl.deleteQuery(query);
    }
    return {
      fresh,
      panels: sourceIndices.length,
      upload_submit_ms: submitted - start,
      texture_upload_ms: uploaded - start,
      gpu_elapsed_ms: gpuElapsedMs,
      completed_ms: performance.now() - start,
      adapter: this.adapter,
    };
  }
  readRanges(): number[][] {
    const gl = this.gl;
    const values = new Float32Array(this.count * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      this.textures[2],
      0,
    );
    gl.readPixels(0, 0, 1, this.count, gl.RGBA, gl.FLOAT, values);
    const error = gl.getError();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (error !== gl.NO_ERROR)
      throw new Error(`GPU range validation readback failed: ${error}`);
    return Array.from({ length: this.count }, (_, i) => [
      values[i * 4],
      values[i * 4 + 1],
    ]);
  }
  /** Read every native count and normalized value for independent validation. */
  readScientificArrays(): {counts:Uint32Array<ArrayBuffer>;values:Float32Array<ArrayBuffer>} {
    if (!this.packed && !this.wideCounts) throw new Error("Exact-count GPU validation requires an integer batch.");
    const gl = this.gl;
    const size = this.count * this.rows * this.cols;
    if (!this.scientificReadback) {
      const program = gl.createProgram()!;
      const compile = (type:number,source:string) => {
        const shader = gl.createShader(type)!;
        gl.shaderSource(shader,source);gl.compileShader(shader);
        if (!gl.getShaderParameter(shader,gl.COMPILE_STATUS))
          throw new Error(gl.getShaderInfoLog(shader) || "Scientific GPU readback shader failed");
        gl.attachShader(program,shader);gl.deleteShader(shader);
      };
      compile(gl.VERTEX_SHADER,`#version 300 es
precision highp float;precision highp int;precision highp usampler2DArray;
uniform highp usampler2DArray images;uniform highp sampler2D normalization;uniform ivec2 shape;uniform float divisor;
flat out uint countOut;out float normalizedOut;
void main(){int p=gl_VertexID;countOut=texelFetch(images,ivec3(p%shape.x,(p/shape.x)%shape.y,p/(shape.x*shape.y)),0).r;
normalizedOut=divisor==0.0?0.0:${this.wideCounts ? "float(countOut)/divisor" : "texelFetch(normalization,ivec2(int(countOut&255u),int(countOut>>8u)),0).r"};
gl_Position=vec4(0,0,0,1);gl_PointSize=1.0;}`);
      compile(gl.FRAGMENT_SHADER,`#version 300 es
precision highp float;out vec4 color;void main(){color=vec4(0);}`);
      gl.transformFeedbackVaryings(program,["countOut","normalizedOut"],gl.SEPARATE_ATTRIBS);
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program,gl.LINK_STATUS))
        throw new Error(gl.getProgramInfoLog(program) || "Scientific GPU readback link failed");
      const buffers = [gl.createBuffer()!,gl.createBuffer()!];
      for (const buffer of buffers) {
        gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER,buffer);
        gl.bufferData(gl.TRANSFORM_FEEDBACK_BUFFER,size*4,gl.STREAM_READ);
      }
      this.scientificReadback={program,buffers,feedback:gl.createTransformFeedback()!};
    }
    const readback = this.scientificReadback;
    gl.bindFramebuffer(gl.FRAMEBUFFER,null);
    gl.bindVertexArray(null);
    gl.useProgram(readback.program);
    gl.uniform1f(gl.getUniformLocation(readback.program,"divisor"),this.lastDivisor ?? 1);
    gl.uniform1i(gl.getUniformLocation(readback.program,"images"),0);
    gl.uniform1i(gl.getUniformLocation(readback.program,"normalization"),5);
    gl.uniform2i(gl.getUniformLocation(readback.program,"shape"),this.cols,this.rows);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK,readback.feedback);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER,0,readback.buffers[0]);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER,1,readback.buffers[1]);
    gl.enable(gl.RASTERIZER_DISCARD);
    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS,0,size);
    gl.endTransformFeedback();
    gl.disable(gl.RASTERIZER_DISCARD);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK,null);
    const counts = new Uint32Array(size),values = new Float32Array(size);
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER,readback.buffers[0]);
    gl.getBufferSubData(gl.TRANSFORM_FEEDBACK_BUFFER,0,counts);
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER,readback.buffers[1]);
    gl.getBufferSubData(gl.TRANSFORM_FEEDBACK_BUFFER,0,values);
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER,null);
    const error = gl.getError();
    if (error !== gl.NO_ERROR) throw new Error(`Full-array GPU validation readback failed: ${error}`);
    return {counts,values};
  }
  destroy() {
    this.sourceValues = null;
    this.previous = null;
    this.disposed = true;
    this.programs.forEach((p) => this.gl.deleteProgram(p));
    this.textures.forEach((t) => this.gl.deleteTexture(t));
    this.gl.deleteBuffer(this.vertices);
    this.gl.deleteFramebuffer(this.framebuffer);
    this.gl.deleteVertexArray(this.vao);
    if (this.scientificReadback) {
      this.gl.deleteProgram(this.scientificReadback.program);
      this.gl.deleteTransformFeedback(this.scientificReadback.feedback);
      this.scientificReadback.buffers.forEach(buffer => this.gl.deleteBuffer(buffer));
    }
  }
}
