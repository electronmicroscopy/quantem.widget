/// <reference types="@webgpu/types" />
// Offline GPU decompression of HDF5 bitshuffle+LZ4 (bslz4) 4D-STEM data.
//
// Ships the NATIVE detector codec (Dectris/Arina write bslz4) so the browser
// downloads ~6x less than raw uint16 and decompresses on WebGPU - no Python, no
// server. Two passes, both verified bit-exact vs h5py on real gold (192x192
// uint16): Pass1 LZ4-decodes each independent block, Pass2 inverts the bit
// transpose. Output is uint16-packed in [scanPos][detPixel] order - the exact
// layout Show4DSTEMCompute reads in uint16 mode, so it feeds masked_sum /
// reduce_frames with no copy.
//
// Per-frame chunk layout (one HDF5 chunk = one diffraction pattern):
//   blockMeta gives, per (frame,block), the absolute byte offset + compressed
//   length of that block's LZ4 stream within the concatenated `compressed`. Each
//   block decompresses to blockElemBytes = blockElems * elemBytes, holds
//   nbits = elemBytes*8 bit-planes of planeBytes = blockElems/8 each.

import { getGPUDevice, onGPULost } from "./device";

// A/B switch for the parallel (Strategy-D, round-based) LZ4 fused kernel vs the serial-thread-0
// fused kernel. Set globalThis.__BSLZ4_PARALLEL=true in the console to measure on real hardware
// without a rebuild. Default false until the parallel kernel is parity-verified + benchmarked.
const BSLZ4_PARALLEL = typeof globalThis !== "undefined" && (globalThis as { __BSLZ4_PARALLEL?: boolean }).__BSLZ4_PARALLEL === true;

// Pass1: one thread per block, LZ4-decode into the `inter` (bitshuffled) buffer.
// Blocks are independent -> embarrassingly parallel. Byte-addressed RMW because
// WGSL storage is u32-only.
const PASS1_WGSL = `
@group(0) @binding(0) var<storage,read> raw: array<u32>;
@group(0) @binding(1) var<storage,read_write> inter: array<u32>;
@group(0) @binding(2) var<storage,read> blkMeta: array<u32>;   // coff,clen per block
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;             // totalBlocks, blockBytes
fn rraw(i:u32)->u32{return (raw[i>>2u]>>((i&3u)*8u))&0xffu;}
fn rout(i:u32)->u32{return (inter[i>>2u]>>((i&3u)*8u))&0xffu;}
fn wout(i:u32,v:u32){let w=i>>2u;let s=(i&3u)*8u;inter[w]=(inter[w]&(~(0xffu<<s)))|((v&0xffu)<<s);}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>){
  let g=gid.x; if(g>=cfg.x){return;}
  let coff=blkMeta[g*2u]; let cend=coff+blkMeta[g*2u+1u]; let base=g*cfg.y;
  var ci=coff; var di=0u;
  loop{ if(ci>=cend){break;}
    let tok=rraw(ci); ci=ci+1u; var nlit=tok>>4u;
    if(nlit==15u){loop{let bb=rraw(ci);ci=ci+1u;nlit=nlit+bb;if(bb!=255u){break;}}}
    var k=0u; loop{if(k>=nlit){break;} wout(base+di+k,rraw(ci+k)); k=k+1u;} ci=ci+nlit; di=di+nlit;
    if(ci>=cend){break;}
    let off=rraw(ci)|(rraw(ci+1u)<<8u); ci=ci+2u; var ml=4u+(tok&0xfu);
    if((tok&0xfu)==15u){loop{let bb=rraw(ci);ci=ci+1u;ml=ml+bb;if(bb!=255u){break;}}}
    var j=0u; loop{if(j>=ml){break;} wout(base+di+j,rout(base+di+j-off)); j=j+1u;} di=di+ml;
  }
}`;

// Pass2: inverse bitshuffle. One thread per GROUP of 8 consecutive pixels - they
// share the same 16 plane-bytes (one byte per bit-plane), so 16 global reads feed
// 8 outputs (8x fewer reads than per-pixel -> ~4x faster, measured). Writes 4
// uint16-packed u32. Plane-major, LSB-first: pixel (group*8 + i) bit b = bit i of
// plane-byte (b*planeBytes + group_byte). cfg = nGroups, strideX, blockElems, planeBytes.
const PASS2_WGSL = `
@group(0) @binding(0) var<storage,read> inter: array<u32>;
@group(0) @binding(1) var<storage,read_write> stack: array<u32>;  // uint16-packed
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;  // nGroups, strideX, blockElems, planeBytes
fn byteAt(o:u32)->u32{return (inter[o>>2u]>>((o&3u)*8u))&0xffu;}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>){
  let grp=gid.y*cfg.y + gid.x; if(grp>=cfg.x){return;}
  let blockElems=cfg.z; let planeBytes=cfg.w; let blockBytes=blockElems*2u;
  let nBlk=__NBLK__; let framePix=__FRAMEPIX__;
  let e0=grp*8u; let frm=e0/framePix; let inFrame=e0%framePix;
  let blk=inFrame/blockElems; let groupByte=(inFrame%blockElems)>>3u;
  let pb=(frm*nBlk+blk)*blockBytes + groupByte;
  var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
  for(var b:u32=0u;b<16u;b=b+1u){
    let byte=byteAt(pb+b*planeBytes); let bit=1u<<b;
    if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
    if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
    if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
    if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
  }
  let o=grp*4u;
  stack[o]=v0|(v1<<16u); stack[o+1u]=v2|(v3<<16u); stack[o+2u]=v4|(v5<<16u); stack[o+3u]=v6|(v7<<16u);
}`;

// Pass2 (uint8): same 8-pixel-group inverse bitshuffle, but clip(0,255) and pack 4
// pixels per u32 - halves resident memory (9.6 GB vs 19 GB for full 512x512). Offline
// default: real detector counts are 0-~50, so clip is near-lossless for the signal.
const PASS2_U8_WGSL = `
@group(0) @binding(0) var<storage,read> inter: array<u32>;
@group(0) @binding(1) var<storage,read_write> stack: array<u32>;  // uint8-packed (4/u32)
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;  // nGroups, strideX, blockElems, planeBytes
fn byteAt(o:u32)->u32{return (inter[o>>2u]>>((o&3u)*8u))&0xffu;}
fn clip8(v:u32)->u32{return select(v,255u,v>255u);}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>){
  let grp=gid.y*cfg.y + gid.x; if(grp>=cfg.x){return;}
  let blockElems=cfg.z; let planeBytes=cfg.w; let blockBytes=blockElems*2u;
  let nBlk=__NBLK__; let framePix=__FRAMEPIX__;
  let e0=grp*8u; let frm=e0/framePix; let inFrame=e0%framePix;
  let blk=inFrame/blockElems; let groupByte=(inFrame%blockElems)>>3u;
  let pb=(frm*nBlk+blk)*blockBytes + groupByte;
  var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
  for(var b:u32=0u;b<16u;b=b+1u){
    let byte=byteAt(pb+b*planeBytes); let bit=1u<<b;
    if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
    if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
    if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
    if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
  }
  let o=grp*2u;
  stack[o]=clip8(v0)|(clip8(v1)<<8u)|(clip8(v2)<<16u)|(clip8(v3)<<24u);
  stack[o+1u]=clip8(v4)|(clip8(v5)<<8u)|(clip8(v6)<<16u)|(clip8(v7)<<24u);
}`;

// Pass2 (uint8 SOURCE): companion encoded from uint8 (typesize 1) -> only 8 bit
// planes, and the LZ4 block is half the bytes -> ~2x faster than the uint16-source
// path, BOTH passes. Output is uint8-packed directly (values already <= 255).
const PASS2_U8SRC_WGSL = `
@group(0) @binding(0) var<storage,read> inter: array<u32>;
@group(0) @binding(1) var<storage,read_write> stack: array<u32>;  // uint8-packed (4/u32)
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;  // nGroups, strideX, blockElems, planeBytes
fn byteAt(o:u32)->u32{return (inter[o>>2u]>>((o&3u)*8u))&0xffu;}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>){
  let grp=gid.y*cfg.y + gid.x; if(grp>=cfg.x){return;}
  let blockElems=cfg.z; let planeBytes=cfg.w; let blockBytes=blockElems;   // uint8: 1 byte/elem
  let nBlk=__NBLK__; let framePix=__FRAMEPIX__;
  let e0=grp*8u; let frm=e0/framePix; let inFrame=e0%framePix;
  let blk=inFrame/blockElems; let groupByte=(inFrame%blockElems)>>3u;
  let pb=(frm*nBlk+blk)*blockBytes + groupByte;
  var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
  for(var b:u32=0u;b<8u;b=b+1u){
    let byte=byteAt(pb+b*planeBytes); let bit=1u<<b;
    if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
    if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
    if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
    if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
  }
  let o=grp*2u;
  stack[o]=v0|(v1<<8u)|(v2<<16u)|(v3<<24u); stack[o+1u]=v4|(v5<<8u)|(v6<<16u)|(v7<<24u);
}`;

// Pass2 (float32 SOURCE): 4-byte elements -> 32 bit-planes. Same 8-pixel-group inverse
// bitshuffle as PASS2_WGSL but loops all 32 planes and reconstructs the FULL 32-bit value
// per pixel, stored as one u32/pixel. The decoded 4 bytes ARE the IEEE-754 float32 bit
// pattern; the reduce/frame shaders bitcast<f32> them. No clip, no pack - full precision.
// cfg = nGroups, strideX, blockElems, planeBytes. blockBytes = blockElems*4 (4-byte elem).
const PASS2_F32_WGSL = `
@group(0) @binding(0) var<storage,read> inter: array<u32>;
@group(0) @binding(1) var<storage,read_write> stack: array<u32>;  // float32 bit-pattern, 1/pixel
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;  // nGroups, strideX, blockElems, planeBytes
fn byteAt(o:u32)->u32{return (inter[o>>2u]>>((o&3u)*8u))&0xffu;}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>){
  let grp=gid.y*cfg.y + gid.x; if(grp>=cfg.x){return;}
  let blockElems=cfg.z; let planeBytes=cfg.w; let blockBytes=blockElems*4u;
  let nBlk=__NBLK__; let framePix=__FRAMEPIX__;
  let e0=grp*8u; let frm=e0/framePix; let inFrame=e0%framePix;
  let blk=inFrame/blockElems; let groupByte=(inFrame%blockElems)>>3u;
  let pb=(frm*nBlk+blk)*blockBytes + groupByte;
  var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
  for(var b:u32=0u;b<32u;b=b+1u){
    let byte=byteAt(pb+b*planeBytes); let bit=1u<<b;
    if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
    if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
    if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
    if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
  }
  let o=grp*8u;
  stack[o]=v0; stack[o+1u]=v1; stack[o+2u]=v2; stack[o+3u]=v3;
  stack[o+4u]=v4; stack[o+5u]=v5; stack[o+6u]=v6; stack[o+7u]=v7;
}`;

const MAX_WG = 65535;

// Copy bytes into a mapped GPU range using the widest aligned element view. V8's
// Uint8Array.set is byte-granular (~3.75 GB/s); a Float64Array view stores 8 bytes/op and is
// ~2-3x faster for multi-GB uploads. `src` is a Uint8Array over a whole file ArrayBuffer at
// byteOffset 0 and the mapped range is a fresh 4-aligned ArrayBuffer, so copy the bulk as
// f64 and the <8 trailing bytes as u8.
function copyWide(dst: ArrayBuffer, src: Uint8Array): void {
  const len = src.byteLength, off = src.byteOffset;
  if ((off & 7) === 0) {
    const n8 = len >>> 3;
    if (n8 > 0) new Float64Array(dst, 0, n8).set(new Float64Array(src.buffer, off, n8));
    const tail = n8 << 3;
    if (tail < len) new Uint8Array(dst, tail, len - tail).set(src.subarray(tail));
    return;
  }
  new Uint8Array(dst).set(src);
}

// FUSED decode (uint16 source -> uint8 output, the offline default): ONE WORKGROUP per
// block. Thread 0 LZ4-decodes the block's blockBytes into workgroup-SHARED `sh` (byte RMW
// on shared is ~100x the old global RMW, and there is no interBuf round-trip), barrier,
// then all 64 threads inverse-bitshuffle the blockElems/8 eight-pixel groups straight from
// `sh` and write uint8-packed output coalesced to `stack`. Bit-exact with PASS1+PASS2_U8:
// same LZ4 token loop, same plane addressing (sh[lg + b*planeBytes]), same pack/clip.
// __NPB__ = blockElems/8 = planeBytes = groups per block; __BE__ = blockElems.
const FUSED_U16U8_WGSL = `
@group(0) @binding(0) var<storage,read> raw: array<u32>;
@group(0) @binding(1) var<storage,read> blkMeta: array<u32>;   // coff,clen per block
@group(0) @binding(2) var<storage,read_write> stack: array<u32>;  // uint8-packed (4/u32)
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;   // totalBlocks, gridX, nBlk, framePix
var<workgroup> sh: array<u32, 2048>;   // up to 8192 decoded (bitshuffled) bytes / block
fn rraw(i:u32)->u32{return (raw[i>>2u]>>((i&3u)*8u))&0xffu;}
fn rsh(i:u32)->u32{return (sh[i>>2u]>>((i&3u)*8u))&0xffu;}
fn wsh(i:u32,v:u32){let w=i>>2u;let s=(i&3u)*8u;sh[w]=(sh[w]&(~(0xffu<<s)))|((v&0xffu)<<s);}
fn clip8(v:u32)->u32{return select(v,255u,v>255u);}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>){
  let g = wid.y*cfg.y + wid.x; let lx = lid.x;
  if(lx==0u && g<cfg.x){
    let coff=blkMeta[g*2u]; let cend=coff+blkMeta[g*2u+1u];
    var ci=coff; var di=0u;
    loop{ if(ci>=cend){break;}
      let tok=rraw(ci); ci=ci+1u; var nlit=tok>>4u;
      if(nlit==15u){loop{let bb=rraw(ci);ci=ci+1u;nlit=nlit+bb;if(bb!=255u){break;}}}
      var k=0u; loop{if(k>=nlit){break;} wsh(di+k,rraw(ci+k)); k=k+1u;} ci=ci+nlit; di=di+nlit;
      if(ci>=cend){break;}
      let off=rraw(ci)|(rraw(ci+1u)<<8u); ci=ci+2u; var ml=4u+(tok&0xfu);
      if((tok&0xfu)==15u){loop{let bb=rraw(ci);ci=ci+1u;ml=ml+bb;if(bb!=255u){break;}}}
      var j=0u; loop{if(j>=ml){break;} wsh(di+j,rsh(di+j-off)); j=j+1u;} di=di+ml;
    }
  }
  workgroupBarrier();
  if(g>=cfg.x){return;}
  let frm = g/cfg.z; let blk = g%cfg.z;
  let pixBase = frm*cfg.w + blk*__BE__;   // first detector pixel of this block
  let oBase = (pixBase>>3u)*2u;            // 2 u32 per 8-pixel group
  // uint8 output == clip8(value): 255 iff ANY bit >=8 is set, else the low byte. So only the
  // 8 LOW planes need a bit-transpose; planes 8..nbits-1 collapse to one OR ("any high bit set"
  // per pixel). Bit-exact for any input, and 2x (uint16) / 4x (uint32) less bitshuffle work.
  for(var lg=lx; lg<__NPB__; lg=lg+64u){
    var hi:u32=0u;
    for(var b:u32=8u;b<__NBITS__;b=b+1u){ hi = hi | rsh(lg + b*__NPB__); }
    var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
    for(var b:u32=0u;b<8u;b=b+1u){
      let byte=rsh(lg + b*__NPB__); let bit=1u<<b;
      if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
      if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
      if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
      if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
    }
    let o=oBase + lg*2u;
    stack[o]=select(v0,255u,(hi&1u)!=0u)|(select(v1,255u,(hi&2u)!=0u)<<8u)|(select(v2,255u,(hi&4u)!=0u)<<16u)|(select(v3,255u,(hi&8u)!=0u)<<24u);
    stack[o+1u]=select(v4,255u,(hi&16u)!=0u)|(select(v5,255u,(hi&32u)!=0u)<<8u)|(select(v6,255u,(hi&64u)!=0u)<<16u)|(select(v7,255u,(hi&128u)!=0u)<<24u);
  }
}`;

// FUSED decode (float32 source -> float32 output, full precision): ONE WORKGROUP per block.
// Thread 0 LZ4-decodes the block into workgroup-SHARED sh (the fast path, no interBuf global
// round-trip), barrier, then all 64 threads inverse-bitshuffle the blockElems/8 eight-pixel
// groups over ALL 32 planes, reconstruct the full 32-bit value and write it as 1 u32/pixel
// (the IEEE-754 bit pattern - reduce/frame shaders bitcast<f32>). __NPB__ = blockElems/8 =
// planeBytes = groups/block; __BE__ = blockElems.
const FUSED_F32_WGSL = `
@group(0) @binding(0) var<storage,read> raw: array<u32>;
@group(0) @binding(1) var<storage,read> blkMeta: array<u32>;   // coff,clen per block
@group(0) @binding(2) var<storage,read_write> stack: array<u32>;  // float32 bit pattern (1/pixel)
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;   // totalBlocks, gridX, nBlk, framePix
var<workgroup> sh: array<u32, 2048>;   // up to 8192 decoded (bitshuffled) bytes / block
fn rraw(i:u32)->u32{return (raw[i>>2u]>>((i&3u)*8u))&0xffu;}
fn rsh(i:u32)->u32{return (sh[i>>2u]>>((i&3u)*8u))&0xffu;}
fn wsh(i:u32,v:u32){let w=i>>2u;let s=(i&3u)*8u;sh[w]=(sh[w]&(~(0xffu<<s)))|((v&0xffu)<<s);}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>){
  let g = wid.y*cfg.y + wid.x; let lx = lid.x;
  if(lx==0u && g<cfg.x){
    let coff=blkMeta[g*2u]; let cend=coff+blkMeta[g*2u+1u];
    var ci=coff; var di=0u;
    loop{ if(ci>=cend){break;}
      let tok=rraw(ci); ci=ci+1u; var nlit=tok>>4u;
      if(nlit==15u){loop{let bb=rraw(ci);ci=ci+1u;nlit=nlit+bb;if(bb!=255u){break;}}}
      var k=0u; loop{if(k>=nlit){break;} wsh(di+k,rraw(ci+k)); k=k+1u;} ci=ci+nlit; di=di+nlit;
      if(ci>=cend){break;}
      let off=rraw(ci)|(rraw(ci+1u)<<8u); ci=ci+2u; var ml=4u+(tok&0xfu);
      if((tok&0xfu)==15u){loop{let bb=rraw(ci);ci=ci+1u;ml=ml+bb;if(bb!=255u){break;}}}
      var j=0u; loop{if(j>=ml){break;} wsh(di+j,rsh(di+j-off)); j=j+1u;} di=di+ml;
    }
  }
  workgroupBarrier();
  if(g>=cfg.x){return;}
  let frm = g/cfg.z; let blk = g%cfg.z;
  let pixBase = frm*cfg.w + blk*__BE__;
  let oBase = pixBase;   // 1 u32 per pixel
  for(var lg=lx; lg<__NPB__; lg=lg+64u){
    var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
    for(var b:u32=0u;b<32u;b=b+1u){
      let byte=rsh(lg + b*__NPB__); let bit=1u<<b;
      if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
      if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
      if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
      if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
    }
    let o=oBase + lg*8u;
    stack[o]=v0; stack[o+1u]=v1; stack[o+2u]=v2; stack[o+3u]=v3;
    stack[o+4u]=v4; stack[o+5u]=v5; stack[o+6u]=v6; stack[o+7u]=v7;
  }
}`;

const FUSED_F32_PIPE_CACHE = new Map<string, GPUComputePipeline>();
function getFusedF32Pipe(device: GPUDevice, blockElems: number): GPUComputePipeline {
  const code = FUSED_F32_WGSL.replace(/__NPB__/g, `${blockElems / 8}u`).replace(/__BE__/g, `${blockElems}u`);
  let p = FUSED_F32_PIPE_CACHE.get(code);
  if (!p) { p = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } }); FUSED_F32_PIPE_CACHE.set(code, p); }
  return p;
}

// Fused float32 decode job: one workgroup/block, float32 stack out (1 u32/pixel, mode 2).
function buildFusedJobF32(device: GPUDevice, spec: Bslz4Spec, preRaw?: GPUBuffer): DecodeJob {
  const { compressed, blockMeta, nFrames, nBlocksPerFrame, blockElems, detSize } = spec;
  const totalBlocks = nFrames * nBlocksPerFrame;
  const stackWords = nFrames * detSize;   // 1 u32/pixel
  let rawBuf: GPUBuffer;
  if (preRaw) { rawBuf = preRaw; }
  else {
    const rawSize = Math.ceil(compressed.byteLength / 4) * 4;
    rawBuf = device.createBuffer({ size: rawSize, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    copyWide(rawBuf.getMappedRange(), compressed);
    rawBuf.unmap();
  }
  const metaBuf = device.createBuffer({ size: blockMeta.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(metaBuf, 0, blockMeta.buffer as ArrayBuffer, blockMeta.byteOffset, blockMeta.byteLength);
  const stack = device.createBuffer({ size: stackWords * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const gx = Math.min(totalBlocks, MAX_WG), gy = Math.ceil(totalBlocks / MAX_WG);
  const cfg = uniform(device, [totalBlocks, gx, nBlocksPerFrame, detSize]);
  const pipe = getFusedF32Pipe(device, blockElems);
  const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: rawBuf } }, { binding: 1, resource: { buffer: metaBuf } },
    { binding: 2, resource: { buffer: stack } }, { binding: 3, resource: { buffer: cfg } } ] });
  return {
    stack, mode: 2,
    record(enc) { const pass = enc.beginComputePass(); pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(gx, gy); pass.end(); },
    releaseTemps() { rawBuf.destroy(); metaBuf.destroy(); cfg.destroy(); },
  };
}

// STRATEGY D (parallel LZ4 via round-based dataflow). PARITY-VERIFIED bit-exact vs the serial
// fused kernel (uint16 AND uint32, nDiff=0 on real gold04/gold06 via verifyFusedD) but it is a
// PERF REGRESSION, kept default-OFF (BSLZ4_PARALLEL) as a reference + harness for future work.
// WHY it loses: round-based dataflow does O(maxDepth x blockBytes) work (every round re-scans
// every match byte) while the serial thread-0 decode is O(blockBytes). With maxDepth ~60 that
// is ~60x more total work; spread over 64 lanes it is roughly break-even on raw ops, and the
// per-round atomics + ~60 workgroupBarriers push it 3-5x SLOWER than serial (measured: 348ms/
// 681ms uint16/uint32 per 10k-frame file vs 118ms/138ms serial). The ONLY structurally-faster
// LZ4 is warp-cooperative copy inside a SINGLE forward pass (silx-style: lanes split each match's
// byte copy as the serial cursor advances), which keeps O(blockBytes) total work - a different
// algorithm than this round-based one, not a tuning of it. Round-based cannot beat serial.
//
// All 64 lanes parse the SAME token stream redundantly (parsing is cheap: ~50 tokens/block,
// pointer chasing in registers) and then cooperate on the byte copies. The LZ4 output is a
// dataflow DAG: literal bytes have
// depth 0; a match byte at di+k reads source di-off+(k%off). Because k%off lands in the FIRST
// `off` bytes of the match's source region (NOT at di+k-off), a period-`off` run does NOT form
// a depth-ml chain - it flattens. Measured on real Arina uint16+uint32 (thousands of blocks):
// the longest dependency chain in any 8 KB block is ~60. So repeated rounds, each a full
// redundant copy of every literal+match byte guarded by "only write a byte whose every source
// is already final," reach the fixed point. The loop runs until a round resolves no byte (the
// `progress` atomic is workgroup-uniform after the post-round barrier, so the early break is in
// UNIFORM control flow) - self-terminating at maxDepth+1 for ANY depth. The per-round
// workgroupBarrier stays uniform; the trap that blanked the prior attempt (barrier inside the
// data-dependent token loop) is gone.
//
// FINALITY without a second buffer: a byte is "final" once it equals its dataflow value and
// will never change again. We track that with a per-byte DONE bitmap in shared memory (1 bit
// per output byte, 8192 bits = 256 u32). Round r: every lane walks the token list; for each
// match byte di+k it checks DONE[src]; if set, it writes the byte to sh and marks DONE[di+k].
// Literal bytes are written + marked DONE in round 0 (their source is the compressed input,
// always available). A byte already DONE is skipped. Once a whole round resolves nothing the
// fixed point is reached (max depth ~60). Each output byte is written EXACTLY ONCE into its
// known-zero shared slot via a single atomicOr (OR-with-0 is identity + commutative, so the
// only lock-free byte write in WGSL's u32-only shared memory); DONE + progress are atomics too.
//
// __NPB__ = blockElems/8 = planeBytes = groups/block; __BE__ = blockElems; __MAXROUNDS__ backstop.
const FUSED_D_WGSL = `
@group(0) @binding(0) var<storage,read> raw: array<u32>;
@group(0) @binding(1) var<storage,read> blkMeta: array<u32>;
@group(0) @binding(2) var<storage,read_write> stack: array<u32>;
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;   // totalBlocks, gridX, nBlk, framePix
var<workgroup> sh: array<atomic<u32>, 2048>;   // decoded bytes, atomic (race-free byte writes)
var<workgroup> done: array<atomic<u32>, 256>;  // 1 bit per output byte: is it final?
var<workgroup> progress: atomic<u32>;          // did this round resolve any byte? (convergence)
fn rraw(i:u32)->u32{return (raw[i>>2u]>>((i&3u)*8u))&0xffu;}
fn rsh(i:u32)->u32{return (atomicLoad(&sh[i>>2u])>>((i&3u)*8u))&0xffu;}
// each output byte is written EXACTLY once into its KNOWN-ZERO slot via a single atomicOr:
// OR-with-0 is identity + commutative, so concurrent different-byte-same-word writes are
// race-free AND order-independent (the only correct lock-free byte write in WGSL).
fn wsh(i:u32,v:u32){atomicOr(&sh[i>>2u], (v&0xffu)<<((i&3u)*8u));}
fn isDone(i:u32)->bool{return ((atomicLoad(&done[i>>5u])>>(i&31u))&1u)!=0u;}
fn setDone(i:u32){atomicOr(&done[i>>5u], 1u<<(i&31u));}
fn clip8(v:u32)->u32{return select(v,255u,v>255u);}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>){
  let g = wid.y*cfg.y + wid.x; let lx = lid.x;
  let live = g < cfg.x;
  let coff = select(0u, blkMeta[g*2u], live);
  let cend = select(0u, coff + blkMeta[g*2u+1u], live);
  // zero sh (so atomicOr-into-zero is exact) and the DONE bitmap. Uniform counted loops.
  for(var w=lx; w<2048u; w=w+64u){ atomicStore(&sh[w], 0u); }
  for(var w=lx; w<256u;  w=w+64u){ atomicStore(&done[w], 0u); }
  workgroupBarrier();
  // ---- ROUND 0: write + mark all LITERAL bytes (source = compressed input, always ready).
  // Owner lane (di+k)&63==lx writes each literal byte exactly once. Match bytes deferred.
  {
    var ci=coff; var di=0u;
    loop{ if(ci>=cend){break;}
      let tok=rraw(ci); ci=ci+1u; var nlit=tok>>4u;
      if(nlit==15u){loop{let bb=rraw(ci);ci=ci+1u;nlit=nlit+bb;if(bb!=255u){break;}}}
      var k=0u; loop{ if(k>=nlit){break;}
        let o=di+k; if((o&63u)==lx){ wsh(o, rraw(ci+k)); setDone(o); }
        k=k+1u; }
      ci=ci+nlit; di=di+nlit;
      if(ci>=cend){break;}
      let off=rraw(ci)|(rraw(ci+1u)<<8u); ci=ci+2u; var ml=4u+(tok&0xfu);
      if((tok&0xfu)==15u){loop{let bb=rraw(ci);ci=ci+1u;ml=ml+bb;if(bb!=255u){break;}}}
      di=di+ml;   // skip match output this round
    }
  }
  workgroupBarrier();
  // ---- ROUNDS 1..MAXROUNDS: resolve match bytes whose source is DONE. FIXED trip count so the
  // two per-round workgroupBarriers sit in UNIFORM control flow (WGSL forbids a barrier whose
  // reachability depends on a non-uniform value - an atomic load is always deemed non-uniform,
  // so an early break on progress is illegal). Instead the expensive token walk is gated by
  // 'active' (a non-uniform branch with NO barrier inside, which IS legal): once a whole round
  // resolves nothing, progress stays 0 forever (resolution is monotonic), so every later round
  // is just two cheap barriers. MAXROUNDS bounds the deepest LZ4 chain (~60 on real Arina).
  var working = 1u;
  for(var r=0u; r<__MAXROUNDS__; r=r+1u){
    if(lx==0u){ atomicStore(&progress, 0u); }
    workgroupBarrier();
    if(working != 0u){
      var ci=coff; var di=0u;
      loop{ if(ci>=cend){break;}
        let tok=rraw(ci); ci=ci+1u; var nlit=tok>>4u;
        if(nlit==15u){loop{let bb=rraw(ci);ci=ci+1u;nlit=nlit+bb;if(bb!=255u){break;}}}
        ci=ci+nlit; di=di+nlit;   // literals already done in round 0
        if(ci>=cend){break;}
        let off=rraw(ci)|(rraw(ci+1u)<<8u); ci=ci+2u; var ml=4u+(tok&0xfu);
        if((tok&0xfu)==15u){loop{let bb=rraw(ci);ci=ci+1u;ml=ml+bb;if(bb!=255u){break;}}}
        var k=0u; loop{ if(k>=ml){break;}
          let o=di+k;
          if((o&63u)==lx && !isDone(o)){
            let src=di-off+(k%off);
            if(isDone(src)){ wsh(o, rsh(src)); setDone(o); atomicStore(&progress, 1u); }
          }
          k=k+1u; }
        di=di+ml;
      }
    }
    workgroupBarrier();
    working = atomicLoad(&progress);   // 0 => converged; gate (not branch-to-barrier) next round
  }
  if(!live){return;}
  // ---- inverse bitshuffle + uint8 pack, identical to FUSED_U16U8 (OR-fold high planes).
  let frm = g/cfg.z; let blk = g%cfg.z;
  let pixBase = frm*cfg.w + blk*__BE__;
  let oBase = (pixBase>>3u)*2u;
  for(var lg=lx; lg<__NPB__; lg=lg+64u){
    var hi:u32=0u;
    for(var b:u32=8u;b<__NBITS__;b=b+1u){ hi = hi | rsh(lg + b*__NPB__); }
    var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
    for(var b:u32=0u;b<8u;b=b+1u){
      let byte=rsh(lg + b*__NPB__); let bit=1u<<b;
      if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
      if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
      if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
      if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
    }
    let o=oBase + lg*2u;
    stack[o]=select(v0,255u,(hi&1u)!=0u)|(select(v1,255u,(hi&2u)!=0u)<<8u)|(select(v2,255u,(hi&4u)!=0u)<<16u)|(select(v3,255u,(hi&8u)!=0u)<<24u);
    stack[o+1u]=select(v4,255u,(hi&16u)!=0u)|(select(v5,255u,(hi&32u)!=0u)<<8u)|(select(v6,255u,(hi&64u)!=0u)<<16u)|(select(v7,255u,(hi&128u)!=0u)<<24u);
  }
}`;

const FUSED_D_PIPE_CACHE = new Map<string, GPUComputePipeline>();
function getFusedDPipe(device: GPUDevice, blockElems: number, nbits: number): GPUComputePipeline {
  const npb = blockElems / 8;
  const code = FUSED_D_WGSL.replace(/__NPB__/g, `${npb}u`).replace(/__BE__/g, `${blockElems}u`).replace(/__NBITS__/g, `${nbits}u`).replace(/__MAXROUNDS__/g, `256u`);
  let p = FUSED_D_PIPE_CACHE.get(code);
  if (!p) { p = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } }); FUSED_D_PIPE_CACHE.set(code, p); }
  return p;
}

const FUSED_PIPE_CACHE = new Map<string, GPUComputePipeline>();
function getFusedPipe(device: GPUDevice, blockElems: number, nbits: number): GPUComputePipeline {
  const npb = blockElems / 8;
  const code = FUSED_U16U8_WGSL.replace(/__NPB__/g, `${npb}u`).replace(/__BE__/g, `${blockElems}u`).replace(/__NBITS__/g, `${nbits}u`);
  let p = FUSED_PIPE_CACHE.get(code);
  if (!p) { p = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } }); FUSED_PIPE_CACHE.set(code, p); }
  return p;
}

// One fused decode job: upload the raw bytes + block table, dispatch one workgroup per
// block (2D grid for the >65535 case). No interBuf. uint16-source -> uint8-output only.
function buildFusedJob(device: GPUDevice, spec: Bslz4Spec, srcDtype: "uint16" | "uint32", preRaw?: GPUBuffer): DecodeJob {
  const { compressed, blockMeta, nFrames, nBlocksPerFrame, blockElems, detSize } = spec;
  const nbits = srcDtype === "uint32" ? 32 : 16;
  const totalBlocks = nFrames * nBlocksPerFrame;
  const stackWords = Math.ceil(nFrames * detSize / 4);
  // raw buffer: either pre-uploaded via the staging pool (batch path) or, for the single
  // decode path, mappedAtCreation here.
  let rawBuf: GPUBuffer;
  if (preRaw) { rawBuf = preRaw; }
  else {
    const rawSize = Math.ceil(compressed.byteLength / 4) * 4;
    rawBuf = device.createBuffer({ size: rawSize, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    copyWide(rawBuf.getMappedRange(), compressed);
    rawBuf.unmap();
  }
  const metaBuf = device.createBuffer({ size: blockMeta.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(metaBuf, 0, blockMeta.buffer as ArrayBuffer, blockMeta.byteOffset, blockMeta.byteLength);
  const stack = device.createBuffer({ size: stackWords * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const gx = Math.min(totalBlocks, MAX_WG), gy = Math.ceil(totalBlocks / MAX_WG);
  const cfg = uniform(device, [totalBlocks, gx, nBlocksPerFrame, detSize]);
  const pipe = getFusedPipe(device, blockElems, nbits);
  const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: rawBuf } }, { binding: 1, resource: { buffer: metaBuf } },
    { binding: 2, resource: { buffer: stack } }, { binding: 3, resource: { buffer: cfg } } ] });
  return {
    stack, mode: 1,
    record(enc) { const pass = enc.beginComputePass(); pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(gx, gy); pass.end(); },
    releaseTemps() { rawBuf.destroy(); metaBuf.destroy(); cfg.destroy(); },
  };
}

// Strategy-D fused job: identical I/O contract to buildFusedJob (same bindings, same stack
// layout, same dispatch grid) but uses the round-based PARALLEL LZ4 kernel. The kernel runs
// until the per-round `progress` atomic reports no byte resolved (convergence early-exit), so
// it self-terminates at maxDepth+1 rounds for ANY LZ4 dependency depth. __MAXROUNDS__ (4096) is
// only a runaway backstop, far above the ~60 worst case on real Arina uint16+uint32 blocks.
function buildFusedJobD(device: GPUDevice, spec: Bslz4Spec, srcDtype: "uint16" | "uint32", preRaw?: GPUBuffer): DecodeJob {
  const { compressed, blockMeta, nFrames, nBlocksPerFrame, blockElems, detSize } = spec;
  const nbits = srcDtype === "uint32" ? 32 : 16;
  const totalBlocks = nFrames * nBlocksPerFrame;
  const stackWords = Math.ceil(nFrames * detSize / 4);
  let rawBuf: GPUBuffer;
  if (preRaw) { rawBuf = preRaw; }
  else {
    const rawSize = Math.ceil(compressed.byteLength / 4) * 4;
    rawBuf = device.createBuffer({ size: rawSize, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    copyWide(rawBuf.getMappedRange(), compressed);
    rawBuf.unmap();
  }
  const metaBuf = device.createBuffer({ size: blockMeta.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(metaBuf, 0, blockMeta.buffer as ArrayBuffer, blockMeta.byteOffset, blockMeta.byteLength);
  const stack = device.createBuffer({ size: stackWords * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const gx = Math.min(totalBlocks, MAX_WG), gy = Math.ceil(totalBlocks / MAX_WG);
  const cfg = uniform(device, [totalBlocks, gx, nBlocksPerFrame, detSize]);
  const pipe = getFusedDPipe(device, blockElems, nbits);
  const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: rawBuf } }, { binding: 1, resource: { buffer: metaBuf } },
    { binding: 2, resource: { buffer: stack } }, { binding: 3, resource: { buffer: cfg } } ] });
  return {
    stack, mode: 1,
    record(enc) { const pass = enc.beginComputePass(); pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(gx, gy); pass.end(); },
    releaseTemps() { rawBuf.destroy(); metaBuf.destroy(); cfg.destroy(); },
  };
}

// Bit-exact parity + GPU-time check for Strategy D vs the serial Fallback, on ONE spec.
// Runs each kernel in its OWN submit (so the wall time around onSubmittedWorkDone is the
// isolated kernel cost, upload excluded - raw is pre-uploaded here), reads back both packed
// uint8 stacks, and reports the exact byte diff. No tolerance: D must be byte-identical to
// Fallback. Returns null if WebGPU is unavailable. Console-driven verify only.
export async function verifyFusedD(spec: Bslz4Spec, srcDtype: "uint16" | "uint32"): Promise<{
  nBytes: number; nDiff: number; maxDiff: number; firstDiffAt: number; meanFrame0: number;
  dErr: string | null; fGpuMs: number; dGpuMs: number;
} | null> {
  const device = await getGPUDevice();
  if (!device) return null;
  const stackWords = Math.ceil(spec.nFrames * spec.detSize / 4);
  const runReadback = async (job: DecodeJob): Promise<{ data: Uint8Array; ms: number }> => {
    const warm = device.createCommandEncoder(); job.record(warm); device.queue.submit([warm.finish()]); await device.queue.onSubmittedWorkDone();
    const enc = device.createCommandEncoder(); job.record(enc);
    const rb = device.createBuffer({ size: stackWords * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(job.stack, 0, rb, 0, stackWords * 4);
    const t0 = performance.now(); device.queue.submit([enc.finish()]); await device.queue.onSubmittedWorkDone();
    const ms = performance.now() - t0;
    await rb.mapAsync(GPUMapMode.READ);
    const data = new Uint8Array(rb.getMappedRange().slice(0)); rb.unmap(); rb.destroy();
    job.releaseTemps(); job.stack.destroy();
    return { data, ms };
  };
  const fRes = await runReadback(buildFusedJob(device, spec, srcDtype));
  device.pushErrorScope("validation");
  const dJob = buildFusedJobD(device, spec, srcDtype);
  const dRes = await runReadback(dJob);
  const dErr = await device.popErrorScope();
  const fBytes = fRes.data, dBytes = dRes.data;
  const n = Math.min(fBytes.length, dBytes.length);
  let nDiff = 0, maxDiff = 0, firstDiffAt = -1;
  for (let i = 0; i < n; i++) { const d = Math.abs(fBytes[i] - dBytes[i]); if (d) { nDiff++; if (d > maxDiff) maxDiff = d; if (firstDiffAt < 0) firstDiffAt = i; } }
  let sum = 0; const f0 = Math.min(spec.detSize, n); for (let i = 0; i < f0; i++) sum += dBytes[i];
  return { nBytes: n, nDiff, maxDiff, firstDiffAt, meanFrame0: sum / f0, dErr: dErr ? dErr.message : null, fGpuMs: fRes.ms, dGpuMs: dRes.ms };
}

export interface Bslz4Spec {
  compressed: Uint8Array;        // concatenated per-frame bslz4 chunks (frame-padded to 4B)
  blockMeta: Uint32Array;        // [coff,clen] per (frame,block), absolute byte offsets
  nFrames: number;
  nBlocksPerFrame: number;
  blockElems: number;            // elements per bitshuffle block (e.g. 4096 for uint16/8192B)
  detSize: number;               // detector pixels per frame (e.g. 192*192)
}

// Compute pipelines are independent of the data (only of the pass2 template), so compile
// them ONCE and reuse across every chunk + dataset - recompiling per chunk was wasteful.
const PIPE_CACHE = new Map<string, { p1: GPUComputePipeline; p2: GPUComputePipeline }>();
function getPipes(device: GPUDevice, srcDtype: "uint8" | "uint16" | "float32", u8: boolean, nBlocksPerFrame: number, detSize: number) {
  const pass2tpl = srcDtype === "float32" ? PASS2_F32_WGSL : srcDtype === "uint8" ? PASS2_U8SRC_WGSL : (u8 ? PASS2_U8_WGSL : PASS2_WGSL);
  const pass2 = pass2tpl.replace("__NBLK__", `${nBlocksPerFrame}u`).replace("__FRAMEPIX__", `${detSize}u`);
  let pipes = PIPE_CACHE.get(pass2);
  if (!pipes) {
    pipes = {
      p1: device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: PASS1_WGSL }), entryPoint: "main" } }),
      p2: device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: pass2 }), entryPoint: "main" } }),
    };
    PIPE_CACHE.set(pass2, pipes);
  }
  return pipes;
}

interface DecodeJob { stack: GPUBuffer; mode: number; record(enc: GPUCommandEncoder): void; releaseTemps(): void; }

// Reusable MAP_WRITE staging pool. mappedAtCreation allocates + zero-inits host-visible
// memory every call (~3.9 GB/s, ~1.7s for a 6.6 GB dataset); a persistent pool pays that
// once, then map/unmap-reuses, so the per-load upload becomes just the (fast, wide) CPU copy
// + a GPU-internal copyBufferToBuffer. Cleared on device loss.
let STAGING: GPUBuffer[] = [];
let STAGING_BYTES = 0;
function ensureStaging(device: GPUDevice, count: number, bytes: number): void {
  if (STAGING.length >= count && STAGING_BYTES >= bytes) return;
  STAGING.forEach((b) => b.destroy());
  STAGING_BYTES = Math.max(bytes, STAGING_BYTES);
  STAGING = Array.from({ length: count }, () => device.createBuffer({ size: STAGING_BYTES, usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC }));
}
onGPULost(() => { STAGING = []; STAGING_BYTES = 0; });

// Upload each spec's compressed bytes into a plain STORAGE buffer via the staging pool:
// map a pooled staging buffer (no per-load alloc), wide-copy the bytes in, then a GPU copy
// to the STORAGE buffer the decoder reads. Returns the raw buffers (caller destroys them).
async function uploadViaStaging(device: GPUDevice, specs: Bslz4Spec[]): Promise<GPUBuffer[]> {
  const align4 = (n: number) => Math.ceil(n / 4) * 4;
  const maxBytes = specs.reduce((m, s) => Math.max(m, align4(s.compressed.byteLength)), 0);
  ensureStaging(device, specs.length, maxBytes);
  const rawBufs = specs.map((s) => device.createBuffer({ size: align4(s.compressed.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }));
  await Promise.all(specs.map(async (s, i) => {
    const sz = align4(s.compressed.byteLength);
    await STAGING[i].mapAsync(GPUMapMode.WRITE, 0, sz);   // reused buffer: maps without re-alloc
    copyWide(STAGING[i].getMappedRange(0, sz), s.compressed);
    STAGING[i].unmap();
  }));
  const enc = device.createCommandEncoder();
  specs.forEach((s, i) => enc.copyBufferToBuffer(STAGING[i], 0, rawBufs[i], 0, align4(s.compressed.byteLength)));
  device.queue.submit([enc.finish()]);   // ordered before the decode submit; next group's mapAsync waits on it
  return rawBufs;
}

// Build (but don't submit) a decode job: allocates the GPU buffers, uploads the compressed
// bytes + block table, and returns a recorder that appends the two compute passes to a
// shared encoder. Batching N jobs into ONE submit + ONE await lets the GPU pipeline the
// chunks instead of draining between each (the per-chunk await was the decode bottleneck).
function buildDecodeJob(device: GPUDevice, spec: Bslz4Spec, dtype: "uint8" | "uint16" | "float32", srcDtype: "uint8" | "uint16" | "float32"): DecodeJob {
  const { compressed, blockMeta, nFrames, nBlocksPerFrame, blockElems, detSize } = spec;
  const f32 = srcDtype === "float32";
  const srcBytes = srcDtype === "uint8" ? 1 : f32 ? 4 : 2;
  const blockBytes = blockElems * srcBytes;
  const planeBytes = blockElems / 8;
  const totalBlocks = nFrames * nBlocksPerFrame;
  const totalElems = nFrames * detSize;
  const u8 = !f32 && (dtype === "uint8" || srcDtype === "uint8");
  // float32: 1 u32/pixel (the IEEE-754 bit pattern). uint16: 2 px/u32. uint8: 4 px/u32.
  const stackWords = f32 ? totalElems : u8 ? Math.ceil(totalElems / 4) : totalElems / 2;
  // Upload the compressed bytes via a mapped-at-creation buffer: writing straight into the
  // GPU-visible mapped range is a single copy, vs writeBuffer's internal CPU staging copy -
  // roughly 2x the host->device throughput, and uploading 7.5 GB/dataset is the decode floor.
  const rawSize = Math.ceil(compressed.byteLength / 4) * 4;
  const rawBuf = device.createBuffer({ size: rawSize, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
  new Uint8Array(rawBuf.getMappedRange()).set(compressed);
  rawBuf.unmap();
  const interBuf = device.createBuffer({ size: totalBlocks * blockBytes, usage: GPUBufferUsage.STORAGE });
  const metaBuf = device.createBuffer({ size: blockMeta.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(metaBuf, 0, blockMeta.buffer as ArrayBuffer, blockMeta.byteOffset, blockMeta.byteLength);
  const stack = device.createBuffer({ size: stackWords * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const cfg1 = uniform(device, [totalBlocks, blockBytes, 0, 0]);
  const nGroups = totalElems / 8;
  const p2wg = Math.ceil(nGroups / 64), gx = Math.min(p2wg, MAX_WG), gy = Math.ceil(p2wg / MAX_WG);
  const cfg2 = uniform(device, [nGroups, gx * 64, blockElems, planeBytes]);
  const { p1, p2 } = getPipes(device, srcDtype, u8, nBlocksPerFrame, detSize);
  const bg1 = device.createBindGroup({ layout: p1.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: rawBuf } }, { binding: 1, resource: { buffer: interBuf } },
    { binding: 2, resource: { buffer: metaBuf } }, { binding: 3, resource: { buffer: cfg1 } } ] });
  const bg2 = device.createBindGroup({ layout: p2.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: interBuf } }, { binding: 1, resource: { buffer: stack } },
    { binding: 2, resource: { buffer: cfg2 } } ] });
  return {
    stack, mode: f32 ? 2 : u8 ? 1 : 0,
    record(enc) {
      const pa = enc.beginComputePass(); pa.setPipeline(p1); pa.setBindGroup(0, bg1); pa.dispatchWorkgroups(Math.ceil(totalBlocks / 64)); pa.end();
      const pb = enc.beginComputePass(); pb.setPipeline(p2); pb.setBindGroup(0, bg2); pb.dispatchWorkgroups(gx, gy); pb.end();
    },
    releaseTemps() { rawBuf.destroy(); interBuf.destroy(); metaBuf.destroy(); cfg1.destroy(); cfg2.destroy(); },
  };
}

// Decode N bslz4 specs into N packed GPU stack buffers, batching the GPU work in groups so
// at most `groupSize` chunks' transient buffers (raw + intermediate) are live at once - one
// submit + one await per group lets the GPU overlap upload and compute across chunks.
export async function decodeBslz4Batch(specs: Bslz4Spec[], dtype: "uint8" | "uint16" | "float32" = "uint8", srcDtype: "uint8" | "uint16" | "uint32" | "float32" = "uint16", groupSize = 14): Promise<{ device: GPUDevice; buffers: GPUBuffer[]; mode: number } | null> {
  const device = await getGPUDevice();
  if (!device) return null;
  const f32 = srcDtype === "float32";
  const fused = dtype === "uint8" && (srcDtype === "uint16" || srcDtype === "uint32");   // uint16/uint32 -> uint8 fast path
  const buffers: GPUBuffer[] = []; let mode = 0;
  for (let g = 0; g < specs.length; g += groupSize) {
    const groupSpecs = specs.slice(g, g + groupSize);
    // Fused paths (uint8 + float32) upload through the reused staging pool (no per-load
    // mappedAtCreation alloc); the non-fused path keeps its own mappedAtCreation upload.
    const raws = (fused || f32) ? await uploadViaStaging(device, groupSpecs) : null;
    const fusedBuild = BSLZ4_PARALLEL ? buildFusedJobD : buildFusedJob;
    const jobs = groupSpecs.map((s, i) => f32 ? buildFusedJobF32(device, s, raws![i]) : fused ? fusedBuild(device, s, srcDtype as "uint16"|"uint32", raws![i]) : buildDecodeJob(device, s, dtype, srcDtype as "uint8"|"uint16"|"float32"));
    const enc = device.createCommandEncoder();
    for (const j of jobs) j.record(enc);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    for (const j of jobs) { j.releaseTemps(); buffers.push(j.stack); mode = j.mode; }
  }
  return { device, buffers, mode };
}

// Decode a bslz4 stack to a packed GPU buffer ([scanPos][detPixel]). dtype "uint8"
// (clip 0-255, 4 px/u32, offline default - half the memory) or "uint16" (lossless,
// 2 px/u32). Layout matches Show4DSTEMCompute.sample() for that mode exactly.
// Returns null if WebGPU is unavailable. Throws (validation) only on misuse.
export async function decodeBslz4ToStack(spec: Bslz4Spec, dtype: "uint8" | "uint16" | "float32" = "uint8", srcDtype: "uint8" | "uint16" | "uint32" | "float32" = "uint16"): Promise<{ device: GPUDevice; buffer: GPUBuffer; mode: number } | null> {
  const device = await getGPUDevice();
  if (!device) return null;
  const fusedOk = dtype === "uint8" && (srcDtype === "uint16" || srcDtype === "uint32");
  const job = srcDtype === "float32"
    ? buildFusedJobF32(device, spec)
    : fusedOk
    ? (BSLZ4_PARALLEL ? buildFusedJobD(device, spec, srcDtype as "uint16"|"uint32", undefined) : buildFusedJob(device, spec, srcDtype as "uint16"|"uint32"))
    : buildDecodeJob(device, spec, dtype, srcDtype as "uint8"|"uint16"|"float32");
  const enc = device.createCommandEncoder();
  job.record(enc);
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  job.releaseTemps();
  return { device, buffer: job.stack, mode: job.mode };
}

function uniform(device: GPUDevice, vals: number[]): GPUBuffer {
  const b = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(b, 0, new Uint32Array(vals).buffer);
  return b;
}
