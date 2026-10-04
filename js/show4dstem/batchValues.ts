/** Exact raw-count transport; normalization is a derived display operation. */
export interface ResidentBatchInfo {
  validation_only?: boolean;
  gpu_validate?: boolean;
  gpu_log?: boolean;
  shape?: number[];
  generation?: string;
  bytes?: number;
  transport?: string;
  received_performance_ms?: number;
  worker_superseded?: number;
  validate_display?: boolean;
  request_id?: number;
  detector_mask_sha256?: string;
  center?: number[];
  inner?: number;
  outer?: number;
  dtype?: string;
  mask_area?: number;
  normalization_offset?: number;
}

export type ResidentScalarType = "<f4" | "<u2" | "<u4";
export type ResidentValues = Float32Array | Uint16Array | Uint32Array;

/** Legacy display buffers omit metadata; explicit scientific codes are strict. */
export function residentScalarType(info?: ResidentBatchInfo): ResidentScalarType {
  const dtype = info?.dtype === undefined ? "<f4" : info.dtype;
  if (dtype !== "<f4" && dtype !== "<u2" && dtype !== "<u4") {
    throw new Error(`Unsupported resident scalar type ${JSON.stringify(dtype)}.`);
  }
  return dtype;
}

/** Borrow the exact count section for readouts/export, without normalizing it. */
export function residentRawValues(bytes: DataView, info?: ResidentBatchInfo): ResidentValues {
  const dtype = residentScalarType(info);
  const itemBytes = dtype === "<u2" ? 2 : 4;
  const countBytes = info?.normalization_offset ?? bytes.byteLength;
  if (!Number.isSafeInteger(countBytes) || countBytes < 0 || countBytes > bytes.byteLength ||
      countBytes % itemBytes !== 0 || bytes.byteOffset % itemBytes !== 0) {
    throw new Error("Resident count section has invalid byte bounds or alignment.");
  }
  if (info?.bytes !== undefined && info.bytes !== bytes.byteLength) {
    throw new Error("Resident metadata does not match the complete payload byte length.");
  }
  if (info?.shape !== undefined) {
    if (!Array.isArray(info.shape) || info.shape.length === 0 ||
        !info.shape.every(size => Number.isSafeInteger(size) && size > 0) ||
        info.shape.reduce((size, next) => size * next, 1) !== countBytes / itemBytes) {
      throw new Error("Resident scientific shape does not match its count section.");
    }
  }
  if (info?.normalization_offset !== undefined &&
      (dtype !== "<u2" || (bytes.byteOffset + countBytes) % 4 !== 0 ||
       bytes.byteLength !== countBytes + 65536 * 4)) {
    throw new Error("Only uint16 batches may carry one complete normalization table.");
  }
  const length = countBytes / itemBytes;
  if (dtype === "<u2") return new Uint16Array(bytes.buffer, bytes.byteOffset, length);
  if (dtype === "<u4") return new Uint32Array(bytes.buffer, bytes.byteOffset, length);
  return new Float32Array(bytes.buffer, bytes.byteOffset, length);
}

export function residentDivisor(info?: ResidentBatchInfo): number {
  if (residentScalarType(info) === "<f4") return 1;
  const divisor = info?.mask_area;
  if (divisor === undefined || !Number.isFinite(divisor) || divisor < 0) {
    throw new Error("Exact count batches require a finite nonnegative detector mask area.");
  }
  return divisor;
}

/** Materialize display values only for CPU fallback or visible summaries. */
export function residentDisplayValues(bytes: DataView, info?: ResidentBatchInfo): Float32Array {
  const counts = residentRawValues(bytes, info);
  if (counts instanceof Float32Array) return counts;
  const divisor = residentDivisor(info);
  const display = new Float32Array(counts.length);
  if (divisor === 0) return display;
  const table = info?.normalization_offset === undefined ? null :
    new Float32Array(bytes.buffer, bytes.byteOffset + info.normalization_offset, 65536);
  for (let i = 0; i < counts.length; i++) display[i] = table ? table[counts[i]] : counts[i] / divisor;
  return display;
}

/** Match the widget CPU's Float32Array assignment of Math.log1p exactly. */
export function residentLogTable(bytes: DataView, offset: number): Float32Array<ArrayBuffer> {
  const normalized=new Float32Array(bytes.buffer,bytes.byteOffset+offset,65536);
  const logarithms=new Float32Array(65536);
  for(let i=0;i<normalized.length;i++)logarithms[i]=Math.log1p(Math.max(0,normalized[i]));
  return logarithms;
}
