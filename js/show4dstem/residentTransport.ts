import * as React from "react";
import { useModel } from "@anywidget/react";
import { residentRawValues, residentScalarType, residentDivisor, type ResidentBatchInfo, type ResidentScalarType } from "./batchValues";
import { residentSocketWorker } from "./residentSocketWorker";
import { CompareBatchWebGL } from "./batchCanvasWebGL";

interface StreamConfig {
  url?: string;
  enabled?: boolean;
  generation?: string;
}
interface BatchFrame { bytes: DataView; info: ResidentBatchInfo }

/** Receive exact batches independently of main-thread pointer and paint work. */
export function useResidentTransport(
  model: ReturnType<typeof useModel>,
  config: StreamConfig | undefined,
  fallbackBytes: DataView,
  fallbackInfo: ResidentBatchInfo | undefined,
) {
  const [frame, setFrame] = React.useState<BatchFrame | null>(null);
  React.useEffect(() => {
    if (!fallbackInfo?.request_id) setFrame(null);
  }, [fallbackInfo]);
  React.useEffect(() => {
    if (!config?.enabled || !config.url) { setFrame(null); return; }
    const url = URL.createObjectURL(new Blob([`(${residentSocketWorker.toString()})()`],{type:"text/javascript"}));
    let worker: Worker;
    let validator: CompareBatchWebGL | null = null;
    let validatorDtype: ResidentScalarType | null = null;
    let legacyValidator: CompareBatchWebGL | null = null;
    try { worker = new Worker(url); }
    catch(error) {
      URL.revokeObjectURL(url);
      model.send({type:"resident_stream_error",error:String(error)});
      return;
    }
    worker.onmessage = event => {
      const message = event.data;
      if (message.type === "frame") {
        const info = message.info as ResidentBatchInfo & {received_epoch_ms:number};
        let dtype: ResidentScalarType;
        try {
          dtype = residentScalarType(info);
          residentRawValues(new DataView(message.bytes), info);
          residentDivisor(info);
        } catch (error) {
          model.send({type:"resident_stream_error",request_id:info.request_id,error:String(error)});
          worker.postMessage({type:"adopted",request_id:info.request_id});
          return; // Drop malformed input while retaining the last complete valid frame.
        }
        if (info.validation_only) {
          const bytes = message.bytes as ArrayBuffer;
          const hash = async (buffer:ArrayBuffer) => {
            const digest = await crypto.subtle.digest("SHA-256",buffer);
            return Array.from(new Uint8Array(digest),value => value.toString(16).padStart(2,"0")).join("");
          };
          void (async () => {
            const gpu: Record<string,unknown> = {};
            if (info.gpu_validate) {
              if (JSON.stringify(info.shape) !== "[66,512,512]") throw new Error("Validation requires all66 native images.");
              if (validatorDtype !== dtype) { validator?.destroy(); validator=null; }
              if (!validator) {
                const canvas = document.createElement("canvas");canvas.width=1;canvas.height=1;
                validator=new CompareBatchWebGL(canvas,66,512,512,dtype);validatorDtype=dtype;
              }
              const activeValidator=validator;
              const rectangles=new Float32Array(66*4);for(let i=0;i<66;i++){rectangles[i*4+2]=1;rectangles[i*4+3]=1;}
              const timing=await activeValidator.render(new DataView(bytes),rectangles,Uint32Array.from({length:66},(_,i)=>i),new Uint8Array(768),
                {log:Boolean(info.gpu_log),auto:false,min:0,max:100,zoom:1,panX:0,panY:0,smooth:false,dtype,divisor:residentDivisor(info),normalizationOffset:info.normalization_offset});
              if (validator !== activeValidator) return;
              const arrays=activeValidator.readScientificArrays();
              const [countsHash,valuesHash]=await Promise.all([hash(arrays.counts.buffer),hash(arrays.values.buffer)]);
              Object.assign(gpu,{gpu_counts_sha256:countsHash,gpu_values_sha256:valuesHash,
                gpu_ranges:activeValidator.readRanges(),gpu_shape:[66,512,512],gpu_adapter:activeValidator.adapter,validation_timing:timing});
              if (info.gpu_log && dtype === "<u2") {
                if (!legacyValidator) {
                  const canvas=document.createElement("canvas");canvas.width=1;canvas.height=1;
                  legacyValidator=new CompareBatchWebGL(canvas,66,512,512,"<u2",false);
                }
                const previous=legacyValidator;
                const legacyTiming=await previous.render(new DataView(bytes),rectangles,Uint32Array.from({length:66},(_,i)=>i),new Uint8Array(768),
                  {log:true,auto:false,min:0,max:100,zoom:1,panX:0,panY:0,smooth:false,dtype,divisor:residentDivisor(info),normalizationOffset:info.normalization_offset});
                if (previous!==legacyValidator) return;
                Object.assign(gpu,{legacy_ranges:previous.readRanges(),legacy_timing:legacyTiming});
              }
            }
            const sha256 = await hash(bytes);
            worker.postMessage({type:"validation",request_id:info.request_id,
              generation:info.generation,sha256,byte_length:bytes.byteLength,...gpu});
          })().catch(error => model.send({type:"resident_stream_error",error:String(error)}));
        } else {
          validator?.destroy();validator=null;legacyValidator?.destroy();legacyValidator=null;
          setFrame({bytes:new DataView(message.bytes),info:{...info,
            transport:"websocket-worker",worker_superseded:message.superseded,
            received_performance_ms:info.received_epoch_ms-performance.timeOrigin}});
        }
        // The transferred ArrayBuffer now belongs to this browser thread.
        // At most one further frame can be in transit; the worker retains
        // only the newest pending complete batch while this thread is busy.
        worker.postMessage({type:"adopted",request_id:info.request_id});
      } else if (message.type === "disconnected") setFrame(null);
      else if (message.type === "error") model.send({type:"resident_stream_error",error:message.error});
    };
    worker.onerror = error => model.send({type:"resident_stream_error",error:error.message});
    worker.postMessage({type:"connect",config:{url:config.url,generation:config.generation}});
    return () => {
      worker.postMessage({type:"dispose"});
      worker.terminate();
      validator?.destroy();
      legacyValidator?.destroy();
      URL.revokeObjectURL(url);
    };
  }, [model,config?.enabled,config?.url,config?.generation]);
  return frame ?? {bytes: fallbackBytes,info: fallbackInfo};
}
