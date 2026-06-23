declare module "jsfive" {
  export class File {
    constructor(buffer: ArrayBuffer | Uint8Array | DataView | unknown, filename?: string);
    get(path: string): unknown;
    keys(): string[];
  }
}
