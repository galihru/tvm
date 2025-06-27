types/pdf-parse.d.ts
declare module "pdf-parse" {
  export interface PdfParseData {
    text: string;
    numpages: number;
    info: any;
    metadata: any;
    version: string;
  }
  function pdf(data: Buffer | Uint8Array): Promise<PdfParseData>;
  export = pdf;
}
