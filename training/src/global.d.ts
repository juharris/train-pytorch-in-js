// Re-export some stuff to help with with writing TypeScript code.
// I tried this so many ways and this seems like the most concise way.
// This is hard to maintain but it's the only way that I could figure out to get it working.
declare module ort {
    export declare namespace InferenceSession {
        export type OnnxValueMapType = import('./ort-decl/inference-session').InferenceSession.OnnxValueMapType
        export type ReturnType = import('./ort-decl/inference-session').InferenceSession.ReturnType
    }
    export const InferenceSession: import('./ort-decl/inference-session').InferenceSessionFactory
    export type InferenceSession = import('./ort-decl/inference-session').InferenceSession
   
    export const Tensor: import('./ort-decl/tensor').TensorConstructor
    export type Tensor = import('./ort-decl/tensor').Tensor
}
