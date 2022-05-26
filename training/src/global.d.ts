// Re-export some stuff to help with with writing TypeScript code.
// I tried this so many ways and this seems like the most concise way.
// This is hard to maintain but it's the only way that I could figure out to get it working.
declare module ort {
	export declare namespace InferenceSession {
		export type OnnxValueMapType = import('./ort-decl').InferenceSession.OnnxValueMapType
		export type ReturnType = import('./ort-decl').InferenceSession.ReturnType
	}
	export const InferenceSession: import('./ort-decl').InferenceSessionFactory
	export type InferenceSession = import('./ort-decl').InferenceSession

	export type OnnxValue = import('./ort-decl').OnnxValue

	export const Tensor: import('./ort-decl').TensorConstructor
	export type Tensor = import('./ort-decl').Tensor
}
