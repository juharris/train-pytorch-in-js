// We load ONNX Runtime Web using a script tag in index.html so declaring the module helps avoid TypeScript errors.
declare module 'ort' {
	declare class Tensor {
		data: any
		shape: number[]
		constructor(type: string, data: any, shape: number[] = undefined)
	}

	declare class InferenceSession {
		static create(model: string): Promise<InferenceSession>
		run(inputs: { [name: string]: Tensor }): Promise<{ [name: string]: Tensor }>
	}
}
