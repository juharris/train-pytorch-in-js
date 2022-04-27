import ort from 'ort'

export function size(shape: number[]): number {
	return shape.reduce((a, b) => a * b)
}

export function randomArray(shape: number[]): Float32Array {
	const result = new Float32Array(size(shape))
	for (let i = 0; i < result.length; ++i) {
		result[i] = Math.random()
	}
	return result
}

export function randomTensor(shape: number[]): ort.Tensor {
	return new ort.Tensor('float32', randomArray(shape), shape)
}
