export function size(shape: readonly number[]): number {
	return shape.reduce((a, b) => a * b)
}

export function randomArray(shape: number[], minimum: number, maximum: number): Float32Array {
	const result = new Float32Array(size(shape))
	for (let i = 0; i < result.length; ++i) {
		result[i] = Math.random() * (maximum - minimum) + minimum
	}
	return result
}

export function randomTensor(shape: number[], minimum: number, maximum: number): ort.Tensor {
	return new ort.Tensor('float32', randomArray(shape, minimum, maximum), shape)
}

/**
 * @param outputLogits batchSize x numClasses
 * @param labels batchSize
 * @returns The number of correct predictions.
 */
export function getNumCorrect(outputLogits: ort.Tensor, labels: ort.Tensor): number {
	let result = 0
	const [batchSize, numClasses] = outputLogits.dims
	for (let i = 0; i < batchSize; ++i) {
		const values = outputLogits.data.slice(i * numClasses, (i + 1) * numClasses) as Float32Array
		const outputLabel = argMax(values)
		if (BigInt(outputLabel) === labels.data[i]) {
			++result
		}
	}
	return result
}


export function argMax(tensor: Float32Array): number {
	let result = 0
	for (let i = 1; i < tensor.length; ++i) {
		if (tensor[i] > tensor[result]) {
			result = i
		}
	}
	return result
}