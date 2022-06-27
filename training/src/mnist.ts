// Modified from https://github.com/cazala/mnist/blob/master/src/mnist.js
// so that we can place the data in a specific folder and avoid out of memory errors
// and use TypeScript.

// Assume the data was loaded when running the Python scripts.
/**
 * Dataset description at https://deepai.org/dataset/mnist.
 */
export class MnistData {
	pixelMax = 255
	pixelMean = 0.1307
	pixelStd = 0.3081

	constructor(
		public batchSize = 64,
		public maxNumTrainSamples = -1,
		public maxNumTestSamples = -1,
	) {
		if (batchSize <= 0) {
			throw new Error("batchSize must be > 0")
		}
	}

	private *batches(data: ort.Tensor[], labels: ort.Tensor[]) {
		for (let batchIndex = 0; batchIndex < data.length; ++batchIndex) {
			yield {
				data: data[batchIndex],
				labels: labels[batchIndex],
			}
		}
	}

	public async * trainingBatches() {
		// Avoid keeping data in memory.
		const trainingData = await this.getData('/data/MNIST/raw/train-images-idx3-ubyte', 2051, 'data', this.maxNumTrainSamples)
		const trainingLabels = await this.getData('/data/MNIST/raw/train-labels-idx1-ubyte', 2049, 'labels', this.maxNumTrainSamples)
		yield* this.batches(trainingData, trainingLabels)
	}

	public async * testBatches() {
		// Avoid keeping data in memory.
		const testData = await this.getData('/data/MNIST/raw/t10k-images-idx3-ubyte', 2051, 'data', this.maxNumTestSamples)
		const testLabels = await this.getData('/data/MNIST/raw/t10k-labels-idx1-ubyte', 2049, 'labels', this.maxNumTestSamples)
		yield* this.batches(testData, testLabels)
	}

	private async getData(url: string, expectedMagicNumber: number, dataType: 'data' | 'labels', maxNumSamples = -1): Promise<ort.Tensor[]> {
		const result = []
		const response = await fetch(url)
		const buffer = await response.arrayBuffer()
		if (buffer.byteLength < 16) {
			throw new Error("Invalid MNIST images file. There aren't enough bytes")
		}
		const magicNumber = new DataView(buffer.slice(0, 4)).getInt32(0, false)
		if (magicNumber !== expectedMagicNumber) {
			throw new Error(`Invalid MNIST images file. The magic number is not ${expectedMagicNumber}. Got ${magicNumber}.`)
		}
		const numDimensions = new DataView(buffer.slice(3, 4)).getUint8(0)
		const shape = []
		for (let i = 0; i < numDimensions; ++i) {
			shape.push(new DataView(buffer.slice(4 + i * 4, 8 + i * 4)).getUint32(0, false))
		}
		const numItems = shape[0]
		const dimensions = shape.slice(1)
		const batchShape: number[] = dataType === 'data' ? [this.batchSize, 1, ...dimensions] : [this.batchSize]
		const dataSize = dimensions.reduce((a, b) => a * b, 1)

		let offset = 4 + 4 * shape.length
		for (let i = 0; i < numItems; i += this.batchSize) {
			if (maxNumSamples > 0 && i > maxNumSamples) {
				break
			}

			if (buffer.byteLength < offset + this.batchSize * dataSize) {
				break
			}
			let batch
			switch (dataType) {
				case 'data':
					const image = new Uint8Array(buffer.slice(offset, offset + this.batchSize * dataSize))
					batch = (new Float32Array(image)).map(v => this.normalize(v))
					batch = new ort.Tensor('float32', batch, batchShape)
					break
				case 'labels':
					const label = new Uint8Array(buffer.slice(offset, offset + this.batchSize * dataSize))
					batch = Array.from(label).map(BigInt)
					batch = new ort.Tensor('int64', new BigInt64Array(batch), batchShape)
					break
			}

			result.push(batch)
			offset += this.batchSize * dataSize
		}

		return result
	}

	private normalize(pixelValue: number): number {
		return ((pixelValue / this.pixelMax) - this.pixelMean) / this.pixelStd
	}
}