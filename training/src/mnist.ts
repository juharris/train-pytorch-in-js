// Modified from https://github.com/cazala/mnist/blob/master/src/mnist.js
// so that we can place the data in a specific folder and avoid out of memory errors
// and use TypeScript.

// Assume the data was loaded when running the Python scripts.
/**
 * Dataset description at https://deepai.org/dataset/mnist.
 */
export class MnistData {
    trainingData?: ort.Tensor[][]
    trainingLabels?: ort.Tensor[][]
    testData?: ort.Tensor[][]
    testLabels?: ort.Tensor[][]

    constructor(
        public batchSize = 64,
        public maxNumTrainSamples = -1,
        public maxNumTestSamples = -1,
    ) {
        if (batchSize <= 0) {
            throw new Error("batchSize must be > 0")
        }
    }

    public async initialize() {
        this.trainingData = await this.getData('/data/MNIST/raw/train-images-idx3-ubyte', 2051, 'data', this.maxNumTrainSamples)
        this.trainingLabels = await this.getData('/data/MNIST/raw/train-labels-idx1-ubyte', 2049, 'labels', this.maxNumTrainSamples)
        this.testData = await this.getData('/data/MNIST/raw/t10k-images-idx3-ubyte', 2051, 'data', this.maxNumTestSamples)
        this.testLabels = await this.getData('/data/MNIST/raw/t10k-labels-idx1-ubyte', 2049, 'labels', this.maxNumTestSamples)
    }

    public * trainingBatches() {
        if (this.trainingData === undefined || this.trainingLabels === undefined) {
            throw new Error("Dataset was not initialized. You must call initialize() first.")
        }
        for (let batchIndex = 0; batchIndex < this.trainingData.length; ++batchIndex) {
            yield {
                data: this.trainingData[batchIndex],
                labels: this.trainingLabels[batchIndex],
            }
        }
    }

    public * testBatches() {
        if (this.testData === undefined || this.testLabels === undefined) {
            throw new Error("Dataset was not initialized. You must call initialize() first.")
        }
        for (let batchIndex = 0; batchIndex < this.testData.length; ++batchIndex) {
            yield {
                data: this.testData[batchIndex],
                labels: this.testLabels[batchIndex],
            }
        }
    }

    private async getData(url: string, expectedMagicNumber: number, dataType: 'data' | 'labels', maxNumSamples = -1): Promise<ort.Tensor[][]> {
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
        // const numRows = shape[1]
        // const numColumns = shape[2]

        const dataSize = shape.slice(1).reduce((a, b) => a * b, 1)

        let batch: ort.Tensor[] = []
        result.push(batch)
        let offset = 4 + 4 * shape.length
        for (let i = 0; i < numItems; ++i) {
            if (maxNumSamples > 0 && i > maxNumSamples) {
                break
            }
            offset += i * dataSize;
            let data
            // FIXME Group accumulated data into a batch.
            // Each batch should be an ort.Tensor of 'float32' for data and 'int64' for labels.
            switch (dataType) {
                case 'data':
                    const image = new Uint8Array(buffer.slice(offset, offset + dataSize))
                    break
                case 'labels':
                    const label = new Uint8Array(buffer.slice(offset, offset + 1))
                    break
            }
            batch.push(data)
            if (batch.length === this.batchSize) {
                batch = []
                result.push(batch)
            }
        }
        return result
    }
}