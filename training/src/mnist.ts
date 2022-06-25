// Modified from https://github.com/cazala/mnist/blob/master/src/mnist.js
// so that we can place the data in a specific folder and avoid out of memory errors
// and use TypeScript.

// Assume the data was loaded when running the Python scripts.
/**
 * Dataset description at https://deepai.org/dataset/mnist.
 */
export class MnistData {
    async load(batchSize = 64, maxNumTrainSamples = -1, maxNumTestSamples = -1) {
        const trainingData = await this.getData('/data/MNIST/raw/train-images-idx3-ubyte', 2051, maxNumTrainSamples)
        const trainingLabels = await this.getData('/data/MNIST/raw/train-labels-idx1-ubyte', 2049, maxNumTrainSamples)
        const testData = await this.getData('/data/MNIST/raw/t10k-images-idx3-ubyte', 2051, maxNumTestSamples)
        const testLabels = await this.getData('/data/MNIST/raw/t10k-labels-idx1-ubyte', 2049, maxNumTestSamples)
        // TODO Batch data by batchSize
        
        
    }

    private async getData(url: string, expectedMagicNumber: number, maxNumSamples = -1): Promise<Uint8Array[]> {
        const response = await fetch(url)
        const buffer = await response.arrayBuffer()
        if (buffer.byteLength < 16) {
            throw new Error("Invalid MNIST images file. There aren't enough bytes")
        }
        const magicNumber = new DataView(buffer.slice(0, 4)).getInt32(0, false)
        if (magicNumber !== expectedMagicNumber) {
            throw new Error(`Invalid MNIST images file. The magic number is not ${expectedMagicNumber}. Got ${magicNumber}.`)
        }
        const numImages = new DataView(buffer.slice(4, 8)).getInt32(0, false)
        const numRows = new DataView(buffer.slice(8, 12)).getInt32(0, false)
        const numColumns = new DataView(buffer.slice(12, 16)).getInt32(0, false)

        const result = []
        for (let i = 16; i < buffer.byteLength; i += numRows * numColumns) {
            if (maxNumSamples > 0 && i > maxNumSamples * numRows * numColumns) {
                break
            }
            const image = new Uint8Array(buffer.slice(i, i + numRows * numColumns))
            result.push(image)
        }
        return result
    }
}