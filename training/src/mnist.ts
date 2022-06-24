// Modified from https://github.com/cazala/mnist/blob/master/src/mnist.js
// so that we can place the data in a specific folder and avoid out of memory errors
// and use TypeScript.

// Assume the data was loaded when running the Python scripts.
export class MnistData {
    async load() {
        const reader = ((await fetch('/data/MNIST/raw/train-images-idx3-ubyte')).body)!.getReader()
        // TODO Use reader.read to get the data.
        const buffer = await reader.read()

        // TODO Load the test data.
    }
}