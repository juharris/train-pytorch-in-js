import { Button, Container, Grid, Link, TextField } from '@mui/material'
import React from 'react'
import Plot from 'react-plotly.js'
import './App.css'
import { Digit } from './components/Digit'
import { MnistData } from './mnist'
import { getNumCorrect, getPredictions, randomTensor, size } from './tensor-utils'

function App() {
	const numRows = 28
	const numCols = 28

	const [initialLearningRate, setInitialLearningRate] = React.useState<number>(3e-4)
	const [gamma, setGamma] = React.useState<number>(1.0)
	const [maxNumTrainSamples, setMaxNumTrainSamples] = React.useState<number>(MnistData.BATCH_SIZE * 100)
	const [maxNumTestSamples, setMaxNumTestSamples] = React.useState<number>(MnistData.BATCH_SIZE * 20)

	const [batchSize, setBatchSize] = React.useState<number>(MnistData.BATCH_SIZE)
	const [numEpochs, setNumEpochs] = React.useState<number>(5)

	const [digits, setDigits] = React.useState<{ pixels: Float32Array, label: number }[]>([])
	const [digitPredictions, setDigitPredictions] = React.useState<number[]>([])


	const [trainingLosses, setTrainingLosses] = React.useState<number[]>([])
	const [testAccuracies, setTestAccuracies] = React.useState<number[]>([])

	const [statusMessage, setStatusMessage] = React.useState("")
	const [errorMessage, setErrorMessage] = React.useState("")
	const [messages, setMessages] = React.useState<string[]>([])

	function showStatusMessage(message: string) {
		console.log(message)
		setStatusMessage(message)
	}

	function showErrorMessage(message: string) {
		setErrorMessage(message)
	}

	function addMessage(message: string) {
		setMessages(messages => [...messages, message])
	}

	async function getSession(url: string): Promise<ort.InferenceSession> {
		showStatusMessage(`Loading ONNX model at "${url}"...`)

		try {
			const result = await ort.InferenceSession.create(url)
			console.debug("Loaded the model. session:", result)
			showStatusMessage(`Loaded the model at "${url}"`)
			return result
		} catch (err) {
			showErrorMessage("Error loading the model: " + err)
			console.error("Error loading the model", err)
			throw err
		}
	}

	async function runModel(
		session: ort.InferenceSession,
		feeds: any,
		isLoggingEnabled = false) {
		const result = await session.run(feeds)
		if (isLoggingEnabled) {
			console.debug("results:", result)

			for (const [k, tensor] of Object.entries(result)) {
				addMessage(`  ${k}: ${(tensor as any).data}`)
			}
		}

		return result
	}

	/**
	 * Run the optimizer.
	 *
	 * @param optimizerSession 
	 * @param runModelResults 
	 * @param weights The weights to optimize. The values will be updated.
	 * @param prevOptimizerOutput 
	 * @param learningRate 
	 * @returns 
	 */
	async function runOptimizer(
		optimizerSession: ort.InferenceSession,
		runModelResults: ort.InferenceSession.ReturnType,
		weights: ort.InferenceSession.OnnxValueMapType,
		prevOptimizerOutput: ort.InferenceSession.ReturnType | undefined,
		learningRate = 0.01,
	): Promise<ort.InferenceSession.ReturnType> {
		const optimizerInputs: { [name: string]: ort.OnnxValue } = {}
		for (const [name, tensor] of Object.entries(weights)) {
			optimizerInputs[name] = tensor
			optimizerInputs[name + '.gradient'] = runModelResults[name + '_grad']
			optimizerInputs[name + '.learning_rate'] = new ort.Tensor('float32', [learningRate])
			// Not used but could be in the future.
			// optimizerInputs[name + '.should_update'] = new ort.Tensor('bool', [true])
			// optimizerInputs[name + '.global_gradient_norm'] = new ort.Tensor('float32', [])
			// Should be float16, but that's not supported.
			// optimizerInputs[name + '.loss_scaler'] = new ort.Tensor('float32', [])
			if (prevOptimizerOutput) {
				for (const suffix of ['.exp_avg', '.exp_avg_sq', '.mixed_precision', '.step']) {
					const prev = prevOptimizerOutput[name + suffix + '.out']
					if (prev) {
						optimizerInputs[name + suffix] = prevOptimizerOutput[name + suffix + '.out']
					}
				}
			} else {
				optimizerInputs[name + '.step'] = new ort.Tensor('int64', new BigInt64Array([1n]))
				optimizerInputs[name + '.exp_avg'] = new ort.Tensor('float32', Array(size((tensor as any).dims)).fill(0), (tensor as any).dims)
				optimizerInputs[name + '.exp_avg_sq'] = new ort.Tensor('float32', Array(size((tensor as any).dims)).fill(0), (tensor as any).dims)
				// Not used but could be in the future.
				// optimizerInputs[name + '.mixed_precision'] = new ort.Tensor('float32', [])
			}
		}

		const output = await optimizerSession.run(optimizerInputs)

		for (const name of Object.keys(weights)) {
			(weights as any)[name] = output[name + '.out']
		}

		return output
	}

	function getPixels(data: Float32Array, numRows: number, numCols: number) {
		const result: number[][] = []
		for (let row = 0; row < numRows; ++row) {
			const rowPixels: number[] = []
			for (let col = 0; col < numCols; ++col) {
				rowPixels.push(data[row * numCols + col])
			}
			result.push(rowPixels)
		}
		return result
	}

	async function updateDigitPredictions(session: ort.InferenceSession, weights: any) {
		// Build a batch.
		const input = new Float32Array(digits.length * numRows * numCols)
		const batchShape = [digits.length, 1, numRows, numCols]
		const labels = []
		for (let i = 0; i < digits.length; ++i) {
			const pixels = digits[i].pixels
			for (let j = 0; j < pixels.length; ++j) {
				input[i * pixels.length + j] = MnistData.normalize(pixels[j])
			}

			// We don't really need to give the real labels since we just need the output.
			labels.push(BigInt(digits[i].label))
		}

		const feeds = {
			input: new ort.Tensor('float32', input, batchShape),
			labels: new ort.Tensor('int64', new BigInt64Array(labels)),
			...weights,
		}
		const runModelResults = await runModel(session, feeds)
		const predictions = getPredictions(runModelResults['output'])
		setDigitPredictions(predictions.slice(0, digits.length))
	}

	async function train() {
		const logIntervalMs = 6 * 1000
		const dataSet = new MnistData(batchSize)
		dataSet.maxNumTrainSamples = maxNumTrainSamples
		dataSet.maxNumTestSamples = maxNumTestSamples

		const modelPrefix = 'mnist_'
		const modelUrl = `${modelPrefix}gradient_graph.onnx`
		const optimizerUrl = `${modelPrefix}optimizer_graph.onnx`
		const session = await getSession(modelUrl)

		// TODO Try to determine these dynamically.
		// There doesn't seem to be a way from the model to get this information.
		const inputSize = numRows * numCols
		const hiddenSize = 128
		const numClasses = 10

		// Initialize weight using distributions explained at https://pytorch.org/docs/stable/generated/torch.nn.Linear.html.
		const weights = {
			'fc1.weight': randomTensor([hiddenSize, inputSize], -Math.sqrt(1 / inputSize), Math.sqrt(1 / inputSize)),
			'fc1.bias': randomTensor([hiddenSize], -Math.sqrt(1 / inputSize), Math.sqrt(1 / inputSize)),
			'fc2.weight': randomTensor([numClasses, hiddenSize], -Math.sqrt(1 / hiddenSize), Math.sqrt(1 / hiddenSize)),
			'fc2.bias': randomTensor([numClasses], -Math.sqrt(1 / hiddenSize), Math.sqrt(1 / hiddenSize)),
		}

		const optimizerSession = await getSession(optimizerUrl)

		let prevOptimizerOutput: ort.InferenceSession.ReturnType | undefined = undefined
		showStatusMessage("Training...")
		let lastLogTime = Date.now()
		const totalNumBatches = dataSet.getNumTrainingBatches()
		const totalNumTestBatches = dataSet.getNumTestBatches()

		const waitAfterLoggingMs = 120

		let learningRate = initialLearningRate
		let numCorrect = 0
		let total = 0
		try {
			for (let epoch = 1; epoch <= numEpochs; ++epoch) {
				updateDigitPredictions(session, weights)

				// Train
				let batchNum = 0
				for await (const batch of dataSet.trainingBatches()) {
					const batchStartTime = Date.now()
					++batchNum
					const feeds = {
						input: batch.data,
						labels: batch.labels,
						...weights,
					}

					const runModelResults = await runModel(session, feeds)
					const loss = runModelResults['loss'].data[0] as number
					const message = `Epoch: ${String(epoch).padStart(2)}/${numEpochs} | Batch: ${String(batchNum).padStart(3)}/${totalNumBatches} | Loss: ${loss.toFixed(4)} | ${((Date.now() - batchStartTime) / 1000).toFixed(3)}s`
					if (isNaN(loss)) {
						console.warn("feeds", feeds)
						console.warn("runModelResults:", runModelResults)
						throw new Error(message)
					}
					setTrainingLosses(losses => losses.concat(loss))
					console.debug(message)
					addMessage(message)
					if (Date.now() - lastLogTime > logIntervalMs) {
						setStatusMessage(message)
						updateDigitPredictions(session, weights)
						// Wait to give the UI a chance to update and respond to inputs.
						await new Promise(resolve => setTimeout(resolve, waitAfterLoggingMs))
						lastLogTime = Date.now()
					}
					prevOptimizerOutput = await runOptimizer(optimizerSession, runModelResults, weights, prevOptimizerOutput, learningRate)
				}

				learningRate *= gamma

				// Test
				let totalTestLoss = 0
				batchNum = 0
				numCorrect = 0
				total = 0
				for await (const batch of dataSet.testBatches()) {
					++batchNum
					const feeds = {
						input: batch.data,
						labels: batch.labels,
						...weights,
					}

					const runModelResults = await runModel(session, feeds)
					const loss = runModelResults['loss'].data[0] as number
					if (isNaN(loss)) {
						console.warn("feeds", feeds)
						console.warn("runModelResults:", runModelResults)
						throw new Error(`Testing | Epoch ${epoch} | Batch ${batchNum}/${totalNumTestBatches} | Loss = ${loss}`)
					}
					totalTestLoss += loss
					numCorrect += getNumCorrect(runModelResults['output'], batch.labels)
					total += batch.labels.dims[0]

					const message = `Epoch: ${String(epoch).padStart(2)}/${numEpochs} | Test Batch: ${String(batchNum).padStart(3)}/${totalNumTestBatches} | Average Test Loss: ${(totalTestLoss / batchNum).toFixed(4)} | Accuracy: ${numCorrect}/${total} (${(100 * (numCorrect / total)).toFixed(1)}%)`
					console.debug(message)
					addMessage(message)
					if (Date.now() - lastLogTime > logIntervalMs) {
						setStatusMessage(message)
						// Wait to give the UI a chance to update and respond to inputs.
						await new Promise(resolve => setTimeout(resolve, waitAfterLoggingMs))
						lastLogTime = Date.now()
					}
				}
				const message = `Epoch: ${String(epoch).padStart(2)}/${numEpochs} | Average Test Loss: ${(totalTestLoss / batchNum).toFixed(4)} | Accuracy: ${numCorrect}/${total} (${(total > 0 ? 100 * (numCorrect / total) : 0).toFixed(1)}%)`
				if (total) {
					const accuracy = numCorrect / total
					setTestAccuracies(accuracies => accuracies.concat(accuracy))
				}
				console.log(message)
				setStatusMessage(message)
				addMessage(message)
				addMessage("")
			}

			let message = "Done training"
			if (total) {
				message += ` | Test Set Accuracy: ${numCorrect}/${total} (${(total > 0 ? 100 * (numCorrect / total) : 0).toFixed(1)}%)`
			}
			showStatusMessage(message)
		} catch (err) {
			showErrorMessage(`Error while training: ${err}`)
			console.error(err)
		}
	}

	function renderPlots() {
		const margin = { t: 20, r: 25, b: 25, l: 40 }
		return (<div className="section">
			<h3>Plots</h3>
			<Grid container spacing={2}>
				<Grid item xs={12} md={6}>
					<h4>Training Loss</h4>
					<Plot
						data={[
							{
								x: trainingLosses.map((_, i) => i),
								y: trainingLosses,
								type: 'scatter',
								mode: 'lines',
							}
						]}
						layout={{ margin, width: 400, height: 320 }}
					/>
				</Grid><Grid item xs={12} md={6}>
					<h4>Test Accuracy (%)</h4>
					<Plot
						data={[
							{
								x: testAccuracies.map((_, i) => i),
								y: testAccuracies.map(a => 100 * a),
								type: 'scatter',
								mode: 'lines+markers',
							}
						]}
						layout={{ margin, width: 400, height: 320 }}
					/>
				</Grid>
			</Grid>
		</div>)
	}

	function renderDigits() {
		return (<div className="section">
			<h4>Test Digits</h4>
			<Grid container spacing={2}>
				{digits.map((digit, digitIndex) => {
					const { pixels, label } = digit
					const rgdPixels = getPixels(pixels, numRows, numCols)
					return (<Grid key={digitIndex} item xs={6} sm={3} md={2}>
						<Digit pixels={rgdPixels} label={label} prediction={digitPredictions[digitIndex]} />
					</Grid>)
				})}
			</Grid>
		</div>)
	}

	function startTraining() {
		setDigitPredictions([])
		setMessages([])
		setStatusMessage("")
		setErrorMessage("")
		train()
	}

	const loadDigits = React.useCallback(async () => {
		const maxNumDigits = 18
		const seenLabels = new Set()
		const dataSet = new MnistData()
		dataSet.maxNumTestSamples = 2 * dataSet.batchSize
		const digits = []
		const normalize = false
		for await (const testBatch of dataSet.testBatches(normalize)) {
			const { data, labels } = testBatch
			const batchSize = labels.dims[0]
			const numRows = data.dims[2]
			const numCols = data.dims[3]
			for (let i = 0; digits.length < maxNumDigits && i < batchSize; ++i) {
				const label = Number(labels.data[i])
				if (seenLabels.size < 10 && seenLabels.has(label)) {
					continue
				}
				seenLabels.add(label)
				const pixels = data.data.slice(i * numRows * numCols, (i + 1) * numRows * numCols) as Float32Array

				digits.push({ pixels, label })
			}

			if (digits.length >= maxNumDigits) {
				break
			}
		}
		setDigits(digits)
	}, [])

	React.useEffect(() => {
		loadDigits()
	}, [loadDigits])

	return (<Container className="App">
		<div className="section">
			<h2>ONNX Runtime Web Training Demo</h2>
			<p>
				In this example, you'll a train classifier in your browser to recognize handwritten digits from the <Link href="https://deepai.org/dataset/mnist" target="_blank" rel="noopener">MNIST Dataset</Link>.
			</p>
			<p>
				You can learn more about how to set up a model that can be trained in your browser at <Link href="https://github.com/juharris/train-pytorch-in-js" target="_blank" rel="noopener">github.com/juharris/train-pytorch-in-js</Link>.
			</p>
		</div>
		<div className="section">
			<h3>Training</h3>
			<p>
				After each epoch, the learning rate will be multiplied by <code>Gamma</code>.
			</p>
			<Grid container spacing={{ xs: 1, md: 2 }}>
				<Grid item xs={12} md={4} >
					<TextField label="Initial Learning Rate"
						type="number"
						value={initialLearningRate}
						onChange={(e) => setInitialLearningRate(Number(e.target.value))}
					/>
				</Grid>
				<Grid item xs={12} md={4}>
					<TextField label="Gamma"
						type="number"
						value={gamma}
						onChange={(e) => setGamma(Number(e.target.value))}
					/>
				</Grid>
			</Grid>
		</div>
		<div className="section">
			<Grid container spacing={{ xs: 1, md: 2 }}>
				<Grid item xs={12} md={4} >
					<TextField label="Number of epochs"
						type="number"
						value={numEpochs}
						onChange={(e) => setNumEpochs(Number(e.target.value))}
					/>
				</Grid>
				<Grid item xs={12} md={4}>
					<TextField label="Batch size"
						type="number"
						value={batchSize}
						onChange={(e) => setBatchSize(Number(e.target.value))}
					/>
				</Grid>
			</Grid>
		</div>
		<div className="section">
			<Grid container spacing={{ xs: 1, md: 2 }}>
				<Grid item xs={12} md={4} >
					<TextField type="number"
						label="Max number of training samples"
						value={maxNumTrainSamples}
						onChange={(e) => setMaxNumTrainSamples(Number(e.target.value))}
					/>
				</Grid>
				<Grid item xs={12} md={4}>
					<TextField type="number"
						label="Max number of test samples"
						value={maxNumTestSamples}
						onChange={(e) => setMaxNumTestSamples(Number(e.target.value))}
					/>
				</Grid>
			</Grid>
		</div>
		<div className="section">
			<Button onClick={startTraining}
				variant="contained">
				Train
			</Button>
		</div>
		{/* TODO Add a button to stop training. How would it work? */}

		{renderPlots()}

		{renderDigits()}

		<pre>{statusMessage}</pre>
		{messages.length > 0 &&
			<div>
				<h3>Logs:</h3>
				<pre>
					{messages.map((m, i) => (<React.Fragment key={i}>
						{m}
						<br />
					</React.Fragment>))}
				</pre>
			</div>}
		{errorMessage &&
			<p className='error'>
				{errorMessage}
			</p>}
	</Container>)
}

export default App
