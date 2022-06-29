import { Button, Container, TextField } from '@mui/material'
import React from 'react'
import './App.css'
import { MnistData } from './mnist'
import { getNumCorrect, randomTensor, size } from './tensor-utils'

function App() {
	const [initialLearningRate, setInitialLearningRate] = React.useState<number>(3e-4)
	const [gamma, setGamma] = React.useState<number>(1.0)
	const [numEpochs, setNumEpochs] = React.useState<number>(3)
	const [messages, setMessages] = React.useState<string[]>([])
	const [statusMessage, setStatusMessage] = React.useState("")
	const [errorMessage, setErrorMessage] = React.useState("")

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

	async function train() {
		const logIntervalMs = 5 * 1000
		const dataSet = new MnistData()
		// TODO Use all the data when we're done debugging.
		dataSet.maxNumTrainSamples = 10000
		dataSet.maxNumTestSamples = 1000

		const modelPrefix = 'mnist_'
		const modelUrl = `/${modelPrefix}gradient_graph.onnx`
		const optimizerUrl = `/${modelPrefix}optimizer_graph.onnx`
		const session = await getSession(modelUrl)

		// TODO Try to determine these dynamically.
		const inputSize = 28 * 28
		const hiddenSize = 128
		const numClasses = 10

		// Initialize weight using distributions explained at https://pytorch.org/docs/stable/generated/torch.nn.Linear.html.
		const weights = {
			'fc1.weight': randomTensor([hiddenSize, inputSize], -Math.sqrt(1 / inputSize), Math.sqrt(1 / inputSize)),
			'fc1.bias': randomTensor([hiddenSize], -Math.sqrt(1 / inputSize), Math.sqrt(1 / inputSize)),
			'fc2.weight': randomTensor([numClasses, hiddenSize], -Math.sqrt(1 / hiddenSize), Math.sqrt(1 / hiddenSize)),
			'fc2.bias': randomTensor([numClasses], -Math.sqrt(1 / hiddenSize), Math.sqrt(1 / hiddenSize)),
		}
		console.debug("weights", weights)

		const optimizerSession = await getSession(optimizerUrl)

		let prevOptimizerOutput: ort.InferenceSession.ReturnType | undefined = undefined
		showStatusMessage("Training...")
		let lastLogTime = Date.now()
		const totalNumBatches = dataSet.getNumTrainingBatches()
		const totalNumTestBatches = dataSet.getNumTestBatches()

		const waitAfterLoggingMs = 120

		let learningRate = initialLearningRate
		try {
			for (let epoch = 1; epoch <= numEpochs; ++epoch) {
				let batchNum = 0
				for await (const batch of dataSet.trainingBatches()) {
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
						throw new Error(`Training | Epoch ${epoch} | Batch ${batchNum}/${totalNumBatches} | Loss = ${loss}`)
					}
					if (Date.now() - lastLogTime > logIntervalMs) {
						const message = `Epoch: ${String(epoch).padStart(2)}/${numEpochs} | Batch: ${String(batchNum).padStart(3)}/${totalNumBatches} | Loss: ${loss.toFixed(4)}`
						addMessage(message)
						setStatusMessage(message)
						lastLogTime = Date.now()
						// Wait to give the UI a chance to update and respond to inputs.
						await new Promise(resolve => setTimeout(resolve, waitAfterLoggingMs))
					}
					prevOptimizerOutput = await runOptimizer(optimizerSession, runModelResults, weights, prevOptimizerOutput, learningRate)
				}

				learningRate *= gamma

				let totalTestLoss = 0
				batchNum = 0
				let numCorrect = 0
				let total = 0
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

					if (Date.now() - lastLogTime > logIntervalMs) {
						const message = `Epoch: ${String(epoch).padStart(2)}/${numEpochs} | Test Batch: ${String(batchNum).padStart(3)}/${totalNumTestBatches} | Average Test Loss: ${(totalTestLoss / batchNum).toFixed(4)} | Accuracy: ${numCorrect}/${total} (${(100 * (numCorrect / total)).toFixed(1)}%)`
						addMessage(message)
						setStatusMessage(message)
						lastLogTime = Date.now()
						// Wait to give the UI a chance to update and respond to inputs.
						await new Promise(resolve => setTimeout(resolve, waitAfterLoggingMs))
					}
				}
				addMessage(`Epoch: ${String(epoch).padStart(2)}/${numEpochs} | Average Test Loss: ${(totalTestLoss / batchNum).toFixed(4)} | Accuracy: ${numCorrect}/${total} (${(total > 0 ? 100 * (numCorrect / total) : 0).toFixed(1)}%)`)
				addMessage("")
			}

			showStatusMessage("Done training")
		} catch (err) {
			showErrorMessage(`Error while training: ${err}`)
			console.error(err)
		}
	}

	function startTraining() {
		setMessages([])
		setErrorMessage("")
		train()
	}

	// Start training when the page loads.
	// FIXME Resolve dependency warning or remove this when we're done debugging.
	React.useEffect(() => {
		startTraining()
	}, [])

	return (<Container className="App">
		<h3>ONNX Runtime Web Training Demo</h3>
		<div className="section">
			<TextField type="number"
				label="Initial Learning Rate"
				value={initialLearningRate}
				onChange={(e) => setInitialLearningRate(Number(e.target.value))}
			/>
		</div>
		<div className="section">
			<p>
				After each epoch, the learning rate will be multiplied by <code>gamma</code>.
			</p>
			<TextField type="number"
				label="Gamma"
				value={gamma}
				onChange={(e) => setGamma(Number(e.target.value))}
			/>
		</div>
		<div className="section">
			<TextField type="number"
				label="Number of epochs"
				value={numEpochs}
				onChange={(e) => setNumEpochs(Number(e.target.value))}
			/>
		</div>
		<div className="section">
			<Button onClick={startTraining}
				variant="contained">
				Start training
			</Button>
		</div>
		{/* TODO Add a button to stop training. */}
		{/* TODO Show some digits and the predicted classes for after every few batches. */}
		<pre>{statusMessage}</pre>
		{messages.length > 0 &&
			<div>
				<h4>Logs:</h4>
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

export default App;
