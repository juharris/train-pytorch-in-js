import React from 'react'
import './App.css'
import { randomTensor, size } from './tensor-utils'

function App() {
	const [numEpochs, setNumEpochs] = React.useState<number>(20)
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

	React.useEffect(() => {
		setMessages([])

		async function getSession(url: string): Promise<ort.InferenceSession> {
			showStatusMessage(`Loading ONNX model at "${url}"...`)

			try {
				const result = await ort.InferenceSession.create(url)
				console.log("Loaded the model. session:", result)
				showStatusMessage(`Loaded the model at "${url}."`)
				return result
			} catch (err) {
				showErrorMessage("Error loading the model: " + err)
				console.error("Error loading the model", err)
				throw err
			}
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
			learningRate = 0.001,
		): Promise<ort.InferenceSession.ReturnType> {
			const optimizerInputs: { [name: string]: ort.OnnxValue } = {}
			for (const [name, tensor] of Object.entries(weights)) {
				optimizerInputs[name] = tensor
				optimizerInputs[name + '.gradient'] = runModelResults[name + '_grad']
				optimizerInputs[name + '.learning_rate'] = new ort.Tensor('float32', [learningRate])
				optimizerInputs[name + '.should_update'] = new ort.Tensor('bool', [true])
				optimizerInputs[name + '.global_gradient_norm'] = new ort.Tensor('float32', [])
				// Should be float16, but that's not supported.
				optimizerInputs[name + '.loss_scaler'] = new ort.Tensor('float32', [])
				if (prevOptimizerOutput) {
					for (const suffix of ['.exp_avg', 'exp_avg_sq', '.mixed_precision', '.step']) {
						optimizerInputs[name + suffix] = prevOptimizerOutput[name + suffix + '.out']
					}
				} else {
					optimizerInputs[name + '.step'] = new ort.Tensor('int64', new BigInt64Array([1n]))
					optimizerInputs[name + '.exp_avg'] = new ort.Tensor('float32', Array(size((tensor as any).dims)).fill(0), (tensor as any).dims)
					optimizerInputs[name + '.exp_avg_sq'] = new ort.Tensor('float32', Array(size((tensor as any).dims)).fill(0), (tensor as any).dims)
					optimizerInputs[name + '.mixed_precision'] = new ort.Tensor('float32', [])
				}
			}

			const output = await optimizerSession.run(optimizerInputs)

			for (const name of Object.keys(weights)) {
				(weights as any)[name] = output[name + '.out']
			}

			return output
		}

		async function train() {
			const modelUrl = '/gradient_graph.onnx'
			const session = await getSession(modelUrl)

			// Set up some sample data to try with our model.
			const dataDimensions = 10
			const batchSize = 1
			const label = BigInt64Array.from([1n])
			const data = randomTensor([batchSize, dataDimensions])
			const labels = new ort.Tensor('int64', label, [batchSize])

			let weights = {
				'fc1.weight': randomTensor([5, 10]),
				'fc1.bias': randomTensor([5]),
				'fc2.weight': randomTensor([2, 5]),
				'fc2.bias': randomTensor([2]),
			}

			const optimizerUrl = '/optimizer_graph.onnx'
			const optimizerSession = await getSession(optimizerUrl)

			let prevOptimizerOutput: ort.InferenceSession.ReturnType | undefined = undefined
			for (let epoch = 1; epoch <= numEpochs; ++epoch) {
				addMessage(`Starting epoch ${epoch} ...`)
				// TODO Loop over batches.
				const feeds = {
					input: data,
					labels: labels,
					...weights,
				}
				const runModelResults = await runModel(session, feeds)
				addMessage(`... loss: ${runModelResults['loss'].data}`)
				prevOptimizerOutput = await runOptimizer(optimizerSession, runModelResults, weights, prevOptimizerOutput)

			}
		}

		train()
	}, [])

	return (<div className="App">
		<h3>Gradient Graph Example</h3>
		<p>{statusMessage}</p>
		{messages.length > 0 ?
			<div>
				<h4>Messages:</h4>
				{messages.map((m, i) =>
					<p key={i}>{m}</p>)
				}
			</div> : null}
		{errorMessage ?
			<p className='error'>
				{errorMessage}
			</p>
			: null}
	</div>)
}

export default App;
