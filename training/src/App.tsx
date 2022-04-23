import React from 'react'
import './App.css'

// We load ONNX Runtime Web using a script tag in index.html.
declare const ort: any

function size(shape: number[]): number{
	return shape.reduce((a, b) => a * b)
}

function randomArray(shape: number[]): Float32Array {
	const result = new Float32Array(size(shape))
	for (let i = 0; i < result.length; ++i) {
		result[i] = Math.random()
	}
	return result
}

function randomTensor(shape: number[]) {
	return new ort.Tensor('float32', randomArray(shape), shape)
}

function App() {
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

	async function runModel(
		session: any,
		feeds: any,
		isLoggingEnabled = false) {
		const result = await session.run(feeds)
		if (isLoggingEnabled) {
			console.debug("results:", result)

			const newMessages: string[] = []
			for (const [k, tensor] of Object.entries(result)) {
				console.debug(k, tensor)
				newMessages.push(`${k}: ${(tensor as any).data}`)
			}
			setMessages(messages => [...messages, ...newMessages])
		}

		return result
	}

	// Load the ONNX model.
	React.useEffect(() => {
		setMessages([])

		async function getSession(url: string) {
			let result
			showStatusMessage(`Loading ONNX model at "${url}"...`)

			try {
				result = await ort.InferenceSession.create(url)
				console.log("Loaded the model. session:", result)
				showStatusMessage(`Loaded the model at "${url}."`)
			} catch (err) {
				showErrorMessage("Error loading the model: " + err)
				console.error("Error loading the model", err)
				throw err
			}

			return result
		}

		async function runOptimizer(
			optimizerSession: any,
			runModelResults: any,
			weights: { [name: string]: any },
			learningRate = 0.001,
		) {
			// TODO Figure out how many steps of the optimizer should be run per batch.
			// Do we restart the step at 1 for each batch?
			const step = 1
			const optimizerInputs: { [name: string]: any } = {}
			for (const [name, tensor] of Object.entries(weights)) {
				// TODO Maybe some should come from output of previous calls to the optimizer (*.out)?
				// Maybe that's just if we run a batch multiple times?
				optimizerInputs[name] = tensor
				optimizerInputs[name + '.gradient'] = runModelResults[name + '_gradient']
				optimizerInputs[name + '.step'] = new ort.Tensor('int64', [step])
				optimizerInputs[name + '.learning_rate'] = new ort.Tensor('float32', [learningRate])
				optimizerInputs[name + '.should_update'] = new ort.Tensor('bool', [true])
				optimizerInputs[name + '.exp_avg'] = new ort.Tensor('float32', Array(size(tensor.shape)).fill(0), tensor.shape)
				optimizerInputs[name + '.exp_avg_sq'] = new ort.Tensor('float32', Array(size(tensor.shape)).fill(0), tensor.shape)
				optimizerInputs[name + '.global_gradient_norm'] = new ort.Tensor('float32', [])
				optimizerInputs[name + '.loss_scaler'] = new ort.Tensor('float16', [])
				optimizerInputs[name + '.mixed_precision'] = new ort.Tensor('float32', [])
			}
			const result = await optimizerSession.run(optimizerInputs)

			// TODO Return new weights.
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

			const weights = {
				'fc1.weight': randomTensor([5, 10]),
				'fc1.bias': randomTensor([5]),
				'fc2.weight': randomTensor([2, 5]),
				'fc2.bias': randomTensor([2]),
			}

			const feeds = {
				input: data,
				labels: labels,
				...weights,
			}

			const optimizerUrl = '/optimizer_graph.onnx'
			const optimizerSession = await getSession(optimizerUrl)

			// TODO Loop over batches and epochs.
			const runModelResults = await runModel(session, feeds, true)
			const newWeights = await runOptimizer(optimizerSession, runModelResults, weights)
		}

		train()
	}, [])

	return (<div className="App">
		<h3>Gradient Graph Example</h3>
		<p>{statusMessage}</p>
		{messages.length > 0 ?
			<div>
				<h4>Messages:</h4>
				<ul>
					{messages.map((m, i) =>
						<li key={i}>{m}</li>)
					}
				</ul>
			</div> : null}
		{errorMessage ?
			<p className='error'>
				{errorMessage}
			</p>
			: null}
	</div>)
}

export default App;
