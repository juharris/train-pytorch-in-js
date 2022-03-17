import React from 'react'
import './App.css'

// We load ONNX Runtime Web using a script tag in index.html.
declare const ort: any

function App() {
	const [messages, setMessages] = React.useState<string[]>([])
	const [statusMessage, setStatusMessage] = React.useState("")
	const [errorMessage, setErrorMessage] = React.useState("")

	function randomTensor(shape: number[]) {
		const values = new Float32Array(shape.reduce((a, b) => a * b))
		for (let i = 0; i < values.length; ++i) {
			values[i] = Math.random()
		}
		return new ort.Tensor('float32', values, shape)
	}

	function showStatusMessage(message: string) {
		console.log(message)
		setStatusMessage(message)
	}

	function addMessage(message: string) {
		console.log(message)
		console.debug("messages:", messages)
		setMessages([...messages, message])
	}

	function showErrorMessage(message: string) {
		console.error(message)
		setErrorMessage(message)
	}

	// Load the ONNX model.
	React.useMemo(async () => {
		const url = '/gradient_graph.onnx'
		showStatusMessage(`Loading ONNX model at "${url}".`)
		let session
		try {
			session = await ort.InferenceSession.create(url)
			console.log("Loaded the model. session:", session)
			showStatusMessage("Loaded the model.")
		} catch (err) {
			showErrorMessage("Error loading the model: " + err)
			console.error("Error loading the model", err)
			return
		}

		// Set up some sample data to try with our model.
		const x = Float32Array.from(Array.from(Array(10)).map(Math.random))
		const label = BigInt64Array.from([1n])
		const batchSize = 1
		const xTensor = new ort.Tensor('float32', x, [batchSize, x.length])
		const labelTensor = new ort.Tensor('int64', label, [batchSize])

		const feeds = {
			input: xTensor,
			labels: labelTensor,
			'fc1.weight': randomTensor([5, 10]),
			'fc1.bias': randomTensor([5]),
			'fc2.weight': randomTensor([2, 5]),
			'fc2.bias': randomTensor([2]),
		}

		const results = await session.run(feeds)
		console.debug("results:", results)
		for (const [k, tensor] of Object.entries(results)) {
			console.debug(k, tensor)
			addMessage(`${k}: ${(tensor as any).data}`)
		}
	}, [])

	return (<div className="App">
		Status: {statusMessage ? <p>{statusMessage}</p> : null}
		{messages ?
			<ul>
				{messages.map((m, i) =>
					<li key={i}>{m}</li>)
				}
			</ul> : null}
		{errorMessage ?
			<p style={{ color: 'red' }}>
				{errorMessage}
			</p>
			: null}
	</div>)
}

export default App;
