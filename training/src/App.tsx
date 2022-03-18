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

	function showErrorMessage(message: string) {
		setErrorMessage(message)
	}

	// Load the ONNX model.
	React.useEffect(() => {
		setMessages([])

		async function loadModel() {
			const url = '/gradient_graph.onnx'
			showStatusMessage(`Loading ONNX model at "${url}"...`)
			let session
			try {
				session = await ort.InferenceSession.create(url)
				console.log("Loaded the model. session:", session)
				showStatusMessage(`Loaded the model at "${url}."`)
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
			const newMessages: string[] = []
			for (const [k, tensor] of Object.entries(results)) {
				console.debug(k, tensor)
				newMessages.push(`${k}: ${(tensor as any).data}`)
			}
			setMessages(messages => [...messages, ...newMessages])
		}

		loadModel()
	}, [])

	return (<div className="App">
		<h3>Gradient Graph Example</h3>
		<p>{statusMessage}</p>
		{messages.length > 0 ?
			<div>
				<ul>
					{messages.map((m, i) =>
						<li key={i}>{m}</li>)
					}
				</ul>
			</div> : null}
		{errorMessage ?
			<p style={{ color: 'red' }}>
				{errorMessage}
			</p>
			: null}
	</div>)
}

export default App;
