import React from 'react'
import logo from './logo.svg'
import './App.css'

// @ts-ignore
import ort from '../public/onnxruntime_web_build_inference_with_training_ops/ort.js'

function App() {

  const [message, setMessage] = React.useState("")
  const [errorMessage, setErrorMessage] = React.useState("")

  function randomTensor(shape: number[]): ort.Tensor {
    const values = new Float32Array(shape.reduce((a, b) => a * b))
    for (let i = 0; i < values.length; ++i) {
      values[i] = Math.random()
    }
    return new ort.Tensor('float32', values, shape)
  }

  async function loadOnnxModel() {
    // const url = '/models/example_ort_model.onnx'
    // const url = '/models/model_with_gradient_graph.onnx'
    // const url = '/models/model_with_training_graph.onnx'
    const url = '/models/gradient_graph_model.onnx'
    showMessage(`Loading ONNX model at "${url}".`)
    let session
    try {
      session = await ort.InferenceSession.create(url)
      console.log("Loaded the model. session:", session)
      showMessage("Loaded the model.")
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
      showMessage(`${k}: ${tensor.data}`)
    }
  }

  function showMessage(message: string) {
    console.log(message)
    setMessage(message)
  }

  function showErrorMessage(message: string) {
    console.error(message)
    setErrorMessage(message)
  }

  const url = '/public/gradient_graph_model.onnx'
  let session
  try {
    session = await ort.InferenceSession.create(url)
    console.log("Loaded the model. session:", session)
    showMessage("Loaded the model.")
  } catch (err) {
    showErrorMessage("Error loading the model: " + err)
    console.error("Error loading the model", err)
    return
  }

  loadOnnxModel()

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        {message ? <p>{message}</p> : null}
        {errorMessage ?
          <p style={{ color: 'red' }}>
            {errorMessage}
          </p>
          : null}
      </header>
    </div>
  );
}

export default App;
