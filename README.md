# train-pytorch-in-js
Convert a PyTorch model to train it in JavaScript using ONNX Runtime Web.

# Steps
0. Define and train your PyTorch model. You probably already did this.
1. Export the model.
2. Load the model in JavaScript.

## 0. Define and train your PyTorch model
You probably already did this.
Here's our simple example:
```python
import torch

class MyModel(torch.nn.Module):
	def __init__(self,
				 input_size: int,
				 hidden_size: int,
				 num_classes: int):
		super(MyModel, self).__init__()

		self.fc1 = torch.nn.Linear(input_size, hidden_size)
		self.relu = torch.nn.ReLU()
		self.fc2 = torch.nn.Linear(hidden_size, num_classes)

	def forward(self, x):
		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)
		return out
```

Let's assume that you trained it.
Training the model in Python isn't required to export it and train it in JavaScript.

## 1. Export the model
We're going to create an ONNX graph that can compute gradients when given training data.

1. Install some dependencies
<!-- TODO Confirm link. -->
PyTorch: see [pytorch.org](https://pytorch.org) for how to install it on your system.

onnxruntime: At this time (March 2022), the utility method to export the gradient graph hasn't been released yet.
It should be in version 1.11 (TODO verify).
Once it's released, you can do `pip install onnxruntime` (see [onnxruntime.ai](https://onnxruntime.ai) for other options).

Until then, you'll need to build onnxruntime yourself.
Here's some commands that should help assuming you're using Linux and have CMake and `conda` setup:
```bash
conda create --name ort-dev python=3.8 numpy h5py
conda activate ort-dev
conda install -c anaconda libstdcxx-ng
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install flake8 pytest
git clone git@github.com:microsoft/onnxruntime.git --recursive TODO recurse submodules
cd onnxruntime
pip install -r requirements-dev.txt
./build.sh --config Release --build_shared_lib --parallel $(expr `nproc` - 1) --enable_training --enable_pybind --build_wheel --skip_submodule_sync --skip_tests

export LD_LIBRARY_PATH="./build/Linux/Release:${LD_LIBRARY_PATH}"
```

2. Export the model
```python
from onnxruntime.training.experimental import export_gradient_graph

# We need a custom loss function to load the graph in an InferenceSession in ONNX Runtime Web.
# You can still make the gradient graph with torch.nn.CrossEntropyLoss() and this test will pass.
def binary_cross_entropy_loss(inp, target):
	return -torch.sum(target * torch.log2(inp[:, 0]) +
					  (1-target) * torch.log2(inp[:, 1]))


loss_fn = binary_cross_entropy_loss

input_size = 10
model = MyModel(input_size=input_size, hidden_size=5, num_classes=2)

# We need a place to save the ONNX graph.
gradient_graph_path = 'gradient_graph.onnx'

# We need example input for the ONNX model.
# It doesn't matter what values are filled in the but the dimensions need to be correct.
batch_size = 32
example_input = torch.randn(
	batch_size, input_size, requires_grad=True)
example_labels = torch.tensor([1])

export_gradient_graph(
			model, loss_fn, example_input, example_labels, gradient_graph_path)
```

You now have an ONNX graph at `gradient_graph.onnx`.
If you want to validate it, see [orttraining_test_experimental_gradient_graph.py](https://github.com/microsoft/onnxruntime/commits/master/orttraining/orttraining/test/python/orttraining_test_experimental_gradient_graph.py) for examples on how you can validate the file.

## 2. Load the model in JavaScript
We'll use [ONNX Runtime Web](TODO) to load the gradient graph.

At this time (March 2022), this only works with custom ONNX Runtine Web builds which have training operators enabled.
The published ONNX Runtime Web doesn't support the certain operators in our graph with gradient calculations such as `GatherGrad`.

1. Build ONNX Runtime Web with training operators enabled.
   1. TODO

2. Setup the example project.
   1. Put the files from the ONNX Runtime Web build (ort.js and others such as the wasm files, if needed) in `onnxruntime_web_build_inference_with_training_ops/`.
   2. `cd training`
   3. `npm install`
   4. `npm run start`
