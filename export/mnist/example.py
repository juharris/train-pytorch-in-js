import os

import onnx
import torch
from onnxruntime.training.experimental import export_gradient_graph
from optim.adam import AdamOnnxGraphBuilder
from torchvision import datasets

from .model import NUM_CLASSES, MnistNet, loss_fn

data_path =  os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
datasets.MNIST(data_path, download=True)

# Batch size for training and testing.
batch_size = 64

input_size = (28, 28)

# How many batches before logging training status
log_interval = 10

model = MnistNet(is_export_mode=True)

# We need a place to save the ONNX graph.
gradient_graph_path = 'mnist_gradient_graph.onnx'

# We need example input for the ONNX model.
# It doesn't matter what values are filled in the but the dimensions need to be correct.
example_input = torch.randn(
    batch_size,1, input_size[0], input_size[1], requires_grad=True)
example_labels = torch.randint(0, NUM_CLASSES, (batch_size,))
# Make sure that we understand how the labels should look.
labels = model(example_input).argmax(dim=1)
assert labels.shape == example_labels.shape

print(f"Writing gradient graph to \"{gradient_graph_path}\".")
export_gradient_graph(
    model, loss_fn, example_input, example_labels, gradient_graph_path)
print(f"Done writing gradient graph to \"{gradient_graph_path}\".")

print("Checking gradient graph...")
onnx_model = onnx.load(gradient_graph_path)
onnx.checker.check_model(onnx_model)
print("✅ Gradient graph should be okay.")

print("Creating Adam optimizer...")
optimizer = AdamOnnxGraphBuilder(model.named_parameters())
onnx_optimizer = optimizer.export()
optimizer_graph_path = 'mnist_optimizer_graph.onnx'
print(f"Writing optimizer graph to \"{optimizer_graph_path}\".")
onnx.save(onnx_optimizer, optimizer_graph_path)

print("Checking optimizer graph...")
onnx_optimizer = onnx.load(optimizer_graph_path)
onnx.checker.check_model(onnx_optimizer)
print("✅ Optimizer graph should be okay.")

print("✅ Done.")
