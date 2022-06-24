import onnx
import torch
from onnxruntime.training.experimental import export_gradient_graph

from optim.adam import AdamOnnxGraphBuilder


# Let's use a simple example.
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

# We need a custom loss function to load the graph in an InferenceSession in ONNX Runtime Web.
# You can still make the gradient graph with torch.nn.CrossEntropyLoss() and this test will pass.


def binary_cross_entropy_loss(output, target):
    return -torch.sum(target * torch.log2(output[:, 0]) +
                      (1-target) * torch.log2(output[:, 1]))


loss_fn = binary_cross_entropy_loss

input_size = 10
num_classes=2
model = MyModel(input_size=input_size, hidden_size=5, num_classes=num_classes)

# We need a place to save the ONNX graph.
gradient_graph_path = 'gradient_graph.onnx'

# We need example input for the ONNX model.
# It doesn't matter what values are filled in the but the dimensions need to be correct.
batch_size = 32
example_input = torch.randn(
    batch_size, input_size, requires_grad=True)
example_labels = torch.randint(0, num_classes, (batch_size,))
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
optimizer_graph_path = 'optimizer_graph.onnx'
print(f"Writing optimizer graph to \"{optimizer_graph_path}\".")
onnx.save(onnx_optimizer, optimizer_graph_path)

print("Checking optimizer graph...")
onnx_optimizer = onnx.load(optimizer_graph_path)
onnx.checker.check_model(onnx_optimizer)
print("✅ Optimizer graph should be okay.")

print("✅ Done.")
