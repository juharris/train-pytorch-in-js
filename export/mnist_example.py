import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxruntime.training.experimental import export_gradient_graph

from optim.adam import AdamOnnxGraphBuilder

# Batch sizes for training and testing
batch_size = 64

input_size = (28, 28)

# How many batches before logging training status
log_interval = 10

# Number of target classes in the MNIST data
num_classes = 10


# Modified from https://kirenz.github.io/deep-learning/docs/mnist-pytorch.html

class MnistNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MnistNet, self).__init__()
        # FIXME Use Linear layers since Conv2d isn't supported in ONNX.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding='valid')
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3,
                               stride=1, padding='valid')
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        return output


# We need a custom loss function to load the graph in an InferenceSession in ONNX Runtime Web.
# You can still make the gradient graph with torch.nn.CrossEntropyLoss() and this test will pass.


def cross_entropy(output, target):
    target = F.one_hot(target, num_classes=num_classes)
    return - torch.sum(target*torch.log(output)) / target.size(0)


loss_fn = cross_entropy


model = MnistNet(num_classes)


# We need a place to save the ONNX graph.
gradient_graph_path = 'mnist_gradient_graph.onnx'

# We need example input for the ONNX model.
# It doesn't matter what values are filled in the but the dimensions need to be correct.
example_input = torch.randn(
    batch_size,1, input_size[0], input_size[1], requires_grad=True)
example_labels = torch.randint(0, num_classes, (batch_size,))

print(f"Writing gradient graph to \"{gradient_graph_path}\".")
export_gradient_graph(
    model, loss_fn, example_input, example_labels, gradient_graph_path, opset_version=15)
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
