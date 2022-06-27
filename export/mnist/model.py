import torch
import torch.nn as nn
import torch.nn.functional as F

# Modified from https://kirenz.github.io/deep-learning/docs/mnist-pytorch.html

NUM_CLASSES=10


# This model had issues exporting to an ONNX file.
# "WARNING: The shape inference of org.pytorch.aten::ATen type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function."
# "Warning: Checker does not support models with experimental ops: ATen"
# "RuntimeError: /onnxruntime_src/orttraining/orttraining/python/orttraining_pybind_state.cc:789 onnxruntime::python::addObjectMethodsForTraining(pybind11::module&, onnxruntime::python::ExecutionProviderRegistrationFn)::<lambda(const pybind11::bytes&, const std::unordered_set<std::basic_string<char> >&, const std::unordered_set<std::basic_string<char> >&, std::string)> [ONNXRuntimeError] : 1 : FAIL : Node (ATen_8) output arg (19) type inference failed"
class MnistConvNet(nn.Module):
    def __init__(self, is_export_mode=False, num_classes=NUM_CLASSES):
        super(MnistConvNet, self).__init__()
        self.is_export_mode = is_export_mode
        # We did have padding='valid' but ONNX doesn't support that.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        if not is_export_mode:
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32*32*3*3, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        if not self.is_export_mode:
            x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        if not self.is_export_mode:
            x = self.dropout2(x)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        return output


class MnistNet(nn.Module):
    """
    A simple multi-layer perceptron model.
    """
    def __init__(self, is_export_mode=False, hidden_size=128, num_classes=NUM_CLASSES):
        super(MnistNet, self).__init__()
        self.is_export_mode = is_export_mode
        self.fc1 = torch.nn.Linear(28*28, hidden_size)
        if not is_export_mode:
            self.dropout1 = nn.Dropout(0.25)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor):
        # Flattening might be causing problems for ONNX Runtime Web, so we might need to try flatten in a pre-processing step.
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        if not self.is_export_mode:
            x = self.dropout1(x)
        x = self.fc2(x)
        # FIXME Don't use softmax because it doesn't work with ONNX Runtime Web.
        output = F.softmax(x, dim=1)
        return output

def cross_entropy(output, target):
    target = F.one_hot(target, NUM_CLASSES)
    # TODO Check if this is right, it ignore the target=0, `torch.log(1 - output)` case, but with it, loss became NaN.
    return - torch.sum(target*torch.log(output)) / target.size(0)


# Original was:
# loss_fn = F.cross_entropy
# but that won't work in ONNX Runtime Web.
loss_fn = cross_entropy