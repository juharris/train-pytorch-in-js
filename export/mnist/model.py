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
        output = softmax(x,dim=1)
        assert output.allclose(F.softmax(x, dim=1)), "The output was not similar to the PyTorch softmax."
        return output


class MnistNet(nn.Module):
    """
    A simple multi-layer perceptron model.
    """
    def __init__(self, is_export_mode=False, hidden_size=128, num_classes=NUM_CLASSES, data_mean=0.1307, data_std=0.3081):
        super(MnistNet, self).__init__()
        self.is_export_mode = is_export_mode
        self.data_mean = data_mean
        self.data_std = data_std
        self.fc1 = torch.nn.Linear(28*28, hidden_size)
        if not is_export_mode:
            self.dropout1 = nn.Dropout(0.25)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        if not self.is_export_mode:
            x = self.dropout1(x)
        x = self.fc2(x)
        output = self.softmax(x, dim=1)
        if not self.is_export_mode:
            assert output.allclose(F.softmax(x, dim=1)), "The output was not similar to the PyTorch softmax."
        return output

    def softmax(self, x, dim):
        # Don't use the built-in softmax because it doesn't work with ONNX Runtime Web.
        # [W:onnxruntime:, graph.cc:2624 InitFunctionBodyForNode] Function body initialization failed for node 'Softmax_4_Grad/SoftmaxGrad_0' optype SoftmaxGrad. Error message /home/juharri/workspace/onnx/onnxruntime/onnxruntime/core/graph/function.cc:788 onnxruntime::FunctionImpl::FunctionImpl(onnxruntime::Graph &, const onnxruntime::NodeIndex &, const onnx::FunctionProto &, const std::unordered_map<std::string, const onnx::FunctionProto *> &, std::vector<std::unique_ptr<onnxruntime::Function>> &, const logging::Logger &, bool) status.IsOK() was false. Resolve subgraph failed:This is an invalid model. In Node, ("0xbc2a58", Squeeze, "", -1) : ("n_as_vector": tensor(int64),"axis_zero": tensor(int64),) -> ("n",) , Error Node (0xbc2a58) has input size 2 not in range [min=1, max=1].

        # Can't use the -max trick for stability because we get an error when exporting the gradient graph.
        # RuntimeError: /onnxruntime_src/orttraining/orttraining/core/graph/gradient_builder_registry.cc:29 onnxruntime::training::GradientDef onnxruntime::training::GetGradientForOp(const onnxruntime::training::GradientGraphConfiguration&, onnxruntime::Graph*, const onnxruntime::Node*, const std::unordered_set<std::basic_string<char> >&, const std::unordered_set<std::basic_string<char> >&, const onnxruntime::logging::Logger&, std::unordered_set<std::basic_string<char> >&) gradient_builder != nullptr was false. The gradient builder has not been registered: ReduceMax for node ReduceMax_4
        # Offset by the max possible value to avoid overflow.
        # output = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
        output = torch.exp(x - (1 - self.data_mean) / self.data_std)
        output = output / output.sum(dim=dim, keepdim=True)
        return output
    
# x = torch.randn(1, 1, 28, 28)

def cross_entropy(output, target):
    target = F.one_hot(target, NUM_CLASSES)
    # Normally you do torch.log(output) and torch.log(1 - output) but that gives NaN.
    return - torch.sum(target*output + (1-target)*(1-output)) / target.size(0)


# Original was:
# loss_fn = F.cross_entropy
# but that won't work in ONNX Runtime Web.
loss_fn = cross_entropy
