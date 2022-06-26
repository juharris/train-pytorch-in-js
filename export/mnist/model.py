import torch
import torch.nn as nn
import torch.nn.functional as F

# Modified from https://kirenz.github.io/deep-learning/docs/mnist-pytorch.html

NUM_CLASSES=10


# This model had issues exporting to an ONNX file.
class MnistConvNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(MnistConvNet, self).__init__()
        # We did have padding='valid' but ONNX doesn't support that.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(32*32*3*3, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor):
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
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        if not self.is_export_mode:
            x = self.dropout1(x)
        x = self.fc2(x)
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