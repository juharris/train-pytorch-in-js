import itertools
import math

import torch
import torch.nn.functional as F
import torch.optim
from torchvision import datasets, transforms

from .model import MnistNet, loss_fn

# Modified from https://kirenz.github.io/deep-learning/docs/mnist-pytorch.html

# Check to make sure the model can be trained and evaluates decently on the test set.


def train(model, device, train_loader, optimizer, epoch, log_interval, train_set_limit=-1):
    model.train()
    if train_set_limit < 0:
        train_set_limit = len(train_loader)
    else:
        train_loader = itertools.islice(train_loader, train_set_limit)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [Batch {batch_idx}/{train_set_limit} ({ 100. * batch_idx / train_set_limit:.0f}%)]\tLoss: { loss.item():.6f}")


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def start_training():
    # Random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)

    # Batch sizes for training and testing
    batch_size = 64
    test_batch_size = batch_size

    # Use -1 to train on all the data.
    train_set_limit = -1
    # train_set_limit = 1000
    train_set_limit = math.floor(float(train_set_limit) / batch_size)

    # How many batches before logging training status
    log_interval = 10

    n_epochs = 2

    learning_rate = 3e-4

    # Get CPU or GPU device for training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Training on {device}.")

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}

    # CUDA settings
    if torch.cuda.is_available():
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # The scaled mean and standard deviation of the MNIST dataset (precalculated)
    data_mean = 0.1307
    data_std = 0.3081

    # Convert input images to tensors and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((data_mean,), (data_std,))
    ])

    # Get the MNIST data from torchvision
    training_data = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    test_data = datasets.MNIST('../data', train=False,
                              transform=transform)

    # Define the data loaders that will handle fetching of data
    train_loader = torch.utils.data.DataLoader(training_data, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)
    model = MnistNet(data_mean=data_mean, data_std=data_std).to(device)

    # Define the optimizer to user for gradient descent
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, n_epochs + 1):
        train(model, device, train_loader, optimizer,
              epoch, log_interval, train_set_limit)
        test(model, device, test_loader)


if __name__ == '__main__':
    start_training()
