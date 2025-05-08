import numpy as np
import scipy as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

from time import time
import os
import random

import matplotlib
import matplotlib.pyplot as plt

from src.structure_selective_model import MyNet
import src.wavelet_loss_trainer
from src.wavelet_transforms import haar_wavelet_transform

SEED = 11111
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)  # If using GPU
torch.cuda.manual_seed_all(SEED)  # For multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(SEED)

os.environ['PYTHONHASHSEED'] = str(SEED)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load full MNIST test set to split into validation and test
full_testset = datasets.MNIST(
    root='./MNIST_data',
    download=True,
    train=False,
    transform=transform
)

# Split the test set into validation and test (e.g., 5k each)
val_size = 5000
test_size = len(full_testset) - val_size
valset, testset = torch.utils.data.random_split(full_testset, [val_size, test_size],
                                                generator=torch.Generator().manual_seed(SEED))

# Load training set
trainset = datasets.MNIST(
    root='./MNIST_data',
    download=True,
    train=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

def main():
    plot_dir = "mark1_0"

    input_size = 784
    output_size = 10
    hidden1_size = 64
    hidden2_size = 36

    lr = 0.01
    structure_loss_weight = 0.005
    epoch = 10

    model = MyNet(input_size, hidden1_size, hidden2_size, output_size)

    params = list(model.parameters())
    optimizer = optim.SGD(params, lr=lr)
    criterion = nn.NLLLoss()


    device = "cpu"

    wavelet_loss_model = src.wavelet_loss_trainer.StructureTrainer(
        model, input_size, hidden1_size, hidden2_size, output_size,
        epoch, lr, optimizer, structure_loss_weight, criterion, device)

    # Train while validating
    wavelet_loss_model.train_epoch(trainloader, valloader)

    # Final test evaluation
    test_accuracy = wavelet_loss_model.eval_epoch(testloader)
    print(f"Final Test Accuracy: {test_accuracy:.4f}")

if __name__ == '__main__':
    main()