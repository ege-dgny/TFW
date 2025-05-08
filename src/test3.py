import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from tqdm import tqdm

from time import time

from torch import dtype
from torchvision import datasets, transforms

SEED = 11111
#random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)  # If using GPU
torch.cuda.manual_seed_all(SEED)  # For multi-GPU
torch.backends.cudnn.deterministic = True  # Ensures deterministic results
torch.backends.cudnn.benchmark = False  # Can slow training but ensures reproducibility
torch.manual_seed(SEED)

#############################
# 1. Data Preparation
#############################

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.MNIST(
    root='./MNIST_data',
    download=True,
    train=True,
    transform=transform
)
valset = datasets.MNIST(
    root='./MNIST_data',
    download=True,
    train=False,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False)


#############################
# 2. Define Custom Network
#############################

class MyNet(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)

    def forward(self, x, return_features=False):

        h1 = F.relu(self.fc1(x))  # shape: (batch_size, hidden1)

        h2 = F.relu(self.fc2(h1))  # shape: (batch_size, hidden2)

        out = F.log_softmax(self.fc3(h2), dim=1)  # shape: (batch_size, output_size)

        if return_features:
            return h1, h2, out
        else:
            return out


#############################
# 3. Instantiate Model & Parameters
#############################
def testalpha(alpha, lr):
    # Dimensions
    input_size = 784  # 28*28
    hidden1_size = 64
    hidden2_size = 32
    output_size = 10

    model = MyNet(input_size, hidden1_size, hidden2_size, output_size)


    HAM1 = np.diag(np.arange(1, hidden1_size+1)/1.0)
    HAM1 -= np.diag(np.arange(1, hidden1_size)/2,1)
    HAM1 -= np.diag(np.arange(1, hidden1_size) / 2, -1)

    HAM2 = np.diag(np.arange(1, hidden2_size+1)/1.0)
    HAM2 -= np.diag(np.arange(1, hidden2_size) / 2, 1)
    HAM2 -= np.diag(np.arange(1, hidden2_size) / 2, -1)

    M1 = torch.from_numpy(HAM1.astype(np.float32))
    M2 = torch.from_numpy(HAM2.astype(np.float32))

    #############################
    # 4. Move to MPS (if available) or CPU
    #############################
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)
    M1 = M1.to(device)
    M2 = M2.to(device)
    #M3 = M3.to(device)

    #############################
    # 5. Define Optimizer & Loss
    #############################

    params = list(model.parameters())

    optimizer = optim.SGD(params, lr=lr)
    criterion = nn.NLLLoss()

    #############################
    # 6. Training Loop
    #############################

    # epochs = 100
    #alpha = 0.25  # Weighting factor for norm-squared terms

    # reg = 5
    trainrep = 50
    number = 1

    time0 = time()
    model.train()

    for e in range(number * (trainrep)):
        running_loss = 0.0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            images = images.view(images.shape[0], -1)

            optimizer.zero_grad()

            h1, h2, out = model(images, return_features=True)

            loss_ce = criterion(out, labels)

            #print(h1.shape)
            #print(M1.shape)
            loss_h1 = torch.matmul(h1, M1).norm(p=1)
            loss_h2 = torch.matmul(h2, M2).norm(p=1)
            #loss_out = torch.matmul(out, M3).norm()

            loss_regularization = loss_h1 + loss_h2# + loss_out

            # Total loss
            # if e % (reg + trainrep) < reg:
            #     loss = alpha * loss_regularization + 1/h1.norm() + 1/h2.norm()
            #     for g in optimizer.param_groups:
            #         g['lr'] = lr*100
            # else:
            loss = loss_ce + alpha * loss_regularization
                # for g in optimizer.param_groups:
                #     g['lr'] = lr

            loss.backward()
            optimizer.step()

            running_loss += loss_ce.item()

        print(f"  {alpha}   {lr}     Epoch {e + 1}/{number * (trainrep)} - Classification Loss: {running_loss / len(trainloader):.4f} - Filter Loss: {loss_regularization}")
        plt.plot(range(hidden1_size), np.mean(h1.detach().numpy(), axis=0), 'r')
        plt.plot(range(hidden2_size), np.mean(h2.detach().numpy(), axis=0), 'b')
        plt.title(f"Alpha: {alpha}      lr = {lr}")
        plt.show()

    #print(f"\nTraining Time (minutes): {(time() - time0) / 60:.2f}")

    #############################
    # 7. Validation Loop
    #############################

    model.eval()

    correct_count, all_count = 0, 0

    for images, labels in valloader:
        images, labels = images.to(device), labels.to(device)

        images = images.view(images.shape[0], -1)

        with torch.no_grad():
            out = model(images)  # returns log-softmax

        ps = torch.exp(out)  # shape: (batch_size, 10)

        ps_cpu = ps.cpu().numpy()
        labels_cpu = labels.cpu().numpy()

        pred_labels = np.argmax(ps_cpu, axis=1)

        correct_count += (pred_labels == labels_cpu).sum()
        all_count += len(labels_cpu)

    #print(f"\nNumber of Images Tested = {all_count}")
    #print(f"Model Accuracy = {correct_count / all_count:.4f}")

    return correct_count / all_count

if __name__ == "__main__":

    print(testalpha(0.000002, 0.005))

    # errors = []
    # for alpha in np.linspace(0.00001, 0.5, num=20):
    #     errors.append(testalpha(alpha))
    # plt.plot(np.linspace(0.00001, 0.5, num=20), errors)
    # plt.xlabel("alpha")
    # plt.ylabel("Accuracy")
    # plt.show()

    # errors = []
    # for lr in np.linspace(-3.2, -2, 10):
    #     errors.append(testalpha(0.1, lr=pow(10, lr)))
    # plt.plot(np.linspace(0.00001, 0.5, num=20), errors)
    # plt.xlabel("lr")
    # plt.ylabel("Accuracy")
    # plt.title("alpha = 0.1")
    # plt.show()