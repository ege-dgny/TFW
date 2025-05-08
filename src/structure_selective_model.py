import torch
import torch.nn as nn
import torch.nn.functional as F


class MyNet(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)

    def forward(self, x, return_features=False):
        print(x.shape)
        # x is expected to be of shape (batch_size, input_size)
        h1 = F.relu(self.fc1(x))  # shape: (batch_size, hidden1)
        h2 = F.relu(self.fc2(h1))  # shape: (batch_size, hidden2)
        out = F.log_softmax(self.fc3(h2), dim=1)  # shape: (batch_size, output_size)
        if return_features:
            return h1, h2, out
        else:
            return out
