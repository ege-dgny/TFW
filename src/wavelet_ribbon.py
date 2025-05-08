import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from src.wavelet_transforms import haar_wavelet_transform

class WaveletRibbon(nn.Module):
    def __init__(self, input_size, output_size, depth, checkpoint_dir,  name, lr=10e-3, wavelet_family="haar"):
        super(WaveletRibbon, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.depth = depth
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint = os.path.join(self.checkpoint_dir, name+"w_r")
        self.wavelet_family = wavelet_family

        self.hidden_size = int(input_size/4)
        self.latent_layers = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size) for _ in range(depth)
        ])
        self.q1 = nn.Linear(self.hidden_size, self.output_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Created Wavelet Ribbon on {self.device}")
        self.to(self.device)


    def forward(self, x, return_features=False):
        init_wav = haar_wavelet_transform(x)
        B, C, H, W = init_wav.shape
        h_mid = H //2
        w_mid = W //2

        ll = init_wav[:, :, :h_mid, :w_mid]
        lh = init_wav[:, :, :h_mid, :w_mid]
        hl = init_wav[:, :, :h_mid, :w_mid]
        hh = init_wav[:, :, :h_mid, :w_mid]

        # x is expected to be of shape (batch_size, input_size)
        h1 = F.relu(self.fc1(x))  # shape: (batch_size, hidden1)
        h2 = F.relu(self.fc2(h1))  # shape: (batch_size, hidden2)
        out = F.log_softmax(self.fc3(h2), dim=1)  # shape: (batch_size, output_size)
        if return_features:
            return h1, h2, out
        else:
            return out

    def wavelet_neuron(self, wavelet):
        pass