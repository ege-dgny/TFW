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
import math
import matplotlib
import matplotlib.pyplot as plt

from src.wavelet_transforms import haar_wavelet_transform

SEED = 11111
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)  # If using GPU
torch.cuda.manual_seed_all(SEED)  # For multi-GPU
torch.backends.cudnn.deterministic = True  # Ensures deterministic results
torch.backends.cudnn.benchmark = False  # Can slow training but ensures reproducibility
torch.manual_seed(SEED)

os.environ['PYTHONHASHSEED'] = str(SEED)

plot_dir = "./mark1_0_random_mask2"
os.makedirs(plot_dir, exist_ok=True)


class StructureTrainer():
    def __init__(self, model, input_size, hidden1_size, hidden2_size, output_size, epochs, learning_rate,
                 optimizer, structure_loss_coeff, criterion=nn.NLLLoss(), device="cpu"):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = epochs
        self.lr = learning_rate
        self.gamma = structure_loss_coeff
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.input_size = input_size

        mask1 = np.zeros((1, 1, 8, 8), dtype=int)
        mask1[:, :, :, 4:] = 0.5
        mask1[:, :, 4:, :] = 0.5
        mask1[:, :, 4:, 4:] = 2


        mask2 = np.zeros((1, 1, 6, 6), dtype=int)
        mask2[:, :, :, 4:] = 0.5
        mask2[:, :, 4:, :] = 0.5
        mask2[:, :, 4:, 4:] = 2


        self.mask1 = torch.tensor(mask1, dtype=torch.float32, device=self.device)
        self.mask2 = torch.tensor(mask2, dtype=torch.float32, device=self.device)

        self.model = self.model.to(self.device)

    def train_epoch(self, train_loader, val_loader=None):
        self.model.train()
        validation_accuracies = []

        for e in range(self.num_epochs):
            running_class_loss = 0.0
            running_struct_loss = 0.0

            for img, label in train_loader:
                img, label = img.to(self.device), label.to(self.device)
                #print(img.shape)
                img = img.view(img.shape[0], -1)

                self.optimizer.zero_grad()
                h1, h2, out = self.model(img, return_features=True)
                classification_loss = self.criterion(out, label)

                h1_img = h1.view(h1.size(0), 1, 8, 8)
                h2_img = h2.view(h2.size(0), 1, 6, 6)

                latent_wavelet1 = haar_wavelet_transform(h1_img)
                latent_wavelet2 = haar_wavelet_transform(h2_img)

                structure_loss1 = latent_wavelet1 * self.mask1
                structure_loss1 = structure_loss1.norm()

                structure_loss2 = latent_wavelet2 * self.mask2
                structure_loss2 = structure_loss2.norm()

                structure_loss = structure_loss1 + structure_loss2

                loss = classification_loss + self.gamma * structure_loss

                loss.backward()
                self.optimizer.step()

                running_class_loss += classification_loss.item()
                running_struct_loss += structure_loss.item()

            # Plot activations
            plt.figure(figsize=(8, 6))
            plt.plot(range(self.hidden1_size), np.mean(h1.detach().cpu().numpy(), axis=0), 'r', label="Hidden Layer 1")
            plt.plot(range(self.hidden2_size), np.mean(h2.detach().cpu().numpy(), axis=0), 'b', label="Hidden Layer 2")
            plt.legend()
            plt.title(f'Epoch {e} - Hidden Layer Activations')
            features_plot_filename = os.path.join(plot_dir, f"epoch_{e}_features.png")
            plt.savefig(features_plot_filename)
            plt.close()

            avg_class_loss = running_class_loss / len(train_loader)
            avg_struct_loss = running_struct_loss / len(train_loader)

            print(f"Epoch {e}: Classification Loss = {avg_class_loss:.4f}, Structure Loss = {avg_struct_loss:.4f}")

            # Evaluate on validation set
            if val_loader:
                val_acc = self.eval_epoch(val_loader)
                validation_accuracies.append(val_acc)
                print(f"Epoch {e}: Validation Accuracy = {val_acc:.4f}")

        # Plot validation accuracy
        if validation_accuracies:
            plt.figure(figsize=(8, 6))
            plt.plot(range(self.num_epochs), validation_accuracies, marker='o', linestyle='-',
                     label='Validation Accuracy')
            plt.title('Validation Accuracy per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(plot_dir, "validation_accuracy.png"))
            plt.show()

        # Plot and save weight matrices
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                weight_matrix = param.detach().cpu().numpy()
                plt.figure(figsize=(8, 6))
                plt.imshow(weight_matrix, aspect='auto', cmap='viridis')
                plt.title(f'Weight Matrix: {name}')
                plt.colorbar()
                weight_plot_filename = os.path.join(plot_dir, f"final_{name.replace('.', '_')}_weights.png")
                plt.savefig(weight_plot_filename)
                plt.show()

        model_path = os.path.join(plot_dir, "trained_model.pt")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        return 1

    def eval_epoch(self, val_loader, model_path=None):
        correct_count, all_count = 0, 0
        self.model.eval()

        for images, labels in val_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            images = images.view(images.shape[0], -1)

            with torch.no_grad():
                out = self.model(images)  # returns log-softmax

            ps = torch.exp(out)  # shape: (batch_size, 10)
            pred_labels = torch.argmax(ps, dim=1)
            correct_count += (pred_labels == labels).sum().item()
            all_count += labels.size(0)

        accuracy = correct_count / all_count

        # Save accuracy to ./plot_dir/accuracy.txt
        accuracy_path = os.path.join(plot_dir, "accuracy.txt")
        os.makedirs(os.path.dirname(accuracy_path), exist_ok=True)
        with open(accuracy_path, "w") as f:
            f.write(f"{accuracy:.6f}\n")

        return accuracy
