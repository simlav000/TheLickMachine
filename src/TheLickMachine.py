import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional

def main():
    load_dataset()



def load_dataset():
    Simon = r"/mnt/c/Users/Simon/OneDrive/School/00 Fall 2024/COMP 451 - Fundamentals of Machine Learning/Project/Data/dataset_vectorized/"
    path = Simon

    load_things_other_than_solo = False

    x_positive_solo  = np.load(path + "solo_positives_vectorized.npy")
    y_positive_solo  = np.ones(x_positive_solo.shape[0])
    x_negative_solo  = np.load(path + "solo_negatives_vectorized.npy")
    y_negative_solo  = np.zeros(x_negative_solo.shape[0])

    if load_things_other_than_solo:
        x_positive_combo = np.load(path + "combo_positives_vectorized.npy")
        y_positive_combo = np.ones(x_positive_combo.shape[0])
        x_negative_combo = np.load(path + "combo_negatives_vectorized.npy")
        y_negative_combo = np.zeros(x_negative_combo.shape[0])

        x_positive_ext   = np.load(path + "external_positives_vectorized.npy")
        y_positive_ext   = np.ones(x_positive_ext.shape[0])
        x_negative_ext   = np.load(path + "external_negatives_vectorized.npy")
        y_negative_ext   = np.zeros(x_negative_ext.shape[0])

        # A tuple (x,y)
        return (
            np.stack([
                x_positive_solo, x_positive_combo, x_positive_ext,
                x_negative_solo, x_negative_combo, x_negative_ext
            ]),
            np.stack([
                y_positive_solo, y_positive_combo, y_positive_ext,
                y_negative_solo, y_negative_combo, y_negative_ext
            ])
            )

    return (
        np.stack([x_positive_solo, x_negative_solo]),
        np.stack([y_positive_solo, y_negative_solo])
        )

class TheLickMachine(nn.Module):
    def __init__(self):
        super(TheLickMachine, self).__init__()

        self.input_shape = (4, 128, 157)

        self.conv_block1 = nn.Sequential(
            # My idea is to start with large kernel sizes since the features 
            # we're looking at are relatively large. We reduce it as we go. We
            # use Same padding to keep the spatial dimensions the same.
            nn.Conv2d(
                in_channels=4,
                out_channels=16,
                kernel_size=9,
                padding=5
            ),


            nn.ReLU(),

            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=7,
                padding=3
            ),

            nn.ReLU(),

            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Placeholder field to show there is a fully connected block
        # It will be defined dynamically in _init_fc so we don't need to
        # manually compute flattened shapes after changing the above convblocks
        self.fully_connected = nn.Sequential()
        self._init_fc()


    def _init_fc(self):
        # Passes a dummy tensor to model to dynamically compute shape of
        # fully-connected block.
        dummy_input = torch.zeros(self.input_shape)
        conv_output = self.conv_block1(dummy_input)
        flattened_size = conv_output.numel() // conv_output.size(0)

        # Define fully connected layers
        self.fully_connected = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(128, 1) # Single output for binary classification
        )


    def forward(self, x, verbose=False):
        input_shape = x.shape
        x = self.conv_block1(x)
        conv1_shape = x.shape
        x = self.fully_connected(x)
        flat_shape = x.shape
        x = torch.sigmoid(x) # Sigmoid for Binary Cross-Entropy Loss
        if verbose:
            print(f"Input shape: {input_shape}")
            print(f"Conv1 shape: {conv1_shape}")
            print(f"Flattened shape: {flat_shape}")
        return x


if __name__ == "__main__":
    #from Vectorize import plot_features
    main()
