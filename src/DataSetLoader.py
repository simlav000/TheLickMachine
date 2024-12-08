import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
# Plot spectogram

# get location of dataset
pwd = os.environ["PWD"]
with open(pwd + "/path.txt", "r") as file:
    pathToDataset = file.read().strip()

transform_train = transforms.Compose([
    # transforms.RandomAffine(degrees=3, translate=(0.08, 0.08)), # Add translation + rot. (helps with overfitting)
    # transforms.Normalize((0.5,), (0.5,))# Normalize data
    ])
# Dataset
class Data(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()
        self.transform = transform

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.transform:
            x = self.transform(x)
        return x, y
    def __len__(self):
        return self.x.shape[0]

# Make functions that return the loaders
def makeSolo(bs=64, shuffle=True, workers=0):
    # Load data
    x_positive_solo = np.load(pathToDataset + "solo_positives_vectorized.npy")
    y_positive_solo = np.ones(x_positive_solo.shape[0])
    x_negative_solo = np.load(pathToDataset + "solo_negatives_vectorized.npy")
    y_negative_solo = np.zeros(x_negative_solo.shape[0])

    # Make loader
    x_solo = np.concatenate((x_positive_solo, x_negative_solo), axis=0)
    y_solo = np.concatenate((y_positive_solo, y_negative_solo), axis=0)
    solo_dataset = Data(x_solo, y_solo, transform_train)
    solo_loader = DataLoader(solo_dataset, batch_size=bs, shuffle=shuffle, num_workers=workers)
    return solo_loader


def makeCombo(bs=64, shuffle=True, workers=0):
    # Load data
    x_positive_combo = np.load(pathToDataset + "combo_positives_vectorized.npy")
    y_positive_combo = np.ones(x_positive_combo.shape[0])
    x_negative_combo = np.load(pathToDataset + "combo_negatives_vectorized.npy")
    y_negative_combo = np.zeros(x_negative_combo.shape[0])

    # Make loader
    x_combo = np.concatenate((x_positive_combo, x_negative_combo), axis=0)
    y_combo = np.concatenate((y_positive_combo, y_negative_combo), axis=0)
    combo_dataset = Data(x_combo, y_combo, transform_train)
    combo_loader = DataLoader(combo_dataset, batch_size=bs, shuffle=shuffle, num_workers=workers)
    return combo_loader

def makeExt(bs=64, shuffle=True, workers=0):
    # Load data
    x_positive_ext = np.load(pathToDataset + "external_positives_vectorized.npy")
    y_positive_ext = np.ones(x_positive_ext.shape[0])
    x_negative_ext = np.load(pathToDataset + "external_negatives_vectorized.npy")
    y_negative_ext = np.zeros(x_negative_ext.shape[0])

    # Make loader
    x_ext = np.concatenate((x_positive_ext, x_negative_ext), axis=0)
    y_ext = np.concatenate((y_positive_ext, y_negative_ext), axis=0)
    ext_dataset = Data(x_ext, y_ext, transform_train)
    ext_loader = DataLoader(ext_dataset, batch_size=bs, shuffle=shuffle, num_workers=workers)
    return ext_loader
