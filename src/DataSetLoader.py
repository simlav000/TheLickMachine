import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
# Plot spectogram

# get location of dataset
pwd = os.environ["PWD"]
with open(pwd + "/path.txt", "r") as file:
    pathToDataset = file.read().strip()

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
def makeLoader(type, bs=64, shuffle=True, workers=0, test_ratio=0.2, transform_train=None, transform_test=None):
    # Load data
    x_positive = np.load(pathToDataset + type + "_positives_vectorized.npy")
    y_positive = np.ones(x_positive.shape[0])
    x_negative = np.load(pathToDataset + type + "_negatives_vectorized.npy")
    y_negative = np.zeros(x_negative.shape[0])

    pts = int(y_positive.shape[0]*(1-test_ratio))#positive training size
    nts = int(y_negative.shape[0]*(1-test_ratio))#negative training size
    # Make loaders
    x_train = np.concatenate((x_positive[0:pts], x_negative[0:nts]), axis=0)
    y_train = np.concatenate((y_positive[0:pts], y_negative[0:nts]), axis=0)

    x_test = np.concatenate((x_positive[pts:], x_negative[nts:]), axis=0)
    y_test = np.concatenate((y_positive[pts:], y_negative[nts:]), axis=0)

    dataset_train = Data(x_train, y_train, transform_train)
    loader_train = DataLoader(dataset_train, batch_size=bs, shuffle=shuffle, num_workers=workers)

    dataset_test = Data(x_test, y_test, transform_test)
    loader_test = DataLoader(dataset_test, batch_size=bs, shuffle=shuffle, num_workers=workers)
    return (loader_train, loader_test)
