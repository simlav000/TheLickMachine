import subprocess
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from DataSetLoader import makeLoader
from TheLickMachine import TheLickMachine
import torch

git_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], stderr=subprocess.STDOUT).decode("utf-8").strip()

# Used for data augmentation
transform_train = transforms.Compose([
    # transforms.RandomAffine(degrees=30, translate=(0.08, 0.08)), # Add translation + rot. (helps with overfitting)
    # transforms.Normalize((0.5,), (0.5,))# Normalize data
    ])
# Create data loader for required type
types = ["solo", "combo", "external"]
type = 0
(loader_train, loader_test) = makeLoader(types[type], workers=2, transform_train=transform_train, test_ratio=0.2)

# Initializing
model = TheLickMachine()
learning_rate = 0.0005
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-6) # With l2 regularization

# Training info
num_epochs = 10
for epoch in range(num_epochs):
    epoch_loss = 0
    for (i, (input_batch, label_batch)) in enumerate(loader_train, 0):
        output_batch = model(input_batch)
        loss_batch = criterion(output_batch, label_batch.view(label_batch.shape[0], 1))
        epoch_loss += loss_batch.item()
        # Zero the gradient
        optimizer.zero_grad()
        # Get gradient
        loss_batch.backward()
        # Gradient descent step
        optimizer.step()
    print(f"epoch: {epoch}, epoch_loss: {epoch_loss}")

loader_train = None
loader_test = None

# Save model to be used later
torch.save(model.state_dict(), git_root+"/model.pth")
