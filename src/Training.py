import subprocess
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from DataSetLoader import makeLoader, mergeLoaders
from TheLickMachine import TheLickMachine
import torch

# Get path to root
git_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], stderr=subprocess.STDOUT).decode("utf-8").strip()

# Get cuda device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device used: {device}")

# Used for data augmentation
transform_train = transforms.Compose([
    # transforms.RandomAffine(degrees=30, translate=(0.08, 0.08)), # Add translation + rot. (helps with overfitting)
    # transforms.Normalize((0.5,), (0.5,))# Normalize data
    ])

# Initializing
model = TheLickMachine()
criterion = nn.BCEWithLogitsLoss()

# Training
def train(loader, model, optimizer, criterion, device=torch.device("cpu"), num_epochs=10):
    model = model.to(device)
    for epoch in range(num_epochs):
        epoch_loss = 0
        for (i, (input_batch, label_batch)) in enumerate(loader, 0):
            # Tranfer to right device
            input_batch = input_batch.to(device)
            label_batch = label_batch.to(device)

            # Start training
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

types = ["solo", "combo", "external"]
# Train model
# Solo
t = 0
(loader_train_solo, _) = makeLoader(types[t], workers=2, transform_train=transform_train, test_ratio=0.2)
# print(f"Training on: {types[t]}")
# learning_rate = 0.0005
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-6) # With l2 regularization
# train(loader_train_solo, model, optimizer, criterion, device=device)
# print()

# Combo
t = 1
(loader_train_combo, _) = makeLoader(types[t], workers=2, transform_train=transform_train, test_ratio=0.2)
# print(f"Training on: {types[t]}")
# learning_rate = 0.0007
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-6) # With l2 regularization
# train(loader_train_combo, model, optimizer, criterion,num_epochs=5, device=device)
# print()

# Combo and Solo
loader_train_combosolo=mergeLoaders(loader_train_solo,loader_train_combo)
print(f"Training on: combo+solo")
learning_rate = 0.0002
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-6) # With l2 regularization
train(loader_train_combosolo, model, optimizer, criterion, num_epochs=20, device=device)
print()

# Save model to be used later
torch.save(model.state_dict(), git_root+"/model.pth")
