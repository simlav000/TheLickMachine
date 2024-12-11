from sys import prefix
import torch
import numpy as np
from TheLickMachine import TheLickMachine
from DataSetLoader import makeLoader
import subprocess
from DataSetLoader import Data
from typing import cast

# get path to root
git_root = (
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.STDOUT
    )
    .decode("utf-8")
    .strip()
)
# Get cuda device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device used: {device}")

# Load model
model = TheLickMachine()
model.load_state_dict(torch.load(git_root + "/model.pth", weights_only=True))
model.eval()  # Set to eval mode

# Model tester
def test(loader, model, prefix="", device=torch.device("cpu")):
    model = model.to(device)
    rounding_lambda = lambda a: round(a)
    with torch.no_grad():
        correct_output = 0
        for i, (input_batch, label_batch) in enumerate(loader, 0):
            # Tranfer to right device
            input_batch = input_batch.to(device)

            # Now test
            output_batch = model(input_batch).view(-1).to("cpu")
            output_batch_rounded = np.vectorize(rounding_lambda)(output_batch)
            correct_output += sum(x == y for x, y in zip(output_batch_rounded, label_batch))

    data = cast(Data, loader.dataset)
    total_labels = len(data)
    accuracy = correct_output / total_labels
    print(f"{prefix}accuracy: {accuracy}")

# Load some data
types = ["solo", "combo", "external"]
t = 0
(_, loader_test) = makeLoader(types[t], workers=2, test_ratio=0.2) # test_raito has to be the same as in the training
test(loader_test, model, prefix="solo ", device=device)

t = 1
(_, loader_test) = makeLoader(types[t], workers=2, test_ratio=0.2) # test_raito has to be the same as in the training
test(loader_test, model, prefix="combo ", device=device)

