import torch
import numpy as np
from TheLickMachine import TheLickMachine
from DataSetLoader import makeLoader
import subprocess
from DataSetLoader import Data
from typing import cast

git_root = (
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.STDOUT
    )
    .decode("utf-8")
    .strip()
)
# Load model
model = TheLickMachine()
model.load_state_dict(torch.load(git_root + "/model.pth", weights_only=True))
model.eval()  # Set to eval mode

# Load some data
types = ["solo", "combo", "external"]
t = 0
(loader_train, loader_test) = makeLoader(types[t], workers=2, test_ratio=0.2)

# Test the model
rounding_lambda = lambda a: round(a)
with torch.no_grad():
    correct_output = 0
    for i, (input_batch, label_batch) in enumerate(loader_test, 0):
        output_batch = np.vectorize(rounding_lambda)(model(input_batch).view(-1))
        correct_output += sum(x == y for x, y in zip(output_batch, label_batch))

data = cast(Data, loader_test.dataset)
total_labels = len(data)
accuracy = correct_output / total_labels
print(f"accuracy: {accuracy}")
