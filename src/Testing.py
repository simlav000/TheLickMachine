from typing import cast
import torch
import numpy as np
from TheLickMachine import TheLickMachine
from DataSetLoader import makeLoader
import subprocess
from DataSetLoader import Data

# get path to root
git_root = (
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.STDOUT
    )
    .decode("utf-8")
    .strip()
)
# Get cuda device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device used: {device}")

# Load model
model = TheLickMachine()
model.load_state_dict(torch.load(git_root + "/cnn.pth", weights_only=True))
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
            correct_output += sum(
                x == y for x, y in zip(output_batch_rounded, label_batch)
            )

    data = cast(Data, loader.dataset)
    total_labels = len(data)
    accuracy = correct_output / total_labels
    print(f"{prefix}accuracy: {accuracy}")


# Load some data
types = ["solo", "combo", "external", "external2"]
t = 0
(loader_train, loader_test) = makeLoader(
    types[t], workers=2, test_ratio=0.2
)  # test_raito has to be the same as in the training
test(loader_test, model, prefix="solo_test ", device=device)
test(loader_train, model, prefix="solo_train ", device=device)
print()

# t = 1
# (loader_train, loader_test) = makeLoader(types[t], workers=2, test_ratio=0.2) # test_raito has to be the same as in the training
# test(loader_test, model, prefix="combo_test ", device=device)
# test(loader_test, model, prefix="combo_train ", device=device)
# print()


t = 3
(loader_train, loader_test) = makeLoader(
    types[t], workers=2, test_ratio=0.2
)  # test_raito has to be the same as in the training
test(loader_test, model, prefix="external_test ", device=device)
test(loader_train, model, prefix="external_train ", device=device)
print()
