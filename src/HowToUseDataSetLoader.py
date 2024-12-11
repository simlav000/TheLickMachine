from PlotFour import plot_four
from DataSetLoader import makeLoader
import torchvision.transforms as transforms

transform_train = transforms.Compose(
    [
        # transforms.RandomAffine(
        #     degrees=30, translate=(0.08, 0.08)
        # ),  # Add translation + rot. (helps with overfitting)
        # transforms.Normalize((0.5,), (0.5,)),  # Normalize data
    ]
)

types = ["solo", "combo", "external"]
type = 2
(loader_train, loader_test) = makeLoader(
    types[type], workers=2, transform_train=transform_train
)
for i, (x, y) in enumerate(loader_test, 0):
    plot_four(x.numpy()[0], name=types[type])
    print(y[0])
loader_train = None
loader_test = None
