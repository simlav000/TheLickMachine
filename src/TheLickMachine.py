import torch
import torch.nn as nn
import torch.nn.functional


class TheLickMachine(nn.Module):
    def __init__(self):
        super(TheLickMachine, self).__init__()

        self.input_shape = (4, 128, 157)
        # 16k samples per second * 5 seconds = 80k total samples
        # 80k samples / 512 samples per successive frames ~ 157 horizontal frames
        #
        # each frame approximates frequency over the next 2048 samples ~ .128 seconds
        # space between frames is 512 samples ~ 0.032 seconds
        #
        # So each bin contains around .128 sec of audio,
        # and the start of each bin is shifted by 0.032 seconds

        # output channel sizes for the convolutional neural network
        self.cout1 = 16
        self.cout2 = 32

        self.conv_block1 = nn.Sequential(
            # My idea is to start with large kernel sizes since the features
            # we're looking at are relatively large. We reduce it as we go. We
            # use Same padding to keep the spatial dimensions the same.
            nn.Conv2d(in_channels=4, out_channels=self.cout1, kernel_size=9, padding=5),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=self.cout1,
                out_channels=self.cout2,
                kernel_size=7,
                padding=3,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.cout2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Placeholder field to show there is a fully connected block
        # It will be defined dynamically in _init_fc so we don't need to
        # manually compute flattened shapes after changing the above convblocks
        self.flattened_size = self.getFlattened()

        # output channel sizes for the dense layers
        self.lout1 = 128

        self.fully_connected = nn.Sequential(
            nn.Linear(self.flattened_size, self.lout1),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(self.lout1, 1),  # Single output for binary classification
        )

    def getFlattened(self):
        # Passes a dummy tensor to model to dynamically compute shape of
        # fully-connected block.
        dummy_input = torch.zeros(self.input_shape).unsqueeze(0)
        conv_output = self.conv_block1(dummy_input)
        flattened_size = conv_output.numel() // conv_output.size(0)
        return flattened_size

    def forward(self, x, verbose=False):
        input_shape = x.shape
        x = self.conv_block1(x)

        x = x.view(
            x.shape[0], self.flattened_size
        )  # Reshape for dense layer (batch_size, flattened_size)

        conv1_shape = x.shape
        x = self.fully_connected(x)

        flat_shape = x.shape
        x = torch.sigmoid(x)  # Sigmoid for Binary Cross-Entropy Loss

        if verbose:
            print(f"Input shape: {input_shape}")
            print(f"Conv1 shape: {conv1_shape}")
            print(f"Flattened shape: {flat_shape}")
        return x
