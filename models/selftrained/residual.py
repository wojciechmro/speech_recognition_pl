import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Initialize a Residual Block, a building block commonly used in ResNet architectures. It allows the network to learn an identity mapping by adding original input (identity) to output of stacked convolutional layers. This addition helps gradients flow directly through the shortcut connection during backpropagation, avoiding gradient vanishing and making it easier to train deeper networks.

        Args:
        in_channels (int): Number of input channels (from previous layer)
        out_channels (int): Number of output channels (to next layer)
        stride (int, optional): Stride of the convolutional layers. Controls the spatial downsampling. Defaults to 1.
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,  # 3x3 kernel
            stride=stride,  # controls the output size reduction
            padding=1,  # ensures output dimensions match input dimensions if stride=1
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        # Downsampling layer (1x1 convolution) is used to match the dimensions of the input and output if the number of channels is different
        self.downsample = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        """
        Forward pass through the Residual Block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            torch.Tensor: Output tensor after applying residual connection and activation, of same shape as input.
        """
        identity = self.downsample(x) if self.downsample else x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        # Add identity (shortcut connection) to avoid vanishing gradients
        out += identity
        return self.relu(out)
