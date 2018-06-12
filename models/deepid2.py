"""
    DeepID2 part with PyTorch.
"""
import torch
import torch.nn as nn


class BasicConv2d(nn.Module):
    def __init__(self, inp, outp, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(inp, outp,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.prelu = nn.PReLU(init=0.25)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        out = self.maxpool(x)

        return out


def conv_4(inp, outp):
    return nn.Sequential(
        nn.Conv2d(inp, outp, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True)
    )


class DeepID2(nn.Module):
    def __init__(self):
        super(DeepID2, self).__init__()
        # special setting
        self.input_size = (128, 128, 3)
        self.mean =None
        self.std = None

        self.feature = nn.Sequential(
            BasicConv2d(3, 20, kernel_size=5, stride=1),
            BasicConv2d(20, 40, kernel_size=3, stride=1),
            BasicConv2d(40, 64, kernel_size=3, stride=1)
        )

        self.conv_4 = conv_4(64, 64)

    def forward(self, x):
        x_pool3 = self.feature(x)
        x = self.conv_4(x_pool3)
        out = torch.add(x_pool3, x)

        return out
