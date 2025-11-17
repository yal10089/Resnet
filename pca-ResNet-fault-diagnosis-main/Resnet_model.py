"""Standalone ResNet implementation used for the fault diagnosis baseline.

This file mirrors the structure from ``model/ResNetModel.py`` but keeps the
interface intentionally small so it can be imported by the standalone training
and testing scripts (``Resnet_train.py`` and ``Resnet_test.py``).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """A simplified residual block for single-channel vibration images."""

    def __init__(self, in_channels: int, out_channels: int, stride, padding: int = 1) -> None:
        super().__init__()
        if isinstance(stride, int):
            stride = [stride, stride]

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = nn.Sequential()
        if stride[0] != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer(x)
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    """Small ResNet variant used in the original paper/code."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv2 = self._make_layer(BasicBlock, 128, [[1, 1], [1, 1]])
        self.conv3 = self._make_layer(BasicBlock, 128, [[2, 1], [1, 1]])
        self.conv4 = self._make_layer(BasicBlock, 128, [[2, 1], [1, 1]])
        self.conv5 = self._make_layer(BasicBlock, 128, [[2, 1], [1, 1]])
        self.fc = nn.Linear(128, num_classes)
        self._init_weights()

    def _make_layer(self, block, out_channels: int, strides) -> nn.Sequential:
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.reshape(out.size(0), -1)
        out = torch.log_softmax(self.fc(out), dim=1)
        return out

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.001)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_parameter_number(net: nn.Module, name: str):
    total_num = sum(p.numel() for p in net.parameters())
    return {"name: {} ->:{}".format(name, total_num)}


if __name__ == "__main__":
    model = ResNet(num_classes=10)
    model.eval()
    inputs = torch.randn(2, 1, 32, 32)
    params = get_parameter_number(model, "resnet_model")
    y = model(inputs)
    print(y.size(), params)
