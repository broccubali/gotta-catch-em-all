import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(
            4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([x, out], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(Bottleneck(in_channels + i * growth_rate, growth_rate))
        self.denseblock = nn.Sequential(*layers)

    def forward(self, x):
        return self.denseblock(x)


class DenseNet(nn.Module):
    def __init__(self, num_blocks, growth_rate=32, reduction=0.5, num_classes=1000):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        num_channels = 2 * growth_rate

        # Initial convolution
        self.conv1 = nn.Conv2d(
            3, num_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        # Dense blocks and transition layers
        self.blocks = nn.ModuleList()
        for i in range(len(num_blocks)):
            block = DenseBlock(num_blocks[i], num_channels, growth_rate)
            self.blocks.append(block)
            num_channels += num_blocks[i] * growth_rate
            if i != len(num_blocks) - 1:
                transition = Transition(num_channels, int(num_channels * reduction))
                self.blocks.append(transition)
                num_channels = int(num_channels * reduction)

        # Final batch normalization
        self.bn2 = nn.BatchNorm2d(num_channels)

        # Linear classifier
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        out = self.pool1(F.relu(self.bn1(self.conv1(x))))
        for block in self.blocks:
            out = block(out)
        out = F.relu(self.bn2(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def DenseNet121(num_classes=1000):
    return DenseNet(num_blocks=[6, 12, 24, 16], growth_rate=32, num_classes=num_classes)


def DenseNet169(num_classes=1000):
    return DenseNet(num_blocks=[6, 12, 32, 32], growth_rate=32, num_classes=num_classes)


def DenseNet201(num_classes=1000):
    return DenseNet(num_blocks=[6, 12, 48, 32], growth_rate=32, num_classes=num_classes)


def DenseNet161(num_classes=1000):
    return DenseNet(num_blocks=[6, 12, 36, 24], growth_rate=48, num_classes=num_classes)



