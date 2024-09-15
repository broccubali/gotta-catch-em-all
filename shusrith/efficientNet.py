import torch
import torch.nn as nn
import torch.nn.functional as F

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, se_ratio=0.25, drop_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.expand_ratio = expand_ratio
        self.stride = stride
        self.drop_rate = drop_rate
        self.has_se = se_ratio is not None and 0 < se_ratio <= 1
        expanded_channels = in_channels * expand_ratio

        # Expansion phase
        self.expand_conv = nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(expanded_channels)
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, stride=stride, 
                                        padding=1, groups=expanded_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded_channels)
        
        # Squeeze and Excite layer
        if self.has_se:
            num_squeezed_channels = max(1, int(in_channels * se_ratio))
            self.se_reduce = nn.Conv2d(expanded_channels, num_squeezed_channels, kernel_size=1)
            self.se_expand = nn.Conv2d(num_squeezed_channels, expanded_channels, kernel_size=1)

        # Output phase
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Dropout
        self.dropout = nn.Dropout2d(p=drop_rate)

    def forward(self, x):
        identity = x
        x = F.relu6(self.bn0(self.expand_conv(x)))
        x = F.relu6(self.bn1(self.depthwise_conv(x)))
        
        # Squeeze and Excite
        if self.has_se:
            se_tensor = F.adaptive_avg_pool2d(x, 1)
            se_tensor = F.relu(self.se_reduce(se_tensor))
            se_tensor = torch.sigmoid(self.se_expand(se_tensor))
            x = x * se_tensor
        
        x = self.bn2(self.project_conv(x))
        
        if self.stride == 1 and x.shape == identity.shape:
            x = x + identity
        
        if self.training and self.drop_rate > 0:
            x = self.dropout(x)
        
        return x


class EfficientNet(nn.Module):
    def __init__(self, num_classes=1000, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2):
        super(EfficientNet, self).__init__()
        
        def round_filters(filters, width_coefficient):
            return int(filters * width_coefficient)
        
        def round_repeats(repeats, depth_coefficient):
            return int(repeats * depth_coefficient)
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, round_filters(32, width_coefficient), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(round_filters(32, width_coefficient)),
            nn.ReLU6(inplace=True),
        )
        
        # Define the architecture using Mobile Inverted Bottleneck Blocks (MBConvBlocks)
        self.blocks = nn.ModuleList([])

        # Block 1
        self.blocks.append(MBConvBlock(in_channels=round_filters(32, width_coefficient), out_channels=round_filters(16, width_coefficient), 
                                       expand_ratio=1, stride=1))
        
        # Block 2
        self.blocks.append(MBConvBlock(in_channels=round_filters(16, width_coefficient), out_channels=round_filters(24, width_coefficient), 
                                       expand_ratio=6, stride=2))
        
        # Block 3
        self.blocks.append(MBConvBlock(in_channels=round_filters(24, width_coefficient), out_channels=round_filters(40, width_coefficient), 
                                       expand_ratio=6, stride=2))

        # Block 4
        self.blocks.append(MBConvBlock(in_channels=round_filters(40, width_coefficient), out_channels=round_filters(80, width_coefficient), 
                                       expand_ratio=6, stride=2))

        # Block 5
        self.blocks.append(MBConvBlock(in_channels=round_filters(80, width_coefficient), out_channels=round_filters(112, width_coefficient), 
                                       expand_ratio=6, stride=1))

        # Block 6
        self.blocks.append(MBConvBlock(in_channels=round_filters(112, width_coefficient), out_channels=round_filters(192, width_coefficient), 
                                       expand_ratio=6, stride=2))

        # Block 7
        self.blocks.append(MBConvBlock(in_channels=round_filters(192, width_coefficient), out_channels=round_filters(320, width_coefficient), 
                                       expand_ratio=6, stride=1))
        
        self.head = nn.Sequential(
            nn.Conv2d(round_filters(320, width_coefficient), round_filters(1280, width_coefficient), kernel_size=1, bias=False),
            nn.BatchNorm2d(round_filters(1280, width_coefficient)),
            nn.ReLU6(inplace=True),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(round_filters(1280, width_coefficient), num_classes)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Example of using EfficientNet
model = EfficientNet(num_classes=1000, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2)
x = torch.randn(1, 3, 224, 224)
output = model(x)
print(output.shape)
