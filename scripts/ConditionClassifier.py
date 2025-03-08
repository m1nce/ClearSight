import torch
import torch.nn as nn

class ConditionClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ConditionClassifier, self).__init__()

        def depthwise_separable_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.conv1 = depthwise_separable_conv(3, 16)
        self.conv2 = depthwise_separable_conv(16, 32, stride=2)
        self.conv3 = depthwise_separable_conv(32, 64, stride=2)
        self.conv4 = depthwise_separable_conv(64, 128, stride=2)
    
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x