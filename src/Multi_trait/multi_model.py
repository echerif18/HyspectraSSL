import sys
import os

parent_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, parent_dir)

from transformation_utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F


####### EfficientNet_B0 #########
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, se_ratio):
        super(SqueezeExcite, self).__init__()
        reduced_channels = max(1, int(in_channels * se_ratio))
        self.se_reduce = nn.Conv1d(in_channels, reduced_channels, 1)
        self.se_expand = nn.Conv1d(reduced_channels, in_channels, 1)
    
    def forward(self, x):
        se = F.adaptive_avg_pool1d(x, 1)
        se = self.se_reduce(se)
        se = F.relu(se)
        se = self.se_expand(se)
        se = torch.sigmoid(se)
        return x * se

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, se_ratio):
        super(MBConvBlock, self).__init__()
        self.expand_ratio = expand_ratio
        self.stride = stride
        self.se_ratio = se_ratio
        self.has_residual = (stride == 1 and in_channels == out_channels)
        self.swish = Swish()  # Use the Swish activation
        
        expanded_channels = in_channels * expand_ratio
        self.expand_conv = nn.Conv1d(in_channels, expanded_channels, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm1d(expanded_channels)
        self.depthwise_conv = nn.Conv1d(expanded_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, groups=expanded_channels, bias=False)
        self.bn1 = nn.BatchNorm1d(expanded_channels)
        self.se = SqueezeExcite(expanded_channels, se_ratio)
        self.project_conv = nn.Conv1d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        identity = x
        
        out = self.expand_conv(x)
        out = self.bn0(out)
        out = self.swish(out)#F.relu(out)
        
        out = self.depthwise_conv(out)
        out = self.bn1(out)
        out = self.swish(out) #F.relu(out)
        
        out = self.se(out)
        
        out = self.project_conv(out)
        out = self.bn2(out)
        
        if self.has_residual:
            out += identity
        return out

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000):
        super(EfficientNetB0, self).__init__()
        self.stem_conv = nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm1d(32)
        self.swish = Swish()  # Use the Swish activation
        
        self.block1a = MBConvBlock(32, 16, expand_ratio=1, stride=1, se_ratio=0.25)
        self.block2a = MBConvBlock(16, 24, expand_ratio=6, stride=2, se_ratio=0.25)
        self.block2b = MBConvBlock(24, 24, expand_ratio=6, stride=1, se_ratio=0.25)
        self.block3a = MBConvBlock(24, 40, expand_ratio=6, stride=2, se_ratio=0.25)
        self.block3b = MBConvBlock(40, 40, expand_ratio=6, stride=1, se_ratio=0.25)
        self.block4a = MBConvBlock(40, 80, expand_ratio=6, stride=2, se_ratio=0.25)
        self.block4b = MBConvBlock(80, 80, expand_ratio=6, stride=1, se_ratio=0.25)
        self.block4c = MBConvBlock(80, 80, expand_ratio=6, stride=1, se_ratio=0.25)
        self.block5a = MBConvBlock(80, 112, expand_ratio=6, stride=1, se_ratio=0.25)
        self.block5b = MBConvBlock(112, 112, expand_ratio=6, stride=1, se_ratio=0.25)
        self.block5c = MBConvBlock(112, 112, expand_ratio=6, stride=1, se_ratio=0.25)
        self.block6a = MBConvBlock(112, 192, expand_ratio=6, stride=2, se_ratio=0.25)
        self.block6b = MBConvBlock(192, 192, expand_ratio=6, stride=1, se_ratio=0.25)
        self.block6c = MBConvBlock(192, 192, expand_ratio=6, stride=1, se_ratio=0.25)
        self.block6d = MBConvBlock(192, 192, expand_ratio=6, stride=1, se_ratio=0.25)
        self.block7a = MBConvBlock(192, 320, expand_ratio=6, stride=1, se_ratio=0.25)
        
        self.head_conv = nn.Conv1d(320, 1280, kernel_size=1, bias=False)
        self.head_bn = nn.BatchNorm1d(1280)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.swish(x) #F.relu(x)
        
        x = self.block1a(x)
        x = self.block2a(x)
        x = self.block2b(x)
        x = self.block3a(x)
        x = self.block3b(x)
        x = self.block4a(x)
        x = self.block4b(x)
        x = self.block4c(x)
        x = self.block5a(x)
        x = self.block5b(x)
        x = self.block5c(x)
        x = self.block6a(x)
        x = self.block6b(x)
        x = self.block6c(x)
        x = self.block6d(x)
        x = self.block7a(x)
        
        x = self.head_conv(x)
        x = self.head_bn(x)
        x = self.swish(x) #F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
