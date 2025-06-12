import torch
import torch.nn as nn

gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu") #device = 'cuda'#'cuda:0' cpu

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class WeightNormalizationConv1D(nn.Module):
    def __init__(self, conv, data_init=False):
        super(WeightNormalizationConv1D, self).__init__()
        self.conv = nn.utils.weight_norm(conv, name='weight', dim=0)
        if data_init:
            self.conv.data = self.conv.data.normal_(mean=0, std=0.02)

    def forward(self, x):
        return self.conv(x)

class ResidualStack(nn.Module):
    def __init__(self, filters):
        super(ResidualStack, self).__init__()
        self.conv1 = WeightNormalizationConv1D(nn.Conv1d(filters, filters, kernel_size=3, dilation=1, padding=1), data_init=False) #
        self.conv2 = WeightNormalizationConv1D(nn.Conv1d(filters, filters, kernel_size=3, dilation=1, padding=1), data_init=False)
        self.conv3 = WeightNormalizationConv1D(nn.Conv1d(filters, filters, kernel_size=3, dilation=3, padding=3), data_init=False)
        self.conv4 = WeightNormalizationConv1D(nn.Conv1d(filters, filters, kernel_size=3, dilation=1, padding=1), data_init=False)
        self.conv5 = WeightNormalizationConv1D(nn.Conv1d(filters, filters, kernel_size=3, dilation=9, padding=9), data_init=False)
        self.conv6 = WeightNormalizationConv1D(nn.Conv1d(filters, filters, kernel_size=3, dilation=1, padding=1), data_init=False)

    def forward(self, x):
        c1 = self.conv1(x)
        lrelu1 = F.leaky_relu(c1)
        c2 = self.conv2(lrelu1)
        add1 = c2 + x

        lrelu2 = F.leaky_relu(add1)
        c3 = self.conv3(lrelu2)
        lrelu3 = F.leaky_relu(c3)
        c4 = self.conv4(lrelu3)
        add2 = c4 + add1

        lrelu4 = F.leaky_relu(add2)
        c5 = self.conv5(lrelu4)
        lrelu5 = F.leaky_relu(c5)
        c6 = self.conv6(lrelu5)
        add3 = c6 + add2

        return add3

class ConvBlock(nn.Module):
    def __init__(self, conv_dim, upsampling_factor):
        super(ConvBlock, self).__init__()
        self.conv_t = WeightNormalizationConv1D(nn.ConvTranspose1d(conv_dim, conv_dim, kernel_size=16, stride=upsampling_factor, padding=6), data_init=False) ##
        self.res_stack = ResidualStack(conv_dim)

    def forward(self, x):
        conv_t = self.conv_t(x)
        lrelu1 = F.leaky_relu(conv_t)
        res_stack = self.res_stack(lrelu1)
        lrelu2 = F.leaky_relu(res_stack)
        return lrelu2

class Generator(nn.Module):
    def __init__(self, latent_dim, input_shape):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        
        n_nodes = 64 * round(input_shape / 4) * 1
        self.fc = nn.Linear(latent_dim, n_nodes)
        self.conv_block = ConvBlock(64, 4)
        self.conv1d_out = WeightNormalizationConv1D(nn.Conv1d(64, 1, kernel_size=7, padding=3), data_init=False) ##

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 64, -1)
        x = self.conv_block(x)
        x = self.conv1d_out(x)
        return torch.tanh(x)


class Discriminator(nn.Module):
    def __init__(self, input_shape, n_classes=1):
        super(Discriminator, self).__init__()
        self.conv1 = spectral_norm(nn.Conv1d(1, 128, kernel_size=3, stride=2, padding=1))
        self.conv2 = spectral_norm(nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1))
        self.conv3 = spectral_norm(nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1))
        self.activation = nn.LeakyReLU(0.2)
        self.batchnorm = nn.BatchNorm1d(128)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.4)
        self.fc = spectral_norm(nn.Linear((input_shape // 8) * 128, n_classes))

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.batchnorm(x)
        x = self.activation(self.conv2(x))
        x = self.batchnorm(x)
        x = self.activation(self.conv3(x))
            
        out_map1 = x
        x = self.batchnorm(x)
        x = self.flatten(x)
        x = self.dropout(x)
        d_out_layer = self.fc(x)
        return d_out_layer, out_map1


class Discriminator_half(nn.Module):
    def __init__(self, input_shape, n_classes=1):
        super(Discriminator_half, self).__init__()
        self.conv1 = spectral_norm(nn.Conv1d(1, 128, kernel_size=3, stride=2, padding=1))
        self.conv2 = spectral_norm(nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1))
        self.conv3 = spectral_norm(nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1))
        self.activation = nn.LeakyReLU(0.2)
        self.batchnorm = nn.BatchNorm1d(128)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.4)
        self.fc = spectral_norm(nn.Linear((input_shape // 8) * 128, n_classes))
        self.globalpool = nn.AdaptiveAvgPool1d(62)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.batchnorm(x)
        x = self.activation(self.conv2(x))
        x = self.batchnorm(x)
        x = self.activation(self.conv3(x))
                    
        x = self.globalpool(x)
        
        out_map1 = x
        x = self.batchnorm(x)
        x = self.flatten(x)
        x = self.dropout(x)
        d_out_layer = self.fc(x)
        return d_out_layer, out_map1