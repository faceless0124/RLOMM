import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1)

    def forward(self, x):
        return self.net(x)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, kernel_size=2,):
        super(TemporalConvNet, self).__init__()
        # 使用最小的padding和dilation保证模型简单
        padding = (kernel_size - 1)
        self.tcn = TemporalBlock(num_inputs, num_outputs, kernel_size, stride=1, dilation=1, padding=padding)

    def forward(self, x):
        return self.tcn(x)
