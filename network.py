import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, X):
        residual = X
        X = F.relu(self.bn1(self.conv1(X)))
        X = self.bn2(self.conv2(X))
        X += residual
        return F.relu(X)


class AlphaZeroNetwork(nn.Module):
    def __init__(self,
                 input_channels: int,
                 height: int,
                 width: int,
                 num_residual_blocks: int,
                 action_size: int,
                 h_channels: int):
        super().__init__()
        self.conv_in = nn.Conv2d(input_channels, h_channels, kernel_size=3, stride=1, padding=1)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(h_channels) for _ in range(num_residual_blocks)]
        )
        self.bn_in = nn.BatchNorm2d(h_channels)

        self.policy_conv = nn.Conv2d(h_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * height * width, action_size)

        self.value_conv = nn.Conv2d(h_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(height * width, h_channels)
        self.value_fc2 = nn.Linear(h_channels, 1)

    def forward(self, X):
        X = F.relu(self.bn_in(self.conv_in(X)))
        X = self.residual_blocks(X)

        policy = F.relu(self.policy_bn(self.policy_conv(X)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        value = F.relu(self.value_bn(self.value_conv(X)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value