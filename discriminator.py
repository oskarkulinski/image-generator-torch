import torch
import torch.nn as nn
import torch.nn.functional as F

import parameters as params


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.LayerNorm(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.LayerNorm(128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.norm3 = nn.LayerNorm(64)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.norm4 = nn.LayerNorm(32)
        self.linear = nn.Linear(32 * 2 * 2, 1)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = F.leaky_relu(x, 0.2)
        x = self.conv3(x)
        x = self.norm3(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = F.leaky_relu(x, 0.2)
        x = self.conv4(x)
        x = self.norm4(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = F.leaky_relu(x, 0.2)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = F.sigmoid(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.uniform_(m.bias, -1, 1)
