import torch
import torch.nn as nn
import torch.nn.functional as F
import parameters as params


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        channels = params.generator_starting_channels
        self.input = nn.Linear(params.noise_dim, channels * 2 * 2)
        self.conv1 = nn.ConvTranspose2d(channels, channels * 2, 3, 2, padding=1, output_padding=1)
        self.norm1 = nn.BatchNorm2d(channels * 2)
        self.conv2 = nn.ConvTranspose2d(channels * 2, channels * 4, 3, 2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(channels * 4)
        self.conv3 = nn.ConvTranspose2d(channels * 4, channels * 8, 3, 2, padding=1, output_padding=1)
        self.norm3 = nn.BatchNorm2d(channels * 8)
        self.conv4 = nn.ConvTranspose2d(channels * 8, channels * 16, 3, 2, padding=1, output_padding=1)
        self.norm4 = nn.BatchNorm2d(channels * 16)
        self.output = nn.Conv2d(channels * 16, 3, 3, 1, 1)

        self._initialize_weights()

    def forward(self, x):
        x = self.input(x)
        x = F.relu(x)
        x = x.view(x.size(0), params.generator_starting_channels, 2, 2)
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.norm4(x)
        x = F.relu(x)
        x = self.output(x)
        x = F.tanh(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
