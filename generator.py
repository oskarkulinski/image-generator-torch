import torch
import torch.nn as nn
import torch.nn.functional as F
import parameters as params

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        channels = params.generator_starting_channels
        self.input = nn.Linear(params.noise_dim, channels*8*8)
        self.conv1 = nn.ConvTranspose2d(channels, channels*2, 3, 2, padding=1, output_padding=1)
        self.norm1 = nn.BatchNorm2d(channels*2)
        self.conv2 = nn.ConvTranspose2d(channels*2, channels*4, 3, 2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(channels*4)
        self.conv3 = nn.ConvTranspose2d(channels*4, channels*8, 3, 2, padding=1, output_padding=1)
        self.norm3 = nn.BatchNorm2d(channels*8)
        self.conv4 = nn.ConvTranspose2d(channels * 8, channels * 16, 3, 2, padding=1, output_padding=1)
        self.norm4 = nn.BatchNorm2d(channels * 16)
        self.output = nn.Conv2d(channels * 16, 3, 3, 1, 1)

    def forward(self, x):
        x = self.input(x)
        x = F.relu(x)
        x = x.view(x.size(0), params.generator_starting_channels, 8, 8)
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

'''def build_generator():
    noise_input = nn.Linear(params.noise_dim)

    dense_1 = tf.keras.layers.Dense(128 * 8 * 8, activation='relu')(noise_input)
    reshape = tf.keras.layers.Reshape((8, 8, 128))(dense_1)

    # 16x16
    conv_1 = tf.keras.layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding='same')(reshape)
    normal_2 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv_1)
    activation_1 = tf.keras.layers.ReLU()(normal_2)

    # 32x32
    conv_2 = tf.keras.layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding='same')(activation_1)
    normal_3 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv_2)
    activation_2 = tf.keras.layers.ReLU()(normal_3)

    # 64x64
    conv_3 = tf.keras.layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')(activation_2)
    normal_4 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv_3)
    activation_3 = tf.keras.layers.ReLU()(normal_4)

    # 128x128
    conv_4 = tf.keras.layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')(activation_3)
    normal_5 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv_4)
    activation_4 = tf.keras.layers.ReLU()(normal_5)

    output = tf.keras.layers.Conv2D(3, kernel_size=3, padding='same', activation='tanh')(activation_4)

    model = tf.keras.Model(noise_input, outputs=output)
    return model
'''