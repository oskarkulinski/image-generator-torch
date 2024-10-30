import torch
import torch.nn as nn
import torch.nn.functional as F

import parameters as params

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.norm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.norm4 = nn.BatchNorm2d(32)
        self.linear = nn.Linear(32 * 8 * 8, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv3(x)
        x = self.norm3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv4(x)
        x = self.norm4(x)
        x = F.leaky_relu(x, 0.2)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = F.sigmoid(x)
        return x

'''def build_discriminator():
    image_input = tf.keras.Input(shape=(params.image_height, params.image_width, 3), name='image_input')

    conv2d_1 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same')(image_input)
    conv2d_1 = t    f.keras.layers.BatchNormalization(momentum=0.8)(conv2d_1)
    conv2d_1 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2d_1)

    conv2d_2 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same')(conv2d_1)
    conv2d_2 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv2d_2)
    conv2d_2 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2d_2)

    conv2d_3 = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(conv2d_2)
    conv2d_3 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv2d_3)
    conv2d_3 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2d_3)

    conv2d_4 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(conv2d_3)
    conv2d_4 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv2d_4)
    conv2d_4 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2d_4)

    flatten = tf.keras.layers.Flatten()(conv2d_3)
    dense_1 = tf.keras.layers.Dense(1, activation='sigmoid')(flatten)

    model = tf.keras.Model(image_input, outputs=dense_1)
    return model
'''