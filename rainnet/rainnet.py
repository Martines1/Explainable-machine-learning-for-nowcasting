import torch
import torch.nn as nn


class RainNet(nn.Module):
    """
    PyTorch equivalent Keras RainNet (v1.0).
    Input: NCHW (N, 4, H, W)   -- Keras version of (H, W, C)
    """

    def __init__(self, in_channels: int = 4):
        super().__init__()

        # --- Encoder ---
        self.conv1f = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=True)
        self.conv1s = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2f = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.conv2s = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3f = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True)
        self.conv3s = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4f = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        self.conv4s = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.drop4 = nn.Dropout(p=0.5)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5f = nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=True)
        self.conv5s = nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=True)
        self.drop5 = nn.Dropout(p=0.5)

        # --- Decoder ---
        self.up6 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv6f = nn.Conv2d(1024 + 512, 512, kernel_size=3, padding=1, bias=True)
        self.conv6s = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)

        self.up7 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv7f = nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1, bias=True)
        self.conv7s = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)

        self.up8 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv8f = nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1, bias=True)
        self.conv8s = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)

        self.up9 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv9f = nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1, bias=True)
        self.conv9s = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.conv9 = nn.Conv2d(64, 2, kernel_size=3, padding=1, bias=True)

        self.out_conv = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        conv1f = self.relu(self.conv1f(x))
        conv1s = self.relu(self.conv1s(conv1f))
        pool1 = self.pool1(conv1s)

        conv2f = self.relu(self.conv2f(pool1))
        conv2s = self.relu(self.conv2s(conv2f))
        pool2 = self.pool2(conv2s)

        conv3f = self.relu(self.conv3f(pool2))
        conv3s = self.relu(self.conv3s(conv3f))
        pool3 = self.pool3(conv3s)

        conv4f = self.relu(self.conv4f(pool3))
        conv4s = self.relu(self.conv4s(conv4f))
        drop4 = self.drop4(conv4s)
        pool4 = self.pool4(drop4)

        conv5f = self.relu(self.conv5f(pool4))
        conv5s = self.relu(self.conv5s(conv5f))
        drop5 = self.drop5(conv5s)

        # Decoder
        up6 = self.up6(drop5)
        up6 = torch.cat([up6, conv4s], dim=1)
        conv6 = self.relu(self.conv6f(up6))
        conv6 = self.relu(self.conv6s(conv6))

        up7 = self.up7(conv6)
        up7 = torch.cat([up7, conv3s], dim=1)
        conv7 = self.relu(self.conv7f(up7))
        conv7 = self.relu(self.conv7s(conv7))

        up8 = self.up8(conv7)
        up8 = torch.cat([up8, conv2s], dim=1)
        conv8 = self.relu(self.conv8f(up8))
        conv8 = self.relu(self.conv8s(conv8))

        up9 = self.up9(conv8)
        up9 = torch.cat([up9, conv1s], dim=1)
        conv9 = self.relu(self.conv9f(up9))
        conv9 = self.relu(self.conv9s(conv9))
        conv9 = self.relu(self.conv9(conv9))

        out = self.out_conv(conv9)
        return out

    def convs_in_keras_order(self):
        names = [
            "conv1f", "conv1s",
            "conv2f", "conv2s",
            "conv3f", "conv3s",
            "conv4f", "conv4s",
            "conv5f", "conv5s",
            "conv6f", "conv6s",
            "conv7f", "conv7s",
            "conv8f", "conv8s",
            "conv9f", "conv9s",
            "conv9", "out_conv",
        ]
        return [getattr(self, n) for n in names]
