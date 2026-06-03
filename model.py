import torch
import torch.nn as nn
import torch.nn.functional as F

# Unet model derived from https://github.com/milesial/Pytorch-UNet and 
# https://github.com/amirhosseinh77/UNet-AerialSegmentation

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout())

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, dropout=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW (channel, height, width)
        # Adjusting borders
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat((x2, x1), dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, scale=1, bilinear=True, dropout=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64 // scale)
        self.down1 = Down(64 // scale, 128 // scale)
        self.down2 = Down(128 // scale, 256 // scale, dropout=dropout)
        self.down3 = Down(256 // scale, 512 // scale, dropout=dropout)
        factor = 2 if bilinear else 1
        self.down4 = Down(512 // scale, 1024 // (factor*scale), dropout=dropout)
        self.up1 = Up(1024 // scale, 512 // (factor*scale), bilinear, dropout=dropout)
        self.up2 = Up(512 // scale, 256 // (factor*scale), bilinear, dropout=dropout)
        self.up3 = Up(256 // scale, 128 // (factor*scale), bilinear)
        self.up4 = Up(128 // scale, 64 // scale, bilinear)
        self.outc = OutConv(64 // scale, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNetPlusPlus(nn.Module):
    def __init__(self, n_channels, n_classes, scale=1, bilinear=True, dropout=False):
        super(UNetPlusPlus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        nb_filter = [64 // scale, 128 // scale, 256 // scale, 512 // scale, 1024 // scale]

        self.pool = nn.MaxPool2d(2, 2)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(nb_filter[0], nb_filter[0], kernel_size=2, stride=2)

        # Fallback to Upsample consistently for dense connections in UNet++
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = DoubleConv(n_channels, nb_filter[0], dropout=dropout)
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1], dropout=dropout)
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2], dropout=dropout)
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3], dropout=dropout)
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4], dropout=dropout)

        self.conv0_1 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0], dropout=dropout)
        self.conv1_1 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1], dropout=dropout)
        self.conv2_1 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2], dropout=dropout)
        self.conv3_1 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3], dropout=dropout)

        self.conv0_2 = DoubleConv(nb_filter[0]*2 + nb_filter[1], nb_filter[0], dropout=dropout)
        self.conv1_2 = DoubleConv(nb_filter[1]*2 + nb_filter[2], nb_filter[1], dropout=dropout)
        self.conv2_2 = DoubleConv(nb_filter[2]*2 + nb_filter[3], nb_filter[2], dropout=dropout)

        self.conv0_3 = DoubleConv(nb_filter[0]*3 + nb_filter[1], nb_filter[0], dropout=dropout)
        self.conv1_3 = DoubleConv(nb_filter[1]*3 + nb_filter[2], nb_filter[1], dropout=dropout)

        self.conv0_4 = DoubleConv(nb_filter[0]*4 + nb_filter[1], nb_filter[0], dropout=dropout)

        self.final = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output
