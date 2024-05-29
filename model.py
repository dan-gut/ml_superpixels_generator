import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up_scale = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x2 = self.up_scale(x2)

        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x


class DownLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownLayer, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(self.pool(x))
        return x


class UpLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpLayer, self).__init__()
        self.up = Up(in_ch, out_ch)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        a = self.up(x1, x2)
        x = self.conv(a)
        return x


class UNet(nn.Module):
    def __init__(self, out_dimensions=2):
        super(UNet, self).__init__()
        self.conv1 = DoubleConv(1, 64)
        self.down1 = DownLayer(64, 128)
        self.down2 = DownLayer(128, 256)
        self.down3 = DownLayer(256, 512)
        self.down4 = DownLayer(512, 1024)
        self.up1 = UpLayer(1024, 512)
        self.up2 = UpLayer(512, 256)
        self.up3 = UpLayer(256, 128)
        self.up4 = UpLayer(128, 64)
        self.last_conv = nn.Conv2d(64, out_dimensions, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x1_up = self.up1(x4, x5)
        x2_up = self.up2(x3, x1_up)
        x3_up = self.up3(x2, x2_up)
        x4_up = self.up4(x1, x3_up)
        output = self.last_conv(x4_up)
        return output


class RepresentationUNet(nn.Module):
    def __init__(self, unet_out_dimensions, patch_size, representation_len=2048):
        super(RepresentationUNet, self).__init__()
        self.unet = UNet(unet_out_dimensions)
        self.softmax = nn.Softmax()
        self.av_pool = nn.AvgPool2d(2, stride=2)
        self.flat = nn.Flatten()
        x_size, y_size = patch_size
        fc_input_size = int(x_size // 2 * y_size // 2)
        self.fc1 = nn.Linear(fc_input_size, representation_len // 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(representation_len // 4, representation_len // 2)
        self.fc3 = nn.Linear(representation_len // 2, representation_len)

        self._patch_coords = None

    def forward(self, x):
        x1 = self.unet(x)
        x2 = self.softmax(x1)
        x3 = self._crop_patch(x2)
        x4 = self.av_pool(x3)
        x5 = torch.sum(x4, dim=1, keepdim=True)
        x6 = self.flat(x5)
        x7 = self.fc1(x6)
        x8 = self.relu(x7)
        x9 = self.fc2(x8)
        x10 = self.relu(x9)
        output = self.fc3(x10)
        return output

    def set_patch_coordinates(self, patch_coordinates):
        self._patch_coords = patch_coordinates

    def _crop_patch(self, input_tensor):
        if self._patch_coords is None:
            raise RuntimeError("Patch coordinates has not been set!")

        output_tensor = input_tensor[:, :, self._patch_coords[0][0]:self._patch_coords[0][1],
                                     self._patch_coords[1][0]:self._patch_coords[1][1]]

        self._patch_coords = None

        return output_tensor


if __name__ == "__main__":
    rep = RepresentationUNet(100, (40, 40))
    rep.set_patch_coordinates([[0, 40], [0, 40]])
    summary(rep.cuda(), (1, 400, 400))
