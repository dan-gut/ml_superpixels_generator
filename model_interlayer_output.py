import numpy as np
import torch
from torch import nn
from torchinfo import summary

from model import UNet


class RepresentationUNet(nn.Module):
    def __init__(self, unet_out_dimensions, patch_size, representation_len=2048, sigmoid=False):
        super(RepresentationUNet, self).__init__()
        self.unet = UNet(unet_out_dimensions)
        self.activation = nn.Sigmoid() if sigmoid else nn.Softmax(dim=1)

        self.fc1 = nn.Linear(unet_out_dimensions, representation_len // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(representation_len // 2, representation_len)

        self.patch_size = patch_size

    def forward(self, x, patches_coords):
        x1 = self.unet(x)
        x2 = self.activation(x1)
        x3 = torch.stack([x2[:, :, i:i+self.patch_size[0], j:j+self.patch_size[1]] for i, j in patches_coords], dim=1)
        x4 = x3.mean(-1).mean(-1)
        x5 = self.fc1(x4)
        x6 = self.relu(x5)
        output = self.fc2(x6)

        return output, x1


    def _get_cropping_grid(self, patches_coords, img_shape):
        grids = []

        xs_min = (patches_coords[:, 0] - img_shape[2]/2) / (img_shape[2]/2)
        xs_max = (patches_coords[:, 1] - img_shape[2]/2) / (img_shape[2]/2)
        ys_min = (patches_coords[:, 2] - img_shape[3]/2) / (img_shape[3]/2)
        ys_max = (patches_coords[:, 3] - img_shape[3]/2) / (img_shape[3]/2)

        for i in range(len(patches_coords)):
            x_grid = torch.linspace(xs_min[i], xs_max[i], self.patch_size[0])
            y_grid = torch.linspace(ys_min[i], ys_max[i], self.patch_size[1])
            meshx, meshy = torch.meshgrid(x_grid, y_grid, indexing='ij')
            grids.append(torch.stack((meshy, meshx), 2))

        return torch.stack(grids)


if __name__ == "__main__":
    model = RepresentationUNet(unet_out_dimensions=1024, patch_size=(40, 40), representation_len=2048)
    batch_size = 4
    num_samples = 4
    summary(model, input_size=(batch_size, 1, 400, 400), patches_coords=[(0, 0) for _ in range(num_samples)])
