import numpy as np
import torch
from torch import nn
from torchinfo import summary

from model import UNet


class RepresentationUNet(nn.Module):
    def __init__(self, unet_out_dimensions, num_of_superpixels, patch_size, representation_len=2048, sigmoid=False):
        super(RepresentationUNet, self).__init__()
        self.unet = UNet(unet_out_dimensions)

        self.activation = nn.Sigmoid() if sigmoid else nn.Softmax(dim=1)
        self.probabilities_activation = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

        middle_feat_no = (unet_out_dimensions + num_of_superpixels)//2
        self.conv1 = nn.Conv2d(unet_out_dimensions, middle_feat_no, 1)
        self.conv2 = nn.Conv2d(middle_feat_no, num_of_superpixels, 1)

        self.fc1 = nn.Linear(unet_out_dimensions, representation_len // 2)
        self.fc2 = nn.Linear(representation_len // 2, representation_len)

        self.patch_size = patch_size

    def forward(self, x, patches_coords=None):
        features_vector = self.unet(x)

        # Head 1 - superpixels probability
        y1 = self.conv1(features_vector)
        y2 = self.relu(y1)
        y3 = self.conv2(y2)
        prob_output = self.probabilities_activation(y3)

        # Head 2 - representation:
        if patches_coords is not None:  # only during training
            x1 = self.activation(features_vector)
            x2 = torch.stack([x1[:, :, i:i+self.patch_size[0], j:j+self.patch_size[1]] for i, j in patches_coords], dim=1)
            x3 = x2.mean(-1).mean(-1)
            x4 = self.fc1(x3)
            x5 = self.relu(x4)
            rep_output = self.fc2(x5)
            return rep_output, features_vector, prob_output
        else:
            return prob_output


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
