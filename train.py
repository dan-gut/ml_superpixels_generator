import os
from pathlib import Path
import random

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image
import torch
import torch.optim as optim
from torch import nn
from torchvision import transforms

from model import RepresentationUNet

random.seed(42)

dino_resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')

model_folder = Path("model")
model_folder.mkdir(exist_ok=True)
model_path = "model/rep_net.pt"
data_base_dir = os.path.join("Data", "images")

test_img_ratio = 0.2

# Model params
patch_size = (40, 40)
unet_output_classes = 1024

# Training params
learning_rate = 0.001
weight_decay = 1e-4
momentum = 0.9
checkpoint_interval = 10
images_per_epoch = 50

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float)])


class RepresentationLoss(nn.Module):
    def __init__(self):
        super(RepresentationLoss, self).__init__()

    def forward(self, pred_rep, target_rep):
        return torch.linalg.norm(pred_rep-target_rep)

    @staticmethod
    def get_dino_rep(input_image, patch_coords):
        input_image = input_image.repeat(3, 1, 1)  # RGB channels are required
        patch = input_image[:, patch_coords[0][0]:patch_coords[0][1], patch_coords[1][0]:patch_coords[1][1]]
        patch = patch.unsqueeze(0)

        rep_model = dino_resnet50.to(device)
        rep_model.eval()

        patch = patch.to(device)
        with torch.no_grad():
            rep = rep_model(patch)

        return rep.detach().cpu()


def dataset_loader(test_set=False):
    files_list = list(os.listdir(data_base_dir))
    files_list = sorted(files_list)
    train_list = files_list[:int((1 - test_img_ratio) * len(files_list))]
    test_list = files_list[int((1 - test_img_ratio) * len(files_list)):]

    while True:
        if test_set:
            chosen_file = random.choice(test_list)
        else:
            chosen_file = random.choice(train_list)

        file_path = os.path.join(data_base_dir, chosen_file)
        image = nib.load(file_path).get_fdata()

        x_min = random.randint(0, image.shape[0] - patch_size[0])
        x_max = x_min + patch_size[0]

        y_min = random.randint(0, image.shape[1] - patch_size[1])
        y_max = y_min + patch_size[1]

        patch_coords = ((x_min, x_max), (y_min, y_max))

        image = np.squeeze(image, axis=2)
        image = Image.fromarray(image.astype(np.uint8))
        image = img_transform(image)

        yield image, patch_coords


def train(epoch_number):
    model = RepresentationUNet(unet_out_dimensions=unet_output_classes, patch_size=patch_size, representation_len=2048)
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    model.to(device)

    if os.path.isfile(model_path):
        print("Loading model")
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        curr_epoch = checkpoint['epoch'] + 1
        losses = checkpoint['losses']
    else:
        curr_epoch = 0
        losses = []

    loss_fun = RepresentationLoss()
    train_loader = dataset_loader(test_set=False)

    for epoch in range(curr_epoch, curr_epoch + epoch_number):
        print(f"Epoch {epoch}")
        epoch_losses = []
        for _ in range(images_per_epoch):
            image, patch_coords = next(train_loader)
            target = RepresentationLoss.get_dino_rep(image, patch_coords)

            image = image.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            model.set_patch_coordinates(patch_coords)

            output = model(image.unsqueeze(0))
            loss = loss_fun(output, target)

            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = np.mean(epoch_losses)
        losses.append(mean_loss)
        print(mean_loss)
        if (epoch + 1) % checkpoint_interval == 0:
            print("Saving model checkpoint")
            torch.save({'epoch': epoch,
                        'losses': losses,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }, model_path)

    print("Saving model")
    torch.save({'epoch': epoch,
                'losses': losses,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, model_path)

    return losses


def test():
    test_loader = dataset_loader(test_set=True)

    model = RepresentationUNet(unet_out_dimensions=unet_output_classes, patch_size=patch_size, representation_len=2048)
    model.to(device)

    if os.path.isfile(model_path):
        print("Loading model")
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])

    loss_fun = RepresentationLoss()
    losses = []

    for _ in range(images_per_epoch):
        image, patch_coords = next(test_loader)
        target = RepresentationLoss.get_dino_rep(image, patch_coords)

        image = image.to(device)
        target = target.to(device)

        model.set_patch_coordinates(patch_coords)
        model.eval()

        output = model(image.unsqueeze(0))

        loss = loss_fun(output, target)
        losses.append(loss.item())

    mean_loss = np.mean(losses)
    print(f"Test loss: {mean_loss}")

    return mean_loss


if __name__ == "__main__":
    torch.cuda.empty_cache()

    train_losses = train(50)
    print(test())

    plt.plot(train_losses)
    plt.grid()
    plt.savefig("loss.png")
