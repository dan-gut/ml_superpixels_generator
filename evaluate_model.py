import os
import random

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from model import RepresentationUNet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_base_dir = os.path.join("Data", "images_resized")

dino_resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')

patch_size = (40, 40)
unet_output_classes = 1024

TEST_CASES_NO = 50


def patches_generator(patches_no=2):
    files_list = list(os.listdir(data_base_dir))
    files_list = sorted(files_list)

    img_transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float)])

    while True:
        chosen_file = random.choice(files_list)

        file_path = os.path.join(data_base_dir, chosen_file)
        image = nib.load(file_path).get_fdata()

        patches_coords = []

        for _ in range(patches_no):
            x_min = random.randint(0, image.shape[0] - patch_size[0])
            x_max = x_min + patch_size[0]

            y_min = random.randint(0, image.shape[1] - patch_size[1])
            y_max = y_min + patch_size[1]

            patches_coords.append((x_min, x_max, y_min, y_max))

        image = np.squeeze(image, axis=2)
        image = Image.fromarray(image.astype(np.uint8))
        image = img_transform(image)

        patches = []

        for i in range(patches_no):
            patch_coords = patches_coords[i]
            patches.append(image[:, patch_coords[0]:patch_coords[1], patch_coords[2]:patch_coords[3]])

        yield image, patches, patches_coords

def get_dino_rep(patch):
    patch = patch.repeat(3, 1, 1)  # RGB channels are required
    patch = patch.unsqueeze(0)

    rep_model = dino_resnet50.to(device)
    rep_model.eval()

    patch = patch.to(device)
    with torch.no_grad():
        rep = rep_model(patch)

    return rep.detach().cpu()

def evaluate_model(model_path, plot_path=None):
    print("Evalutating...")
    patch_gen = patches_generator(patches_no=2)

    model = RepresentationUNet(unet_out_dimensions=unet_output_classes, patch_size=patch_size, representation_len=2048)
    model.to(device)

    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    model2dino = []
    model2model = []
    dino2dino = []
    for _ in range(TEST_CASES_NO):
        image, patches, patches_coords = next(patch_gen)
        patch_1, patch_2 = patches
        patch_1_coords, patch_2_coords = patches_coords

        dino_rep_1 = get_dino_rep(patch_1)
        dino_rep_2 = get_dino_rep(patch_2)

        image = image.to(device)

        model_rep_1 = model(image.unsqueeze(0), torch.tensor(patch_1_coords).unsqueeze(0))
        model_rep_1 = model_rep_1.detach().cpu()

        model_rep_2 = model(image.unsqueeze(0), torch.tensor(patch_2_coords).unsqueeze(0))
        model_rep_2 = model_rep_2.detach().cpu()

        model2dino.append(torch.linalg.norm(model_rep_1 - dino_rep_1))
        model2model.append(torch.linalg.norm(model_rep_1 - model_rep_2))
        dino2dino.append(torch.linalg.norm(dino_rep_1 - dino_rep_2))

    if plot_path is not None:
        plt.clf()
        plt.plot(model2model, 'o', label="mod2mod")
        plt.plot(model2dino, 'o', label="mod2dino")
        plt.plot(dino2dino, 'o', label="dino2dino")
        plt.legend()
        plt.grid()
        plt.savefig(plot_path)

    return model2model, model2dino, dino2dino


if __name__ == "__main__":
    random.seed(42)

    model2model, model2dino, dino2dino = evaluate_model("model/rep_net_batch.pt","model_batch_evaluation.png")

    print(f"Model to model mean: {np.mean(model2model)}\n"
          f"Model to dino mean: {np.mean(model2dino)}\n"
          f"Dino to dino mean: {np.mean(dino2dino)}")
