import argparse
import os
from pathlib import Path
import random

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import mark_boundaries
import torch
from torch import nn, unique
from torchvision import transforms

from model_interlayer_output import RepresentationUNet


random.seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_transform = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float)])


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--plot_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--contrastive_rate", type=float, default=0.5)
    parser.add_argument("--classes", type=float, default=1024)
    parser.add_argument("--patch_size", type=int, nargs="+", default=[40, 40])
    parser.add_argument("--sigmoid", action="store_true", default=False)
    parser.add_argument("--patches_choice", type=str, default="sample", choices=["sample", "grid"])
    parser.add_argument("--step_size", type=int, nargs="+", default=[40, 40], help="for grid option")
    parser.add_argument("--num_samples", type=int, default=40, help="for sample")
    args = parser.parse_args()
    return args


def get_patches_coords(image_shape, patches_choice, patch_size, step_size, num_samples):
    if patches_choice == "grid":
        begin_x = torch.randint(low=0, high=max(patch_size[0], step_size[0]), size=(1,)).item()
        begin_y = torch.randint(low=0, high=max(patch_size[1], step_size[1]), size=(1,)).item()
        coords_x = torch.arange(
            start=begin_x, end=max(begin_x + 1, image_shape[0] - patch_size[0]), step=step_size[0]
        )
        coords_y = torch.arange(
            start=begin_y, end=max(begin_y + 1, image_shape[1] - patch_size[1]), step=step_size[1]
        )
        coords = [(x, y) for x in coords_x for y in coords_y]
    else:
        corrds_x = torch.randint(low=0, high=image_shape[0] - patch_size[0], size=(num_samples,))
        corrds_y = torch.randint(low=0, high=image_shape[1] - patch_size[1], size=(num_samples,))
        coords = list(zip(corrds_x, corrds_y))
    return coords


def load_model(args):
    model_name = (
        f"bs{args.batch_size}_lr{args.lr}_c{args.classes}_cr{args.contrastive_rate}"
        f"_ps{'_'.join([str(p) for p in args.patch_size])}"
    )
    if args.patches_choice == "grid":
        model_name += f"_grid_ss{'_'.join([str(s) for s in args.step_size])}"
    elif args.patches_choice == "sample":
        model_name += f"_sample_ns{args.num_samples}"
    else:
        assert False, args.patches_choice
    if args.sigmoid:
        model_name += "_sig"

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir.joinpath(f"{model_name}.pt")
    print(model_path)

    representation_len = 2048
    model = RepresentationUNet(
        unet_out_dimensions=args.classes,
        patch_size=args.patch_size,
        representation_len=representation_len,
        sigmoid=args.sigmoid,
    )

    print("Loading model")
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, model_name


def load_image(image_path):
    if image_path.endswith(".nii.gz"):
        image = nib.load(image_path).get_fdata()
        image = np.squeeze(image, axis=2)
        image = Image.fromarray(image.astype(np.uint8))
        image = img_transform(image)
    elif image_path.endswith(".png"):
        image = Image.open(image_path)
        image = img_transform(image)
    else:
        raise ValueError("Unsupported image type")

    return image


def get_unique_sp(data):
    uv = np.unique(data)
    s = ndi.generate_binary_structure(2, 2)
    cum_num = 0
    result = np.zeros_like(data)
    for v in uv[1:]:
        labeled_array, num_features = ndi.label((data == v).astype(int), structure=s)
        result += np.where(labeled_array > 0, labeled_array + cum_num, 0).astype(result.dtype)
        cum_num += num_features

    return result


def plot_sp_img(image, plot_path):
    cmap = plt.cm.tab20
    num_colors = cmap.N
    colors = cmap(np.arange(num_colors))

    mapped_colors = colors[image % num_colors]

    image_rgb = (mapped_colors[:, :, :3] * 255).astype(np.uint8)

    plt.imshow(image_rgb)
    plt.savefig(plot_path)


def generate_superpixels(args):
    model, model_name = load_model(args)
    model.eval()

    for img_name in os.listdir(args.data_dir):
        img = load_image(os.path.join(args.data_dir, img_name))

        patches_coords = get_patches_coords(img[0].shape, args.patches_choice, args.patch_size, args.step_size,
                                            args.num_samples)
        _, unet_output = model(img.unsqueeze(0), patches_coords)

        act = nn.Sigmoid() if args.sigmoid else nn.Softmax(dim=1)

        output_img = act(unet_output)
        output_img = output_img.squeeze(0)
        sp_img = torch.argmax(output_img, dim=0)

        unique_sp_img = get_unique_sp(sp_img.numpy())

        output_sp_path = Path(os.path.join(args.output_dir, model_name, f"{img_name.split('.')[0]}.png"))
        output_sp_path.parent.mkdir(parents=True, exist_ok=True)
        output_img = Image.fromarray(unique_sp_img.astype(np.uint16))
        output_img.save(output_sp_path)

        if args.plot_dir is not None:
            output_sp_plot1_path = Path(os.path.join(args.plot_dir, model_name, f"{img_name.split('.')[0]}_1.png"))
            output_sp_plot2_path = Path(os.path.join(args.plot_dir, model_name, f"{img_name.split('.')[0]}_2.png"))
            output_sp_plot1_path.parent.mkdir(parents=True, exist_ok=True)

            plt.imshow(mark_boundaries(img.squeeze(0).numpy(), sp_img.numpy(), color=(0, 1, 0)))
            plt.savefig(output_sp_plot1_path)
            plot_sp_img(unique_sp_img, output_sp_plot2_path)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = args_parser()
    generate_superpixels(args)
