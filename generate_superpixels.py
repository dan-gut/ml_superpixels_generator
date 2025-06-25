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
from tqdm import tqdm

random.seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from train import img_transform
import torch.nn.functional as F

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--plot_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--contrastive_rate", type=float, default=0.5)
    parser.add_argument("--classes", type=int, default=1024)
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


def load_model(args, in_ch):
    model_name = args.model_dir.replace("model/", "").replace(".pt","")
    model_path = args.model_dir
    representation_len = 2048
    model = RepresentationUNet(
        in_ch=in_ch,
        unet_out_dimensions=args.classes,
        patch_size=args.patch_size,
        representation_len=representation_len,
        sigmoid=args.sigmoid,
    )

    print("Loading model")
    checkpoint = torch.load(f"{model_path}", map_location=torch.device(device))
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

from train import MNIST_DATASETS
from torchvision import models
from torchvision.utils import save_image
def generate_superpixels(args):
    if args.data_dir.startswith("medmnist"):
        dataset = args.data_dir.split("_")[1]
        split = args.data_dir.split("_")[2]
        assert split == "test"
        print(f"dataset {dataset} split {split}")
        train_dataset = MNIST_DATASETS[dataset](split=split, size=224, download=True, transform=img_transform,
                                                root=f"/net/tscratch/people/plgmpro/data/medmnist/", as_rgb=False)
        rgb_train_dataset = MNIST_DATASETS[dataset](split=split, size=224, download=True, transform=img_transform,
                                                root=f"/net/tscratch/people/plgmpro/data/medmnist/", as_rgb=True)
        # load resnet
        resnet = models.resnet18(weights="IMAGENET1K_V1")
        resnet_path = f"resnets/resnet18_{dataset}_mnist.pt"
        if train_dataset[0][1].shape[-1] != 1:
            num_classes = train_dataset[0][1].shape[-1]
        else:
            num_classes = max([y.max() + 1 for _, y in train_dataset])
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
        resnet.load_state_dict(torch.load(resnet_path, map_location="cpu")["model"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resnet.to(device)
        resnet.eval()
        print("Resnet loaded")
        all_imgs = {}
        for i, ((image, y), (rgb_image, rgb_y)) in enumerate(zip(train_dataset, rgb_train_dataset)):
            assert np.all(y == rgb_y), i
            if len(y) != 1:
                y_label = "".join([str(yi) for yi in y.tolist()])
                y = 0
            else:
                y = y[0]
                y_label = y
            with torch.no_grad():
                pred_y = resnet(rgb_image.unsqueeze(0).to(device))
            if pred_y.shape[-1] == 14:
                pred_y = "".join([str(yi) for yi in (F.sigmoid(pred_y) > 0.5).squeeze(0).long().tolist()])
            else:
                pred_y = torch.argmax(pred_y, 1).detach().cpu().item()
            if y in all_imgs:
                all_imgs[y].append((i, y_label, pred_y, image))
            else:
                all_imgs[y] = [(i, y_label, pred_y, image)]
        images = []
        for y in all_imgs:
            images.extend(all_imgs[y][:50])
        indexes, labels, preds, images = zip(*images)

    else:
        images = [load_image(os.path.join(args.data_dir, img_name)) for img_name in os.listdir(args.data_dir)]
        indexes = range(len(images))
        preds = range(len(images))
        labels = range(len(images))

    model, model_name = load_model(args, images[0].shape[0])
    model.eval()
    for i, label, pred, img in tqdm(zip(indexes, labels, preds, images), total=len(images)):
        img_name  = f"test_idx{i:03}_label{label}_pred{pred}.png"
        patches_coords = get_patches_coords(img[0].shape, args.patches_choice, args.patch_size, args.step_size,
                                            args.num_samples)
        _, unet_output = model(img.unsqueeze(0), patches_coords)

        act = nn.Sigmoid() if args.sigmoid else nn.Softmax(dim=1)

        output_img = act(unet_output)
        output_img = output_img.squeeze(0)
        sp_img = torch.argmax(output_img, dim=0)

        unique_sp_img = get_unique_sp(sp_img.numpy())



        output_path = Path(os.path.join(args.output_dir, model_name))
        img_path = Path(os.path.join(output_path, img_name))
        img_path.parent.mkdir(parents=True, exist_ok=True)
        if img.shape[0] == 1:
            save_image(img, img_path)
            #plt.imsave(img_path, img[0].numpy(), cmap="gray", vmin=0, vmax=1)
        else:
            save_image(img, img_path)
            #plt.imsave(img_path, img.permute(1,2,0).numpy(), vmin=0, vmax=1)

        img_path = Path(os.path.join(output_path, img_name.replace(".png", "_sp.png")))
        output_img = Image.fromarray(sp_img.numpy().astype(np.uint16))
        output_img.save(img_path)
        
        img_path = Path(os.path.join(output_path, img_name.replace(".png", "_sp_uq.png")))
        output_img = Image.fromarray(unique_sp_img.astype(np.uint16))
        output_img.save(img_path)
        if img.shape[0] == 3:
            img = img.mean(0).unsqueeze(0)

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
