import argparse
import os
from pathlib import Path
import random
from tqdm import tqdm

import nibabel as nib
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from model import RepresentationUNet


random.seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_transform = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float)])


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--model_dir", type=str, default="./model")
    parser.add_argument("--runs_dir", type=str, default="./runs")
    parser.add_argument("--epochs", type=int, default=50)
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


class RepresentationLoss(nn.Module):
    def __init__(self, contrastive_rate=0.5):
        super(RepresentationLoss, self).__init__()
        self.contrastive_rate = contrastive_rate

    def forward(self, pred_reps, target_reps):
        similarity = torch.linalg.norm(pred_reps - target_reps, dim=-1)
        similarity = similarity.mean()

        pred_scores = torch.stack(
            [
                torch.linalg.norm(pred_reps - pred_reps[:, i: i + 1], dim=-1)
                for i in range(pred_reps.shape[1])
            ], dim=1,
        )
        target_scores = torch.stack(
            [
                torch.linalg.norm(target_reps - target_reps[:, i: i + 1], dim=-1)
                for i in range(target_reps.shape[1])
            ], dim=1
        )
        contrastiveness = torch.nn.functional.relu(target_scores - pred_scores)
        contrastiveness = contrastiveness.sum(-1) / (contrastiveness.shape[-1] - 1)
        contrastiveness = contrastiveness.mean()

        total_loss = (1 - self.contrastive_rate) * similarity + self.contrastive_rate * contrastiveness

        return {
            "similarity": similarity,
            "contrastiveness": contrastiveness,
            "target_scores": target_scores.mean(),
            "pred_scores": pred_scores.mean(),
            "total": total_loss,
        }

    @staticmethod
    def get_dino_rep(rep_model, input_images, patch_coords, patch_size):
        input_images = input_images.repeat(1, 3, 1, 1)  # RGB channels are required
        patches = torch.stack(
            [input_images[:, :, x : x + patch_size[0], y : y + patch_size[1]] for x, y in patch_coords], dim=1
        )
        patches = patches.reshape(-1, patches.shape[-3], patches.shape[-2], patches.shape[-1])
        with torch.no_grad():
            rep = rep_model(patches)
        rep = rep.reshape(input_images.shape[0], len(patch_coords), rep.shape[-1])
        return rep


class RepDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files_list = list(os.listdir(data_dir))
        self.files_list = sorted(self.files_list)

        self.images = []
        for file in self.files_list:
            file_path = os.path.join(self.data_dir, file)
            image = nib.load(file_path).get_fdata()
            image = np.squeeze(image, axis=2)
            image = Image.fromarray(image.astype(np.uint8))
            image = img_transform(image)
            self.images.append(image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


def train(args):

    # Training params
    weight_decay = 1e-4
    momentum = 0.9
    checkpoint_interval = 10

    dino_resnet50 = torch.hub.load("facebookresearch/dino:main", "dino_resnet50", source="github")

    rep_model = dino_resnet50.to(device)
    rep_model.eval()


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
    model_path = model_dir.joinpath(model_name)
    print(model_path)

    loss_fun = RepresentationLoss(contrastive_rate=args.contrastive_rate)
    dataset = RepDataset(args.data_dir)
    train_idx = torch.arange(0, int(0.8 * len(dataset)))
    test_idx = torch.arange(int(0.8 * len(dataset)), len(dataset))
    dataset_train, dataset_test = (
        Subset(dataset, train_idx),
        Subset(dataset, test_idx),
    )

    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    data_loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    writer = SummaryWriter(args.runs_dir)
    representation_len = 2048
    model = RepresentationUNet(
        unet_out_dimensions=args.classes,
        patch_size=args.patch_size,
        representation_len=representation_len,
        sigmoid=args.sigmoid,
    )
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=weight_decay, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    model.to(device)

    if os.path.isfile(model_path):
        print("Loading model")
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        loaded_epoch_no = checkpoint["epoch"]
    else:
        loaded_epoch_no = -1

    try:
        for epoch in range(loaded_epoch_no + 1, loaded_epoch_no + 1 + args.epochs):
            torch.cuda.empty_cache()

            model.train()
            epoch_losses = {}
            for images in tqdm(data_loader_train, total=len(data_loader_train)):
                images = images.to(device)
                patches_coords = get_patches_coords(
                    images[0][0].shape, args.patches_choice, args.patch_size, args.step_size, args.num_samples
                )
                target = RepresentationLoss.get_dino_rep(rep_model, images, patches_coords, args.patch_size)
                optimizer.zero_grad()
                output = model(images, patches_coords)
                loss = loss_fun(output, target)
                tot_loss = loss["total"]
                tot_loss.backward()
                optimizer.step()
                for k, v in loss.items():
                    if f"train_{k}" in epoch_losses:
                        epoch_losses[f"train_{k}"].append(v.item())
                    else:
                        epoch_losses[f"train_{k}"] = [v.item()]
            model.eval()
            with torch.no_grad():
                for images in tqdm(data_loader_test, total=len(data_loader_test)):
                    images = images.to(device)
                    patches_coords = get_patches_coords(
                        images[0][0].shape, args.patches_choice, args.patch_size, args.step_size, args.num_samples
                    )
                    target = RepresentationLoss.get_dino_rep(rep_model, images, patches_coords, args.patch_size)
                    output = model(images, patches_coords)
                    loss = loss_fun(output, target)
                    for k, v in loss.items():
                        if f"test_{k}" in epoch_losses:
                            epoch_losses[f"test_{k}"].append(v.item())
                        else:
                            epoch_losses[f"test_{k}"] = [v.item()]

            loss_str = f"epoch: {epoch}\n"
            for k, v in epoch_losses.items():
                loss_str += f"{k}: {np.mean(v):.4f}\n"
            print(loss_str)
            writer.add_scalars("loss", {k: np.mean(v) for k, v in epoch_losses.items()}, global_step=epoch)
            if (epoch + 1) % checkpoint_interval == 0:
                print("Saving model checkpoint")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    },
                    model_path,
                )
            scheduler.step()
    except KeyboardInterrupt:
        pass
    finally:
        print("Saving model")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            model_path,
        )

        writer.close()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = args_parser()
    train(args)
