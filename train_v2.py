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
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from model import RepresentationUNet
from evaluate import evaluate_model

random.seed(42)

dino_resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')

model_folder = Path("model")
model_folder.mkdir(exist_ok=True)
model_path = "model/rep_net_v2.pt"
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
images_per_epoch = 5
contrastive_rate = 0.5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float)])

class RepresentationLoss(nn.Module):
    def __init__(self, contrastive_rate=0.5):
        super(RepresentationLoss, self).__init__()
        self.contrastive_rate = contrastive_rate

    def forward(self, pred_reps, target_reps):
        assert len(pred_reps) == len(target_reps) == 2

        similarity = (torch.linalg.norm(pred_reps[0]-target_reps[0]) + torch.linalg.norm(pred_reps[1]-target_reps[1])) / 2

        pred_dist = torch.linalg.norm(pred_reps[1] - pred_reps[0])
        target_dist = torch.linalg.norm(target_reps[1] - target_reps[0])

        contrastiveness = torch.nn.functional.relu(target_dist - pred_dist)

        total_loss = (1 - self.contrastive_rate) * similarity + self.contrastive_rate * contrastiveness

        return {"Similarity": (1 - self.contrastive_rate) * similarity,
                "Contrastiveness": self.contrastive_rate * contrastiveness,
                "Total Loss": total_loss}

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


def dataset_loader(test_set=False, patches_no=2):
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

        patches_coords = []
        for _ in range(patches_no):
            x_min = random.randint(0, image.shape[0] - patch_size[0])
            x_max = x_min + patch_size[0]

            y_min = random.randint(0, image.shape[1] - patch_size[1])
            y_max = y_min + patch_size[1]

            patches_coords.append(((x_min, x_max), (y_min, y_max)))

        image = np.squeeze(image, axis=2)
        image = Image.fromarray(image.astype(np.uint8))
        image = img_transform(image)

        yield image, patches_coords


def train(epochs_number):
    writer = SummaryWriter(os.path.join("runs", f"cr={contrastive_rate}"))

    model = RepresentationUNet(unet_out_dimensions=unet_output_classes, patch_size=patch_size, representation_len=2048)
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    model.to(device)

    if os.path.isfile(model_path):
        print("Loading model")
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loaded_epoch_no = checkpoint['epoch']
        losses = checkpoint['losses']
    else:
        loaded_epoch_no = -1
        losses = []

    loss_fun = RepresentationLoss(contrastive_rate=contrastive_rate)
    train_loader = dataset_loader(test_set=False, patches_no=2)

    try:
        for epoch in range(loaded_epoch_no + 1, loaded_epoch_no + 1 + epochs_number):
            torch.cuda.empty_cache()
            print(f"Epoch {epoch}")
            epoch_losses = []
            epoch_sim_losses = []
            epoch_con_losses = []
            for _ in range(images_per_epoch):
                image, patches_coords = next(train_loader)

                target_0 = RepresentationLoss.get_dino_rep(image, patches_coords[0])
                target_1 = RepresentationLoss.get_dino_rep(image, patches_coords[1])

                image = image.to(device)
                target_0 = target_0.to(device)
                target_1 = target_1.to(device)

                optimizer.zero_grad()
                model.set_patch_coordinates(patches_coords[0])
                output_0 = model(image.unsqueeze(0))

                model.set_patch_coordinates(patches_coords[1])
                output_1 = model(image.unsqueeze(0))

                loss = loss_fun([output_0, output_1], [target_0, target_1])

                sim_loss = loss["Similarity"]
                con_loss = loss["Contrastiveness"]
                tot_loss = loss["Total Loss"]

                tot_loss.backward()
                optimizer.step()
                epoch_losses.append(tot_loss.item())
                epoch_sim_losses.append(sim_loss.item())
                epoch_con_losses.append(con_loss.item())

            mean_loss = np.mean(epoch_losses)
            losses.append(mean_loss)
            mean_sim_loss = np.mean(epoch_sim_losses)
            mean_con_loss = np.mean(epoch_con_losses)
            print(f"Loss = {mean_loss} ({mean_sim_loss}+{mean_con_loss})")

            writer.add_scalars("Loss",
                               {"Similarity": mean_sim_loss,
                                "Contrastiveness": mean_con_loss,
                                "Total": mean_loss},
                               global_step=epoch)
            if (epoch + 1) % checkpoint_interval == 0:
                print("Saving model checkpoint")
                torch.save({'epoch': epoch,
                            'losses': losses,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, model_path)
    except KeyboardInterrupt:
        pass
    finally:
        print("Saving model")
        torch.save({'epoch': epoch,
                    'losses': losses,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, model_path)

        writer.close()

    return losses


if __name__ == "__main__":
    torch.cuda.empty_cache()
    train_losses = train(10)

    torch.cuda.empty_cache()
    _ = evaluate_model("model/rep_net_v2.pt", "plots/eval_cr05_epoch10.png")

    torch.cuda.empty_cache()
    train_losses.extend(train(40))

    torch.cuda.empty_cache()
    _ = evaluate_model("model/rep_net_v2.pt", "plots/eval_cr05_epoch50.png")

    torch.cuda.empty_cache()
    train_losses.extend(train(50))

    torch.cuda.empty_cache()
    _ = evaluate_model("model/rep_net_v2.pt", "plots/val_cr05_epoch100.png")

    plt.clf()
    plt.plot(train_losses)
    plt.grid()
    plt.savefig("loss.png")
