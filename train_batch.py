import os
from pathlib import Path
import random

import nibabel as nib
import numpy as np
from PIL import Image
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import tqdm

from model_batch import RepresentationUNet

random.seed(42)

dino_resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')

model_folder = Path("model")
model_folder.mkdir(exist_ok=True)
model_path = "model/rep_net_v2.pt"
data_base_dir = os.path.join("Data", "images_resized")

test_img_ratio = 0.2

# Model params
patch_size = (40, 40)
unet_output_classes = 1024

# Training params
learning_rate = 0.001
weight_decay = 1e-4
momentum = 0.9
checkpoint_interval = 10
contrastive_rate = 0.5
batches_per_epoch = 5
batch_size = 4

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

        similarity = (torch.linalg.norm(pred_reps[0]-target_reps[0], dim=1) +
                      torch.linalg.norm(pred_reps[1]-target_reps[1], dim=1)) / 2

        similarity = torch.sum(similarity)  # sum over a batch

        pred_dist = torch.linalg.norm(pred_reps[1] - pred_reps[0], dim=1)
        target_dist = torch.linalg.norm(target_reps[1] - target_reps[0], dim=1)

        contrastiveness = torch.nn.functional.relu(target_dist - pred_dist)

        contrastiveness = torch.sum(contrastiveness)  # sum over a batch

        total_loss = (1 - self.contrastive_rate) * similarity + self.contrastive_rate * contrastiveness

        return {"Similarity": (1 - self.contrastive_rate) * similarity,
                "Contrastiveness": self.contrastive_rate * contrastiveness,
                "Total Loss": total_loss}

    @staticmethod
    def get_dino_rep(input_image, patch_coords):
        input_image = input_image.repeat(3, 1, 1)  # RGB channels are required
        patch = input_image[:, patch_coords[0]:patch_coords[1], patch_coords[2]:patch_coords[3]]
        patch = patch.unsqueeze(0)

        rep_model = dino_resnet50.to(device)
        rep_model.eval()

        patch = patch.to(device)
        with torch.no_grad():
            rep = rep_model(patch)

        return rep.detach().cpu()


class RepDataset(IterableDataset):
    def __init__(self, data_dir, patches_no=2, test_set=False, test_ratio=0.2):
        self.files_list = list(os.listdir(data_dir))
        self.files_list = sorted(self.files_list)

        self.train_list = self.files_list[:int((1 - test_ratio) * len(self.files_list))]
        self.test_list = self.files_list[int((1 - test_ratio) * len(self.files_list)):]

        self.test_set = test_set
        self.patches_no = patches_no

    def __iter__(self):
        while True:
            if self.test_set:
                chosen_file = random.choice(self.test_list)
            else:
                chosen_file = random.choice(self.train_list)

            file_path = os.path.join(data_base_dir, chosen_file)
            image = nib.load(file_path).get_fdata()

            patches_coords = []
            for _ in range(self.patches_no):
                x_min = random.randint(0, image.shape[0] - patch_size[0])
                x_max = x_min + patch_size[0]

                y_min = random.randint(0, image.shape[1] - patch_size[1])
                y_max = y_min + patch_size[1]

                patches_coords.append(np.array((x_min, x_max, y_min, y_max)))

            image = np.squeeze(image, axis=2)
            image = Image.fromarray(image.astype(np.uint8))
            image = img_transform(image)

            yield image, patches_coords


def get_target_representations(images, patches_coords):
    targets_0 = []
    targets_1 = []

    for i, image in enumerate(images):
        targets_0.append(RepresentationLoss.get_dino_rep(image, patches_coords[0][i]))
        targets_1.append(RepresentationLoss.get_dino_rep(image, patches_coords[1][i]))

    return torch.stack(targets_0).squeeze(1), torch.stack(targets_1).squeeze(1)


def train(epochs_number, data_loader, loss_fun):
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

    try:
        for epoch in range(loaded_epoch_no + 1, loaded_epoch_no + 1 + epochs_number):
            torch.cuda.empty_cache()
            epoch_losses = []
            epoch_sim_losses = []
            epoch_con_losses = []

            for _ in tqdm.trange(batches_per_epoch, unit="batch", desc=f"Epoch {epoch}"):
                image, patches_coords = next(iter(data_loader))

                target_0, target_1 = get_target_representations(image, patches_coords)

                image = image.to(device)
                target_0 = target_0.to(device)
                target_1 = target_1.to(device)

                optimizer.zero_grad()

                output_0 = model(image, patches_coords[0])
                output_1 = model(image, patches_coords[1])

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
    loss_fun = RepresentationLoss(contrastive_rate=contrastive_rate)
    dataset = RepDataset(data_base_dir, patches_no=2, test_set=False, test_ratio=0.2)
    data_loader = DataLoader(dataset, batch_size=batch_size)

    train(30, data_loader, loss_fun)
