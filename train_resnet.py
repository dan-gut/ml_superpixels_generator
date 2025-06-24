import argparse
import copy
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

from torchvision import datasets, models
from sklearn.metrics import roc_auc_score

from train import img_transform, MNIST_DATASETS
import torch.nn.functional as F

from medmnist.evaluator import getAUC, getACC

random.seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="", choices=MNIST_DATASETS.keys()
    )
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()
    return args


def eval(model, loader, task):
    model.eval()
    correct = 0
    total = 0
    ys, preds = list(), list()
    with torch.no_grad():
        for inputs, labels in tqdm(loader, total=len(loader)):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            predicted = torch.argmax(outputs, 1)
            if (len(labels.shape) == 2) and (labels.shape[-1] != 1):
                l = labels.argmax(-1)
            elif (len(labels.shape) == 2) and (labels.shape[-1] == 1):
                l = labels.squeeze(-1)
            else:
                l = labels
            total += l.size(-1)
            correct += (predicted == l).sum().item()

            ys.append(labels.detach().cpu())
            logit = F.softmax(outputs, dim=1)
            preds.append(logit.detach().cpu())
    y_true = torch.cat(ys).numpy()
    y_pred_prob = torch.cat(preds).numpy()
    auc = getAUC(y_true, y_pred_prob, task=task)
    acc = getACC(y_true, y_pred_prob, task=task)
    return acc, auc


def train(model, loader, optimizer, scheduler, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        assert inputs.shape[1] == 3
        optimizer.zero_grad()
        outputs = model(inputs)
        if labels.shape[-1] == 1:
            loss = criterion(outputs, labels.squeeze(-1))
        else:
            labels = labels.argmax(-1)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    scheduler.step()

    return running_loss / len(loader), correct / total


MNIST_TASKS = {
    "pneumonia": "binary-class",
    "path": "multi-class",
    "chest": "multi-label, binary-class",
    "derma": "multi-class",
    "oct": "multi-class",
    "retina": "regression",
    "breast": "binary-class",
    "blood": "multi-class",
    "tissue": "multi-class",
    "organa": "multi-class",
    "organc": "multi-class",
    "organs": "multi-class"
}

def main(args):
    print(args)
    batch_size = 128
    lr = 0.001
    epochs = 100
    task = MNIST_TASKS[args.dataset]
    dataset_train = MNIST_DATASETS[args.dataset](
        split="train",
        size=224,
        download=True,
        root="/net/tscratch/people/plgmpro/data/medmnist/",
        transform=img_transform,
        as_rgb=True,
    )
    dataset_val = MNIST_DATASETS[args.dataset](
        split="val",
        size=224,
        download=True,
        root="/net/tscratch/people/plgmpro/data/medmnist/",
        transform=img_transform,
        as_rgb=True,
    )
    dataset_test = MNIST_DATASETS[args.dataset](
        split="test",
        size=224,
        download=True,
        root="/net/tscratch/people/plgmpro/data/medmnist/",
        transform=img_transform,
        as_rgb=True,
    )
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    if dataset_train[0][1].shape[-1] != 1:
        num_classes = dataset_train[0][1].shape[-1]
    else:
        num_classes = max([y.max() + 1 for _, y in dataset_train])

    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    #model.load_state_dict(torch.load(args.save_path, map_location="cpu")["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 75], gamma=0.1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    accs = list()
    val_accs = list()
    max_acc = 0
    save_model = None
    for epoch in range(1, epochs+1):
        loss, train_acc = train(model, train_loader, optimizer, scheduler, criterion)
        val_acc, val_roc_auc = eval(model, val_loader, task)
        test_acc, test_roc_auc = eval(model, test_loader, task)
        val_accs.append(val_acc)
        accs.append((train_acc, val_acc, val_roc_auc, test_acc, test_roc_auc))
        if val_acc > max_acc:
            max_acc = val_acc
            save_model = copy.deepcopy(model)

        print(
            f"Epoch {epoch} Loss {loss:.4f} Train acc {train_acc:.2f} Val acc {val_acc:.2f} auc {val_roc_auc:.2f} Test {test_acc:.2f} auc {test_roc_auc:.2f}"
        )

    save_model = model
    train_acc, train_roc_auc = eval(save_model, train_loader, task)
    val_acc, val_roc_auc = eval(save_model, val_loader, task)
    test_acc, test_roc_auc = eval(save_model, test_loader, task)
    print(
        f"Final Train acc {train_acc:.2f} auc {train_roc_auc:.2f} Val acc {val_acc:.2f} auc {val_roc_auc:.2f} Test acc {test_acc:.2f} auc {test_roc_auc:.2f}"
    )
    torch.save({"args": vars(args), "model": save_model.state_dict()}, args.save_path)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = args_parser()
    main(args)
