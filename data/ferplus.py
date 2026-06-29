"""
Carregamento do dataset FER+ (versao em imagens, organizada por pastas de classe).

Layout esperado (ImageFolder):
    <data_dir>/train/<emocao>/*.png
    <data_dir>/test/<emocao>/*.png
(aceita tambem 'valid'/'val' como split de validacao)

A versao do Kaggle (arnabkumarroy02/ferplus) ja vem em imagens 112x112 com 8
classes: anger, contempt, disgust, fear, happy, neutral, sad, surprise.
Se voce usar a rota oficial do Microsoft FER+ (fer2013new.csv), rode antes
scripts/prepare_ferplus.py para gerar essa estrutura de pastas.
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

import config


def build_transforms(img_size=config.IMG_SIZE, train=True):
    if train:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN, config.STD),
        ])
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])


def _find_split(data_dir, names):
    for n in names:
        p = os.path.join(data_dir, n)
        if os.path.isdir(p):
            return p
    return None


def get_dataloaders(data_dir=config.DATA_DIR, img_size=config.IMG_SIZE,
                    batch_size=128, num_workers=4, balanced_sampler=False):
    """Retorna (train_loader, val_loader, classes, class_weights)."""
    train_dir = _find_split(data_dir, ["train", "Training", "FER2013Train"])
    val_dir = _find_split(data_dir, ["test", "valid", "val", "Valid",
                                     "PublicTest", "FER2013Valid", "FER2013Test"])
    if train_dir is None or val_dir is None:
        raise FileNotFoundError(
            f"Nao encontrei subpastas de train/test em '{data_dir}'. "
            f"Estrutura esperada: <data_dir>/train/<classe>/*  e  <data_dir>/test/<classe>/*")

    train_ds = datasets.ImageFolder(train_dir, transform=build_transforms(img_size, True))
    val_ds = datasets.ImageFolder(val_dir, transform=build_transforms(img_size, False))
    classes = train_ds.classes

    # pesos por classe (FER+ e desbalanceado)
    counts = np.bincount([y for _, y in train_ds.samples], minlength=len(classes))
    class_weights = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)),
                                 dtype=torch.float32)

    if balanced_sampler:
        sample_w = class_weights[[y for _, y in train_ds.samples]]
        sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=4, pin_memory=False, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=False, drop_last=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=False)
    return train_loader, val_loader, classes, class_weights


if __name__ == "__main__":
    tl, vl, classes, w = get_dataloaders(num_workers=0, batch_size=8)
    xb, yb = next(iter(tl))
    print("classes:", classes)
    print("batch:", xb.shape, yb.shape, "| pesos:", w.tolist())
