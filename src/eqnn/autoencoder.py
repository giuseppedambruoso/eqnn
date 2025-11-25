import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import datasets, transforms

# -------------------
# 1️⃣ Carica ResNet-18 e tronca al layer3
# -------------------
resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet18.eval()

model_16x16 = nn.Sequential(
    resnet18.conv1,
    resnet18.bn1,
    resnet18.relu,
    resnet18.maxpool,
    resnet18.layer1,
    resnet18.layer2,
    resnet18.layer3,
)

# -------------------
# 2️⃣ Dati EuroSAT (train e test)
# -------------------
train_dir = "../../data/EuroSAT_split/train"
test_dir = "../../data/EuroSAT_split/test"

preprocess = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_set = datasets.ImageFolder(train_dir, transform=preprocess)
test_set = datasets.ImageFolder(test_dir, transform=preprocess)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

# -------------------
# 3️⃣ Funzione comune per generare nuovi dataset
# -------------------


def generate_feature_dataset(loader, dataset, output_root):
    os.makedirs(output_root, exist_ok=True)

    for idx, (img, label) in enumerate(loader):
        class_name = dataset.classes[label]

        # Crea dir output per classe
        class_out = os.path.join(output_root, class_name)
        os.makedirs(class_out, exist_ok=True)

        with torch.no_grad():
            fmap = model_16x16(img)  # → [1, C, 16, 16]

        # Somma i canali → [16, 16]
        summed = fmap.sum(dim=1).squeeze().cpu()

        # Normalizza a [0,1]
        min_val, max_val = summed.min(), summed.max()
        norm = (summed - min_val) / (max_val - min_val + 1e-8)

        np_img = norm.numpy()

        # Salva immagine 16x16
        out_path = os.path.join(class_out, f"{idx}.png")
        plt.imsave(out_path, np_img, cmap="viridis")

        if idx % 500 == 0:
            print(f"[{idx}] salvata {out_path}")

    print(f"✔ Dataset creato in: {output_root}")


# -------------------
# 4️⃣ Genera nuovo train e nuovo test
# -------------------

generate_feature_dataset(train_loader, train_set, "EuroSAT_16x16/train")
generate_feature_dataset(test_loader, test_set, "EuroSAT_16x16/test")
