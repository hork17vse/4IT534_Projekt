import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from PIL import Image
from torchvision.models import EfficientNet_B0_Weights, ResNet18_Weights, MobileNet_V3_Large_Weights
from sklearn.metrics import roc_auc_score
import numpy as np

# Konfigurace
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cesty k datasetu
DATASET_PATH_1 = "data/HAM10000_images_part_1"
DATASET_PATH_2 = "data/HAM10000_images_part_2"
METADATA_PATH = "data/HAM10000_metadata.csv"

# Načtení metadat a definice labels a label_to_idx - JEN JEDNOU!
metadata_df = pd.read_csv(METADATA_PATH)
labels = metadata_df['dx'].unique()  # Unikátní labely pro celý dataset
label_to_idx = {label: idx for idx, label in enumerate(labels)}
num_classes = len(labels)  # Počet tříd pro všechny modely

# Transformace obrázků
transform = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Vlastní dataset třída
class SkinCancerDataset(Dataset):
    def __init__(self, metadata_df, transform=None):
        self.metadata_df = metadata_df
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for _, row in metadata_df.iterrows():
            image_id = row['image_id'] + ".jpg"
            image_path = os.path.join(DATASET_PATH_1, image_id)
            if not os.path.exists(image_path):
                image_path = os.path.join(DATASET_PATH_2, image_id)

            if os.path.exists(image_path):
                self.image_paths.append(image_path)
                self.labels.append(label_to_idx[row['dx']])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Použít celý dataset (nebo větší část) - zakomentujte subset_size
subset_size = int(0.15 * len(metadata_df))
metadata_df = metadata_df.sample(n=subset_size, random_state=42).reset_index(drop=True)

dataset = SkinCancerDataset(metadata_df, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Výběr modelů
MODEL_NAMES = ["efficientnet", "resnet", "mobilenet"]

for MODEL_NAME in MODEL_NAMES:
    print(f"\nTrénování modelu: {MODEL_NAME}")
    print(f"Trénuji {MODEL_NAME} s {num_classes} třídami: {labels}")  # Používáme num_classes

    if MODEL_NAME == "efficientnet":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        for param in model.features[-5:].parameters():  # Fine-tuning posledních 5 vrstev
            param.requires_grad = True
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.classifier[1].in_features, num_classes)  # num_classes!
        )
    elif MODEL_NAME == "resnet":
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        for param in model.layer3.parameters():  # Fine-tuning posledních vrstev
            param.requires_grad = True
        for param in model.layer4.parameters():  # Přidání další vrstvy k fine-tuningu
            param.requires_grad = True
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, num_classes)  # num_classes!
        )
    elif MODEL_NAME == "mobilenet":
        model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        for param in model.features[-7:].parameters():  # Fine-tuning posledních 7 vrstev
            param.requires_grad = True
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(960, num_classes)  # num_classes!
        )
    else:
        raise ValueError("Neplatný model")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_acc = 0.0
    best_auc = 0.0  # Pro AUC
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        all_labels = []  # Pro labely všech vzorků
        all_probs = []  # Pro predikce (pravděpodobnosti) všech vzorků

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)  # Výstup modelu (logits)

                # Spočítáme predikce (argmax)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # KLÍČOVÁ ZMĚNA:
                # 1. Aplikujeme softmax pro získání pravděpodobností
                probabilities = torch.softmax(outputs, dim=1)

                # 2. Uložíme labely a pravděpodobnosti
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())

        acc = 100 * correct / total

        # Po skončení testovacího cyklu:
        # Spočítáme AUC z uložených pravděpodobností a labelů.
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')

        print(f"Model {MODEL_NAME} - Epoch {epoch + 1}/{EPOCHS}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {acc:.2f}%, AUC: {auc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"best_{MODEL_NAME}.pth")

    print(f"Model {MODEL_NAME} - Nejlepší přesnost: {best_acc:.2f}%\n")