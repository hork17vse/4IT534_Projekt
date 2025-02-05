import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from PIL import Image
from torchvision.models import ResNet18_Weights
from sklearn.metrics import roc_auc_score
import numpy as np
import time
from sklearn.model_selection import StratifiedShuffleSplit
import random

# Vytvoření složky pro ukládání modelu, pokud neexistuje
MODEL_SAVE_DIR = "model"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Konfigurace
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed pro reprodukovatelnost
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Cesty k datasetu
DATASET_PATH_1 = "data/HAM10000_images_part_1"
DATASET_PATH_2 = "data/HAM10000_images_part_2"
METADATA_PATH = "data/HAM10000_metadata.csv"

# Načtení metadat a definice labelů a label_to_idx
metadata_df = pd.read_csv(METADATA_PATH)
labels = metadata_df['dx'].unique()
label_to_idx = {label: idx for idx, label in enumerate(labels)}
num_classes = len(labels)

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

# Vytvoření datasetu a rozdělení na trénovací, evaluační a testovací sady
dataset = SkinCancerDataset(metadata_df, transform=transform)

# Rozdělení datasetu pomocí StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED) # 0.2 = eval + test
train_idx, eval_test_idx = next(sss.split(metadata_df, metadata_df['dx']))
train_df = metadata_df.iloc[train_idx]
eval_test_df = metadata_df.iloc[eval_test_idx]

sss_eval_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED) # 0.5 from 0.2 = 0.1 test and 0.1 eval
eval_idx, test_idx = next(sss_eval_test.split(eval_test_df, eval_test_df['dx']))
eval_df = eval_test_df.iloc[eval_idx]
test_df = eval_test_df.iloc[test_idx]

train_dataset = SkinCancerDataset(train_df, transform=transform)
eval_dataset = SkinCancerDataset(eval_df, transform=transform)
test_dataset = SkinCancerDataset(test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Definice modelu ResNet18
MODEL_NAME = "resnet"

print(f"\nTrénování modelu: {MODEL_NAME}")
print(f"Trénuji {MODEL_NAME} s {num_classes} třídami: {labels}")

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# Zmrazení většiny vrstev (např. prvních 5)
for param in model.parameters():
    param.requires_grad = False

# Rozmrazení posledních vrstev (postupně)
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, num_classes)
)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

best_acc = 0.0
best_auc = 0.0

for epoch in range(EPOCHS):
    start_time = time.time()

    # Trénování
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{i+1}/{len(train_loader)}], Loss: {running_loss / (i+1):.4f}")

    scheduler.step()

    # Evaluace a testování (spojené v jeden blok po každé epoše)
    model.eval()
    eval_correct = 0
    eval_total = 0
    eval_all_labels = []
    eval_all_probs = []

    test_correct = 0  # Přesunuto sem
    test_total = 0    # Přesunuto sem
    test_all_labels = [] # Přesunuto sem
    test_all_probs = [] # Přesunuto sem

    with torch.no_grad():
        for images, labels in eval_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            eval_total += labels.size(0)
            eval_correct += (predicted == labels).sum().item()
            probabilities = torch.softmax(outputs, dim=1)
            eval_all_labels.extend(labels.cpu().numpy())
            eval_all_probs.extend(probabilities.cpu().numpy())

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            probabilities = torch.softmax(outputs, dim=1)
            test_all_labels.extend(labels.cpu().numpy())
            test_all_probs.extend(probabilities.cpu().numpy())

        test_acc = 100 * test_correct / test_total
        test_auc = roc_auc_score(test_all_labels, test_all_probs, multi_class='ovr')

        eval_acc = 100 * eval_correct / eval_total  # Výpočet eval_acc
        eval_auc = roc_auc_score(eval_all_labels, eval_all_probs, multi_class='ovr')  # Výpočet eval_auc

    end_time = time.time()
    epoch_time = end_time - start_time

    print(
        f"Model {MODEL_NAME} - Epoch {epoch + 1}/{EPOCHS}, Train Loss: {running_loss / len(train_loader):.4f}, Eval Accuracy: {eval_acc:.2f}%, Eval AUC: {eval_auc:.4f}, Test Accuracy: {test_acc:.2f}%, Test AUC: {test_auc:.4f}, Time: {epoch_time:.2f} seconds")

    # Ukládání modelu na základě evaluace
    if eval_acc > best_acc:
        best_acc = eval_acc
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f"best_{MODEL_NAME}.pth"))

print(f"Model {MODEL_NAME} - Nejlepší přesnost (evaluace): {best_acc:.2f}%\n")