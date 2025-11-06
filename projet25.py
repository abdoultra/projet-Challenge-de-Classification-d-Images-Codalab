import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from torchvision import transforms

#############################
# 1. FUNCTIONS DEFINITIONS
#############################

def load_data():
    with open("starting_k/dataset_images_train", 'rb') as fo:
        dataset = pickle.load(fo)
    print("Clés disponibles dans dataset:", dataset.keys())
    X = dataset['data']
    y = dataset['target']
    return X, y

def show_image(flat_img):
    img = flat_img.reshape(3, 32, 32).transpose(1, 2, 0).astype(np.uint8)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def tsne_visualization(X, y, n_samples=3000):
    print("Calcul t-SNE (cela peut prendre un peu de temps)...")
    tsne = TSNE(n_components=2, random_state=42)
    X_reduced = tsne.fit_transform(X[:n_samples])
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y[:n_samples], cmap='tab10')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("t-SNE visualization of 3000 samples")
    plt.show()

#############################
# 2. DATASET CLASS
#############################

class ImageDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data.reshape(-1, 3, 32, 32).astype(np.float32)
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        img = torch.tensor(img)
        if self.transform:
            img = self.transform(img)
        return img, label

#############################
# 3. CNN MODEL
#############################

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)

        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#############################
# 4. TRAIN / EVAL FUNCTIONS
#############################

def train(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total

#############################
# 5. MAIN EXECUTION FLOW
#############################

if __name__ == "__main__":

    # ======================
    # Charger les données
    # ======================
    X, y = load_data()

    if X is None or y is None:
        print("Erreur de chargement")
        exit()

    show_image(X[0])  # Affiche la première image (optionnel)
    # tsne_visualization(X, y)  # Optionnel (long)

    # ======================
    # Train / Validation split
    # ======================
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalisation
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_valid = (X_valid - mean) / std

    # ======================
    # Data Augmentation sur le training set
    # ======================
    augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    ])

    # Dataset & Dataloader
    train_dataset = ImageDataset(X_train, y_train, transform=augmentation)
    valid_dataset = ImageDataset(X_valid, y_valid)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64)

    # ======================
    # CNN Model & Training
    # ======================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    best_acc = 0
    epochs_no_improve = 0
    n_epochs = 50
    patience = 7  # early stopping

    # ======================
    # Training loop
    # ======================
    for epoch in range(n_epochs):
        loss = train(model, train_loader, optimizer, criterion, device)
        acc = evaluate(model, valid_loader, device)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{n_epochs}] Loss: {loss:.4f} | Validation Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs")

        if epochs_no_improve >= patience:
            print("Early stopping triggered!")
            break

    # ======================
    # Générer la soumission
    # ======================
    with open("public_dat/data_images_test", 'rb') as fo:
        data_test = pickle.load(fo)

    X_test = data_test['data']
    X_test = (X_test - mean) / std

    X_test_dataset = ImageDataset(X_test, np.zeros(X_test.shape[0]))
    X_test_loader = DataLoader(X_test_dataset, batch_size=64)

    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    all_preds = []
    with torch.no_grad():
        for data, _ in X_test_loader:
            data = data.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())

    # ======================
    # Sauvegarde du fichier CSV pour Codalab
    # ======================

    # 1. Sauvegarde brute sans index ni header
    np.savetxt("images_test_predictions.csv", np.array(all_preds), fmt='%d')

    print("✅ Submission file created: images_test_predictions.csv")
