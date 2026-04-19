import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")


def add_label_noise(labels, noise_ratio, num_classes=10):
    noisy_labels = labels.copy()
    n = len(labels)
    num_noisy = int(noise_ratio * n)

    indices = np.random.choice(n, num_noisy, replace=False)

    for i in indices:
        original = noisy_labels[i]
        new_label = np.random.randint(0, num_classes)

        while new_label == original:
            new_label = np.random.randint(0, num_classes)

        noisy_labels[i] = new_label

    return noisy_labels

class NoisyCIFAR10(torch.utils.data.Dataset):
    def __init__(self, dataset, noise_ratio):
        self.dataset = dataset
        self.original_labels = np.array(dataset.targets)
        self.noisy_labels = add_label_noise(self.original_labels, noise_ratio)

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        y = self.noisy_labels[idx]
        return x, y

    def __len__(self):
        return len(self.dataset)
    

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Augmentations
# =========================
augment = T.Compose([
    T.RandomResizedCrop(32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.4, 0.4, 0.4, 0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.247, 0.243, 0.261]),
])

# =========================
# Dataset wrapper
# =========================
class VICRegDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        return augment(x), augment(x)

    def __len__(self):
        return len(self.dataset)

# =========================
# Load CIFAR-10
# =========================
base_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=None
)

dataset = VICRegDataset(base_dataset)

loader = DataLoader(dataset, batch_size=216, shuffle=True, num_workers=2)

# =========================
# Encoder (ResNet18)
# =========================
encoder = models.resnet18(pretrained=False)
encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
encoder.maxpool = nn.Identity()
encoder.fc = nn.Identity()
encoder = encoder.to(device)

# =========================
# Projection Head
# =========================
class Projector(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=512, out_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

projector = Projector().to(device)

# =========================
# VICReg Loss Components
# =========================
def invariance_loss(z1, z2):
    return F.mse_loss(z1, z2)

def variance_loss(z):
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return torch.mean(F.relu(1 - std))

def covariance_loss(z):
    z = z - z.mean(dim=0)
    N, D = z.size()

    cov = (z.T @ z) / (N - 1)

    off_diag = cov.flatten()[1:].view(D - 1, D + 1)[:, :-1]
    return (off_diag ** 2).sum() / D

def vicreg_loss(z1, z2):
    sim = invariance_loss(z1, z2)
    var = variance_loss(z1) + variance_loss(z2)
    cov = covariance_loss(z1) + covariance_loss(z2)

    return 25 * sim + 25 * var + 1 * cov

def extract_features_encoder(model, dataset, num_samples=1500):
    model.eval()

    features = []
    labels = []

    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            z = model(x)

            features.append(z.cpu())
            labels.append(y)

            if sum(f.size(0) for f in features) >= num_samples:
                break

    features = torch.cat(features, dim=0)[:num_samples]
    labels = torch.cat(labels, dim=0)[:num_samples]

    return features.numpy(), labels.numpy()

def plot_tsne(features, labels, title, filename):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(8, 8))

    for i in range(10):
        idx = labels == i
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=str(i), s=5)

    plt.legend()
    plt.title(title)
    plt.savefig(filename)
    plt.show()

# =========================
# Optimizer
# =========================
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(projector.parameters()),
    lr=1e-3
)

# =========================
# Training Loop
# =========================
max_epochs = 300
patience = 20
min_delta = 1e-3

best_loss = float("inf")
patience_counter = 0

for epoch in range(max_epochs):
    total_loss = 0

    for x1, x2 in loader:
        x1, x2 = x1.to(device), x2.to(device)

        z1 = projector(encoder(x1))
        z2 = projector(encoder(x2))

        loss = vicreg_loss(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # Early stopping check
    if best_loss - avg_loss > min_delta:
        best_loss = avg_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"\nConverged at epoch {epoch+1}")
        break

print("\nChecking cosine similarity (SSL)")

x1, x2 = next(iter(loader))
x1, x2 = x1.to(device), x2.to(device)

with torch.no_grad():
    z1 = encoder(x1)
    z2 = encoder(x2)

cos_sim = F.cosine_similarity(z1, z2).mean()
print("Cosine similarity:", cos_sim.item())

print("\nGenerating t-SNE for SSL (VICReg encoder)")

viz_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465],
                [0.247, 0.243, 0.261])
])
)

features_ssl, labels_ssl = extract_features_encoder(
    encoder, viz_dataset, num_samples=1500
)

plot_tsne(features_ssl, labels_ssl,
          "t-SNE (SSL - VICReg Encoder)",
          "tsne_ssl.png")

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465],
                [0.247, 0.243, 0.261])
])
)
torch.save(encoder.state_dict(), "vicreg_encoder.pth")

dataset_10 = NoisyCIFAR10(train_dataset, 0.1)
dataset_30 = NoisyCIFAR10(train_dataset, 0.3)
dataset_50 = NoisyCIFAR10(train_dataset, 0.5)

loader_clean = DataLoader(train_dataset, batch_size=256, shuffle=True)
loader_10 = DataLoader(dataset_10, batch_size=256, shuffle=True)
loader_30 = DataLoader(dataset_30, batch_size=256, shuffle=True)
loader_50 = DataLoader(dataset_50, batch_size=256, shuffle=True)

print("Original label:", train_dataset.targets[0])
print("Noisy label (10%):", dataset_10.noisy_labels[0])

for param in encoder.parameters():
    param.requires_grad = False

encoder.eval()
classifier = nn.Linear(512, 10).to(device)
criterion = nn.CrossEntropyLoss()
def train_linear_probe(encoder, classifier, loader):
    classifier.train()
    optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    max_epochs = 200
    patience = 15
    min_delta = 1e-3

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(max_epochs):
        total_loss = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                features = encoder(x)

            outputs = classifier(features)
            loss = criterion(outputs, y)

            optimizer_cls.zero_grad()
            loss.backward()
            optimizer_cls.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Linear Probe] Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Linear probe converged at epoch {epoch+1}")
            break

def evaluate(encoder, classifier, loader):
    classifier.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            features = encoder(x)
            outputs = classifier(features)

            preds = outputs.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = 100 * correct / total
    return accuracy
test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
     transform=T.Compose([
        T.ToTensor(),
        T.Normalize([0.4914,0.4822,0.4465],
                    [0.247,0.243,0.261])
    ])

)

test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

print("\nTraining on CLEAN labels")
train_linear_probe(encoder, classifier, loader_clean)

acc_clean = evaluate(encoder, classifier, test_loader)
print("Accuracy (clean):", acc_clean)

classifier = nn.Linear(512, 10).to(device)  # reset

print("\nTraining on 10% noisy labels")
train_linear_probe(encoder, classifier, loader_10)

acc_10 = evaluate(encoder, classifier, test_loader)
print("Accuracy (10% noise):", acc_10)

classifier = nn.Linear(512, 10).to(device)

print("\nTraining on 30% noisy labels")
train_linear_probe(encoder, classifier, loader_30)

acc_30 = evaluate(encoder, classifier, test_loader)
print("Accuracy (30% noise):", acc_30)
classifier = nn.Linear(512, 10).to(device)

print("\nTraining on 50% noisy labels")
train_linear_probe(encoder, classifier, loader_50)

acc_50 = evaluate(encoder, classifier, test_loader)
print("Accuracy (50% noise):", acc_50)

# =========================
# SUPERVISED BASELINE
# =========================
def get_supervised_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 10)
    return model.to(device) 

def train_supervised(model, loader):
    model.train()

    optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4
)
    criterion = nn.CrossEntropyLoss()

    max_epochs = 200
    patience = 15
    min_delta = 1e-3

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(max_epochs):
        total_loss = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Supervised] Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Supervised converged at epoch {epoch+1}")
            break

def evaluate_supervised(model, loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            preds = outputs.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return 100 * correct / total

# =========================
# SUPERVISED EXPERIMENTS
# =========================

# CLEAN
model = get_supervised_model()
print("\nSupervised: CLEAN labels")
train_supervised(model, loader_clean)
acc_sup_clean = evaluate_supervised(model, test_loader)
print("Supervised Accuracy (clean):", acc_sup_clean)

print("\nChecking cosine similarity (Supervised)")

x, _ = next(iter(loader_clean))
x = x.to(device)

with torch.no_grad():
    z = model(x)

# Compare same batch (not augmentations)
cos_sim_sup = F.cosine_similarity(z[:-1], z[1:]).mean()
print("Supervised cosine similarity:", cos_sim_sup.item())

print("\nGenerating t-SNE for Supervised Model")

features_sup, labels_sup = extract_features_encoder(
    model, viz_dataset, num_samples=1500
)

plot_tsne(features_sup, labels_sup,
          "t-SNE (Supervised Model)",
          "tsne_supervised.png")
# 10%
model = get_supervised_model()
print("\nSupervised: 10% noise")
train_supervised(model, loader_10)
acc_sup_10 = evaluate_supervised(model, test_loader)
print("Supervised Accuracy (10%):", acc_sup_10)

# 30%
model = get_supervised_model()
print("\nSupervised: 30% noise")
train_supervised(model, loader_30)
acc_sup_30 = evaluate_supervised(model, test_loader)
print("Supervised Accuracy (30%):", acc_sup_30)

# 50%
model = get_supervised_model()
print("\nSupervised: 50% noise")
train_supervised(model, loader_50)
acc_sup_50 = evaluate_supervised(model, test_loader)
print("Supervised Accuracy (50%):", acc_sup_50)

def print_results_table():
    print("\nFinal Results Table\n")

    headers = ["Model", "CLEAN", "10%", "30%", "50%"]
    row_ssl = ["SSL (VICReg)", acc_clean, acc_10, acc_30, acc_50]
    row_sup = ["Supervised", acc_sup_clean, acc_sup_10, acc_sup_30, acc_sup_50]

    print("{:<20} {:<10} {:<10} {:<10} {:<10}".format(*headers))
    print("-" * 60)

    print("{:<20} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f}".format(*row_ssl))
    print("{:<20} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f}".format(*row_sup))

print_results_table()