import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision import models
import random
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# TRANSFORMS
# =====================
base_transform = T.Compose([
    T.RandomResizedCrop(32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.4,0.4,0.4,0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize([0.4914,0.4822,0.4465],
                [0.247,0.243,0.261])
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.4914,0.4822,0.4465],
                [0.247,0.243,0.261])
])

# =====================
# TRANSFORM CLASSES
# =====================
class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]

class RandomImageTransform:
    def __init__(self, base_transform, dataset):
        self.base_transform = base_transform
        self.dataset = dataset

    def __call__(self, x):
        x1 = self.base_transform(x)

        idx = random.randint(0, len(self.dataset)-1)
        rand_img, _ = self.dataset[idx]
        x2 = self.base_transform(rand_img)

        return [x1, x2]

# =====================
# DATA
# =====================
base_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=None
)

dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True,
    transform=TwoCropsTransform(base_transform)
)

loader_ssl = DataLoader(dataset, batch_size=128, shuffle=True,
                        num_workers=4, pin_memory=True)

train_dataset = torchvision.datasets.CIFAR10(
    "./data", train=True, transform=test_transform
)
test_dataset = torchvision.datasets.CIFAR10(
    "./data", train=False, transform=test_transform
)

loader_clean = DataLoader(train_dataset, batch_size=128, shuffle=True,
                          num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=128,
                         num_workers=4, pin_memory=True)

# =====================
# VICReg LOSS
# =====================
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
    return 25*invariance_loss(z1,z2) + \
           25*(variance_loss(z1)+variance_loss(z2)) + \
           covariance_loss(z1)+covariance_loss(z2)

# =====================
# MODEL
# =====================
def get_encoder():
    encoder = models.resnet18(weights=None)
    encoder.conv1 = nn.Conv2d(3,64,3,1,1,bias=False)
    encoder.maxpool = nn.Identity()
    encoder.fc = nn.Identity()
    return encoder.to(device)

class Projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,512)
        )
    def forward(self,x):
        return self.net(x)

# =====================
# TRAIN FUNCTION (NEW SETUP)
# =====================
def train_ssl(t, corruption_window=20, recovery_window=10):
    encoder = get_encoder()
    projector = Projector().to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()), lr=1e-3
    )

    corrupt_start = t
    corrupt_end = t + corruption_window
    total_epochs = t + corruption_window + recovery_window

    for epoch in range(total_epochs):
        start_time = time.time()
        total_loss = 0

        # phase selection
        if epoch < corrupt_start:
            dataset.transform = TwoCropsTransform(base_transform)
            phase = "CLEAN"

        elif corrupt_start <= epoch < corrupt_end:
            dataset.transform = RandomImageTransform(base_transform, base_dataset)
            phase = f"CORRUPT"

        else:
            dataset.transform = TwoCropsTransform(base_transform)
            phase = "RECOVERY"

        for (x1, x2), _ in loader_ssl:
            x1, x2 = x1.to(device), x2.to(device)

            z1 = projector(encoder(x1))
            z2 = projector(encoder(x2))

            loss = vicreg_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # ===== METRICS =====
        with torch.no_grad():
            cos_sim = F.cosine_similarity(z1, z2).mean().item()
            std = z1.std(dim=0).mean().item()
            align = F.mse_loss(z1, z2).item()

        if (epoch+1) % 5 == 0:
            epoch_time = time.time() - start_time
            print(f"[t={t}] Epoch {epoch+1} | {phase} | Loss: {total_loss/len(loader_ssl):.4f} ")

    return encoder

# =====================
# LINEAR PROBE
# =====================
def train_linear_probe(encoder, loader):
    classifier = nn.Linear(512,10).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    encoder.eval()

    for epoch in range(10):
        for x,y in loader:
            x,y = x.to(device), y.to(device)

            with torch.no_grad():
                features = encoder(x)

            loss = criterion(classifier(features), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return classifier

def evaluate(encoder, classifier, loader):
    correct,total = 0,0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            preds = classifier(encoder(x)).argmax(1)
            correct += (preds==y).sum().item()
            total += y.size(0)
    return 100*correct/total

# =====================
# RUN EXPERIMENTS
# =====================
t_values = [5, 10, 15, 20, 25, 30]

results = {}

# Existing setup (corruption=20, recovery=10)
for t in t_values:
    print(f"\n===== EXPERIMENT t = {t} | (C=20, R=10) =====")

    encoder = train_ssl(t, corruption_window=20, recovery_window=10)

    clf = train_linear_probe(encoder, loader_clean)
    acc = evaluate(encoder, clf, test_loader)

    results[f"t={t},C=20,R=10"] = acc
    print(f"[t={t}] Accuracy: {acc:.2f}")


# NEW: corruption=10, recovery=10
for t in t_values:
    print(f"\n===== EXPERIMENT t = {t} | (C=10, R=10) =====")

    encoder = train_ssl(t, corruption_window=10, recovery_window=10)

    clf = train_linear_probe(encoder, loader_clean)
    acc = evaluate(encoder, clf, test_loader)

    results[f"t={t},C=10,R=10"] = acc
    print(f"[t={t}] Accuracy: {acc:.2f}")


# NEW: corruption=20, recovery=20
for t in t_values:
    print(f"\n===== EXPERIMENT t = {t} | (C=20, R=20) =====")

    encoder = train_ssl(t, corruption_window=20, recovery_window=20)

    clf = train_linear_probe(encoder, loader_clean)
    acc = evaluate(encoder, clf, test_loader)

    results[f"t={t},C=20,R=20"] = acc
    print(f"[t={t}] Accuracy: {acc:.2f}")


print("\nFINAL RESULTS:")
for k, v in results.items():
    print(k, ":", v)