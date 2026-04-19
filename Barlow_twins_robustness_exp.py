import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def add_label_noise(labels, noise_ratio, num_classes=10):
    noisy = labels.copy()
    n = len(labels)
    num_noisy = int(noise_ratio * n)

    idx = np.random.choice(n, num_noisy, replace=False)

    for i in idx:
        new_label = np.random.randint(0, num_classes)
        while new_label == noisy[i]:
            new_label = np.random.randint(0, num_classes)
        noisy[i] = new_label

    return noisy

#-------------label noise-----------------

class NoisyDataset(Dataset):
    def __init__(self, dataset, noise_ratio):
        self.dataset = dataset
        self.labels = np.array(dataset.targets)
        self.noisy_labels = add_label_noise(self.labels, noise_ratio)

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        return x, self.noisy_labels[idx]

    def __len__(self):
        return len(self.dataset)
    
#-------------Barlow twins augmentation-----------------

class BarlowTransform:
    def __init__(self):
        self.transform = T.Compose([
            T.RandomResizedCrop(32, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize([0.4914,0.4822,0.4465],
                        [0.247,0.243,0.261])
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)
    
#-------------Dataset wrapper-----------------

class SSLDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = BarlowTransform()

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        return self.transform(x)

    def __len__(self):
        return len(self.dataset)
    
#-------------Model (Encoder + Projector)-----------------
    
class BarlowModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = models.resnet18(weights=None)
        self.encoder.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.encoder.maxpool = nn.Identity()
        self.encoder.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return z
    
#-------------New for Cosine similarity evaluation-----------------

from torchvision.transforms.functional import to_pil_image

def compute_cosine(model, loader):
    model.eval()

    cos_y = 0
    cos_z = 0
    count = 0

    transform = BarlowTransform()

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)

            # convert each image to PIL first
            x1 = torch.stack([
                transform(to_pil_image(img.cpu()))[0] for img in x
            ]).to(device)

            x2 = torch.stack([
                transform(to_pil_image(img.cpu()))[1] for img in x
            ]).to(device)

            # encoder output
            y1 = model.encoder(x1)
            y2 = model.encoder(x2)

            # projector output
            z1 = model.projector(y1)
            z2 = model.projector(y2)

            cos_y += F.cosine_similarity(y1, y2, dim=1).mean().item()
            cos_z += F.cosine_similarity(z1, z2, dim=1).mean().item()

            count += 1

    return cos_y / count, cos_z / count

#-------------Barlow Twins Loss-----------------

def off_diagonal(x):
    n, m = x.shape
    return x.flatten()[:-1].view(n-1, n+1)[:,1:].flatten()


def barlow_loss(z1, z2, lambd=5e-3):
    z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-9)
    z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-9)

    N = z1.size(0)
    c = (z1.T @ z2) / N

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag_term = off_diagonal(c).pow_(2).sum()

    return on_diag + lambd * off_diag_term

#-------------Training loop-----------------

def train_ssl(model, loader, epochs=50):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total_loss = 0

        for x1, x2 in loader:
            x1, x2 = x1.to(device), x2.to(device)

            z1 = model(x1)
            z2 = model(x2)

            loss = barlow_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[SSL] Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

#-------------Linear Probe-----------------

#def train_linear_probe(encoder, loader):
#    classifier = nn.Linear(512, 10).to(device)
#    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
#    criterion = nn.CrossEntropyLoss()

#    encoder.eval()

#    for epoch in range(50):
#        for x, y in loader:
#            x, y = x.to(device), y.to(device)

#            with torch.no_grad():
#                feats = encoder(x)

#            out = classifier(feats)
#            loss = criterion(out, y)

#            optimizer.zero_grad()
#            loss.backward()
#            optimizer.step()

#    return classifier

#-------------Evaluation-----------------

#def evaluate(encoder, classifier, loader):
#    encoder.eval()
#    classifier.eval()

#    correct = 0
#    total = 0

#    with torch.no_grad():
#        for x, y in loader:
#            x, y = x.to(device), y.to(device)

#            feats = encoder(x)
#            out = classifier(feats)

#            pred = out.argmax(1)
#            correct += (pred == y).sum().item()
#            total += y.size(0)

#    return 100 * correct / total

#-------------Supervised Baseline-----------------

#def train_supervised(loader):
#    model = models.resnet18(weights=None)
#    model.fc = nn.Linear(512, 10)
#    model = model.to(device)

#    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#    criterion = nn.CrossEntropyLoss()

#    for epoch in range(100):
#        for x, y in loader:
#            x, y = x.to(device), y.to(device)

#            out = model(x)
#            loss = criterion(out, y)

#            optimizer.zero_grad()
#            loss.backward()
#            optimizer.step()

#    return model

#def evaluate_supervised(model, loader):
#    model.eval()
#    correct, total = 0, 0

#    with torch.no_grad():
#        for x, y in loader:
#            x, y = x.to(device), y.to(device)

#            out = model(x)  # already gives final predictions
#            preds = out.argmax(dim=1)

#            correct += (preds == y).sum().item()
#            total += y.size(0)

#    return 100 * correct / total

#-------------Main-----------------

# Load dataset
train_base = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
#---------------------------------------------------------
# For faster experimentation, we can use a subset of the training data for SSL pretraining. In practice, you would use the full dataset.
from torch.utils.data import Subset

indices = np.random.choice(len(train_base), 500, replace=False)
train_base = Subset(train_base, indices)
#---------------------------------------------------------
test_set = torchvision.datasets.CIFAR10(
    root="./data", train=False,
    transform=T.Compose([
        T.ToTensor(),
        T.Normalize([0.4914,0.4822,0.4465],
                    [0.247,0.243,0.261])
    ])
)

ssl_data = SSLDataset(train_base)
ssl_loader = DataLoader(ssl_data, batch_size=256, shuffle=True)

# Train SSL
model = BarlowModel().to(device)
train_ssl(model, ssl_loader)

# Prepare datasets
transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.4914,0.4822,0.4465],
                [0.247,0.243,0.261])
])

train_clean = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform)
train_10 = NoisyDataset(train_clean, 0.1)
train_30 = NoisyDataset(train_clean, 0.3)
train_50 = NoisyDataset(train_clean, 0.5)

loaders = {
    "clean": DataLoader(train_clean, batch_size=256, shuffle=True),
    "10%": DataLoader(train_10, batch_size=256, shuffle=True),
    "30%": DataLoader(train_30, batch_size=256, shuffle=True),
    "50%": DataLoader(train_50, batch_size=256, shuffle=True),
}

test_loader = DataLoader(test_set, batch_size=256)

# SSL results
#results_ssl = {}

print("\nCOSINE SIMILARITY RESULTS\n")

for k, loader in loaders.items():
    print(f"\nNoise Level: {k}")

    cos_y, cos_z = compute_cosine(model, loader)

    print(f"Before projector (y): {cos_y:.4f}")
    print(f"After projector (z):  {cos_z:.4f}")

# Supervised results
#results_sup = {}

#for k, loader in loaders.items():
#    sup_model = train_supervised(loader)
#    acc = evaluate_supervised(sup_model, test_loader)
#    results_sup[k] = acc
#    print(f"Supervised ({k}): {acc:.2f}")
