import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA (SimCLR Augmentation)
class SimCLRTransform:
    def __init__(self, size=32):
        self.transform = T.Compose([
            T.RandomResizedCrop(size=size),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=5),
            T.ToTensor()
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)

def get_train_loader(batch_size=128):
    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True,
        transform=SimCLRTransform()
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# MODEL (Modified ResNet)
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)

        resnet.conv1 = nn.Conv2d(3,64,3,1,1,bias=False)
        resnet.maxpool = nn.Identity()

        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        return torch.flatten(self.encoder(x),1)

class ProjectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Linear(128,64)
        )

    def forward(self,x):
        return self.mlp(x)

class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.projector = ProjectionHead()

    def forward(self,x):
        h = self.encoder(x)
        z = self.projector(h)
        return h,z

# LOSS 
def nt_xent_loss(z_i, z_j, temperature=0.5):
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    z = torch.cat([z_i, z_j], dim=0)
    sim = torch.matmul(z, z.T) / temperature

    mask = torch.eye(sim.size(0), dtype=torch.bool).to(device)
    sim = sim.masked_fill(mask, -1e4)

    positives = torch.cat([
        torch.diag(sim, z_i.size(0)),
        torch.diag(sim, -z_i.size(0))
    ])

    exp_sim = torch.exp(sim)
    denom = exp_sim.sum(dim=1)

    loss = -torch.log(torch.exp(positives)/denom)
    return loss.mean()

# TRAIN SIMCLR
def train_simclr(model, loader, epochs=250):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")

    accum_steps = 4
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        optimizer.zero_grad()

        pbar = tqdm(loader)

        for i, ((x_i,x_j),_) in enumerate(pbar):
            x_i,x_j = x_i.to(device), x_j.to(device)

            with torch.amp.autocast("cuda"):
                _,z_i = model(x_i)
                _,z_j = model(x_j)
                loss = nt_xent_loss(z_i,z_j) / accum_steps

            scaler.scale(loss).backward()

            if (i+1)%accum_steps==0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()*accum_steps
            pbar.set_postfix(loss=loss.item()*accum_steps)

        print(f"Epoch {epoch+1}: {total_loss/len(loader):.4f}")

# LINEAR EVAL
class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512,10)

    def forward(self,x):
        return self.fc(x)

def train_linear(model, classifier, loader, epochs=250):
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.eval()
    classifier.train()

    for epoch in range(epochs):
        total=0
        for x,y in loader:
            x,y = x.to(device), y.to(device)

            with torch.no_grad():
                h = model.encoder(x)

            out = classifier(h)
            loss = criterion(out,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()

        print(f"Linear Epoch {epoch+1}: {total/len(loader):.4f}")

def evaluate(model, classifier, loader):
    correct=0
    total=0

    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            h = model.encoder(x)
            out = classifier(h)
            _,pred = torch.max(out,1)

            total += y.size(0)
            correct += (pred==y).sum().item()

    print(f"Accuracy: {100*correct/total:.2f}%")

# LABEL NOISE
def add_label_noise(dataset, noise):
    labels = dataset.targets.copy()
    n = int(len(labels)*noise)

    idx = random.sample(range(len(labels)), n)

    for i in idx:
        new = random.randint(0,9)
        while new==labels[i]:
            new = random.randint(0,9)
        labels[i]=new

    dataset.targets = labels
    return dataset

# MAIN
if __name__=="__main__":

    BATCH=128

    # TRAIN SIMCLR 
    loader = get_train_loader(BATCH)
    model = SimCLR().to(device)

    train_simclr(model, loader, epochs=250)

    torch.save(model.state_dict(),"simclr_model.pth")

    # DATA 
    transform = T.ToTensor()

    train_clean = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_data = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    test_loader = DataLoader(test_data, batch_size=BATCH)

    # SSL MODEL
    ssl_model = SimCLR().to(device)
    ssl_model.load_state_dict(torch.load("simclr_model.pth"))
    ssl_model.eval()

    for p in ssl_model.encoder.parameters():
        p.requires_grad=False

    # EXP1
    for noise in [0.1,0.3,0.5]:

        print(f"\n===== Noise {noise*100}% =====")

        noisy = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        noisy = add_label_noise(noisy, noise)

        loader_noisy = DataLoader(noisy, batch_size=BATCH, shuffle=True)

        # supervised
        sup = nn.Sequential(Encoder(), nn.Linear(512,10)).to(device)
        opt = optim.Adam(sup.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(20):
            for x,y in loader_noisy:
                x,y = x.to(device), y.to(device)
                out = sup(x)
                loss = loss_fn(out,y)

                opt.zero_grad()
                loss.backward()
                opt.step()

        correct=0
        total=0
        for x,y in test_loader:
            x,y = x.to(device), y.to(device)
            out = sup(x)
            _,pred = torch.max(out,1)
            total+=y.size(0)
            correct+=(pred==y).sum().item()

        print(f"Supervised Acc: {100*correct/total:.2f}%")

        # SSL
        classifier = LinearClassifier().to(device)
        train_linear(ssl_model, classifier,
                     DataLoader(train_clean, batch_size=BATCH, shuffle=True),
                     epochs=20)

        print("SSL:")
        evaluate(ssl_model, classifier, test_loader)