import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA (SimCLR Augmentation)
class SimCLRTransform:
    def __init__(self, size=32):
        self.transform = T.Compose([
            T.RandomResizedCrop(size=size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=5),
            T.ToTensor()
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


def get_train_loader(batch_size=128):
    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=SimCLRTransform(size=32)
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2
    )

    print(f"Total samples: {len(dataset)}")    
    print(f"Total batches: {len(loader)}")

    return loader


# MODEL
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        resnet.maxpool = nn.Identity()

        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        return torch.flatten(self.encoder(x), start_dim=1)


class ProjectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        return self.mlp(x)


class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.projector = ProjectionHead()

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z


# LOSS
def nt_xent_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.size(0)

    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    z = torch.cat([z_i, z_j], dim=0)

    sim_matrix = torch.matmul(z, z.T)
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)

    sim_matrix = sim_matrix / temperature
    sim_matrix = sim_matrix.masked_fill(mask, -1e4)

    positives = torch.cat([
        torch.diag(sim_matrix, batch_size),
        torch.diag(sim_matrix, -batch_size)
    ])

    exp_sim = torch.exp(sim_matrix)
    loss = -torch.log(torch.exp(positives) / exp_sim.sum(dim=1))

    return loss.mean()


# TRAIN SIMCLR
def train_simclr(model, loader, epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()
    accum_steps = 4

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        optimizer.zero_grad()

        h_sim_total, z_sim_total, count = 0, 0, 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

        for i, ((x_i, x_j), _) in enumerate(pbar):

            x_i, x_j = x_i.to(device), x_j.to(device)

            with torch.cuda.amp.autocast():
                h_i, z_i = model(x_i)
                h_j, z_j = model(x_j)

                loss = nt_xent_loss(z_i, z_j) / accum_steps

            # similarity
            h_sim = torch.sum(F.normalize(h_i, dim=1) * F.normalize(h_j, dim=1), dim=1).mean()
            z_sim = torch.sum(F.normalize(z_i, dim=1) * F.normalize(z_j, dim=1), dim=1).mean()

            h_sim_total += h_sim.item()
            z_sim_total += z_sim.item()
            count += 1

            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps

        print(f"\nEpoch {epoch+1}")
        print(f"Loss: {total_loss/len(loader):.4f}")
        print(f"Cosine(h_i, h_j): {h_sim_total/count:.4f}")
        print(f"Cosine(z_i, z_j): {z_sim_total/count:.4f}")

    return model


# MAIN
if __name__ == "__main__":

    print("Starting SimCLR training on CIFAR-10...")

    train_loader = get_train_loader()
    model = SimCLR().to(device)

    model = train_simclr(model, train_loader, epochs=50)

    torch.save(model.state_dict(), "simclr_model.pth")

    print(" Training complete and model saved!")