import torch, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

loader = DataLoader(datasets.MNIST('./data', train=True, download=True,
                    transform=transforms.ToTensor()), batch_size=128, shuffle=True)

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(784, 400), nn.ReLU())
        self.mu  = nn.Linear(400, 20)
        self.var = nn.Linear(400, 20)
        self.dec = nn.Sequential(nn.Linear(20, 400), nn.ReLU(), nn.Linear(400, 784), nn.Sigmoid())

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.var(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)  # reparameterization
        return self.dec(z), mu, logvar

model = VAE()
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    for x, _ in loader:
        recon, mu, logvar = model(x)
        bce = nn.functional.binary_cross_entropy(recon, x.view(-1, 784), reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
        loss = bce + kld
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"Epoch {epoch+1} Loss: {loss.item():.1f}")

print("Done! VAE generates new MNIST digits by sampling z ~ N(0,1) → decoder")
