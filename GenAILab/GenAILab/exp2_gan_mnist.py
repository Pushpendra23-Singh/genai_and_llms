import torch, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

loader = DataLoader(datasets.MNIST('./data', train=True, download=True,
                    transform=transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize([0.5],[0.5])])), batch_size=128, shuffle=True)

G = nn.Sequential(nn.Linear(100,256), nn.ReLU(), nn.Linear(256,784), nn.Tanh())
D = nn.Sequential(nn.Linear(784,256), nn.LeakyReLU(0.2), nn.Linear(256,1), nn.Sigmoid())

opt_G = torch.optim.Adam(G.parameters(), lr=2e-4)
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4)
loss  = nn.BCELoss()

for epoch in range(10):
    for real, _ in loader:
        real = real.view(-1, 784)
        bs   = real.size(0)

        # Train Discriminator
        z     = torch.randn(bs, 100)
        fake  = G(z).detach()
        d_loss = loss(D(real), torch.ones(bs,1)) + loss(D(fake), torch.zeros(bs,1))
        opt_D.zero_grad(); d_loss.backward(); opt_D.step()

        # Train Generator
        z     = torch.randn(bs, 100)
        g_loss = loss(D(G(z)), torch.ones(bs,1))
        opt_G.zero_grad(); g_loss.backward(); opt_G.step()

    print(f"Epoch {epoch+1} | D: {d_loss.item():.3f} | G: {g_loss.item():.3f}")

print("Done! Generator now creates fake MNIST digits from random noise z ~ N(0,1)")
