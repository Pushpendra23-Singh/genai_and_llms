import torch, torch.nn as nn, math

class MultiHeadAttention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dk = h, d//h
        self.qkv = nn.Linear(d, 3*d)
        self.out  = nn.Linear(d, d)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q,k,v = [t.view(B,T,self.h,self.dk).transpose(1,2) for t in (q,k,v)]
        scores = q @ k.transpose(-2,-1) / math.sqrt(self.dk)
        if mask is not None: scores = scores.masked_fill(mask==0, -1e9)
        return self.out((scores.softmax(-1) @ v).transpose(1,2).reshape(B,T,-1))

class TransformerBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.attn = MultiHeadAttention(d, h)
        self.ff   = nn.Sequential(nn.Linear(d,4*d), nn.ReLU(), nn.Linear(4*d,d))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x, mask=None):
        x = self.n1(x + self.attn(x, mask))
        return self.n2(x + self.ff(x))

class MiniTransformer(nn.Module):
    def __init__(self, vocab, d=64, h=4, layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.Sequential(*[TransformerBlock(d,h) for _ in range(layers)])
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        return self.head(self.blocks(self.emb(x)))

# Demo: train on tiny toy sequence
model = MiniTransformer(vocab=10)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    x = torch.randint(0, 9, (4, 8))     # batch=4, seq_len=8
    y = x + 1                            # predict next token (toy task)
    loss = nn.functional.cross_entropy(model(x).view(-1,10), y.view(-1))
    opt.zero_grad(); loss.backward(); opt.step()

print(f"Final loss: {loss.item():.4f}")
print("Transformer built from scratch — Attention → Add&Norm → FFN")
