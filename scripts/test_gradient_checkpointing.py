import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from functools import partial

torch.manual_seed(0)

# ----------------------------
# Tiny Transformer-ish blocks
# ----------------------------
class ToyBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, idx: int):
        super().__init__()
        self.idx = idx
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        # Print so we can count how many times this block runs per training step
        print(f"[FWD] ToyBlock {self.idx}")

        # Self-attention
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out

        # MLP
        h = self.ln2(x)
        x = x + self.mlp(h)
        return x


class ToyTransformer(nn.Module):
    def __init__(self, vocab_size=128, d_model=64, n_heads=4, n_layers=2, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 256, d_model))  # max seq len 256 (overkill for the demo)
        self.blocks = nn.ModuleList([ToyBlock(d_model, n_heads, idx=i) for i in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def _run_block(self, block, x):
        # Helper so we can pass a single-Tensor callable to checkpoint()
        return block(x)

    def forward(self, idx):  # idx: (B, T) int64
        x = self.emb(idx)
        x = x + self.pos[:, : x.size(1)]
        for block in self.blocks:
            if self.use_checkpoint:
                # checkpoint() requires a function that takes only Tensor args
                x = checkpoint(partial(self._run_block, block), x)
            else:
                x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab)
        return logits


# ----------------------------
# Tiny training loop
# ----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Toggle here to compare behavior
    USE_CHECKPOINT = True  # set False to see single forward per block per step

    model = ToyTransformer(use_checkpoint=USE_CHECKPOINT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    B, T, vocab = 4, 16, 128
    steps = 2

    for step in range(steps):
        print(f"\n===== TRAIN STEP {step} (checkpointing={USE_CHECKPOINT}) =====")
        # Dummy inputs/targets
        x = torch.randint(0, vocab, (B, T), device=device)
        y = torch.randint(0, vocab, (B, T), device=device)

        opt.zero_grad(set_to_none=True)
        logits = model(x)             # <-- prints from each block forward
        loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1))
        loss.backward()               # <-- with checkpointing, block forwards run again here
        opt.step()

        print(f"loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()
