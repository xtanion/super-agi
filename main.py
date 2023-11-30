import torch
from src.transformer import Transformer
from src.utils import (get_batch, get_data, encode, decode, estimate_loss, vocab_size)  # noqa

# Hyperparameters
batch_size = 64
block_size = 256

max_iters = 500  # to be tuned later
eval_interval = 100
learning_rate = 3e-4

device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
head_size = 16
dropout = 0.2


model = Transformer(vocab_size, n_embd, n_head, n_layer, block_size,
                    device)
m = model.to(device)  # running with CUDA
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# creating a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")  # noqa

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
