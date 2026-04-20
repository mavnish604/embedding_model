import torch

from model import MiniTransformer
from model import mean_pooling


model = MiniTransformer(vocab_size=30522)

input_ids = torch.randint(0, 30522, (2, 128))
mask = torch.ones_like(input_ids)

out = model(input_ids, mask)

print(out.shape)

emb = mean_pooling(out, mask)
print(emb.shape)
