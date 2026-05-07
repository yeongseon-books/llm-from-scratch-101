import numpy as np
from common import TinyGPT, TinyGPTConfig

np.random.seed(0)
cfg = TinyGPTConfig(vocab_size=50, block_size=8, n_embd=16, n_head=2, n_layer=2)
model = TinyGPT(cfg)
idx = np.random.randint(0, 50, size=(2, 8))
print(model.forward(idx).shape)
