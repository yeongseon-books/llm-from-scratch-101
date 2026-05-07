import numpy as np
from common import Embedding, positional_encoding

np.random.seed(0)
emb = Embedding(50, 16)
idx = np.array([[1, 2, 3]])
out = emb(idx) + positional_encoding(3, 16)[None, :, :]
print(out.shape)
