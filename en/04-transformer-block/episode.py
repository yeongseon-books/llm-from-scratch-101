import numpy as np
from common import TransformerBlock

np.random.seed(0)
x = np.random.randn(2, 6, 16)
blk = TransformerBlock(16, 2)
print(blk(x).shape)
