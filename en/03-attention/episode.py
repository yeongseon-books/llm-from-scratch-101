import numpy as np
from common import scaled_dot_product_attention

np.random.seed(0)
q = np.random.randn(1, 2, 4, 8)
k = np.random.randn(1, 2, 4, 8)
v = np.random.randn(1, 2, 4, 8)
out, w = scaled_dot_product_attention(q, k, v)
print(out.shape, w.shape)
