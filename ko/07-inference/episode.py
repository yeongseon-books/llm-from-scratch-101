import numpy as np
from common import sample_topk

np.random.seed(0)
logits = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
print(sample_topk(logits, top_k=3, temperature=0.8))
