import numpy as np
from common import cross_entropy

np.random.seed(0)
x = np.eye(4)
y = np.array([0, 1, 2, 3])
w = np.zeros((4, 4))
lr = 0.5
for _ in range(30):
    logits = x @ w
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    grad = probs
    grad[np.arange(4), y] -= 1
    grad /= 4
    w -= lr * (x.T @ grad)
print(round(cross_entropy(x @ w, y), 4))
