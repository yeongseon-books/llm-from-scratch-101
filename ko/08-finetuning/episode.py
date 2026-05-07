import numpy as np

np.random.seed(0)
x = np.eye(3)
y = np.array([0, 1, 2])
w = np.zeros((3, 3))
base = w.copy()
for _ in range(20):
    logits = x @ w
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    grad = probs
    grad[np.arange(3), y] -= 1
    grad /= 3
    w -= 0.3 * (x.T @ grad)
print(float(np.abs(w - base).sum()))
