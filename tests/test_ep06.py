import numpy as np

from common import cross_entropy


def test_training_loss_decreases() -> None:
    x = np.eye(4)
    y = np.array([0, 1, 2, 3])
    w = np.zeros((4, 4))
    start = cross_entropy(x @ w, y)
    for _ in range(30):
        logits = x @ w
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        grad = probs
        grad[np.arange(4), y] -= 1
        grad /= 4
        w -= 0.5 * (x.T @ grad)
    end = cross_entropy(x @ w, y)
    assert end < start
