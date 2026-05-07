import numpy as np


def test_finetuning_updates_parameters() -> None:
    x = np.eye(3)
    y = np.array([0, 1, 2])
    w = np.zeros((3, 3))
    base = w.copy()
    for _ in range(10):
        logits = x @ w
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        grad = probs
        grad[np.arange(3), y] -= 1
        grad /= 3
        w -= 0.3 * (x.T @ grad)
    assert float(np.abs(w - base).sum()) > 0.0
