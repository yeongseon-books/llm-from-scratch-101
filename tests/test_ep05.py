import numpy as np

from common import TinyGPT, TinyGPTConfig


def test_tiny_gpt_forward_shape() -> None:
    cfg = TinyGPTConfig(vocab_size=30, block_size=6, n_embd=12, n_head=2, n_layer=1)
    model = TinyGPT(cfg)
    idx = np.random.randint(0, cfg.vocab_size, size=(3, 6))
    logits = model.forward(idx)
    assert logits.shape == (3, 6, cfg.vocab_size)
