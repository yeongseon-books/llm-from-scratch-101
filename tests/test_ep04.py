import numpy as np

from common import TransformerBlock


def test_transformer_block_preserves_shape() -> None:
    x = np.random.randn(2, 5, 16)
    y = TransformerBlock(16, 2)(x)
    assert y.shape == x.shape
