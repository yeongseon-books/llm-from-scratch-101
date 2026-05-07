import numpy as np

from common import scaled_dot_product_attention


def test_attention_shape_and_row_sum() -> None:
    np.random.seed(0)
    q = np.random.randn(1, 2, 4, 8)
    k = np.random.randn(1, 2, 4, 8)
    v = np.random.randn(1, 2, 4, 8)
    out, w = scaled_dot_product_attention(q, k, v)
    assert out.shape == (1, 2, 4, 8)
    assert np.allclose(w.sum(axis=-1), 1.0)
