from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

np.random.seed(0)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    probs = softmax(logits, axis=-1)
    idx = np.arange(targets.shape[0])
    picked = probs[idx, targets]
    return float(-np.mean(np.log(np.clip(picked, 1e-12, 1.0))))


class CharTokenizer:
    def __init__(self, corpus: str) -> None:
        chars = sorted(set(corpus))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.itos[i] for i in token_ids)


class Embedding:
    def __init__(self, vocab_size: int, emb_dim: int) -> None:
        self.weight = np.random.randn(vocab_size, emb_dim) * 0.02

    def __call__(self, idx: np.ndarray) -> np.ndarray:
        return self.weight[idx]


def positional_encoding(seq_len: int, emb_dim: int) -> np.ndarray:
    pos = np.arange(seq_len)[:, None]
    div = np.exp(np.arange(0, emb_dim, 2) * (-math.log(10000.0) / emb_dim))
    pe = np.zeros((seq_len, emb_dim))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe


def scaled_dot_product_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    d = q.shape[-1]
    scores = (q @ np.swapaxes(k, -1, -2)) / math.sqrt(d)
    if causal:
        t = q.shape[-2]
        mask = np.triu(np.ones((t, t), dtype=bool), k=1)
        scores = np.where(mask, -1e9, scores)
    weights = softmax(scores, axis=-1)
    out = weights @ v
    return out, weights


class MultiHeadAttention:
    def __init__(self, n_embd: int, n_head: int) -> None:
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.wq = np.random.randn(n_embd, n_embd) * 0.02
        self.wk = np.random.randn(n_embd, n_embd) * 0.02
        self.wv = np.random.randn(n_embd, n_embd) * 0.02
        self.wo = np.random.randn(n_embd, n_embd) * 0.02

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        b, t, _ = x.shape
        q = (
            (x @ self.wq)
            .reshape(b, t, self.n_head, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            (x @ self.wk)
            .reshape(b, t, self.n_head, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            (x @ self.wv)
            .reshape(b, t, self.n_head, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        out, weights = scaled_dot_product_attention(q, k, v, causal=True)
        merged = out.transpose(0, 2, 1, 3).reshape(b, t, self.n_embd)
        return merged @ self.wo, weights


def _layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


class TransformerBlock:
    def __init__(self, n_embd: int, n_head: int, mlp_ratio: int = 2) -> None:
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.w1 = np.random.randn(n_embd, n_embd * mlp_ratio) * 0.02
        self.w2 = np.random.randn(n_embd * mlp_ratio, n_embd) * 0.02

    def __call__(self, x: np.ndarray) -> np.ndarray:
        a, _ = self.attn(_layer_norm(x))
        x = x + a
        h = np.tanh(_layer_norm(x) @ self.w1)
        x = x + (h @ self.w2)
        return x


@dataclass
class TinyGPTConfig:
    vocab_size: int = 50
    block_size: int = 8
    n_embd: int = 16
    n_head: int = 2
    n_layer: int = 2


class TinyGPT:
    def __init__(self, config: TinyGPTConfig) -> None:
        self.config = config
        self.tok_emb = Embedding(config.vocab_size, config.n_embd)
        self.blocks = [
            TransformerBlock(config.n_embd, config.n_head)
            for _ in range(config.n_layer)
        ]
        self.pos = positional_encoding(config.block_size, config.n_embd)
        self.lm_head = np.random.randn(config.n_embd, config.vocab_size) * 0.02

    def forward(self, idx: np.ndarray) -> np.ndarray:
        b, t = idx.shape
        x = self.tok_emb(idx) + self.pos[:t][None, :, :]
        for block in self.blocks:
            x = block(x)
        return x @ self.lm_head


def sample_topk(logits: np.ndarray, top_k: int = 5, temperature: float = 1.0) -> int:
    scaled = logits / max(temperature, 1e-8)
    top_idx = np.argsort(scaled)[-top_k:]
    top_logits = scaled[top_idx]
    probs = softmax(top_logits)
    return int(np.random.choice(top_idx, p=probs))
