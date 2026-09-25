"""LoRA from scratch in NumPy: a tiny MLP, frozen base + low-rank adapters, Adam, merging.

Model: x -> Linear(16, 64) -> ReLU -> Linear(64, 4) -> softmax.

LoRA on a linear layer y = x W + b (W frozen, shape d_in x d_out):
    y = x W + b + (alpha / r) * (x A) B,   A: d_in x r (small random),  B: r x d_out (zeros)
Only A and B are trained; B = 0 at init so the adapted model starts exactly at the base.
Merging: W_merged = W + (alpha / r) * A @ B -> zero inference overhead.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def init_mlp(d_in: int, d_hid: int, d_out: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    return {
        "W1": rng.normal(0, np.sqrt(2 / d_in), (d_in, d_hid)), "b1": np.zeros(d_hid),
        "W2": rng.normal(0, np.sqrt(2 / d_hid), (d_hid, d_out)), "b2": np.zeros(d_out),
    }


def init_lora(base: Dict[str, np.ndarray], r: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    ad = {}
    for layer in ("1", "2"):
        d_in, d_out = base["W" + layer].shape
        ad["A" + layer] = rng.normal(0, 1 / np.sqrt(d_in), (d_in, r))
        ad["B" + layer] = np.zeros((r, d_out))
    return ad


def _eff(base, lora, layer, scale):
    W = base["W" + layer]
    if lora is None:
        return W
    return W + scale * lora["A" + layer] @ lora["B" + layer]


def forward(base, X, lora: Optional[Dict[str, np.ndarray]] = None, scale: float = 1.0):
    """Forward pass; the LoRA path is computed as (x A) B, never materialising A @ B."""
    h_pre = X @ base["W1"] + base["b1"]
    if lora is not None:
        h_pre = h_pre + scale * (X @ lora["A1"]) @ lora["B1"]
    h = np.maximum(h_pre, 0)
    logits = h @ base["W2"] + base["b2"]
    if lora is not None:
        logits = logits + scale * (h @ lora["A2"]) @ lora["B2"]
    return logits, (X, h_pre, h)


def softmax_xent(logits: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    n = len(y)
    loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
    g = p.copy()
    g[np.arange(n), y] -= 1
    return float(loss), g / n


def backward(base, cache, dlogits, lora=None, scale=1.0, train_base=True):
    X, h_pre, h = cache
    grads = {}
    W2e = _eff(base, lora, "2", scale)
    dh = dlogits @ W2e.T
    dh_pre = dh * (h_pre > 0)
    if train_base:
        grads.update(W2=h.T @ dlogits, b2=dlogits.sum(0), W1=X.T @ dh_pre, b1=dh_pre.sum(0))
    if lora is not None:
        # dL/dB = scale * (hA)^T dY ; dL/dA = scale * h^T (dY B^T)
        grads["B2"] = scale * (h @ lora["A2"]).T @ dlogits
        grads["A2"] = scale * h.T @ (dlogits @ lora["B2"].T)
        grads["B1"] = scale * (X @ lora["A1"]).T @ dh_pre
        grads["A1"] = scale * X.T @ (dh_pre @ lora["B1"].T)
    return grads


class Adam:
    def __init__(self, params: Dict[str, np.ndarray], lr: float = 1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps, self.t = params, lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads: Dict[str, np.ndarray]) -> None:
        self.t += 1
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def accuracy(base, X, y, lora=None, scale=1.0) -> float:
    return float((forward(base, X, lora, scale)[0].argmax(1) == y).mean())


def train(base, X, y, *, epochs: int, lr: float, batch: int, rng: np.random.Generator,
          lora=None, scale=1.0, train_base=True, X_val=None, y_val=None) -> List[Dict[str, float]]:
    """Mini-batch Adam. If lora is given and train_base=False, only A/B are updated (base frozen)."""
    params = base if train_base else lora
    opt = Adam(params, lr=lr)
    hist = []
    n = len(y)
    for ep in range(epochs):
        idx = rng.permutation(n)
        tot = 0.0
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            logits, cache = forward(base, X[b], lora, scale)
            loss, g = softmax_xent(logits, y[b])
            grads = backward(base, cache, g, lora, scale, train_base)
            opt.step({k: v for k, v in grads.items() if k in params})
            tot += loss * len(b)
        rec = {"epoch": ep + 1, "train_loss": round(tot / n, 5)}
        if X_val is not None:
            rec["val_acc"] = round(accuracy(base, X_val, y_val, lora, scale), 4)
        hist.append(rec)
    return hist


def merge_lora(base, lora, scale) -> Dict[str, np.ndarray]:
    """Fold adapters into the weights: W' = W + scale * A @ B (biases unchanged)."""
    merged = {k: v.copy() for k, v in base.items()}
    for layer in ("1", "2"):
        merged["W" + layer] = base["W" + layer] + scale * lora["A" + layer] @ lora["B" + layer]
    return merged


def n_params(d: Dict[str, np.ndarray]) -> int:
    return int(sum(v.size for v in d.values()))
