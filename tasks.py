"""Synthetic pretrain task A and a *shifted* downstream task B.

Both tasks are 4-class problems on x ~ N(0, I_16), labelled by a fixed random
"teacher" MLP. Task B's teacher equals task A's teacher plus a rank-2 update of its
first-layer weights -- i.e. the distribution shift itself is low-rank, which is
exactly the situation LoRA's low-rank hypothesis is built for.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

D_IN, D_HID_TEACHER, N_CLASSES = 16, 32, 4


def make_teachers(seed: int = 42, shift_rank: int = 2, shift_scale: float = 0.7) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1 / np.sqrt(D_IN), (D_IN, D_HID_TEACHER))
    W2 = rng.normal(0, 1 / np.sqrt(D_HID_TEACHER), (D_HID_TEACHER, N_CLASSES))
    U = rng.normal(0, 1, (D_IN, shift_rank))
    V = rng.normal(0, 1, (shift_rank, D_HID_TEACHER))
    delta = shift_scale * (U @ V) / np.sqrt(D_IN * shift_rank)
    return {"W1_A": W1, "W1_B": W1 + delta, "W2": W2, "delta_rank": np.array(shift_rank)}


def _label(X: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    return np.argmax(np.tanh(X @ W1) @ W2, axis=1)


def make_task(teachers: Dict[str, np.ndarray], which: str, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, D_IN))
    W1 = teachers["W1_A"] if which == "A" else teachers["W1_B"]
    return X, _label(X, W1, teachers["W2"])
