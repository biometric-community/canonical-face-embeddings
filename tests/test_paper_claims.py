"""The paper's central claim, reproduced with this repo's maps on synthetic embeddings.

Abstract: embeddings of different CNNs are related by a linear or rotation map, so
features of one CNN can be compared with those of another.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from canemb.eval import tar_at_far
from canemb.mapping.maps import apply_map, fit_linear, fit_rotation

DIM, IDS = 64, 300


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def world():
    """Two CNNs that see the same identity space through different rotations."""
    rng = np.random.RandomState(0)
    rotation, _ = np.linalg.qr(rng.normal(size=(DIM, DIM)))
    ids = rng.normal(size=(IDS, DIM))

    def cnn_a(x):
        return unit(x + 0.3 * rng.normal(size=x.shape))

    def cnn_b(x):
        return unit((x + 0.3 * rng.normal(size=x.shape)) @ rotation.T)

    return ids, cnn_a, cnn_b, rng


def cross_tar(world, fit) -> float:
    ids, cnn_a, cnn_b, rng = world
    enroll = rng.normal(size=(2000, DIM))
    mapping = fit(cnn_a(enroll), cnn_b(enroll))
    same = (ids, ids)
    different = (ids, np.roll(ids, 1, axis=0))
    scores, labels = [], []
    for (left, right), label in ((same, 1), (different, 0)):
        a = apply_map(torch.tensor(cnn_a(left)), mapping).numpy()
        scores.append((a * cnn_b(right)).sum(axis=1))
        labels.append(np.full(len(left), label))
    return tar_at_far(np.concatenate(scores), np.concatenate(labels), 0.01)


def test_linear_map_supports_cross_cnn_verification(world):
    assert cross_tar(world, fit_linear) >= 0.9


def test_rotation_map_supports_cross_cnn_verification(world):
    assert cross_tar(world, fit_rotation) >= 0.9


def test_unmapped_embeddings_cannot_be_compared(world):
    ids, cnn_a, cnn_b, _ = world
    same = (cnn_a(ids) * cnn_b(ids)).sum(axis=1)
    different = (cnn_a(ids) * cnn_b(np.roll(ids, 1, axis=0))).sum(axis=1)
    labels = np.concatenate([np.ones(IDS, dtype=int), np.zeros(IDS, dtype=int)])
    assert tar_at_far(np.concatenate([same, different]), labels, 0.01) <= 0.2
