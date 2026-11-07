"""Maps, losses and metrics, checked against Eqs. (2)-(3) of the paper."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from canemb.eval import best_acc_eer, tar_at_far
from canemb.mapping.maps import apply_map, fit_linear, fit_rotation
from canemb.models.fr_net import ArcFace, CosFace


def random_rotation(d: int, seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.RandomState(seed).normal(size=(d, d)))
    return q if np.linalg.det(q) > 0 else q @ np.diag([-1.0] + [1.0] * (d - 1))


@pytest.fixture
def embeddings():
    rng = np.random.RandomState(0)
    src = rng.normal(size=(500, 16))
    return src / np.linalg.norm(src, axis=1, keepdims=True)


def test_linear_map_recovers_a_known_transform(embeddings):
    true_map = np.random.RandomState(1).normal(size=(8, 16))
    assert np.allclose(fit_linear(embeddings, embeddings @ true_map.T), true_map, atol=1e-8)


def test_rotation_map_recovers_a_known_rotation(embeddings):
    true_map = random_rotation(16, 2)
    assert np.allclose(fit_rotation(embeddings, embeddings @ true_map.T), true_map, atol=1e-8)


def test_rotation_map_follows_eq_3(embeddings):
    tgt = embeddings @ random_rotation(16, 3).T + 0.1 * np.random.RandomState(4).normal(size=embeddings.shape)
    u, _, vh = np.linalg.svd(embeddings.T @ tgt)
    sign = np.sign(np.linalg.det(u) * np.linalg.det(vh))
    expected = u @ np.diag([1.0] * 15 + [sign]) @ vh
    assert np.allclose(fit_rotation(embeddings, tgt).T, expected, atol=1e-8)


def test_rotation_map_is_a_proper_rotation_even_for_a_mirrored_target(embeddings):
    """Eq. (3): the last singular value is flipped so no mirroring occurs."""
    mirror = random_rotation(16, 5) @ np.diag([-1.0] + [1.0] * 15)
    m = fit_rotation(embeddings, embeddings @ mirror.T)
    assert np.allclose(m @ m.T, np.eye(16), atol=1e-8)
    assert np.linalg.det(m) == pytest.approx(1.0)


def test_rotation_maps_are_symmetric_when_swapping_source_and_target(embeddings):
    """Sec. 4: the penalty is nearly symmetric, from the orthonormal solution."""
    tgt = embeddings @ random_rotation(16, 6).T + 0.05 * np.random.RandomState(7).normal(size=embeddings.shape)
    assert np.allclose(fit_rotation(embeddings, tgt), fit_rotation(tgt, embeddings).T, atol=1e-8)


def test_linear_fit_is_at_least_as_good_as_rotation(embeddings):
    """Sec. 4: rotation adds a constraint, so its enrollment error cannot be lower."""
    tgt = embeddings @ np.random.RandomState(8).normal(size=(16, 16)).T
    linear = np.linalg.norm(embeddings @ fit_linear(embeddings, tgt).T - tgt)
    rotation = np.linalg.norm(embeddings @ fit_rotation(embeddings, tgt).T - tgt)
    assert linear <= rotation


def test_mapped_embeddings_are_normalised():
    out = apply_map(torch.randn(4, 16), np.random.RandomState(9).normal(size=(8, 16)))
    assert torch.allclose(out.norm(dim=1), torch.ones(4), atol=1e-5)


def test_cosface_subtracts_the_margin_on_the_target_class():
    torch.manual_seed(0)
    head = CosFace(8, 5, s=30.0, m=0.35)
    emb, labels = F.normalize(torch.randn(3, 8), dim=1), torch.tensor([0, 1, 2])
    cosine = emb @ F.normalize(head.weight, dim=1).t()
    expected = 30.0 * cosine
    expected[torch.arange(3), labels] -= 30.0 * 0.35
    assert torch.allclose(head(emb, labels), expected, atol=1e-4)


def test_arcface_adds_the_angular_margin_on_the_target_class():
    torch.manual_seed(1)
    head = ArcFace(8, 5, s=30.0, m=0.5)
    emb, labels = F.normalize(torch.randn(3, 8), dim=1), torch.tensor([0, 1, 2])
    cosine = emb @ F.normalize(head.weight, dim=1).t()
    expected = 30.0 * cosine
    target = cosine[torch.arange(3), labels].acos() + 0.5
    expected[torch.arange(3), labels] = 30.0 * target.cos()
    assert torch.allclose(head(emb, labels), expected, atol=1e-3)


def test_tar_at_far_matches_a_worked_example():
    scores = np.concatenate([np.linspace(0, 1, 1001), [0.5, 0.995, 2.0, 3.0]])
    labels = np.concatenate([np.zeros(1001, dtype=int), np.ones(4, dtype=int)])
    assert tar_at_far(scores, labels, 0.01) == pytest.approx(0.75)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def test_eer_is_the_point_where_fmr_equals_fnmr():
    lo, hi = 0.0, 1.0
    for _ in range(80):
        t = 0.5 * (lo + hi)
        fmr, fnmr = 1.0 - _norm_cdf(t), _norm_cdf((t - 1.0) / 0.1)
        lo, hi = (t, hi) if fmr > fnmr else (lo, t)
    expected = 1.0 - _norm_cdf(t)

    rng = np.random.RandomState(3)
    scores = np.concatenate([rng.normal(1.0, 0.1, 200000), rng.normal(0.0, 1.0, 200000)])
    labels = np.concatenate([np.ones(200000, dtype=int), np.zeros(200000, dtype=int)])
    assert best_acc_eer(scores, labels)[1] == pytest.approx(expected, abs=0.01)
