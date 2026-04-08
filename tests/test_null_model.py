"""Tests for the null-model sampler."""

import math
import numpy as np
from koide_soddy.null_model import sample_koide_triples, compute_F_squared, clopper_pearson_ci


def test_sampler_produces_valid_triples():
    """Sampler returns the requested number of triples."""
    triples = sample_koide_triples(100, seed=42)
    assert triples.shape == (100, 3)


def test_all_triples_satisfy_koide():
    """All sampled triples satisfy |Q - 2/3| < 1e-5."""
    triples = sample_koide_triples(1000, seed=42)
    for m1, m2, m3 in triples:
        s = m1 + m2 + m3
        sr = math.sqrt(m1) + math.sqrt(m2) + math.sqrt(m3)
        Q = s / (sr * sr)
        assert abs(Q - 2.0 / 3.0) < 1e-5, f"Q = {Q}, violates Koide"


def test_ordering_m1_lt_m2_lt_m3():
    """All triples have m1 < m2 < m3."""
    triples = sample_koide_triples(1000, seed=42)
    assert np.all(triples[:, 0] < triples[:, 1])
    assert np.all(triples[:, 1] < triples[:, 2])


def test_sampler_reproducible():
    """Same seed produces identical triples."""
    t1 = sample_koide_triples(100, seed=123)
    t2 = sample_koide_triples(100, seed=123)
    np.testing.assert_array_equal(t1, t2)


def test_F_squared_positive():
    """F^2 values are positive for all sampled triples."""
    triples = sample_koide_triples(100, seed=42)
    F2 = compute_F_squared(triples)
    assert np.all(F2 > 0)


def test_clopper_pearson_basic():
    """Clopper-Pearson CI contains the observed proportion."""
    lo, hi = clopper_pearson_ci(50, 1000, alpha=0.05)
    assert lo < 0.05 < hi
    # Edge cases
    lo0, hi0 = clopper_pearson_ci(0, 100)
    assert lo0 == 0.0
    lo_all, hi_all = clopper_pearson_ci(100, 100)
    assert hi_all == 1.0
