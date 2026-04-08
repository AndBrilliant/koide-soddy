"""Null-model sampler: generate Koide-satisfying triples and test for quark mass matches."""

import math
import numpy as np
from scipy import stats

from koide_soddy.leptons import F_from_soddy


def _solve_m2_for_koide(m1: float, m3: float) -> float | None:
    """Solve for m2 such that Q(m1, m2, m3) = 2/3 exactly.

    The Koide condition Q = (m1+m2+m3)/(sqrt(m1)+sqrt(m2)+sqrt(m3))^2 = 2/3
    is equivalent to: (sum_m) * 3 = 2 * (sum_sqrt)^2

    Let x = sqrt(m2). Then:
        sum_m = m1 + x^2 + m3
        sum_sqrt = sqrt(m1) + x + sqrt(m3)
        3*(m1 + x^2 + m3) = 2*(sqrt(m1) + x + sqrt(m3))^2

    Expanding RHS: 2*(m1 + x^2 + m3 + 2*sqrt(m1)*x + 2*sqrt(m3)*x + 2*sqrt(m1*m3))
    So: 3*(m1+x^2+m3) = 2*(m1+x^2+m3) + 4*x*(sqrt(m1)+sqrt(m3)) + 4*sqrt(m1*m3)
    => (m1+x^2+m3) = 4*x*(sqrt(m1)+sqrt(m3)) + 4*sqrt(m1*m3)
    => x^2 - 4*(sqrt(m1)+sqrt(m3))*x + (m1+m3-4*sqrt(m1*m3)) = 0

    Quadratic in x = sqrt(m2).
    """
    s1, s3 = math.sqrt(m1), math.sqrt(m3)

    a = 1.0
    b = -4.0 * (s1 + s3)
    c = m1 + m3 - 4.0 * s1 * s3

    disc = b * b - 4.0 * a * c
    if disc < 0:
        return None

    sqrt_disc = math.sqrt(disc)
    # Two roots for x = sqrt(m2)
    x_plus = (-b + sqrt_disc) / (2.0 * a)
    x_minus = (-b - sqrt_disc) / (2.0 * a)

    # Need x > 0 and m1 < m2 = x^2 < m3
    for x in [x_minus, x_plus]:
        if x <= 0:
            continue
        m2 = x * x
        if m1 < m2 < m3:
            return m2

    return None


def sample_koide_triples(n: int, seed: int = 42) -> np.ndarray:
    """Generate n valid Koide-satisfying triples (m1, m2, m3).

    Sampling:
      1. m3 log-uniform in [1, 10^4] MeV
      2. m1 log-uniform in [10^-3, 10^-1] * m3
      3. Solve for m2 via Koide condition
      4. Verify |Q - 2/3| < 10^-5

    Returns array of shape (n, 3) with columns [m1, m2, m3].
    """
    rng = np.random.default_rng(seed)
    triples = []

    while len(triples) < n:
        # Batch sampling for efficiency
        batch_size = min(n - len(triples), 10000) * 2  # oversample

        # m3 log-uniform in [1, 10^4]
        log_m3 = rng.uniform(math.log(1.0), math.log(1e4), batch_size)
        m3_batch = np.exp(log_m3)

        # m1 log-uniform in [10^-3, 10^-1] * m3
        log_ratio = rng.uniform(math.log(1e-3), math.log(1e-1), batch_size)
        m1_batch = np.exp(log_ratio) * m3_batch

        for m1, m3 in zip(m1_batch, m3_batch):
            if len(triples) >= n:
                break

            m2 = _solve_m2_for_koide(float(m1), float(m3))
            if m2 is None:
                continue

            # Verify Koide precision
            s = m1 + m2 + m3
            sr = math.sqrt(m1) + math.sqrt(m2) + math.sqrt(m3)
            Q = s / (sr * sr)
            if abs(Q - 2.0 / 3.0) < 1e-5:
                triples.append((m1, m2, m3))

    return np.array(triples[:n])


def compute_F_squared(triples: np.ndarray) -> np.ndarray:
    """Compute F^2 for each triple. Returns array of F^2 values in MeV."""
    F2 = np.empty(len(triples))
    for i, (m1, m2, m3) in enumerate(triples):
        F = F_from_soddy(float(m1), float(m2), float(m3))
        F2[i] = F * F
    return F2


def compute_natural_scale(triples: np.ndarray) -> np.ndarray:
    """Compute mu* = m1 + m2 + m3 for each triple."""
    return triples.sum(axis=1)


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper-Pearson) binomial confidence interval.

    Returns (lower, upper) bounds at the (1-alpha) confidence level.
    """
    if n == 0:
        return (0.0, 1.0)
    if k == 0:
        lo = 0.0
    else:
        lo = stats.beta.ppf(alpha / 2, k, n - k + 1)
    if k == n:
        hi = 1.0
    else:
        hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))
