"""Unit tests for Beta EIG computation — verifies against known reference values."""

import numpy as np
import pytest

from scovox_eval.metrics.eig import beta_eig, beta_variance, logodds_entropy


def _try_scipy():
    try:
        from scipy.special import digamma  # noqa: F401
        return True
    except ImportError:
        return False


class TestBetaEIG:
    @pytest.mark.skipif(not _try_scipy(), reason="scipy not installed")
    def test_uniform_prior_max_eig(self):
        """Beta(1,1) should have maximum EIG ~ 0.6137."""
        eig = beta_eig(np.array([1.0]), np.array([1.0]))
        assert eig[0] == pytest.approx(0.6137, abs=0.01)

    @pytest.mark.skipif(not _try_scipy(), reason="scipy not installed")
    def test_high_evidence_low_eig(self):
        """Beta(200,2) should have near-zero EIG."""
        eig = beta_eig(np.array([200.0]), np.array([2.0]))
        assert eig[0] < 0.001

    @pytest.mark.skipif(not _try_scipy(), reason="scipy not installed")
    def test_eig_decreases_with_evidence(self):
        """EIG should decrease as evidence increases at p=0.5."""
        a = np.array([1.0, 5.0, 50.0, 100.0])
        b = a.copy()
        eig = beta_eig(a, b)
        assert np.all(np.diff(eig) < 0)

    @pytest.mark.skipif(not _try_scipy(), reason="scipy not installed")
    def test_vectorised(self):
        a = np.array([1.0, 10.0, 100.0])
        b = np.array([1.0, 10.0, 100.0])
        eig = beta_eig(a, b)
        assert eig.shape == (3,)
        assert np.all(eig >= 0)


class TestBetaVariance:
    def test_uniform_prior(self):
        """Beta(1,1) variance = 1/12."""
        var = beta_variance(np.array([1.0]), np.array([1.0]))
        assert var[0] == pytest.approx(1 / 12, abs=1e-6)

    def test_symmetric_high_evidence(self):
        """Beta(100,100) variance << Beta(1,1) variance."""
        var_low = beta_variance(np.array([1.0]), np.array([1.0]))
        var_high = beta_variance(np.array([100.0]), np.array([100.0]))
        assert var_high[0] < var_low[0] / 10


class TestLogOddsEntropy:
    def test_max_entropy_at_half(self):
        """Entropy peaks at p=0.5."""
        ent = logodds_entropy(np.array([0.5]))
        assert ent[0] == pytest.approx(np.log(2), abs=1e-6)

    def test_low_entropy_near_extremes(self):
        ent = logodds_entropy(np.array([0.01, 0.99]))
        assert np.all(ent < 0.1)
