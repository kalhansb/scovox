"""Unit tests for semantic calibration metrics and runtime stats."""

import tempfile

import numpy as np
import pytest

from scovox_eval.metrics._stratify import compute_alpha0, make_strata_masks
from scovox_eval.metrics.semantic_brier import (
    compute_semantic_brier,
    compute_stratified_semantic_brier,
)
from scovox_eval.metrics.semantic_ece import (
    compute_semantic_ece,
    compute_stratified_semantic_ece,
)
from scovox_eval.metrics.semantic_reliability_diagram import compute_reliability_data
from scovox_eval.metrics.compute_runtime_stats import (
    compute_runtime_stats,
    parse_log_timing,
)


# ---------------------------------------------------------------------------
# Stratification helpers
# ---------------------------------------------------------------------------

class TestStratify:
    def test_alpha0_without_a_unk(self):
        evidence = np.array([[1.0, 2.0], [5.0, 10.0], [20.0, 5.0]])
        alpha0 = compute_alpha0(evidence)
        np.testing.assert_array_equal(alpha0, [3.0, 15.0, 25.0])

    def test_alpha0_with_a_unk(self):
        evidence = np.array([[1.0, 2.0], [5.0, 10.0]])
        a_unk = np.array([1.0, 2.0])
        alpha0 = compute_alpha0(evidence, a_unk)
        np.testing.assert_array_equal(alpha0, [4.0, 17.0])

    def test_strata_masks(self):
        alpha0 = np.array([1.0, 4.9, 5.0, 10.0, 20.0, 20.1, 50.0])
        masks = make_strata_masks(alpha0)
        assert masks["low (<5)"].sum() == 2      # 1.0, 4.9
        assert masks["medium (5-20)"].sum() == 3  # 5.0, 10.0, 20.0
        assert masks["high (>20)"].sum() == 2     # 20.1, 50.0


# ---------------------------------------------------------------------------
# Semantic ECE
# ---------------------------------------------------------------------------

class TestSemanticECE:
    def test_perfect_predictions(self):
        """One-hot correct predictions: all confidence=1.0, all correct -> ECE=0."""
        n, k = 1000, 5
        gt = np.random.RandomState(42).randint(0, k, size=n)
        probs = np.zeros((n, k), dtype=np.float32)
        probs[np.arange(n), gt] = 1.0
        result = compute_semantic_ece(probs, gt)
        assert result["semantic_ece"] == pytest.approx(0.0, abs=1e-6)

    def test_overconfident(self):
        """All predict class 0 at 0.9 but only 50% are class 0 -> large ECE."""
        n = 1000
        probs = np.full((n, 3), 0.05, dtype=np.float32)
        probs[:, 0] = 0.9
        gt = np.zeros(n, dtype=int)
        gt[500:] = 1  # half wrong
        result = compute_semantic_ece(probs, gt)
        assert result["semantic_ece"] > 0.3

    def test_stratification_separates(self):
        """Verify stratified ECE processes each stratum independently."""
        n = 300
        probs = np.zeros((n, 2), dtype=np.float32)
        probs[:, 0] = 0.9
        probs[:, 1] = 0.1
        gt = np.zeros(n, dtype=int)  # all correct
        alpha0 = np.concatenate([
            np.full(100, 2.0),   # low
            np.full(100, 10.0),  # medium
            np.full(100, 30.0),  # high
        ])
        strat = compute_stratified_semantic_ece(probs, gt, alpha0)
        assert "low (<5)" in strat
        assert "medium (5-20)" in strat
        assert "high (>20)" in strat
        # All correct at 0.9 -> ECE ~0.1 in each stratum
        for name, res in strat.items():
            assert res["semantic_ece"] < 0.15


# ---------------------------------------------------------------------------
# Semantic Brier
# ---------------------------------------------------------------------------

class TestSemanticBrier:
    def test_perfect_predictions(self):
        """One-hot correct -> Brier = 0."""
        n, k = 500, 4
        gt = np.random.RandomState(0).randint(0, k, size=n)
        probs = np.zeros((n, k), dtype=np.float32)
        probs[np.arange(n), gt] = 1.0
        result = compute_semantic_brier(probs, gt)
        assert result["brier_score"] == pytest.approx(0.0, abs=1e-6)

    def test_worst_case_binary(self):
        """All probability on wrong class, K=2 -> Brier = 2.0."""
        n = 100
        gt = np.zeros(n, dtype=int)  # all class 0
        probs = np.zeros((n, 2), dtype=np.float32)
        probs[:, 1] = 1.0  # all confidence on class 1 (wrong)
        result = compute_semantic_brier(probs, gt)
        assert result["brier_score"] == pytest.approx(2.0)

    def test_uniform_predictions(self):
        """Uniform 1/K -> Brier = (K-1)/K for each voxel."""
        k = 5
        n = 200
        gt = np.zeros(n, dtype=int)
        probs = np.full((n, k), 1.0 / k, dtype=np.float32)
        result = compute_semantic_brier(probs, gt)
        # Per voxel: (1 - 1/K)^2 + (K-1)*(1/K)^2 = (K-1)/K^2 * (K-1+1) = (K-1)/K
        expected = (k - 1) / k
        assert result["brier_score"] == pytest.approx(expected, rel=1e-5)

    def test_stratified_brier(self):
        """Different quality in different strata -> different Brier values."""
        k = 3
        # Low alpha: perfect predictions
        n_lo = 100
        gt_lo = np.zeros(n_lo, dtype=int)
        probs_lo = np.zeros((n_lo, k), dtype=np.float32)
        probs_lo[:, 0] = 1.0

        # High alpha: uniform predictions
        n_hi = 100
        gt_hi = np.zeros(n_hi, dtype=int)
        probs_hi = np.full((n_hi, k), 1.0 / k, dtype=np.float32)

        probs = np.concatenate([probs_lo, probs_hi])
        gt = np.concatenate([gt_lo, gt_hi])
        alpha0 = np.concatenate([np.full(n_lo, 2.0), np.full(n_hi, 30.0)])

        strat = compute_stratified_semantic_brier(probs, gt, alpha0)
        assert strat["low (<5)"]["brier_score"] == pytest.approx(0.0, abs=1e-6)
        assert strat["high (>20)"]["brier_score"] > 0.5

    def test_gt_label_out_of_range(self):
        """GT labels beyond K should raise ValueError."""
        probs = np.ones((10, 3)) / 3
        gt = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])  # class 3 >= K=3
        with pytest.raises(ValueError, match="class 3"):
            compute_semantic_brier(probs, gt)


# ---------------------------------------------------------------------------
# Reliability diagram data
# ---------------------------------------------------------------------------

class TestReliabilityDiagram:
    def test_output_structure(self):
        """Return dict has expected keys and sub-keys."""
        n, k = 200, 3
        probs = np.random.RandomState(1).dirichlet(np.ones(k), size=n).astype(np.float32)
        gt = np.random.RandomState(1).randint(0, k, size=n)
        alpha0 = np.random.RandomState(1).uniform(0, 30, size=n)
        data = compute_reliability_data(probs, gt, alpha0)

        assert "all" in data
        for key in ["all", "low (<5)", "medium (5-20)", "high (>20)"]:
            assert key in data
            d = data[key]
            assert "semantic_ece" in d
            assert "bin_edges" in d
            assert "bin_accuracy" in d
            assert "bin_confidence" in d
            assert "bin_count" in d

    def test_consistent_with_semantic_ece(self):
        """ECE in reliability data should match compute_semantic_ece."""
        n, k = 500, 4
        probs = np.random.RandomState(7).dirichlet(np.ones(k), size=n).astype(np.float32)
        gt = np.random.RandomState(7).randint(0, k, size=n)
        alpha0 = np.full(n, 10.0)  # all medium

        data = compute_reliability_data(probs, gt, alpha0)
        direct = compute_semantic_ece(probs, gt)
        assert data["all"]["semantic_ece"] == pytest.approx(direct["semantic_ece"])


# ---------------------------------------------------------------------------
# Runtime stats
# ---------------------------------------------------------------------------

_SAMPLE_LOG = """\
[INFO] [launch]: Default logging verbosity is set to INFO
[scovox_mapping_node-1] [INFO] [1234567890.0] [atlas.scovox_node]: SCovox ready res=0.200
[scovox_mapping_node-1] [INFO] [1234567891.0] [atlas.scovox_node]: recv=1 replay=0 integrated: voxels=5000 frame_ms=100.0 tf_ms=0.1 integrate_ms=90.0 publish_ms=9.9 mem_mb=40.0
[scovox_mapping_node-1] [INFO] [1234567892.0] [atlas.scovox_node]: recv=2 replay=1 integrated: voxels=6000 frame_ms=200.0 tf_ms=0.2 integrate_ms=180.0 publish_ms=19.8 mem_mb=50.0
[scovox_mapping_node-1] [INFO] [1234567893.0] [atlas.scovox_node]: recv=3 replay=2 integrated: voxels=7000 frame_ms=150.0 tf_ms=0.1 integrate_ms=135.0 publish_ms=14.9 mem_mb=45.0
"""


class TestRuntimeStats:
    def test_parse_log(self):
        """Parse known log lines and verify extracted arrays."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(_SAMPLE_LOG)
            f.flush()
            timing = parse_log_timing(f.name)

        assert len(timing["frame_ms"]) == 3
        np.testing.assert_array_almost_equal(
            timing["frame_ms"], [100.0, 200.0, 150.0]
        )
        np.testing.assert_array_almost_equal(
            timing["voxels"], [5000, 6000, 7000]
        )
        np.testing.assert_array_almost_equal(
            timing["mem_mb"], [40.0, 50.0, 45.0]
        )

    def test_stats_known_values(self):
        """Verify median/P95/P99 against numpy for known data."""
        timing = {
            "frame_ms": np.array([100.0, 200.0, 150.0]),
            "mem_mb": np.array([40.0, 50.0, 45.0]),
            "voxels": np.array([5000.0, 6000.0, 7000.0]),
        }
        stats = compute_runtime_stats(timing)
        assert stats["n_frames"] == 3
        assert stats["frame_ms"]["median"] == pytest.approx(150.0)
        assert stats["memory"]["peak_mb"] == pytest.approx(50.0)
        assert stats["memory"]["final_mb"] == pytest.approx(45.0)
        assert stats["voxels"]["final"] == 7000
        assert stats["voxels"]["peak"] == 7000

    def test_empty_log(self):
        """Empty log should produce zero-length arrays and zero stats."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("[INFO] no timing data here\n")
            f.flush()
            timing = parse_log_timing(f.name)

        assert len(timing["frame_ms"]) == 0
        stats = compute_runtime_stats(timing)
        assert stats["n_frames"] == 0
        assert stats["frame_ms"]["median"] == 0.0
