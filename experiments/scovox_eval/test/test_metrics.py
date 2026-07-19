"""Unit tests for evaluation metrics — known-answer synthetic cases."""

import numpy as np
import pytest

from scovox_eval.metrics.ece import compute_ece
from scovox_eval.metrics.miou import compute_miou


class TestECE:
    def test_perfect_calibration(self):
        """Perfectly calibrated predictions have ECE = 0."""
        np.random.seed(42)
        n = 10_000
        prob = np.random.rand(n).astype(np.float32)
        gt = (np.random.rand(n) < prob).astype(float)
        result = compute_ece(prob, gt, n_bins=20)
        assert result["ece"] < 0.05  # statistical noise

    def test_constant_overconfident(self):
        """All predictions at 0.9 but only 50% actually occupied -> large ECE."""
        n = 1000
        prob = np.full(n, 0.9)
        gt = np.zeros(n)
        gt[:500] = 1.0
        result = compute_ece(prob, gt, n_bins=20)
        assert result["ece"] > 0.3

    def test_brier_perfect(self):
        """Perfect binary predictions have Brier = 0."""
        prob = np.array([1.0, 0.0, 1.0, 0.0])
        gt = np.array([1.0, 0.0, 1.0, 0.0])
        result = compute_ece(prob, gt)
        assert result["brier"] == pytest.approx(0.0)


class TestMIoU:
    def test_perfect_agreement(self):
        labels = np.array([0, 1, 2, 0, 1, 2])
        result = compute_miou(labels, labels)
        assert result["miou"] == pytest.approx(1.0)

    def test_no_overlap(self):
        pred = np.array([0, 0, 0])
        gt = np.array([1, 1, 1])
        result = compute_miou(pred, gt, num_classes=2)
        assert result["miou"] == pytest.approx(0.0)

    def test_ignore_label(self):
        pred = np.array([0, 1, 0, 1])
        gt = np.array([0, 1, 255, 255])
        result = compute_miou(pred, gt, ignore_label=255)
        assert result["miou"] == pytest.approx(1.0)

    def test_partial_overlap(self):
        pred = np.array([0, 0, 1, 1])
        gt = np.array([0, 1, 1, 1])
        result = compute_miou(pred, gt, num_classes=2)
        # Class 0: intersection=1, union=2, IoU=0.5
        # Class 1: intersection=2, union=3, IoU=0.667
        assert result["per_class_iou"][0] == pytest.approx(0.5)
        assert result["per_class_iou"][1] == pytest.approx(2 / 3)
