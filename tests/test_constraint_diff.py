"""
Unit tests for constraint_diff module.

Tests the Constraint-Diff evaluation metrics for comparing
predicted constraints against ground truth.
"""

import pytest
from ordinal_spatial.evaluation.constraint_diff import (
    ConstraintDiffMetrics,
    compute_constraint_diff,
    compute_constraint_diff_batch,
)


class TestConstraintDiffMetrics:
    """Tests for ConstraintDiffMetrics dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        metrics = ConstraintDiffMetrics()
        assert metrics.n_ground_truth == 0
        assert metrics.n_predicted == 0
        assert metrics.n_correct == 0
        assert metrics.n_missing == 0
        assert metrics.n_spurious == 0
        assert metrics.n_violated == 0
        assert metrics.f1 == 0.0

    def test_to_dict(self):
        """Test serialization to dict."""
        metrics = ConstraintDiffMetrics(
            n_ground_truth=10,
            n_predicted=8,
            n_correct=6,
            n_missing=4,
            n_spurious=2,
            precision=0.75,
            recall=0.6,
            f1=0.67,
        )
        d = metrics.to_dict()
        assert d["n_ground_truth"] == 10
        assert d["n_predicted"] == 8
        assert d["precision"] == 0.75

    def test_summary(self):
        """Test summary generation."""
        metrics = ConstraintDiffMetrics(
            n_ground_truth=10,
            n_predicted=8,
            n_correct=6,
            n_missing=4,
            precision=0.75,
            recall=0.6,
        )
        summary = metrics.summary()
        assert "GT: 10" in summary
        assert "Pred: 8" in summary


class TestComputeConstraintDiff:
    """Tests for compute_constraint_diff function."""

    def test_empty_both(self):
        """Test with empty constraint sets."""
        result = compute_constraint_diff({}, {})
        assert result.n_ground_truth == 0
        assert result.n_predicted == 0
        assert result.n_correct == 0
        assert result.f1 == 0.0

    def test_perfect_match_qrr(self):
        """Test perfect match for QRR constraints."""
        gt = {
            "qrr": [
                {"pair1": ["a", "b"], "pair2": ["c", "d"], "comparator": "<", "metric": "dist3D"}
            ]
        }
        pred = {
            "qrr": [
                {"pair1": ["a", "b"], "pair2": ["c", "d"], "comparator": "<", "metric": "dist3D"}
            ]
        }
        result = compute_constraint_diff(pred, gt, constraint_types=["qrr"])
        assert result.n_ground_truth == 1
        assert result.n_predicted == 1
        assert result.n_correct == 1
        assert result.n_missing == 0
        assert result.n_spurious == 0
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0

    def test_missing_constraint(self):
        """Test detection of missing constraints."""
        gt = {
            "qrr": [
                {"pair1": ["a", "b"], "pair2": ["c", "d"], "comparator": "<", "metric": "dist3D"},
                {"pair1": ["e", "f"], "pair2": ["g", "h"], "comparator": ">", "metric": "dist3D"},
            ]
        }
        pred = {
            "qrr": [
                {"pair1": ["a", "b"], "pair2": ["c", "d"], "comparator": "<", "metric": "dist3D"}
            ]
        }
        result = compute_constraint_diff(pred, gt, constraint_types=["qrr"])
        assert result.n_ground_truth == 2
        assert result.n_predicted == 1
        assert result.n_missing == 1
        assert result.missing_rate == 0.5
        assert result.recall == 0.5

    def test_spurious_constraint(self):
        """Test detection of spurious (hallucinated) constraints."""
        gt = {
            "qrr": [
                {"pair1": ["a", "b"], "pair2": ["c", "d"], "comparator": "<", "metric": "dist3D"}
            ]
        }
        pred = {
            "qrr": [
                {"pair1": ["a", "b"], "pair2": ["c", "d"], "comparator": "<", "metric": "dist3D"},
                {"pair1": ["x", "y"], "pair2": ["z", "w"], "comparator": ">", "metric": "dist3D"},
            ]
        }
        result = compute_constraint_diff(pred, gt, constraint_types=["qrr"])
        assert result.n_ground_truth == 1
        assert result.n_predicted == 2
        assert result.n_spurious == 1
        assert result.spurious_rate == 0.5
        assert result.precision == 0.5

    def test_violated_constraint(self):
        """Test detection of violated (flipped direction) constraints."""
        gt = {
            "qrr": [
                {"pair1": ["a", "b"], "pair2": ["c", "d"], "comparator": "<", "metric": "dist3D"}
            ]
        }
        pred = {
            "qrr": [
                {"pair1": ["a", "b"], "pair2": ["c", "d"], "comparator": ">", "metric": "dist3D"}
            ]
        }
        result = compute_constraint_diff(pred, gt, constraint_types=["qrr"])
        assert result.n_ground_truth == 1
        assert result.n_predicted == 1
        assert result.n_correct == 0
        assert result.n_violated == 1
        assert result.violated_rate == 1.0

    def test_axial_constraints(self):
        """Test axial constraint comparison."""
        gt = {
            "axial": [
                {"object1": "a", "object2": "b", "relation": "left_of"},
                {"object1": "c", "object2": "d", "relation": "above"},
            ]
        }
        pred = {
            "axial": [
                {"object1": "a", "object2": "b", "relation": "left_of"}
            ]
        }
        result = compute_constraint_diff(pred, gt, constraint_types=["axial"])
        assert result.n_ground_truth == 2
        assert result.n_predicted == 1
        assert result.n_correct == 1
        assert result.n_missing == 1

    def test_size_constraints(self):
        """Test size constraint comparison."""
        gt = {
            "size": [
                {"object1": "a", "object2": "b", "comparator": "<"}
            ]
        }
        pred = {
            "size": [
                {"object1": "a", "object2": "b", "comparator": "<"}
            ]
        }
        result = compute_constraint_diff(pred, gt, constraint_types=["size"])
        assert result.n_correct == 1
        assert result.f1 == 1.0

    def test_topology_constraints(self):
        """Test topology constraint comparison."""
        gt = {
            "topology": [
                {"object1": "a", "object2": "b", "relation": "disjoint"}
            ]
        }
        pred = {
            "topology": [
                {"object1": "a", "object2": "b", "relation": "disjoint"}
            ]
        }
        result = compute_constraint_diff(pred, gt, constraint_types=["topology"])
        assert result.n_correct == 1
        assert result.f1 == 1.0

    def test_multiple_constraint_types(self):
        """Test with multiple constraint types."""
        gt = {
            "qrr": [
                {"pair1": ["a", "b"], "pair2": ["c", "d"], "comparator": "<", "metric": "dist3D"}
            ],
            "axial": [
                {"object1": "a", "object2": "b", "relation": "left_of"}
            ],
        }
        pred = {
            "qrr": [
                {"pair1": ["a", "b"], "pair2": ["c", "d"], "comparator": "<", "metric": "dist3D"}
            ],
            "axial": [
                {"object1": "a", "object2": "b", "relation": "left_of"}
            ],
        }
        result = compute_constraint_diff(pred, gt, constraint_types=["qrr", "axial"])
        assert result.n_ground_truth == 2
        assert result.n_correct == 2
        assert result.f1 == 1.0
        assert "qrr" in result.by_type
        assert "axial" in result.by_type

    def test_comparator_normalization(self):
        """Test that different comparator formats are normalized."""
        gt = {
            "qrr": [
                {"pair1": ["a", "b"], "pair2": ["c", "d"], "comparator": "lt", "metric": "dist3D"}
            ]
        }
        pred = {
            "qrr": [
                {"pair1": ["a", "b"], "pair2": ["c", "d"], "comparator": "<", "metric": "dist3D"}
            ]
        }
        result = compute_constraint_diff(pred, gt, constraint_types=["qrr"])
        assert result.n_correct == 1
        assert result.f1 == 1.0

    def test_pair_order_canonicalization(self):
        """Test that pair order is canonicalized."""
        gt = {
            "qrr": [
                {"pair1": ["b", "a"], "pair2": ["c", "d"], "comparator": "<", "metric": "dist3D"}
            ]
        }
        pred = {
            "qrr": [
                {"pair1": ["a", "b"], "pair2": ["c", "d"], "comparator": "<", "metric": "dist3D"}
            ]
        }
        result = compute_constraint_diff(pred, gt, constraint_types=["qrr"])
        assert result.n_correct == 1


class TestComputeConstraintDiffBatch:
    """Tests for batch computation."""

    def test_batch_empty(self):
        """Test batch with empty list."""
        result = compute_constraint_diff_batch([], [])
        assert result.n_ground_truth == 0

    def test_batch_multiple_scenes(self):
        """Test batch with multiple scenes."""
        predictions = [
            {"qrr": [{"pair1": ["a", "b"], "pair2": ["c", "d"], "comparator": "<", "metric": "dist3D"}]},
            {"qrr": [{"pair1": ["e", "f"], "pair2": ["g", "h"], "comparator": ">", "metric": "dist3D"}]},
        ]
        ground_truths = [
            {"qrr": [{"pair1": ["a", "b"], "pair2": ["c", "d"], "comparator": "<", "metric": "dist3D"}]},
            {"qrr": [{"pair1": ["e", "f"], "pair2": ["g", "h"], "comparator": ">", "metric": "dist3D"}]},
        ]
        result = compute_constraint_diff_batch(predictions, ground_truths, constraint_types=["qrr"])
        assert result.n_ground_truth == 2
        assert result.n_correct == 2
        assert result.f1 == 1.0


class TestEdgeCases:
    """Edge case tests."""

    def test_no_gt_with_predictions(self):
        """Test when GT is empty but predictions exist."""
        gt = {"qrr": []}
        pred = {
            "qrr": [
                {"pair1": ["a", "b"], "pair2": ["c", "d"], "comparator": "<", "metric": "dist3D"}
            ]
        }
        result = compute_constraint_diff(pred, gt, constraint_types=["qrr"])
        assert result.n_ground_truth == 0
        assert result.n_predicted == 1
        assert result.n_spurious == 1
        assert result.recall == 0.0

    def test_gt_with_no_predictions(self):
        """Test when predictions are empty but GT exists."""
        gt = {
            "qrr": [
                {"pair1": ["a", "b"], "pair2": ["c", "d"], "comparator": "<", "metric": "dist3D"}
            ]
        }
        pred = {"qrr": []}
        result = compute_constraint_diff(pred, gt, constraint_types=["qrr"])
        assert result.n_ground_truth == 1
        assert result.n_predicted == 0
        assert result.n_missing == 1
        assert result.precision == 0.0

    def test_approximately_equal_match(self):
        """Test approximately equal comparator matching."""
        gt = {
            "qrr": [
                {"pair1": ["a", "b"], "pair2": ["c", "d"], "comparator": "~=", "metric": "dist3D"}
            ]
        }
        pred = {
            "qrr": [
                {"pair1": ["a", "b"], "pair2": ["c", "d"], "comparator": "eq", "metric": "dist3D"}
            ]
        }
        result = compute_constraint_diff(pred, gt, constraint_types=["qrr"])
        assert result.n_correct == 1
