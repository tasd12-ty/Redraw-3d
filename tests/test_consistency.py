"""
Unit tests for consistency checking in ORDINAL-SPATIAL benchmark.
"""

import pytest
from ordinal_spatial.evaluation.consistency import (
    ConsistencyStatus,
    ConsistencyReport,
    ConsistencyChecker,
    check_qrr_consistency,
    check_trr_consistency,
    check_full_consistency,
    find_minimal_conflict,
)


class TestCheckQRRConsistency:
    """Tests for QRR consistency checking."""

    def test_empty_constraints(self):
        """Test with no constraints."""
        report = check_qrr_consistency([])
        assert report.is_consistent is True
        assert len(report.cycles) == 0

    def test_single_constraint(self):
        """Test with single constraint."""
        constraints = [
            {"pair1": ["A", "B"], "pair2": ["C", "D"], "comparator": "<"}
        ]
        report = check_qrr_consistency(constraints)
        assert report.is_consistent is True

    def test_consistent_chain(self):
        """Test consistent transitive chain."""
        constraints = [
            {"pair1": ["A", "B"], "pair2": ["C", "D"], "comparator": "<"},
            {"pair1": ["C", "D"], "pair2": ["E", "F"], "comparator": "<"},
        ]
        report = check_qrr_consistency(constraints)
        assert report.is_consistent is True

    def test_simple_cycle(self):
        """Test detection of simple contradiction cycle."""
        constraints = [
            {"pair1": ["A", "B"], "pair2": ["C", "D"], "comparator": "<"},
            {"pair1": ["C", "D"], "pair2": ["E", "F"], "comparator": "<"},
            {"pair1": ["E", "F"], "pair2": ["A", "B"], "comparator": "<"},
        ]
        report = check_qrr_consistency(constraints)
        assert report.is_consistent is False
        assert len(report.cycles) > 0

    def test_approx_equality_no_edge(self):
        """Test that approximate equality doesn't create edges."""
        constraints = [
            {"pair1": ["A", "B"], "pair2": ["C", "D"], "comparator": "~="},
        ]
        report = check_qrr_consistency(constraints)
        assert report.is_consistent is True
        assert report.stats["n_edges"] == 0

    def test_mixed_comparators(self):
        """Test mixture of strict and approximate comparisons."""
        constraints = [
            {"pair1": ["A", "B"], "pair2": ["C", "D"], "comparator": "<"},
            {"pair1": ["C", "D"], "pair2": ["E", "F"], "comparator": "~="},  # No cycle contribution
            {"pair1": ["E", "F"], "pair2": ["G", "H"], "comparator": "<"},
        ]
        report = check_qrr_consistency(constraints)
        assert report.is_consistent is True


class TestConsistencyChecker:
    """Tests for incremental consistency checker."""

    def test_incremental_add(self):
        """Test adding constraints incrementally."""
        checker = ConsistencyChecker()

        # Add first constraint
        cycle = checker.add_qrr(
            {"pair1": ["A", "B"], "pair2": ["C", "D"], "comparator": "<"}
        )
        assert cycle is None

        # Add second constraint
        cycle = checker.add_qrr(
            {"pair1": ["C", "D"], "pair2": ["E", "F"], "comparator": "<"}
        )
        assert cycle is None

    def test_detect_cycle_on_add(self):
        """Test cycle detection when adding conflicting constraint."""
        checker = ConsistencyChecker()

        checker.add_qrr(
            {"pair1": ["A", "B"], "pair2": ["C", "D"], "comparator": "<"}
        )
        checker.add_qrr(
            {"pair1": ["C", "D"], "pair2": ["E", "F"], "comparator": "<"}
        )

        # This should create a cycle
        cycle = checker.add_qrr(
            {"pair1": ["E", "F"], "pair2": ["A", "B"], "comparator": "<"}
        )
        assert cycle is not None

    def test_reset(self):
        """Test resetting the checker."""
        checker = ConsistencyChecker()
        checker.add_qrr(
            {"pair1": ["A", "B"], "pair2": ["C", "D"], "comparator": "<"}
        )
        assert len(checker.constraints) == 1

        checker.reset()
        assert len(checker.constraints) == 0


class TestCheckTRRConsistency:
    """Tests for TRR consistency checking."""

    def test_empty_constraints(self):
        """Test with no constraints."""
        report = check_trr_consistency([])
        assert report.is_consistent is True

    def test_single_constraint(self):
        """Test with single constraint."""
        constraints = [
            {"target": "A", "ref1": "B", "ref2": "C", "hour": 3}
        ]
        report = check_trr_consistency(constraints)
        assert report.is_consistent is True

    def test_consistent_symmetric_pair(self):
        """Test consistent symmetric TRR pair."""
        # If A is at 3 o'clock from B->C, then from C->B, A should be at ~9 o'clock
        constraints = [
            {"target": "A", "ref1": "B", "ref2": "C", "hour": 3},
            {"target": "A", "ref1": "C", "ref2": "B", "hour": 9},  # 3+6=9
        ]
        report = check_trr_consistency(constraints, tolerance_hours=1)
        assert report.is_consistent is True

    def test_inconsistent_symmetric_pair(self):
        """Test inconsistent symmetric TRR pair."""
        # If A is at 3 o'clock from B->C, then from C->B, A at 3 is inconsistent
        constraints = [
            {"target": "A", "ref1": "B", "ref2": "C", "hour": 3},
            {"target": "A", "ref1": "C", "ref2": "B", "hour": 3},  # Should be ~9
        ]
        report = check_trr_consistency(constraints, tolerance_hours=1)
        # This should detect the inconsistency
        assert report.is_consistent is False


class TestFindMinimalConflict:
    """Tests for minimal conflict identification."""

    def test_no_conflict(self):
        """Test when there's no conflict."""
        constraints = [
            {"pair1": ["A", "B"], "pair2": ["C", "D"], "comparator": "<"},
        ]
        new = {"pair1": ["E", "F"], "pair2": ["G", "H"], "comparator": "<"}

        involved = find_minimal_conflict(constraints, new)
        assert len(involved) == 0

    def test_find_conflict(self):
        """Test finding constraints involved in conflict."""
        constraints = [
            {"pair1": ["A", "B"], "pair2": ["C", "D"], "comparator": "<"},
            {"pair1": ["C", "D"], "pair2": ["E", "F"], "comparator": "<"},
        ]
        # This creates a cycle
        new = {"pair1": ["E", "F"], "pair2": ["A", "B"], "comparator": "<"}

        involved = find_minimal_conflict(constraints, new)
        assert len(involved) > 0
        assert new in involved


class TestFullConsistency:
    """Tests for combined QRR+TRR consistency."""

    def test_both_consistent(self):
        """Test when both QRR and TRR are consistent."""
        qrr = [
            {"pair1": ["A", "B"], "pair2": ["C", "D"], "comparator": "<"},
        ]
        trr = [
            {"target": "A", "ref1": "B", "ref2": "C", "hour": 3},
        ]
        report = check_full_consistency(qrr, trr)
        assert report.is_consistent is True

    def test_qrr_inconsistent(self):
        """Test when QRR is inconsistent."""
        qrr = [
            {"pair1": ["A", "B"], "pair2": ["C", "D"], "comparator": "<"},
            {"pair1": ["C", "D"], "pair2": ["E", "F"], "comparator": "<"},
            {"pair1": ["E", "F"], "pair2": ["A", "B"], "comparator": "<"},
        ]
        trr = []
        report = check_full_consistency(qrr, trr)
        assert report.is_consistent is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
