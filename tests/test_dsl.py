"""
Unit tests for the ORDINAL-SPATIAL DSL module.

Tests cover:
- Comparator algebra
- QRR predicate computation
- TRR predicate computation
- Edge cases and boundary conditions
"""

import pytest
import numpy as np
from typing import Dict

# Import DSL components
from ordinal_spatial.dsl.comparators import (
    Comparator,
    TolerancePreset,
    TOLERANCE_STRICT,
    TOLERANCE_STANDARD,
    TOLERANCE_RELAXED,
    compare,
    compare_ratio,
    ordinal_distance,
    is_flip,
    difficulty_from_ratio,
    ComparatorChain,
)

from ordinal_spatial.dsl.predicates import (
    MetricType,
    QRRConstraint,
    TRRConstraint,
    compute_qrr,
    compute_trr,
    extract_all_qrr,
    extract_all_trr,
    clock_angular_error,
    angle_to_hour,
    hour_to_quadrant,
    compute_angle_2d,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_objects() -> Dict[str, Dict]:
    """Create sample objects for testing."""
    return {
        "obj_A": {
            "id": "obj_A",
            "shape": "cube",
            "color": "red",
            "size": "large",
            "position_3d": [0.0, 0.0, 0.0],
            "position_2d": [500, 500],
            "depth": 5.0,
        },
        "obj_B": {
            "id": "obj_B",
            "shape": "sphere",
            "color": "blue",
            "size": "medium",
            "position_3d": [1.0, 0.0, 0.0],
            "position_2d": [600, 500],
            "depth": 5.0,
        },
        "obj_C": {
            "id": "obj_C",
            "shape": "cylinder",
            "color": "green",
            "size": "small",
            "position_3d": [0.0, 2.0, 0.0],
            "position_2d": [500, 300],
            "depth": 3.0,
        },
        "obj_D": {
            "id": "obj_D",
            "shape": "cone",
            "color": "yellow",
            "size": "medium",
            "position_3d": [3.0, 3.0, 0.0],
            "position_2d": [700, 200],
            "depth": 2.0,
        },
    }


# =============================================================================
# Comparator Tests
# =============================================================================

class TestComparator:
    """Tests for Comparator enum and operations."""

    def test_comparator_values(self):
        """Test comparator string values."""
        assert str(Comparator.LT) == "<"
        assert str(Comparator.APPROX) == "~="
        assert str(Comparator.GT) == ">"

    def test_comparator_ordinal(self):
        """Test ordinal values for distance calculation."""
        assert Comparator.LT.ordinal == 0
        assert Comparator.APPROX.ordinal == 1
        assert Comparator.GT.ordinal == 2

    def test_comparator_flip(self):
        """Test flipping comparators."""
        assert Comparator.LT.flip() == Comparator.GT
        assert Comparator.GT.flip() == Comparator.LT
        assert Comparator.APPROX.flip() == Comparator.APPROX

    def test_comparator_from_string(self):
        """Test parsing comparators from strings."""
        assert Comparator.from_string("<") == Comparator.LT
        assert Comparator.from_string(">") == Comparator.GT
        assert Comparator.from_string("~=") == Comparator.APPROX
        assert Comparator.from_string("≈") == Comparator.APPROX
        assert Comparator.from_string("lt") == Comparator.LT

    def test_comparator_from_string_invalid(self):
        """Test parsing invalid comparator strings."""
        with pytest.raises(ValueError):
            Comparator.from_string("invalid")


class TestCompare:
    """Tests for the compare function."""

    def test_compare_strict_less(self):
        """Test strict less than comparison."""
        assert compare(1.0, 2.0, tau=0.10) == Comparator.LT
        assert compare(0.5, 1.0, tau=0.10) == Comparator.LT

    def test_compare_strict_greater(self):
        """Test strict greater than comparison."""
        assert compare(2.0, 1.0, tau=0.10) == Comparator.GT
        assert compare(1.0, 0.5, tau=0.10) == Comparator.GT

    def test_compare_approximately_equal(self):
        """Test approximate equality within tolerance."""
        # 1.0 vs 1.05 with tau=0.10 should be approximately equal
        assert compare(1.0, 1.05, tau=0.10) == Comparator.APPROX
        assert compare(1.05, 1.0, tau=0.10) == Comparator.APPROX

        # 1.0 vs 1.09 with tau=0.10 should be approximately equal
        assert compare(1.0, 1.09, tau=0.10) == Comparator.APPROX

    def test_compare_boundary_cases(self):
        """Test boundary cases at tolerance threshold."""
        # With tau=0.10, threshold = 0.10 * max(a,b)
        # For 1.0 vs 1.11: threshold = 0.10 * 1.11 = 0.111, diff = 0.11 < 0.111 -> APPROX
        assert compare(1.0, 1.11, tau=0.10) == Comparator.APPROX
        assert compare(1.0, 1.10, tau=0.10) == Comparator.APPROX
        # Clear LT case: 1.0 vs 1.15 -> threshold = 0.115, diff = 0.15 > 0.115 -> LT
        assert compare(1.0, 1.15, tau=0.10) == Comparator.LT

    def test_compare_zero_values(self):
        """Test handling of zero values."""
        assert compare(0.0, 0.0, tau=0.10) == Comparator.APPROX
        assert compare(0.0, 1.0, tau=0.10) == Comparator.LT
        assert compare(1.0, 0.0, tau=0.10) == Comparator.GT

    def test_compare_invalid_tau(self):
        """Test invalid tolerance values."""
        with pytest.raises(ValueError):
            compare(1.0, 2.0, tau=0.0)
        with pytest.raises(ValueError):
            compare(1.0, 2.0, tau=1.0)
        with pytest.raises(ValueError):
            compare(1.0, 2.0, tau=-0.1)

    def test_compare_negative_values(self):
        """Test handling of negative values."""
        with pytest.raises(ValueError):
            compare(-1.0, 2.0, tau=0.10)

    def test_compare_ratio(self):
        """Test compare with ratio output."""
        comp, ratio = compare_ratio(1.0, 2.0, tau=0.10)
        assert comp == Comparator.LT
        assert ratio == pytest.approx(0.5)

        comp, ratio = compare_ratio(2.0, 1.0, tau=0.10)
        assert comp == Comparator.GT
        assert ratio == pytest.approx(2.0)


class TestOrdinalDistance:
    """Tests for ordinal distance calculations."""

    def test_ordinal_distance_same(self):
        """Test distance between same comparators."""
        assert ordinal_distance(Comparator.LT, Comparator.LT) == 0
        assert ordinal_distance(Comparator.APPROX, Comparator.APPROX) == 0
        assert ordinal_distance(Comparator.GT, Comparator.GT) == 0

    def test_ordinal_distance_adjacent(self):
        """Test distance between adjacent comparators."""
        assert ordinal_distance(Comparator.LT, Comparator.APPROX) == 1
        assert ordinal_distance(Comparator.APPROX, Comparator.GT) == 1

    def test_ordinal_distance_flip(self):
        """Test distance for complete flip."""
        assert ordinal_distance(Comparator.LT, Comparator.GT) == 2
        assert ordinal_distance(Comparator.GT, Comparator.LT) == 2


class TestIsFlip:
    """Tests for flip detection."""

    def test_is_flip_true(self):
        """Test detection of flips."""
        assert is_flip(Comparator.LT, Comparator.GT) is True
        assert is_flip(Comparator.GT, Comparator.LT) is True

    def test_is_flip_false(self):
        """Test non-flips."""
        assert is_flip(Comparator.LT, Comparator.LT) is False
        assert is_flip(Comparator.LT, Comparator.APPROX) is False
        assert is_flip(Comparator.APPROX, Comparator.GT) is False


class TestDifficultyFromRatio:
    """Tests for difficulty level calculation."""

    def test_difficulty_easy(self):
        """Test easy difficulty (large ratio)."""
        assert difficulty_from_ratio(3.0) == 1
        assert difficulty_from_ratio(2.5) == 1

    def test_difficulty_medium(self):
        """Test medium difficulty."""
        assert difficulty_from_ratio(1.8) == 2
        assert difficulty_from_ratio(1.4) == 3

    def test_difficulty_hard(self):
        """Test hard difficulty (small ratio)."""
        assert difficulty_from_ratio(1.2) == 4
        assert difficulty_from_ratio(1.1) == 5
        assert difficulty_from_ratio(1.02) == 6

    def test_difficulty_inverse_ratio(self):
        """Test that ratios < 1 are normalized."""
        # 0.5 -> 1/0.5 = 2.0 -> difficulty 1 (but boundary check needed)
        # Actually 2.0 is at boundary, might be level 1 or 2
        assert difficulty_from_ratio(0.5) in [1, 2]  # 2.0 is at boundary
        assert difficulty_from_ratio(0.6) == 2  # Equivalent to ~1.67


class TestTolerancePreset:
    """Tests for tolerance presets."""

    def test_preset_values(self):
        """Test preset tolerance values."""
        assert TOLERANCE_STRICT.tau == 0.05
        assert TOLERANCE_STANDARD.tau == 0.10
        assert TOLERANCE_RELAXED.tau == 0.20

    def test_preset_invalid(self):
        """Test invalid preset creation."""
        with pytest.raises(ValueError):
            TolerancePreset("invalid", tau=0.0, description="test")
        with pytest.raises(ValueError):
            TolerancePreset("invalid", tau=1.0, description="test")


class TestComparatorChain:
    """Tests for comparator chain transitivity."""

    def test_chain_all_lt(self):
        """Test chain with all LT."""
        chain = ComparatorChain()
        chain.add(Comparator.LT).add(Comparator.LT)
        assert chain.implies() == Comparator.LT

    def test_chain_all_gt(self):
        """Test chain with all GT."""
        chain = ComparatorChain()
        chain.add(Comparator.GT).add(Comparator.GT)
        assert chain.implies() == Comparator.GT

    def test_chain_all_approx(self):
        """Test chain with all APPROX."""
        chain = ComparatorChain()
        chain.add(Comparator.APPROX).add(Comparator.APPROX)
        assert chain.implies() == Comparator.APPROX

    def test_chain_mixed_inconsistent(self):
        """Test inconsistent chain."""
        chain = ComparatorChain()
        chain.add(Comparator.LT).add(Comparator.GT)
        assert chain.implies() is None
        assert chain.is_consistent() is False


# =============================================================================
# QRR Predicate Tests
# =============================================================================

class TestQRRConstraint:
    """Tests for QRR constraint dataclass."""

    def test_qrr_creation(self):
        """Test QRR constraint creation."""
        qrr = QRRConstraint(
            pair1=("obj_A", "obj_B"),
            pair2=("obj_C", "obj_D"),
            metric=MetricType.DIST_3D,
            comparator=Comparator.LT,
        )
        assert qrr.pair1 == ("obj_A", "obj_B")
        assert qrr.pair2 == ("obj_C", "obj_D")
        assert qrr.metric == MetricType.DIST_3D
        assert qrr.comparator == Comparator.LT

    def test_qrr_pair_sorting(self):
        """Test that pairs are sorted canonically."""
        qrr = QRRConstraint(
            pair1=("obj_B", "obj_A"),  # Reversed
            pair2=("obj_D", "obj_C"),  # Reversed
            metric=MetricType.DIST_3D,
            comparator=Comparator.LT,
        )
        assert qrr.pair1 == ("obj_A", "obj_B")
        assert qrr.pair2 == ("obj_C", "obj_D")

    def test_qrr_flip(self):
        """Test flipping QRR constraint."""
        qrr = QRRConstraint(
            pair1=("obj_A", "obj_B"),
            pair2=("obj_C", "obj_D"),
            metric=MetricType.DIST_3D,
            comparator=Comparator.LT,
            ratio=0.5,
        )
        flipped = qrr.flip()
        assert flipped.pair1 == ("obj_C", "obj_D")
        assert flipped.pair2 == ("obj_A", "obj_B")
        assert flipped.comparator == Comparator.GT
        assert flipped.ratio == pytest.approx(2.0)

    def test_qrr_to_dict(self):
        """Test serialization to dict."""
        qrr = QRRConstraint(
            pair1=("obj_A", "obj_B"),
            pair2=("obj_C", "obj_D"),
            metric=MetricType.DIST_3D,
            comparator=Comparator.LT,
        )
        d = qrr.to_dict()
        assert d["pair1"] == ["obj_A", "obj_B"]
        assert d["metric"] == "dist3D"
        assert d["comparator"] == "<"

    def test_qrr_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "pair1": ["obj_A", "obj_B"],
            "pair2": ["obj_C", "obj_D"],
            "metric": "dist3D",
            "comparator": "<",
        }
        qrr = QRRConstraint.from_dict(d)
        assert qrr.pair1 == ("obj_A", "obj_B")
        assert qrr.metric == MetricType.DIST_3D
        assert qrr.comparator == Comparator.LT


class TestComputeQRR:
    """Tests for QRR computation from objects."""

    def test_compute_qrr_dist3d(self, sample_objects):
        """Test computing QRR with 3D distance metric."""
        # obj_A at (0,0,0), obj_B at (1,0,0) -> dist = 1.0
        # obj_C at (0,2,0), obj_D at (3,3,0) -> dist = sqrt((3-0)^2 + (3-2)^2) = sqrt(10) ≈ 3.16
        qrr = compute_qrr(
            sample_objects,
            pair1=("obj_A", "obj_B"),
            pair2=("obj_C", "obj_D"),
            metric=MetricType.DIST_3D,
            tau=0.10,
        )
        assert qrr.comparator == Comparator.LT  # 1.0 < 3.16

    def test_compute_qrr_approximately_equal(self, sample_objects):
        """Test QRR returning approximately equal."""
        # Create objects with similar distances
        objects = {
            "A": {"position_3d": [0, 0, 0]},
            "B": {"position_3d": [1, 0, 0]},  # dist(A,B) = 1.0
            "C": {"position_3d": [0, 0, 0]},
            "D": {"position_3d": [0, 1.05, 0]},  # dist(C,D) = 1.05
        }
        qrr = compute_qrr(
            objects,
            pair1=("A", "B"),
            pair2=("C", "D"),
            metric=MetricType.DIST_3D,
            tau=0.10,
        )
        assert qrr.comparator == Comparator.APPROX  # Within 10% tolerance


# =============================================================================
# TRR Predicate Tests
# =============================================================================

class TestTRRConstraint:
    """Tests for TRR constraint dataclass."""

    def test_trr_creation(self):
        """Test TRR constraint creation."""
        trr = TRRConstraint(
            target="obj_A",
            ref1="obj_B",
            ref2="obj_C",
            hour=3,
        )
        assert trr.target == "obj_A"
        assert trr.ref1 == "obj_B"
        assert trr.ref2 == "obj_C"
        assert trr.hour == 3
        # Hour 3 should be in quadrant 2 (hours 3-5)
        assert trr.quadrant == hour_to_quadrant(3)
        assert trr.quadrant == 2

    def test_trr_invalid_hour(self):
        """Test invalid hour values."""
        with pytest.raises(ValueError):
            TRRConstraint(target="A", ref1="B", ref2="C", hour=0)
        with pytest.raises(ValueError):
            TRRConstraint(target="A", ref1="B", ref2="C", hour=13)


class TestAngleToHour:
    """Tests for angle to clock hour conversion."""

    def test_angle_to_hour_twelve(self):
        """Test 0 degrees maps to hour 12."""
        assert angle_to_hour(0) == 12
        assert angle_to_hour(10) == 12
        assert angle_to_hour(350) == 12

    def test_angle_to_hour_three(self):
        """Test 90 degrees maps to hour 3."""
        assert angle_to_hour(90) == 3
        assert angle_to_hour(80) == 3
        assert angle_to_hour(100) == 3

    def test_angle_to_hour_six(self):
        """Test 180 degrees maps to hour 6."""
        assert angle_to_hour(180) == 6

    def test_angle_to_hour_nine(self):
        """Test 270 degrees maps to hour 9."""
        assert angle_to_hour(270) == 9


class TestHourToQuadrant:
    """Tests for hour to quadrant conversion."""

    def test_quadrant_1(self):
        """Test hours in quadrant 1."""
        assert hour_to_quadrant(12) == 1
        assert hour_to_quadrant(1) == 1
        assert hour_to_quadrant(2) == 1

    def test_quadrant_2(self):
        """Test hours in quadrant 2."""
        assert hour_to_quadrant(3) == 2
        assert hour_to_quadrant(4) == 2
        assert hour_to_quadrant(5) == 2

    def test_quadrant_3(self):
        """Test hours in quadrant 3."""
        assert hour_to_quadrant(6) == 3
        assert hour_to_quadrant(7) == 3
        assert hour_to_quadrant(8) == 3

    def test_quadrant_4(self):
        """Test hours in quadrant 4."""
        assert hour_to_quadrant(9) == 4
        assert hour_to_quadrant(10) == 4
        assert hour_to_quadrant(11) == 4


class TestClockAngularError:
    """Tests for clock angular error calculation."""

    def test_same_hour(self):
        """Test error between same hours."""
        assert clock_angular_error(3, 3) == 0

    def test_adjacent_hours(self):
        """Test error between adjacent hours."""
        assert clock_angular_error(3, 4) == 30
        assert clock_angular_error(12, 1) == 30

    def test_opposite_hours(self):
        """Test error between opposite hours."""
        assert clock_angular_error(12, 6) == 180
        assert clock_angular_error(3, 9) == 180

    def test_wraparound(self):
        """Test wraparound handling."""
        # 12 vs 11 should be 30 degrees, not 330
        assert clock_angular_error(12, 11) == 30
        assert clock_angular_error(1, 11) == 60


class TestComputeAngle2D:
    """Tests for 2D angle computation."""

    def test_angle_toward_reference(self):
        """Test angle when target is toward reference."""
        target = np.array([2.0, 0.0])
        ref1 = np.array([0.0, 0.0])
        ref2 = np.array([1.0, 0.0])

        angle = compute_angle_2d(target, ref1, ref2)
        assert angle == pytest.approx(0.0, abs=1.0)  # Should be ~0 (12 o'clock)

    def test_angle_perpendicular(self):
        """Test angle when target is perpendicular."""
        target = np.array([0.0, 1.0])
        ref1 = np.array([0.0, 0.0])
        ref2 = np.array([1.0, 0.0])

        angle = compute_angle_2d(target, ref1, ref2)
        # In standard coords, perpendicular up is 90 degrees (3 o'clock)
        assert angle == pytest.approx(90.0, abs=1.0)


class TestExtractAllConstraints:
    """Tests for batch constraint extraction."""

    def test_extract_all_qrr(self, sample_objects):
        """Test extracting all QRR constraints."""
        constraints = extract_all_qrr(
            sample_objects,
            metric=MetricType.DIST_3D,
            tau=0.10,
            disjoint_only=True,
        )
        # With 4 objects: C(4,2)=6 pairs, C(6,2)/2=15 pair comparisons
        # But disjoint only: need to exclude overlapping pairs
        # For 4 objects, 3 disjoint pair comparisons
        assert len(constraints) == 3

    def test_extract_all_trr(self, sample_objects):
        """Test extracting all TRR constraints."""
        constraints = extract_all_trr(sample_objects)
        # With 4 objects: P(4,3) = 24 ordered triples
        assert len(constraints) == 24


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
