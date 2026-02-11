"""
约束比较引擎测试。

Tests for evaluation/comparison.py ConstraintComparator.
"""

import pytest
from ordinal_spatial.evaluation.comparison import (
    ConstraintComparator,
    ComparisonResult,
    ConstraintMatch,
    _qrr_canonical,
    _axial_canonical,
    _size_canonical,
    _topology_canonical,
    _occlusion_canonical,
    _closer_canonical,
    _trr_canonical,
)


# -------------------------------------------------------
# Canonical Key Tests
# -------------------------------------------------------

class TestCanonicalKeys:

    def test_qrr_canonical_normal(self):
        c = {"pair1": ["A", "B"], "pair2": ["C", "D"],
             "metric": "dist3D", "comparator": "<"}
        key, comp = _qrr_canonical(c)
        assert key == (("A", "B"), ("C", "D"), "dist3D")
        assert comp == "<"

    def test_qrr_canonical_pair_flip(self):
        """Swapped pair order → comparator flips."""
        c = {"pair1": ["C", "D"], "pair2": ["A", "B"],
             "metric": "dist3D", "comparator": "<"}
        key, comp = _qrr_canonical(c)
        assert key == (("A", "B"), ("C", "D"), "dist3D")
        assert comp == ">"  # 翻转

    def test_axial_canonical_normal(self):
        c = {"obj1": "A", "obj2": "B",
             "relation": "left_of"}
        key, rel = _axial_canonical(c)
        assert key == ("A", "B")
        assert rel == "left_of"

    def test_axial_canonical_obj_flip(self):
        """obj1 > obj2 → swap + invert relation."""
        c = {"obj1": "B", "obj2": "A",
             "relation": "left_of"}
        key, rel = _axial_canonical(c)
        assert key == ("A", "B")
        assert rel == "right_of"

    def test_size_canonical(self):
        c = {"bigger": "X", "smaller": "Y"}
        key, rel = _size_canonical(c)
        assert key == ("X", "Y")
        assert rel == "bigger"

    def test_topology_canonical_sorted(self):
        c = {"obj1": "Z", "obj2": "A", "relation": "disjoint"}
        key, rel = _topology_canonical(c)
        assert key == ("A", "Z")

    def test_occlusion_canonical_directed(self):
        c = {"occluder": "A", "occluded": "B"}
        key, _ = _occlusion_canonical(c)
        assert key == ("A", "B")

    def test_closer_canonical(self):
        c = {"anchor": "A", "closer": "B", "farther": "C"}
        key, val = _closer_canonical(c)
        assert key == ("A", ("B", "C"))
        assert val == "B"

    def test_trr_canonical(self):
        c = {"target": "A", "ref1": "B", "ref2": "C",
             "hour": 3}
        key, hour = _trr_canonical(c)
        assert key == ("A", "B", "C")
        assert hour == 3


# -------------------------------------------------------
# QRR Comparison
# -------------------------------------------------------

class TestCompareQRR:

    def setup_method(self):
        self.c = ConstraintComparator()

    def test_correct(self):
        pred = {"qrr": [
            {"pair1": ["A", "B"], "pair2": ["C", "D"],
             "metric": "dist3D", "comparator": "<"}
        ]}
        tgt = {"qrr": [
            {"pair1": ["A", "B"], "pair2": ["C", "D"],
             "metric": "dist3D", "comparator": "<"}
        ]}
        r = self.c.compare(pred, tgt, ["qrr"])
        assert r.n_correct == 1
        assert r.n_wrong == 0
        assert r.f1 == 1.0

    def test_flip(self):
        pred = {"qrr": [
            {"pair1": ["A", "B"], "pair2": ["C", "D"],
             "metric": "dist3D", "comparator": "<"}
        ]}
        tgt = {"qrr": [
            {"pair1": ["A", "B"], "pair2": ["C", "D"],
             "metric": "dist3D", "comparator": ">"}
        ]}
        r = self.c.compare(pred, tgt, ["qrr"])
        assert r.n_wrong == 1
        wrong_m = [
            m for m in r.matches if m.category == "wrong_value"
        ]
        assert wrong_m[0].details["is_flip"]
        assert wrong_m[0].details["sub_type"] == "flip"

    def test_shift(self):
        pred = {"qrr": [
            {"pair1": ["A", "B"], "pair2": ["C", "D"],
             "metric": "dist3D", "comparator": "~="}
        ]}
        tgt = {"qrr": [
            {"pair1": ["A", "B"], "pair2": ["C", "D"],
             "metric": "dist3D", "comparator": "<"}
        ]}
        r = self.c.compare(pred, tgt, ["qrr"])
        assert r.n_wrong == 1
        wrong_m = [
            m for m in r.matches if m.category == "wrong_value"
        ]
        assert wrong_m[0].details["sub_type"] == "shift"

    def test_missing(self):
        pred = {"qrr": []}
        tgt = {"qrr": [
            {"pair1": ["A", "B"], "pair2": ["C", "D"],
             "metric": "dist3D", "comparator": "<"}
        ]}
        r = self.c.compare(pred, tgt, ["qrr"])
        assert r.n_missing == 1

    def test_hallucinated(self):
        pred = {"qrr": [
            {"pair1": ["X", "Y"], "pair2": ["W", "Z"],
             "metric": "dist3D", "comparator": "<"}
        ]}
        tgt = {"qrr": []}
        r = self.c.compare(pred, tgt, ["qrr"])
        assert r.n_hallucinated == 1


# -------------------------------------------------------
# Axial Comparison
# -------------------------------------------------------

class TestCompareAxial:

    def setup_method(self):
        self.c = ConstraintComparator()

    def test_correct(self):
        pred = {"axial": [
            {"obj1": "A", "obj2": "B", "relation": "left_of"}
        ]}
        tgt = {"axial": [
            {"obj1": "A", "obj2": "B", "relation": "left_of"}
        ]}
        r = self.c.compare(pred, tgt, ["axial"])
        assert r.n_correct == 1

    def test_flip(self):
        pred = {"axial": [
            {"obj1": "A", "obj2": "B", "relation": "left_of"}
        ]}
        tgt = {"axial": [
            {"obj1": "A", "obj2": "B", "relation": "right_of"}
        ]}
        r = self.c.compare(pred, tgt, ["axial"])
        assert r.n_wrong == 1

    def test_obj_order_normalized(self):
        """Different obj order, same relation → still matches."""
        pred = {"axial": [
            {"obj1": "B", "obj2": "A", "relation": "right_of"}
        ]}
        tgt = {"axial": [
            {"obj1": "A", "obj2": "B", "relation": "left_of"}
        ]}
        r = self.c.compare(pred, tgt, ["axial"])
        # B right_of A → canonical (A,B,left_of) == target
        assert r.n_correct == 1


# -------------------------------------------------------
# Occlusion Comparison
# -------------------------------------------------------

class TestCompareOcclusion:

    def setup_method(self):
        self.c = ConstraintComparator()

    def test_correct(self):
        pred = {"occlusion": [
            {"occluder": "A", "occluded": "B"}
        ]}
        tgt = {"occlusion": [
            {"occluder": "A", "occluded": "B"}
        ]}
        r = self.c.compare(pred, tgt, ["occlusion"])
        assert r.n_correct == 1

    def test_reversed_is_wrong(self):
        """Reversed occlusion direction → wrong_value."""
        pred = {"occlusion": [
            {"occluder": "B", "occluded": "A"}
        ]}
        tgt = {"occlusion": [
            {"occluder": "A", "occluded": "B"}
        ]}
        r = self.c.compare(pred, tgt, ["occlusion"])
        assert r.n_wrong == 1


# -------------------------------------------------------
# Size, Topology, Closer, TRR
# -------------------------------------------------------

class TestCompareOtherTypes:

    def setup_method(self):
        self.c = ConstraintComparator()

    def test_size_correct(self):
        pred = {"size": [{"bigger": "A", "smaller": "B"}]}
        tgt = {"size": [{"bigger": "A", "smaller": "B"}]}
        r = self.c.compare(pred, tgt, ["size"])
        assert r.n_correct == 1

    def test_topology_wrong(self):
        pred = {"topology": [
            {"obj1": "A", "obj2": "B", "relation": "disjoint"}
        ]}
        tgt = {"topology": [
            {"obj1": "A", "obj2": "B", "relation": "touching"}
        ]}
        r = self.c.compare(pred, tgt, ["topology"])
        assert r.n_wrong == 1

    def test_closer_flip(self):
        pred = {"closer": [
            {"anchor": "A", "closer": "B", "farther": "C"}
        ]}
        tgt = {"closer": [
            {"anchor": "A", "closer": "C", "farther": "B"}
        ]}
        r = self.c.compare(pred, tgt, ["closer"])
        assert r.n_wrong == 1

    def test_trr_correct(self):
        pred = {"trr": [
            {"target": "A", "ref1": "B", "ref2": "C",
             "hour": 3}
        ]}
        tgt = {"trr": [
            {"target": "A", "ref1": "B", "ref2": "C",
             "hour": 3}
        ]}
        r = self.c.compare(pred, tgt, ["trr"])
        assert r.n_correct == 1

    def test_trr_shift(self):
        pred = {"trr": [
            {"target": "A", "ref1": "B", "ref2": "C",
             "hour": 4}
        ]}
        tgt = {"trr": [
            {"target": "A", "ref1": "B", "ref2": "C",
             "hour": 3}
        ]}
        r = self.c.compare(pred, tgt, ["trr"])
        assert r.n_wrong == 1
        m = r.matches[0]
        assert m.details["sub_type"] == "shift"


# -------------------------------------------------------
# Multi-type + GT Annotation
# -------------------------------------------------------

class TestMultiType:

    def test_multiple_types(self):
        pred = {
            "qrr": [{"pair1": ["A", "B"], "pair2": ["C", "D"],
                      "metric": "dist3D", "comparator": "<"}],
            "axial": [{"obj1": "A", "obj2": "B",
                       "relation": "left_of"}],
            "occlusion": [{"occluder": "X", "occluded": "Y"}],
        }
        tgt = {
            "qrr": [{"pair1": ["A", "B"], "pair2": ["C", "D"],
                      "metric": "dist3D", "comparator": ">"}],
            "axial": [{"obj1": "A", "obj2": "B",
                       "relation": "right_of"}],
            "size": [{"bigger": "A", "smaller": "B"}],
        }
        c = ConstraintComparator()
        r = c.compare(pred, tgt)
        # qrr: flip → wrong
        # axial: flip → wrong
        # size: missing
        # occlusion: hallucinated
        assert r.n_wrong == 2
        assert r.n_missing == 1
        assert r.n_hallucinated == 1
        assert "qrr" in r.by_type
        assert "axial" in r.by_type

    def test_gt_annotation(self):
        """hallucinated 约束附注 gt_exists。"""
        pred = {"qrr": [
            {"pair1": ["X", "Y"], "pair2": ["W", "Z"],
             "metric": "dist3D", "comparator": "<"}
        ]}
        tgt = {"qrr": []}
        gt = {"qrr": [
            {"pair1": ["X", "Y"], "pair2": ["W", "Z"],
             "metric": "dist3D", "comparator": "<"}
        ]}
        c = ConstraintComparator(gt_for_annotation=gt)
        r = c.compare(pred, tgt, ["qrr"])
        assert r.n_hallucinated == 1
        h = [m for m in r.matches
             if m.category == "hallucinated"][0]
        assert h.details["gt_exists"] is True
        assert h.details["gt_match"] is True

    def test_gt_annotation_no_match(self):
        """GT 中存在但值不同。"""
        pred = {"qrr": [
            {"pair1": ["X", "Y"], "pair2": ["W", "Z"],
             "metric": "dist3D", "comparator": "<"}
        ]}
        tgt = {"qrr": []}
        gt = {"qrr": [
            {"pair1": ["X", "Y"], "pair2": ["W", "Z"],
             "metric": "dist3D", "comparator": ">"}
        ]}
        c = ConstraintComparator(gt_for_annotation=gt)
        r = c.compare(pred, tgt, ["qrr"])
        h = [m for m in r.matches
             if m.category == "hallucinated"][0]
        assert h.details["gt_exists"] is True
        assert h.details["gt_match"] is False


# -------------------------------------------------------
# Batch + Serialization
# -------------------------------------------------------

class TestBatchAndSerialization:

    def test_batch(self):
        c = ConstraintComparator()
        preds = [
            {"qrr": [{"pair1": ["A", "B"],
                       "pair2": ["C", "D"],
                       "metric": "dist3D",
                       "comparator": "<"}]},
            {"qrr": []},
        ]
        tgts = [
            {"qrr": [{"pair1": ["A", "B"],
                       "pair2": ["C", "D"],
                       "metric": "dist3D",
                       "comparator": "<"}]},
            {"qrr": [{"pair1": ["E", "F"],
                       "pair2": ["G", "H"],
                       "metric": "dist3D",
                       "comparator": ">"}]},
        ]
        r = c.compare_batch(preds, tgts, ["s1", "s2"],
                            ["qrr"])
        assert r.n_correct == 1
        assert r.n_missing == 1

    def test_to_dict(self):
        c = ConstraintComparator()
        pred = {"qrr": [
            {"pair1": ["A", "B"], "pair2": ["C", "D"],
             "metric": "dist3D", "comparator": "<"}
        ]}
        tgt = {"qrr": [
            {"pair1": ["A", "B"], "pair2": ["C", "D"],
             "metric": "dist3D", "comparator": "<"}
        ]}
        r = c.compare(pred, tgt, ["qrr"])
        d = r.to_dict()
        assert d["n_correct"] == 1
        assert d["f1"] == 1.0

    def test_empty_comparison(self):
        c = ConstraintComparator()
        r = c.compare({}, {})
        assert r.n_correct == 0
        assert r.n_missing == 0
        assert r.f1 == 0.0

    def test_summary(self):
        c = ConstraintComparator()
        r = c.compare(
            {"qrr": [{"pair1": ["A", "B"],
                       "pair2": ["C", "D"],
                       "metric": "dist3D",
                       "comparator": "<"}]},
            {"qrr": [{"pair1": ["A", "B"],
                       "pair2": ["C", "D"],
                       "metric": "dist3D",
                       "comparator": "<"}]},
            ["qrr"],
        )
        s = r.summary()
        assert "F1:1.000" in s
