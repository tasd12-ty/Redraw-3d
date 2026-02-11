"""
元数据工具测试。

Tests for utils/metadata.py normalization functions.
"""

import pytest
from ordinal_spatial.utils.metadata import (
    get_world_constraints,
    get_view_constraints,
    get_merged_gt,
)


class TestGetWorldConstraints:

    def test_new_format(self):
        """New {world, views} format."""
        meta = {
            "constraints": {
                "world": {"qrr": [1], "topology": [2]},
                "views": [],
            }
        }
        w = get_world_constraints(meta)
        assert w["qrr"] == [1]
        assert w["topology"] == [2]

    def test_legacy_flat_format(self):
        """Legacy flat constraints dict (no world key)."""
        meta = {
            "constraints": {
                "qrr": [1], "trr": [2],
            }
        }
        w = get_world_constraints(meta)
        assert w["qrr"] == [1]
        assert w["trr"] == [2]

    def test_empty(self):
        w = get_world_constraints({})
        assert w == {}


class TestGetViewConstraints:

    def test_existing_view(self):
        meta = {
            "constraints": {
                "world": {},
                "views": [
                    {"axial_2d": [1]},
                    {"axial_2d": [2]},
                ],
            }
        }
        v = get_view_constraints(meta, 1)
        assert v["axial_2d"] == [2]

    def test_out_of_range(self):
        meta = {"constraints": {"views": []}}
        v = get_view_constraints(meta, 5)
        assert v == {}

    def test_no_views_key(self):
        meta = {"constraints": {"qrr": []}}
        v = get_view_constraints(meta, 0)
        assert v == {}


class TestGetMergedGt:

    def test_world_only(self):
        """view_idx=None → only world constraints."""
        meta = {
            "constraints": {
                "world": {
                    "qrr": [1], "topology": [2], "size": [],
                    "closer": [], "axial": [3],
                    "occlusion": [4], "trr": [5],
                },
                "views": [
                    {"axial_2d": [99], "occlusion": [88],
                     "trr": [77]},
                ],
            }
        }
        gt = get_merged_gt(meta)
        assert gt["qrr"] == [1]
        assert gt["axial"] == [3]  # world.axial
        assert gt["occlusion"] == [4]
        assert gt["trr"] == [5]

    def test_with_view_index(self):
        """view_idx=0 → world invariant + views[0] dependent."""
        meta = {
            "constraints": {
                "world": {
                    "qrr": [1], "topology": [2], "size": [3],
                    "closer": [4], "axial": [99],  # 不应被用
                },
                "views": [
                    {"axial_2d": [10], "occlusion": [11],
                     "trr": [12]},
                ],
            }
        }
        gt = get_merged_gt(meta, view_idx=0)
        # 视角不变
        assert gt["qrr"] == [1]
        assert gt["topology"] == [2]
        # 视角相关来自 views[0]
        assert gt["axial"] == [10]
        assert gt["occlusion"] == [11]
        assert gt["trr"] == [12]

    def test_view_fallback(self):
        """view_idx out of range → fallback to world."""
        meta = {
            "constraints": {
                "world": {
                    "qrr": [], "topology": [], "size": [],
                    "closer": [], "axial": [1],
                    "occlusion": [2], "trr": [],
                },
                "views": [],
            }
        }
        gt = get_merged_gt(meta, view_idx=5)
        assert gt["axial"] == [1]
        assert gt["occlusion"] == [2]
