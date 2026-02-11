"""
视角相关约束的单元测试。

Tests for view-specific constraint extraction, diff, and GT loading.
"""

import pytest
from ordinal_spatial.generation.constraint_extractor import (
    ConstraintExtractor,
    ExtractionConfig,
    extract_axial_2d_from_view,
    extract_occlusion_from_view,
)
from ordinal_spatial.dsl.schema import (
    AxialRelation,
)
from ordinal_spatial.evaluation.constraint_diff import (
    _compute_occlusion_diff,
    compute_grouped_constraint_diff,
)
from ordinal_spatial.tasks.t2_extraction import get_gt_for_view


# -------------------------------------------------------
# Fixtures
# -------------------------------------------------------

def _make_objects(specs):
    """Helper to build objects_for_view dict from compact specs.

    specs: list of (obj_id, px, py, depth, size)
    """
    objs = {}
    for obj_id, px, py, depth, size in specs:
        objs[obj_id] = {
            "pixel_coords": [px, py, depth],
            "size": size,
        }
    return objs


# -------------------------------------------------------
# extract_axial_2d_from_view
# -------------------------------------------------------

class TestExtractAxial2D:

    def test_left_right(self):
        """obj_0 (px=100) LEFT_OF obj_1 (px=350)."""
        objs = _make_objects([
            ("obj_0", 100, 200, 0.5, "large"),
            ("obj_1", 350, 200, 0.5, "large"),
        ])
        axials = extract_axial_2d_from_view(
            objs, tau=0.10, image_width=480
        )
        lr = [a for a in axials
              if a.relation in (AxialRelation.LEFT_OF,
                                AxialRelation.RIGHT_OF)]
        assert len(lr) == 1
        assert lr[0].obj1 == "obj_0"
        assert lr[0].obj2 == "obj_1"
        assert lr[0].relation == AxialRelation.LEFT_OF

    def test_above_below(self):
        """obj_0 (py=50) ABOVE obj_1 (py=280) — y-down."""
        objs = _make_objects([
            ("obj_0", 200, 50, 0.5, "medium"),
            ("obj_1", 200, 280, 0.5, "medium"),
        ])
        axials = extract_axial_2d_from_view(
            objs, tau=0.10, image_height=320
        )
        ud = [a for a in axials
              if a.relation in (AxialRelation.ABOVE,
                                AxialRelation.BELOW)]
        assert len(ud) == 1
        assert ud[0].relation == AxialRelation.ABOVE

    def test_in_front_of(self):
        """obj_0 (depth=0.3) IN_FRONT_OF obj_1 (depth=0.8)."""
        objs = _make_objects([
            ("obj_0", 200, 200, 0.3, "small"),
            ("obj_1", 200, 200, 0.8, "small"),
        ])
        axials = extract_axial_2d_from_view(objs, tau=0.10)
        fb = [a for a in axials
              if a.relation in (AxialRelation.IN_FRONT_OF,
                                AxialRelation.BEHIND)]
        assert len(fb) == 1
        assert fb[0].relation == AxialRelation.IN_FRONT_OF

    def test_tau_threshold_skips(self):
        """Within tau threshold → no constraint emitted."""
        # tau=0.10, image_width=480 → tau_px=48
        # dx = 40 < 48 → skipped
        objs = _make_objects([
            ("obj_0", 200, 200, 0.50, "medium"),
            ("obj_1", 240, 200, 0.50, "medium"),
        ])
        axials = extract_axial_2d_from_view(
            objs, tau=0.10, image_width=480
        )
        lr = [a for a in axials
              if a.relation in (AxialRelation.LEFT_OF,
                                AxialRelation.RIGHT_OF)]
        assert len(lr) == 0

    def test_missing_pixel_coords_skipped(self):
        """Objects without pixel_coords are skipped."""
        objs = {
            "obj_0": {"pixel_coords": [100, 200, 0.5], "size": "large"},
            "obj_1": {"size": "large"},  # 无 pixel_coords
        }
        axials = extract_axial_2d_from_view(objs)
        assert len(axials) == 0

    def test_view_flip(self):
        """Different views produce flipped L/R."""
        # View 0: obj_0 左, obj_1 右
        v0 = _make_objects([
            ("obj_0", 100, 200, 0.5, "large"),
            ("obj_1", 380, 200, 0.5, "large"),
        ])
        # View 1: obj_0 右, obj_1 左 (相机旋转 180°)
        v1 = _make_objects([
            ("obj_0", 380, 200, 0.5, "large"),
            ("obj_1", 100, 200, 0.5, "large"),
        ])

        a0 = extract_axial_2d_from_view(v0, tau=0.10)
        a1 = extract_axial_2d_from_view(v1, tau=0.10)

        lr0 = [a for a in a0
               if a.relation in (AxialRelation.LEFT_OF,
                                 AxialRelation.RIGHT_OF)]
        lr1 = [a for a in a1
               if a.relation in (AxialRelation.LEFT_OF,
                                 AxialRelation.RIGHT_OF)]

        assert len(lr0) == 1 and len(lr1) == 1
        assert lr0[0].relation == AxialRelation.LEFT_OF
        assert lr1[0].relation == AxialRelation.RIGHT_OF


# -------------------------------------------------------
# extract_occlusion_from_view
# -------------------------------------------------------

class TestExtractOcclusion:

    def test_close_objects_with_depth_diff(self):
        """Overlapping + depth gap → occlusion detected."""
        objs = _make_objects([
            ("obj_0", 200, 200, 0.3, "large"),   # 近
            ("obj_1", 220, 210, 0.7, "large"),    # 远, 像素距离 ~22
        ])
        # large radius=60, 所以 60+60=120 > 22 → 重叠
        occ = extract_occlusion_from_view(objs, tau_depth=0.05)
        assert len(occ) == 1
        assert occ[0].occluder == "obj_0"
        assert occ[0].occluded == "obj_1"
        assert occ[0].partial is True

    def test_far_apart_no_occlusion(self):
        """Objects far apart in pixel space → no occlusion."""
        objs = _make_objects([
            ("obj_0", 50, 50, 0.3, "small"),
            ("obj_1", 400, 300, 0.8, "small"),
        ])
        # small radius=25, 距离远大于 25+25=50
        occ = extract_occlusion_from_view(objs, tau_depth=0.05)
        assert len(occ) == 0

    def test_same_depth_no_occlusion(self):
        """Overlapping but same depth → no occlusion."""
        objs = _make_objects([
            ("obj_0", 200, 200, 0.50, "large"),
            ("obj_1", 210, 205, 0.51, "large"),  # 仅 0.01 差
        ])
        occ = extract_occlusion_from_view(objs, tau_depth=0.05)
        assert len(occ) == 0


# -------------------------------------------------------
# _build_view_objects
# -------------------------------------------------------

class TestBuildViewObjects:

    def test_override_position_2d(self):
        """position_2d and depth overridden correctly."""
        config = ExtractionConfig(tau=0.10)
        ext = ConstraintExtractor(config)

        base = {
            "obj_0": {
                "position_3d": [1, 2, 0],
                "position_2d": [100, 200],
            },
            "obj_1": {
                "position_3d": [-1, 3, 0],
                "position_2d": [300, 150],
            },
        }
        view_data = {
            "camera": {"camera_id": "view_1"},
            "objects": [
                {"id": "obj_0", "pixel_coords": [380, 210, 0.45]},
                {"id": "obj_1", "pixel_coords": [120, 170, 0.55]},
            ],
        }

        result = ext._build_view_objects(base, view_data)
        assert result["obj_0"]["position_2d"] == [380, 210]
        assert result["obj_0"]["depth"] == 0.45
        assert result["obj_1"]["pixel_coords"] == [120, 170, 0.55]
        # 原始 base 不被修改
        assert base["obj_0"]["position_2d"] == [100, 200]


# -------------------------------------------------------
# _extract_view_constraints 集成
# -------------------------------------------------------

class TestExtractViewConstraintsIntegration:

    def test_axial_2d_and_occlusion_populated(self):
        """_extract_view_constraints fills axial_2d and occlusion."""
        scene = {
            "scene_id": "test_001",
            "objects": [
                {
                    "id": "obj_0", "shape": "cube", "color": "red",
                    "size": "large",
                    "position_3d": [1, 2, 0],
                    "position_2d": [100, 200],
                },
                {
                    "id": "obj_1", "shape": "sphere", "color": "blue",
                    "size": "small",
                    "position_3d": [-1, 3, 0],
                    "position_2d": [300, 150],
                },
            ],
            "views": [
                {
                    "camera": {"camera_id": "view_0"},
                    "objects": [
                        {"id": "obj_0",
                         "pixel_coords": [100, 200, 0.4]},
                        {"id": "obj_1",
                         "pixel_coords": [350, 180, 0.6]},
                    ],
                },
                {
                    "camera": {"camera_id": "view_1"},
                    "objects": [
                        {"id": "obj_0",
                         "pixel_coords": [380, 210, 0.5]},
                        {"id": "obj_1",
                         "pixel_coords": [120, 170, 0.55]},
                    ],
                },
            ],
        }
        config = ExtractionConfig(tau=0.10)
        ext = ConstraintExtractor(config)
        osd = ext.extract(scene)

        assert len(osd.views) == 2
        # View 0: obj_0(px=100) LEFT_OF obj_1(px=350)
        v0_lr = [
            a for a in osd.views[0].axial_2d
            if a.relation in (AxialRelation.LEFT_OF,
                              AxialRelation.RIGHT_OF)
        ]
        assert any(
            a.obj1 == "obj_0" and a.relation == AxialRelation.LEFT_OF
            for a in v0_lr
        )

        # View 1: obj_0(px=380) RIGHT_OF obj_1(px=120)
        v1_lr = [
            a for a in osd.views[1].axial_2d
            if a.relation in (AxialRelation.LEFT_OF,
                              AxialRelation.RIGHT_OF)
        ]
        assert any(
            a.obj1 == "obj_0" and a.relation == AxialRelation.RIGHT_OF
            for a in v1_lr
        )


# -------------------------------------------------------
# get_gt_for_view
# -------------------------------------------------------

class TestGetGtForView:

    def test_correct_split(self):
        """View-invariant from world, view-dependent from views."""
        scene = {
            "constraints": {
                "world": {
                    "qrr": [{"pair1": ["A", "B"], "comparator": "<"}],
                    "topology": [{"obj1": "A", "obj2": "B", "rel": "disjoint"}],
                    "size": [{"obj1": "A", "obj2": "B", "rel": "bigger"}],
                    "closer": [],
                    "axial": [{"obj1": "A", "obj2": "B", "relation": "left_of"}],
                    "occlusion": [],
                    "trr": [],
                },
                "views": [
                    {
                        "axial_2d": [{"obj1": "A", "obj2": "B", "relation": "right_of"}],
                        "occlusion": [{"occluder": "A", "occluded": "B"}],
                        "trr": [{"target": "A", "ref1": "B", "ref2": "C"}],
                    },
                    {
                        "axial_2d": [{"obj1": "A", "obj2": "B", "relation": "left_of"}],
                        "occlusion": [],
                        "trr": [],
                    },
                ],
            },
        }
        gt = get_gt_for_view(scene, view_index=0)
        # 视角不变来自 world
        assert gt["qrr"] == [{"pair1": ["A", "B"], "comparator": "<"}]
        assert gt["topology"] == [{"obj1": "A", "obj2": "B", "rel": "disjoint"}]
        # 视角相关来自 views[0]
        assert gt["axial"] == [{"obj1": "A", "obj2": "B", "relation": "right_of"}]
        assert gt["occlusion"] == [{"occluder": "A", "occluded": "B"}]

    def test_fallback_no_views(self):
        """No views → fallback to world.axial."""
        scene = {
            "constraints": {
                "world": {
                    "qrr": [],
                    "topology": [],
                    "size": [],
                    "closer": [],
                    "axial": [{"obj1": "A", "obj2": "B", "relation": "left_of"}],
                    "occlusion": [],
                    "trr": [],
                },
            },
        }
        gt = get_gt_for_view(scene, view_index=0)
        assert gt["axial"] == [{"obj1": "A", "obj2": "B", "relation": "left_of"}]
        assert gt["occlusion"] == []

    def test_view_index_out_of_range(self):
        """view_index beyond available views → fallback."""
        scene = {
            "constraints": {
                "world": {
                    "qrr": [], "topology": [], "size": [],
                    "closer": [],
                    "axial": [{"rel": "world"}],
                    "occlusion": [{"rel": "world"}],
                    "trr": [],
                },
                "views": [
                    {"axial_2d": [{"rel": "v0"}], "occlusion": [], "trr": []},
                ],
            },
        }
        gt = get_gt_for_view(scene, view_index=5)
        assert gt["axial"] == [{"rel": "world"}]


# -------------------------------------------------------
# _compute_occlusion_diff
# -------------------------------------------------------

class TestComputeOcclusionDiff:

    def test_perfect_match(self):
        gt = [{"occluder": "A", "occluded": "B"}]
        pred = [{"occluder": "A", "occluded": "B"}]
        m = _compute_occlusion_diff(pred, gt)
        assert m.n_correct == 1
        assert m.n_missing == 0
        assert m.n_spurious == 0
        assert m.n_violated == 0
        assert m.f1 == 1.0

    def test_missing(self):
        gt = [
            {"occluder": "A", "occluded": "B"},
            {"occluder": "C", "occluded": "D"},
        ]
        pred = [{"occluder": "A", "occluded": "B"}]
        m = _compute_occlusion_diff(pred, gt)
        assert m.n_correct == 1
        assert m.n_missing == 1

    def test_violated_reversal(self):
        """Pred has (B,A), GT has (A,B) → violated, not spurious."""
        gt = [{"occluder": "A", "occluded": "B"}]
        pred = [{"occluder": "B", "occluded": "A"}]
        m = _compute_occlusion_diff(pred, gt)
        assert m.n_correct == 0
        assert m.n_violated == 1
        assert m.n_spurious == 0
        assert m.n_missing == 1

    def test_spurious(self):
        """Pred has extra constraint not in GT."""
        gt = []
        pred = [{"occluder": "X", "occluded": "Y"}]
        m = _compute_occlusion_diff(pred, gt)
        assert m.n_spurious == 1
        assert m.n_correct == 0


# -------------------------------------------------------
# compute_grouped_constraint_diff
# -------------------------------------------------------

class TestGroupedConstraintDiff:

    def test_returns_three_groups(self):
        pred = {
            "qrr": [], "axial": [], "topology": [],
            "size": [], "occlusion": [],
        }
        gt = {
            "qrr": [], "axial": [], "topology": [],
            "size": [], "occlusion": [],
        }
        result = compute_grouped_constraint_diff(pred, gt)
        assert "view_invariant" in result
        assert "view_dependent" in result
        assert "overall" in result

    def test_vi_vs_vd_separation(self):
        """Axial error only affects view_dependent, not view_invariant."""
        pred = {
            "qrr": [{"pair1": ["A", "B"], "pair2": ["C", "D"],
                      "metric": "dist3D", "comparator": "<"}],
            "axial": [{"obj1": "A", "obj2": "B",
                       "relation": "right_of"}],  # 错误
            "topology": [],
            "size": [],
            "occlusion": [],
        }
        gt = {
            "qrr": [{"pair1": ["A", "B"], "pair2": ["C", "D"],
                      "metric": "dist3D", "comparator": "<"}],
            "axial": [{"obj1": "A", "obj2": "B",
                       "relation": "left_of"}],
            "topology": [],
            "size": [],
            "occlusion": [],
        }
        result = compute_grouped_constraint_diff(pred, gt)
        # QRR 完全正确 → view_invariant F1 = 1.0
        assert result["view_invariant"].n_correct == 1
        # Axial 全部违反 → view_dependent 正确为 0
        assert result["view_dependent"].n_correct == 0
