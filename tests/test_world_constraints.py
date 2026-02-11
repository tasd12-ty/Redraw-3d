"""
世界约束提取测试。

Tests for world-level (3D) constraint extraction functions added
to generation/constraint_extractor.py.
"""

import pytest
from ordinal_spatial.generation.constraint_extractor import (
    ConstraintExtractor,
    ExtractionConfig,
    extract_topology_from_scene,
    extract_size_from_scene,
    extract_closer_from_scene,
    extract_axial_3d_from_scene,
    extract_occlusion_3d_from_scene,
)
from ordinal_spatial.dsl.schema import AxialRelation


# -------------------------------------------------------
# Fixtures
# -------------------------------------------------------

def _objs_3d(specs):
    """Build objects dict from compact specs.

    Each spec: (id, x, y, z, size).
    """
    d = {}
    for obj_id, x, y, z, size in specs:
        d[obj_id] = {
            "id": obj_id,
            "position_3d": [x, y, z],
            "position": [x, y, z],
            "size": size,
        }
    return d


# -------------------------------------------------------
# Topology
# -------------------------------------------------------

class TestExtractTopology:

    def test_far_apart_disjoint(self):
        objs = _objs_3d([
            ("A", 0, 0, 0, "small"),
            ("B", 10, 10, 0, "small"),
        ])
        cs = extract_topology_from_scene(objs)
        assert len(cs) == 1
        assert cs[0].relation == "disjoint"

    def test_close_touching(self):
        # radius(medium) = 0.5*0.5 = 0.25
        # combined = 0.5, dist ≈ 0.45 → 0.8*0.5=0.4 < 0.45 < 0.6=1.2*0.5
        objs = _objs_3d([
            ("A", 0, 0, 0, "medium"),
            ("B", 0.45, 0, 0, "medium"),
        ])
        cs = extract_topology_from_scene(objs)
        assert len(cs) == 1
        assert cs[0].relation == "touching"

    def test_overlapping(self):
        # Same position → dist=0 → overlapping
        objs = _objs_3d([
            ("A", 0, 0, 0, "large"),
            ("B", 0.1, 0, 0, "large"),
        ])
        cs = extract_topology_from_scene(objs)
        assert len(cs) == 1
        assert cs[0].relation == "overlapping"


# -------------------------------------------------------
# Size
# -------------------------------------------------------

class TestExtractSize:

    def test_different_sizes(self):
        objs = _objs_3d([
            ("A", 0, 0, 0, "large"),
            ("B", 5, 0, 0, "small"),
        ])
        cs = extract_size_from_scene(objs)
        assert len(cs) == 1
        assert cs[0].bigger == "A"
        assert cs[0].smaller == "B"

    def test_same_size_no_constraint(self):
        objs = _objs_3d([
            ("A", 0, 0, 0, "medium"),
            ("B", 5, 0, 0, "medium"),
        ])
        cs = extract_size_from_scene(objs)
        assert len(cs) == 0

    def test_three_objects(self):
        objs = _objs_3d([
            ("A", 0, 0, 0, "large"),
            ("B", 3, 0, 0, "medium"),
            ("C", 6, 0, 0, "small"),
        ])
        cs = extract_size_from_scene(objs)
        # A>B, A>C, B>C
        assert len(cs) == 3


# -------------------------------------------------------
# Closer
# -------------------------------------------------------

class TestExtractCloser:

    def test_basic_closer(self):
        objs = _objs_3d([
            ("A", 0, 0, 0, "medium"),
            ("B", 1, 0, 0, "medium"),
            ("C", 10, 0, 0, "medium"),
        ])
        cs = extract_closer_from_scene(objs)
        # A → B closer than C (d=1 vs d=10)
        anchor_a = [c for c in cs if c.anchor == "A"]
        assert any(
            c.closer == "B" and c.farther == "C"
            for c in anchor_a
        )

    def test_equidistant_no_constraint(self):
        """Approx equal distances → no closer constraint."""
        objs = _objs_3d([
            ("A", 0, 0, 0, "medium"),
            ("B", 1, 0, 0, "medium"),
            ("C", -1, 0, 0, "medium"),
        ])
        cs = extract_closer_from_scene(objs)
        # B and C are equidistant from A → no A→B>C
        anchor_a = [
            c for c in cs
            if c.anchor == "A"
            and set([c.closer, c.farther]) == {"B", "C"}
        ]
        assert len(anchor_a) == 0


# -------------------------------------------------------
# Axial 3D
# -------------------------------------------------------

class TestExtractAxial3D:

    def test_left_right(self):
        objs = _objs_3d([
            ("A", -5, 0, 0, "medium"),
            ("B", 5, 0, 0, "medium"),
        ])
        cs = extract_axial_3d_from_scene(objs)
        lr = [c for c in cs if c.relation in (
            AxialRelation.LEFT_OF, AxialRelation.RIGHT_OF
        )]
        assert len(lr) == 1
        assert lr[0].obj1 == "A"
        assert lr[0].relation == AxialRelation.LEFT_OF

    def test_above_below(self):
        objs = _objs_3d([
            ("A", 0, 0, 5, "medium"),
            ("B", 0, 0, -5, "medium"),
        ])
        cs = extract_axial_3d_from_scene(objs)
        ab = [c for c in cs if c.relation in (
            AxialRelation.ABOVE, AxialRelation.BELOW
        )]
        assert len(ab) == 1
        assert ab[0].obj1 == "A"
        assert ab[0].relation == AxialRelation.ABOVE

    def test_front_behind(self):
        objs = _objs_3d([
            ("A", 0, -5, 0, "medium"),  # Y 小 = 前
            ("B", 0, 5, 0, "medium"),
        ])
        cs = extract_axial_3d_from_scene(objs)
        fb = [c for c in cs if c.relation in (
            AxialRelation.IN_FRONT_OF, AxialRelation.BEHIND
        )]
        assert len(fb) == 1
        assert fb[0].obj1 == "A"
        assert fb[0].relation == AxialRelation.IN_FRONT_OF

    def test_within_tau_no_constraint(self):
        """差异小于 tau → 不生成约束。"""
        objs = _objs_3d([
            ("A", 0, 0, 0, "medium"),
            ("B", 0.05, 0, 0, "medium"),
        ])
        cs = extract_axial_3d_from_scene(objs)
        assert len(cs) == 0


# -------------------------------------------------------
# Occlusion 3D
# -------------------------------------------------------

class TestExtractOcclusion3D:

    def test_overlapping_with_depth_diff(self):
        """X/Z 重叠 + Y 深度差 → 遮挡。"""
        objs = _objs_3d([
            ("A", 0, -5, 0, "medium"),   # A 在前
            ("B", 0.5, 5, 0, "medium"),   # B 在后, X/Z 近
        ])
        cs = extract_occlusion_3d_from_scene(objs)
        assert len(cs) == 1
        assert cs[0].occluder == "A"
        assert cs[0].occluded == "B"

    def test_far_apart_no_occlusion(self):
        """X 差距大 → 不遮挡。"""
        objs = _objs_3d([
            ("A", 0, -5, 0, "medium"),
            ("B", 5, 5, 0, "medium"),
        ])
        cs = extract_occlusion_3d_from_scene(objs)
        assert len(cs) == 0

    def test_same_depth_no_occlusion(self):
        """Y 差距小 → 不遮挡。"""
        objs = _objs_3d([
            ("A", 0, 0, 0, "medium"),
            ("B", 0.5, 0.05, 0, "medium"),
        ])
        cs = extract_occlusion_3d_from_scene(objs)
        assert len(cs) == 0


# -------------------------------------------------------
# Full ConstraintExtractor integration
# -------------------------------------------------------

class TestConstraintExtractorWorldFull:

    def test_extract_all_7_types(self):
        """ConstraintExtractor.extract() 输出包含全部 7 类 world 约束。"""
        scene = {
            "scene_id": "test_scene",
            "objects": [
                {"id": "A", "shape": "cube", "color": "red",
                 "size": "large", "position_3d": [-3, -2, 0]},
                {"id": "B", "shape": "sphere", "color": "blue",
                 "size": "small", "position_3d": [3, 2, 0]},
                {"id": "C", "shape": "cylinder", "color": "green",
                 "size": "medium", "position_3d": [0, 0, 2]},
            ],
        }
        config = ExtractionConfig(
            tau=0.10,
            include_qrr=True,
            include_trr=True,
            metrics=[],  # 空 → 不触发 QRR dist3D
        )
        from ordinal_spatial.dsl.predicates import MetricType
        config.metrics = [MetricType.DIST_3D]

        ext = ConstraintExtractor(config)
        osd = ext.extract(scene)

        w = osd.world
        # 3 objects → C(3,2)=3 pairs
        assert len(w.topology) == 3
        # size: large>medium, large>small, medium>small
        assert len(w.size) == 3
        # axial: at least some L/R, F/B relations
        assert len(w.axial) > 0
        # closer: 3 objects → permutations
        assert len(w.closer) > 0
        # trr: 3 objects → P(3,3)=6 triples
        assert len(w.trr) > 0
        # model_dump works
        d = w.model_dump()
        assert "trr" in d
        assert "occlusion" in d

    def test_world_constraints_model_dump_roundtrip(self):
        """WorldConstraints model_dump/validate 包含 trr 和 occlusion。"""
        from ordinal_spatial.dsl.schema import WorldConstraints
        w = WorldConstraints(
            qrr=[], topology=[], size=[], closer=[],
            axial=[], trr=[], occlusion=[],
        )
        d = w.model_dump()
        assert set(d.keys()) == {
            "qrr", "topology", "size", "closer",
            "axial", "trr", "occlusion",
        }
        # 可以从 dict 重建
        w2 = WorldConstraints.model_validate(d)
        assert w2 == w
