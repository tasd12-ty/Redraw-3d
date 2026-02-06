"""
VLM Constraint Agent 单元测试。

Unit tests for VLM Constraint Agent.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock

from ordinal_spatial.agents.base import ConstraintSet, ObjectInfo
from ordinal_spatial.agents.vlm_constraint_agent import (
    VLMConstraintAgent,
    VLMAgentConfig,
)
from ordinal_spatial.agents.prompts.constraint_extraction import (
    build_single_view_prompt,
    build_multi_view_prompt,
    get_system_prompt,
)
from ordinal_spatial.dsl.schema import (
    AxialConstraint,
    AxialRelation,
    SizeConstraint,
    CloserConstraint,
    TopologyConstraint,
    OcclusionConstraint,
    QRRConstraintSchema,
)


# =============================================================================
# Test ConstraintSet
# =============================================================================

class TestConstraintSet:
    """Tests for ConstraintSet data class."""

    def test_empty_constraint_set(self):
        """Test creating an empty constraint set."""
        cs = ConstraintSet()
        assert cs.objects == []
        assert cs.total_constraints() == 0
        assert cs.confidence == 0.0

    def test_constraint_set_with_objects(self):
        """Test constraint set with objects."""
        obj1 = ObjectInfo(id="cube1", type="cube", color="red", size_class="large")
        obj2 = ObjectInfo(id="sphere1", type="sphere", color="blue", size_class="small")

        cs = ConstraintSet(objects=[obj1, obj2], confidence=0.9)
        assert len(cs.objects) == 2
        assert cs.confidence == 0.9

    def test_constraint_set_with_constraints(self):
        """Test constraint set with various constraint types."""
        axial = AxialConstraint(obj1="a", obj2="b", relation=AxialRelation.LEFT_OF)
        size = SizeConstraint(bigger="a", smaller="b")
        closer = CloserConstraint(anchor="a", closer="b", farther="c")
        topology = TopologyConstraint(obj1="a", obj2="b", relation="disjoint")

        cs = ConstraintSet(
            axial=[axial],
            size=[size],
            closer=[closer],
            topology=[topology],
        )

        assert len(cs.axial) == 1
        assert len(cs.size) == 1
        assert len(cs.closer) == 1
        assert len(cs.topology) == 1
        assert cs.total_constraints() == 4

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        obj = ObjectInfo(id="cube1", type="cube", color="red", size_class="large")
        axial = AxialConstraint(obj1="cube1", obj2="sphere1", relation=AxialRelation.LEFT_OF)

        cs = ConstraintSet(objects=[obj], axial=[axial], confidence=0.85)

        # Convert to dict
        data = cs.to_dict()
        assert "objects" in data
        assert "constraints" in data
        assert data["confidence"] == 0.85

        # Convert back
        cs2 = ConstraintSet.from_dict(data)
        assert len(cs2.objects) == 1
        assert len(cs2.axial) == 1
        assert cs2.confidence == 0.85

    def test_summary(self):
        """Test summary generation."""
        cs = ConstraintSet(
            objects=[ObjectInfo(id="a", type="cube", color="red")],
            axial=[AxialConstraint(obj1="a", obj2="b", relation=AxialRelation.LEFT_OF)],
            confidence=0.9,
        )
        summary = cs.summary()
        assert "Objects: 1" in summary
        assert "Axial:    1" in summary
        assert "Confidence: 0.90" in summary

    def test_count_by_arity(self):
        """Test arity-based counting with expected combinatorial values."""
        objs = [
            ObjectInfo(id="a", type="cube", color="red"),
            ObjectInfo(id="b", type="sphere", color="blue"),
            ObjectInfo(id="c", type="cylinder", color="green"),
            ObjectInfo(id="d", type="cube", color="yellow"),
            ObjectInfo(id="e", type="sphere", color="gray"),
        ]
        cs = ConstraintSet(
            objects=objs,
            axial=[AxialConstraint(obj1="a", obj2="b", relation=AxialRelation.LEFT_OF)],
            topology=[TopologyConstraint(obj1="a", obj2="b", relation="disjoint")],
            size=[SizeConstraint(bigger="a", smaller="b")],
            closer=[CloserConstraint(anchor="a", closer="b", farther="c")],
            qrr=[QRRConstraintSchema(pair1=["a", "b"], pair2=["c", "d"], metric="dist3D", comparator="<")],
        )
        counts = cs.count_by_arity()

        # N=5
        assert counts["n_objects"] == 5
        # C(5,2)=10
        assert counts["binary"]["expected_pairs"] == 10
        # C(5,3)=10
        assert counts["ternary"]["expected_triples"] == 10
        # 3*C(5,4)=15
        assert counts["quaternary"]["expected_pure_qrr"] == 15

        # Actual counts
        assert counts["binary"]["axial"] == 1
        assert counts["binary"]["topology"] == 1
        assert counts["binary"]["size"] == 1
        assert counts["binary"]["total"] == 3
        assert counts["ternary"]["closer"] == 1
        assert counts["ternary"]["total"] == 1
        assert counts["quaternary"]["qrr"] == 1
        assert counts["grand_total"] == 5

    def test_count_by_arity_small(self):
        """Test arity counting with 3 objects (no QRR possible)."""
        objs = [
            ObjectInfo(id="a", type="cube", color="red"),
            ObjectInfo(id="b", type="sphere", color="blue"),
            ObjectInfo(id="c", type="cylinder", color="green"),
        ]
        cs = ConstraintSet(objects=objs)
        counts = cs.count_by_arity()

        assert counts["n_objects"] == 3
        assert counts["binary"]["expected_pairs"] == 3  # C(3,2)
        assert counts["ternary"]["expected_triples"] == 1  # C(3,3)
        assert counts["quaternary"]["expected_pure_qrr"] == 0  # N<4

    def test_to_dict_includes_counts(self):
        """Test that to_dict includes counts field."""
        cs = ConstraintSet(
            objects=[ObjectInfo(id="a", type="cube", color="red")],
        )
        data = cs.to_dict()
        assert "counts" in data
        assert "binary" in data["counts"]
        assert "ternary" in data["counts"]
        assert "quaternary" in data["counts"]


# =============================================================================
# Test Prompt Building
# =============================================================================

class TestPromptBuilding:
    """Tests for prompt building functions."""

    def test_get_system_prompt_single(self):
        """Test single-view system prompt."""
        prompt = get_system_prompt("single", tau=0.10)
        assert "Single-View Constraint Extraction" in prompt
        assert "tau" in prompt.lower()

    def test_get_system_prompt_multi(self):
        """Test multi-view system prompt."""
        prompt = get_system_prompt("multi", tau=0.10)
        assert "Multi-View Constraint Extraction" in prompt

    def test_build_single_view_prompt(self):
        """Test building single-view prompt."""
        objects = [
            {"id": "cube1", "color": "red", "shape": "cube", "size": "large"},
            {"id": "sphere1", "color": "blue", "shape": "sphere", "size": "small"},
        ]

        prompts = build_single_view_prompt(objects, tau=0.10)

        assert "system" in prompts
        assert "user" in prompts
        assert "cube1" in prompts["user"]
        assert "red" in prompts["user"]

    def test_build_single_view_prompt_no_objects(self):
        """Test building prompt without known objects."""
        prompts = build_single_view_prompt(None, tau=0.10)

        assert "system" in prompts
        assert "user" in prompts
        assert "identify all visible objects" in prompts["user"].lower()

    def test_build_multi_view_prompt(self):
        """Test building multi-view prompt."""
        prompts = build_multi_view_prompt(n_views=3, objects=None, tau=0.10)

        assert "system" in prompts
        assert "user" in prompts
        assert "3" in prompts["user"]


# =============================================================================
# Test VLM Agent
# =============================================================================

class TestVLMAgent:
    """Tests for VLMConstraintAgent."""

    def test_config_defaults(self):
        """Test default configuration."""
        config = VLMAgentConfig()
        assert config.model == "google/gemma-3-27b-it"
        assert config.temperature == 0.0
        assert config.retry_count == 3

    def test_normalize_comparator(self):
        """Test comparator normalization."""
        agent = VLMConstraintAgent()

        assert agent._normalize_comparator("<") == "<"
        assert agent._normalize_comparator("lt") == "<"
        assert agent._normalize_comparator("less") == "<"
        assert agent._normalize_comparator(">") == ">"
        assert agent._normalize_comparator("gt") == ">"
        assert agent._normalize_comparator("~=") == "~="
        assert agent._normalize_comparator("approx") == "~="
        assert agent._normalize_comparator("equal") == "~="

    def test_extract_json_direct(self):
        """Test JSON extraction from clean JSON."""
        agent = VLMConstraintAgent()
        text = '{"objects": [], "constraints": {}, "confidence": 0.9}'
        data = agent._extract_json(text)
        assert data["confidence"] == 0.9

    def test_extract_json_markdown(self):
        """Test JSON extraction from markdown code block."""
        agent = VLMConstraintAgent()
        text = '''Here is the result:
```json
{"objects": [], "constraints": {}, "confidence": 0.85}
```
'''
        data = agent._extract_json(text)
        assert data["confidence"] == 0.85

    def test_build_constraint_set(self):
        """Test building constraint set from parsed data."""
        agent = VLMConstraintAgent()

        data = {
            "objects": [
                {"id": "cube1", "type": "cube", "color": "red", "size_class": "large"},
                {"id": "sphere1", "type": "sphere", "color": "blue", "size_class": "small"},
            ],
            "constraints": {
                "axial": [
                    {"obj1": "cube1", "obj2": "sphere1", "relation": "left_of"}
                ],
                "size": [
                    {"bigger": "cube1", "smaller": "sphere1"}
                ],
                "closer": [
                    {"anchor": "cube1", "closer": "sphere1", "farther": "cone1"}
                ],
                "topology": [
                    {"obj1": "cube1", "obj2": "sphere1", "relation": "disjoint"}
                ],
                "occlusion": [],
                "qrr": [],
                "trr": [],
            },
            "confidence": 0.9,
        }

        cs = agent._build_constraint_set(data)

        assert len(cs.objects) == 2
        assert len(cs.axial) == 1
        assert len(cs.size) == 1
        assert len(cs.closer) == 1
        assert len(cs.topology) == 1
        assert cs.confidence == 0.9

    def test_validate_constraints_axial_consistency(self):
        """Test axial constraint consistency validation."""
        agent = VLMConstraintAgent(VLMAgentConfig(validate_consistency=True))

        # Create constraint set with consistent axial relations
        cs = ConstraintSet(
            axial=[
                AxialConstraint(obj1="a", obj2="b", relation=AxialRelation.LEFT_OF),
            ],
        )

        agent._validate_constraints(cs)
        assert "consistency_issues" not in cs.metadata

    def test_validate_constraints_size_cycle(self):
        """Test size constraint cycle detection."""
        agent = VLMConstraintAgent(VLMAgentConfig(validate_consistency=True))

        # Create constraint set with size cycle
        cs = ConstraintSet(
            size=[
                SizeConstraint(bigger="a", smaller="b"),
                SizeConstraint(bigger="b", smaller="a"),  # Contradiction!
            ],
        )

        agent._validate_constraints(cs)
        assert "consistency_issues" in cs.metadata
        assert len(cs.metadata["consistency_issues"]) > 0

    @patch('ordinal_spatial.agents.vlm_constraint_agent.VLMConstraintAgent.client')
    def test_call_api_success(self, mock_client):
        """Test successful API call."""
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='{"objects": [], "constraints": {}, "confidence": 0.9}'))]
        mock_client.chat.completions.create.return_value = mock_response

        agent = VLMConstraintAgent()
        result = agent._call_api([{"role": "user", "content": "test"}])

        assert "objects" in result
        assert "confidence" in result


# =============================================================================
# Test Schema Integration
# =============================================================================

class TestSchemaIntegration:
    """Tests for schema type integration."""

    def test_axial_constraint_inverse(self):
        """Test axial constraint inverse."""
        ac = AxialConstraint(obj1="a", obj2="b", relation=AxialRelation.LEFT_OF)
        inv = ac.inverse()

        assert inv.obj1 == "b"
        assert inv.obj2 == "a"
        assert inv.relation == AxialRelation.RIGHT_OF

    def test_closer_to_qrr(self):
        """Test closer constraint conversion to QRR style."""
        cc = CloserConstraint(anchor="A", closer="B", farther="C")
        qrr = cc.to_qrr_style()

        assert qrr["pair1"] == ["A", "B"]
        assert qrr["pair2"] == ["A", "C"]
        assert qrr["comparator"] == "<"

    def test_size_constraint_inverse(self):
        """Test size constraint inverse."""
        sc = SizeConstraint(bigger="a", smaller="b")
        inv = sc.inverse()

        assert inv.bigger == "b"
        assert inv.smaller == "a"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
