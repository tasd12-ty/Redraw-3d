"""
VLM 约束提取智能体。

VLM-based Constraint Extraction Agent.

This agent uses Vision-Language Models to extract spatial constraints
from single or multiple images according to the formal DSL.

Features:
- Single-view extraction (Task-3)
- Multi-view extraction (Task-2)
- Automatic JSON parsing with fallback
- Constraint validation and consistency checking
- Support for multiple VLM providers via OpenRouter
"""

import json
import base64
import re
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

from ordinal_spatial.agents.base import ConstraintAgent, ConstraintSet, ObjectInfo
from ordinal_spatial.agents.prompts.constraint_extraction import (
    build_single_view_prompt,
    build_multi_view_prompt,
)
from ordinal_spatial.dsl.schema import (
    QRRConstraintSchema,
    TRRConstraintSchema,
    TopologyConstraint,
    OcclusionConstraint,
    AxialConstraint,
    AxialRelation,
    SizeConstraint,
    CloserConstraint,
)


logger = logging.getLogger(__name__)


@dataclass
class VLMAgentConfig:
    """
    VLM 智能体配置。

    Configuration for VLM Constraint Agent.

    Attributes:
        model: VLM model name
        api_base: API base URL
        api_key: API key (or set OPENROUTER_API_KEY env var)
        temperature: Sampling temperature (0 for deterministic)
        max_tokens: Maximum tokens in response
        timeout: Request timeout in seconds
        retry_count: Number of retries on failure
        retry_delay: Delay between retries
        validate_consistency: Whether to check constraint consistency
    """
    model: str = "google/gemma-3-27b-it"
    api_base: str = "https://openrouter.ai/api/v1"
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = 120.0
    retry_count: int = 3
    retry_delay: float = 2.0
    validate_consistency: bool = True


class VLMConstraintAgent(ConstraintAgent):
    """
    基于 VLM 的约束提取智能体。

    VLM-based constraint extraction agent.

    Uses vision-language models to extract spatial constraints from images,
    outputting a formal Qualitative Scene Program (QSP).
    """

    def __init__(self, config: Optional[VLMAgentConfig] = None):
        """
        初始化智能体。

        Initialize the agent.

        Args:
            config: Agent configuration
        """
        self.config = config or VLMAgentConfig()
        self._client = None

    @property
    def client(self):
        """
        延迟初始化 API 客户端。

        Lazy initialization of API client.
        """
        if self._client is None:
            try:
                from openai import OpenAI
                import os

                api_key = self.config.api_key or os.environ.get("OPENROUTER_API_KEY")
                if not api_key:
                    raise ValueError(
                        "API key required. Set OPENROUTER_API_KEY env var or pass in config."
                    )

                self._client = OpenAI(
                    base_url=self.config.api_base,
                    api_key=api_key,
                    timeout=self.config.timeout,
                )
            except ImportError as e:
                raise ImportError(
                    "openai package required. Install with: pip install openai"
                ) from e

        return self._client

    def extract_from_single_view(
        self,
        image: Union[str, Path, bytes],
        objects: Optional[List[Dict[str, Any]]] = None,
        tau: float = 0.10,
    ) -> ConstraintSet:
        """
        从单视角图像提取约束 (Task-3)。

        Extract constraints from a single-view image (Task-3).

        Args:
            image: Image path, bytes, or base64 string
            objects: Optional list of known objects
            tau: Tolerance parameter

        Returns:
            ConstraintSet with extracted constraints
        """
        logger.info("Extracting constraints from single view")

        # Build prompt
        prompts = build_single_view_prompt(objects, tau)

        # Build messages with image
        messages = self._build_messages(prompts, [image])

        # Call VLM
        max_tokens = self._resolve_max_tokens(objects)
        response = self._call_api(messages, max_tokens=max_tokens)

        # Parse response
        result = self._parse_response(response)

        # Add metadata
        result.metadata["mode"] = "single_view"
        result.metadata["tau"] = tau
        result.metadata["model"] = self.config.model

        # Validate if configured
        if self.config.validate_consistency:
            self._validate_constraints(result)

        return result

    def extract_from_multi_view(
        self,
        images: List[Union[str, Path, bytes]],
        objects: Optional[List[Dict[str, Any]]] = None,
        tau: float = 0.10,
    ) -> ConstraintSet:
        """
        从多视角图像提取约束 (Task-2)。

        Extract constraints from multiple views (Task-2).

        Args:
            images: List of images
            objects: Optional list of known objects
            tau: Tolerance parameter

        Returns:
            ConstraintSet with view-invariant and view-dependent constraints
        """
        logger.info(f"Extracting constraints from {len(images)} views")

        # Build prompt
        prompts = build_multi_view_prompt(len(images), objects, tau)

        # Build messages with all images
        messages = self._build_messages(prompts, images)

        # Call VLM
        max_tokens = self._resolve_max_tokens(objects)
        response = self._call_api(messages, max_tokens=max_tokens)

        # Parse response
        result = self._parse_response(response)

        # Add metadata
        result.metadata["mode"] = "multi_view"
        result.metadata["n_views"] = len(images)
        result.metadata["tau"] = tau
        result.metadata["model"] = self.config.model

        # Validate if configured
        if self.config.validate_consistency:
            self._validate_constraints(result)

        return result

    def _build_messages(
        self,
        prompts: Dict[str, str],
        images: List[Union[str, Path, bytes]],
    ) -> List[Dict]:
        """
        构建 API 消息列表。

        Build API message list with images.
        """
        messages = []

        # System message
        if prompts.get("system"):
            messages.append({
                "role": "system",
                "content": prompts["system"]
            })

        # User message with images
        user_content = []

        # Add all images
        for i, image in enumerate(images):
            # 多视角输入时，为每张图添加视角标签以减少歧义
            if len(images) > 1:
                user_content.append({
                    "type": "text",
                    "text": f"View {i}:"
                })
            image_data = self._encode_image(image)
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data}",
                    "detail": "high"
                }
            })

        # Add text prompt
        user_content.append({
            "type": "text",
            "text": prompts["user"]
        })

        messages.append({
            "role": "user",
            "content": user_content
        })

        return messages

    def _resolve_max_tokens(
        self,
        objects: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        根据物体数量动态设置输出 token 上限。

        Resolve max_tokens by object count.
        """
        if not objects:
            return self.config.max_tokens

        n_objects = len(objects)
        if n_objects <= 5:
            computed = 4096
        elif n_objects <= 8:
            computed = 8192
        else:
            computed = 16384
        # 不低于用户配置值，避免缩减用户显式设置的上限
        return max(self.config.max_tokens, computed)

    def _encode_image(self, image: Union[str, Path, bytes]) -> str:
        """
        将图像编码为 base64。

        Encode image to base64.
        """
        if isinstance(image, bytes):
            return base64.b64encode(image).decode()
        elif isinstance(image, (str, Path)):
            path = Path(image)
            if path.exists():
                return base64.b64encode(path.read_bytes()).decode()
            else:
                # Assume already base64
                return str(image)
        else:
            raise ValueError(f"Invalid image type: {type(image)}")

    def _call_api(
        self,
        messages: List[Dict],
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        调用 VLM API。

        Call VLM API with retry logic.
        """
        import time

        last_error = None

        for attempt in range(self.config.retry_count):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=max_tokens or self.config.max_tokens,
                )
                return response.choices[0].message.content

            except Exception as e:
                last_error = e
                logger.warning(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.retry_count - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))

        raise RuntimeError(
            f"API call failed after {self.config.retry_count} attempts: {last_error}"
        )

    def _parse_response(self, response: str) -> ConstraintSet:
        """
        解析 VLM 响应为 ConstraintSet。

        Parse VLM response into ConstraintSet.
        """
        try:
            data = self._extract_json(response)
            data = self._normalize_response_data(data)
            return self._build_constraint_set(data)

        except Exception as e:
            logger.warning(f"Failed to parse response: {e}")
            logger.debug(f"Raw response: {response[:500]}...")
            return ConstraintSet(
                confidence=0.0,
                metadata={"parse_error": str(e), "raw_response": response}
            )

    def _normalize_response_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        规范化模型输出，补齐关键缺省字段。

        Normalize model response and fill required defaults.
        """
        constraints = data.get("constraints")
        if not isinstance(constraints, dict):
            return data

        qrr_constraints = constraints.get("qrr")
        if not isinstance(qrr_constraints, list):
            return data

        for item in qrr_constraints:
            if not isinstance(item, dict):
                continue
            metric = item.get("metric")
            if not metric:
                item["metric"] = "dist3D"
        return data

    def _extract_json(self, text: str) -> Dict:
        """
        从文本中提取 JSON。

        Extract JSON from text, handling various formats.
        """
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try markdown code block
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        # Try to find JSON object
        json_pattern = r'\{[\s\S]*\}'
        matches = re.findall(json_pattern, text)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        raise ValueError("Could not extract JSON from response")

    def _build_constraint_set(self, data: Dict) -> ConstraintSet:
        """
        从解析的数据构建 ConstraintSet。

        Build ConstraintSet from parsed data.
        """
        # Parse objects
        objects = []
        for obj in data.get("objects", []):
            objects.append(ObjectInfo(
                id=obj.get("id", f"obj_{len(objects)}"),
                type=obj.get("type", "unknown"),
                color=obj.get("color", "unknown"),
                size_class=obj.get("size_class", "medium"),
                position_2d=obj.get("position_2d"),
                description=obj.get("description"),
            ))

        constraints = data.get("constraints", {})

        # Parse axial constraints
        axial = []
        _axial_values = {r.value for r in AxialRelation}
        for c in constraints.get("axial", []):
            try:
                obj1 = c.get("obj1") or c.get("target")
                obj2 = c.get("obj2") or c.get("reference")
                rel_str = c.get("relation", "left_of")
                # Fix swapped obj2/relation (VLM sometimes swaps them)
                if obj2 in _axial_values and rel_str not in _axial_values:
                    obj2, rel_str = rel_str, obj2
                relation = AxialRelation(rel_str)
                axial.append(AxialConstraint(
                    obj1=obj1,
                    obj2=obj2,
                    relation=relation,
                ))
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Invalid axial constraint: {c}, error: {e}")

        # Parse topology constraints
        topology = []
        for c in constraints.get("topology", []):
            try:
                obj1 = c.get("obj1") or c.get("target")
                obj2 = c.get("obj2") or c.get("reference")
                topology.append(TopologyConstraint(
                    obj1=obj1,
                    obj2=obj2,
                    relation=c.get("relation", "disjoint"),
                ))
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Invalid topology constraint: {c}, error: {e}")

        # Parse occlusion constraints
        occlusion = []
        for c in constraints.get("occlusion", []):
            try:
                occluder = c.get("occluder") or c.get("target")
                occluded = c.get("occluded") or c.get("reference")
                occlusion.append(OcclusionConstraint(
                    occluder=occluder,
                    occluded=occluded,
                    partial=c.get("partial", False),
                ))
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Invalid occlusion constraint: {c}, error: {e}")

        # Parse size constraints
        size = []
        for c in constraints.get("size", []):
            try:
                bigger = c.get("bigger")
                smaller = c.get("smaller")
                if not bigger and c.get("relation") == "bigger":
                    bigger = c.get("target")
                    smaller = c.get("reference")
                elif not bigger and c.get("relation") == "smaller":
                    bigger = c.get("reference")
                    smaller = c.get("target")
                size.append(SizeConstraint(
                    bigger=bigger,
                    smaller=smaller,
                ))
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Invalid size constraint: {c}, error: {e}")

        # Parse closer constraints
        closer = []
        for c in constraints.get("closer", []):
            try:
                farther = c.get("farther") or c.get("further")
                closer.append(CloserConstraint(
                    anchor=c["anchor"],
                    closer=c["closer"],
                    farther=farther,
                ))
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Invalid closer constraint: {c}, error: {e}")

        # Parse QRR constraints
        qrr = []
        for c in constraints.get("qrr", []):
            try:
                qrr.append(QRRConstraintSchema(
                    pair1=c["pair1"],
                    pair2=c["pair2"],
                    metric=c.get("metric", "dist3D"),
                    comparator=self._normalize_comparator(c.get("comparator", "~=")),
                ))
            except (KeyError, ValueError) as e:
                logger.warning(f"Invalid QRR constraint: {c}, error: {e}")

        # Parse TRR constraints
        trr = []
        for c in constraints.get("trr", []):
            try:
                hour = int(c.get("hour", 12))
                hour = max(1, min(12, hour))
                trr.append(TRRConstraintSchema(
                    target=c["target"],
                    ref1=c["ref1"],
                    ref2=c["ref2"],
                    hour=hour,
                ))
            except (KeyError, ValueError) as e:
                logger.warning(f"Invalid TRR constraint: {c}, error: {e}")

        return ConstraintSet(
            objects=objects,
            axial=axial,
            topology=topology,
            occlusion=occlusion,
            size=size,
            closer=closer,
            qrr=qrr,
            trr=trr,
            confidence=float(data.get("confidence", 0.5)),
        )

    def _normalize_comparator(self, comp: str) -> str:
        """
        规范化比较器。

        Normalize comparator to standard form.
        """
        comp = str(comp).strip().lower()
        if comp in ("<", "lt", "less", "less_than"):
            return "<"
        elif comp in (">", "gt", "greater", "greater_than"):
            return ">"
        else:
            return "~="

    def _validate_constraints(self, result: ConstraintSet) -> None:
        """
        验证约束一致性。

        Validate constraint consistency.
        """
        issues = []

        # Check axial consistency (if A left_of B, B should not be left_of A)
        axial_pairs = {}
        for c in result.axial:
            key = (c.obj1, c.obj2)
            if key in axial_pairs:
                issues.append(f"Duplicate axial constraint for ({c.obj1}, {c.obj2})")
            axial_pairs[key] = c.relation

            # Check inverse
            inv_key = (c.obj2, c.obj1)
            if inv_key in axial_pairs:
                inv = c.inverse()
                if axial_pairs[inv_key] != inv.relation:
                    issues.append(
                        f"Inconsistent axial: {c.obj1} {c.relation.value} {c.obj2} "
                        f"conflicts with {c.obj2} {axial_pairs[inv_key].value} {c.obj1}"
                    )

        # Check size transitivity (if A > B and B > C, then A > C)
        # This is a simplified check
        size_graph = {}
        for c in result.size:
            if c.bigger not in size_graph:
                size_graph[c.bigger] = set()
            size_graph[c.bigger].add(c.smaller)

            # Check for cycles
            if c.smaller in size_graph and c.bigger in size_graph.get(c.smaller, set()):
                issues.append(f"Size cycle: {c.bigger} > {c.smaller} but also {c.smaller} > {c.bigger}")

        if issues:
            result.metadata["consistency_issues"] = issues
            logger.warning(f"Constraint consistency issues: {issues}")
