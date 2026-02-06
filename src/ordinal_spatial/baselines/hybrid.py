"""
ORDINAL-SPATIAL 基准测试的混合预测-验证-修复基线。

本基线结合了 VLM 预测与一致性检查，
允许模型修正不一致的预测结果。

工作流程：
1. VLM 生成初始约束预测
2. 一致性检查器验证约束（检测循环）
3. 如果不一致，格式化冲突信息并要求 VLM 修复
4. 重复直到一致或达到最大迭代次数

优势：
- 自动检测并修复逻辑矛盾
- 提高约束集质量
- 减少传递性违反
- 提供迭代修复记录

适用场景：
- T2 任务（完整约束提取）
- 需要高一致性的应用
- 分析 VLM 的自我修正能力

Hybrid predict-verify-repair baseline for ORDINAL-SPATIAL benchmark.

This baseline combines VLM prediction with consistency checking,
allowing the model to revise inconsistent predictions.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import logging

from ordinal_spatial.baselines.vlm_direct import VLMDirectBaseline, VLMConfig
from ordinal_spatial.evaluation.consistency import (
    ConsistencyChecker,
    check_qrr_consistency,
    ConsistencyReport,
)
from ordinal_spatial.prompts import build_repair_prompt

logger = logging.getLogger(__name__)


@dataclass
class HybridConfig:
    """
    混合基线的配置。

    Configuration for hybrid baseline.
    """
    vlm_config: VLMConfig = field(default_factory=VLMConfig)
    max_repair_iterations: int = 3
    require_consistency: bool = True
    log_iterations: bool = True


@dataclass
class RepairIteration:
    """
    修复迭代的记录。

    Record of a repair iteration.
    """
    iteration: int
    prediction: Dict[str, Any]
    consistency_report: ConsistencyReport
    conflicts: List[str]
    is_final: bool


class HybridBaseline:
    """
    带预测-验证-修复循环的混合基线。

    处理流程：
    1. VLM 生成初始约束预测
    2. 一致性检查器验证约束
    3. 如果不一致，格式化冲突信息并要求 VLM 修复
    4. 重复直到一致或达到最大迭代次数

    Hybrid baseline with predict-verify-repair loop.

    Process:
    1. VLM generates initial constraint prediction
    2. Consistency checker validates constraints
    3. If inconsistent, format conflict info and ask VLM to repair
    4. Repeat until consistent or max iterations reached
    """

    def __init__(self, config: HybridConfig = None):
        """
        初始化混合基线。

        参数:
            config: 混合配置

        Initialize hybrid baseline.

        Args:
            config: Hybrid configuration
        """
        self.config = config or HybridConfig()
        self.vlm = VLMDirectBaseline(self.config.vlm_config)
        self.checker = ConsistencyChecker()

    def extract_constraints(
        self,
        image: Union[str, bytes],
        objects: Dict[str, Dict],
        tau: float = 0.10,
    ) -> Dict[str, Any]:
        """
        带验证和修复的约束提取。

        参数:
            image: 图像路径或 base64 编码的字节
            objects: 物体数据字典
            tau: 容差

        返回:
            最终（理想情况下一致的）约束预测结果

        Extract constraints with verification and repair.

        Args:
            image: Image path or base64-encoded bytes
            objects: Object data dictionary
            tau: Tolerance

        Returns:
            Final (ideally consistent) constraint prediction
        """
        iterations = []

        # Initial prediction
        prediction = self.vlm.extract_constraints(image, objects, tau)
        iteration = 0

        while iteration < self.config.max_repair_iterations:
            # Check consistency
            qrr_constraints = prediction.get("qrr", [])
            report = check_qrr_consistency(qrr_constraints)

            # Format conflicts
            conflicts = self._format_conflicts(report)

            iterations.append(RepairIteration(
                iteration=iteration,
                prediction=prediction.copy(),
                consistency_report=report,
                conflicts=conflicts,
                is_final=report.is_consistent,
            ))

            if self.config.log_iterations:
                logger.info(
                    f"Iteration {iteration}: "
                    f"consistent={report.is_consistent}, "
                    f"n_constraints={len(qrr_constraints)}, "
                    f"n_conflicts={len(conflicts)}"
                )

            # If consistent, we're done
            if report.is_consistent:
                break

            # Repair
            iteration += 1
            if iteration < self.config.max_repair_iterations:
                prediction = self._repair(image, objects, prediction, conflicts, tau)

        # Add metadata
        prediction["iterations"] = [
            {
                "iteration": it.iteration,
                "is_consistent": it.consistency_report.is_consistent,
                "n_conflicts": len(it.conflicts),
            }
            for it in iterations
        ]
        prediction["final_consistent"] = iterations[-1].consistency_report.is_consistent if iterations else False
        prediction["n_iterations"] = len(iterations)

        return prediction

    def _repair(
        self,
        image: Union[str, bytes],
        objects: Dict[str, Dict],
        prediction: Dict[str, Any],
        conflicts: List[str],
        tau: float,
    ) -> Dict[str, Any]:
        """
        要求 VLM 修复不一致的预测。

        参数:
            image: 图像
            objects: 物体数据
            prediction: 当前（不一致的）预测
            conflicts: 冲突描述
            tau: 容差

        返回:
            修复后的预测

        Ask VLM to repair inconsistent predictions.

        Args:
            image: Image
            objects: Object data
            prediction: Current (inconsistent) prediction
            conflicts: Conflict descriptions
            tau: Tolerance

        Returns:
            Repaired prediction
        """
        # Build repair prompt
        prompts = build_repair_prompt(prediction, conflicts, objects)

        # Build messages with image
        messages = self.vlm._build_messages(prompts, image)

        # Call API
        response = self.vlm._call_api(messages)

        # Parse response
        result = self.vlm._parse_osd_response(response)

        # Keep track of revisions
        result["repair_iteration"] = True
        result["previous_conflicts"] = conflicts

        return result

    def _format_conflicts(self, report: ConsistencyReport) -> List[str]:
        """
        将冲突循环格式化为人类可读的字符串。

        Format conflict cycles as human-readable strings.
        """
        conflicts = []

        for cycle in report.cycles:
            # Format the cycle as a readable string
            parts = []
            for i, (pair, edge) in enumerate(zip(cycle.nodes, cycle.edges)):
                parts.append(f"dist{pair} {edge}")

            cycle_str = " → ".join(parts)
            if cycle.nodes:
                cycle_str += f" → dist{cycle.nodes[0]}"

            conflicts.append(f"Cycle detected: {cycle_str}")

        return conflicts

    def predict_qrr_batch(
        self,
        image: Union[str, bytes],
        objects: Dict[str, Dict],
        queries: List[Dict],
        tau: float = 0.10,
    ) -> List[Dict[str, Any]]:
        """
        带一致性强制的批量 QRR 查询预测。

        该方法不是独立预测每个查询，而是提取完整的约束集，
        并在所有预测之间检查一致性。

        参数:
            image: 图像
            objects: 物体数据
            queries: QRR 查询列表
            tau: 容差

        返回:
            预测列表

        Predict multiple QRR queries with consistency enforcement.

        Instead of predicting each query independently, this method
        extracts a full constraint set and checks consistency across
        all predictions.

        Args:
            image: Image
            objects: Object data
            queries: List of QRR queries
            tau: Tolerance

        Returns:
            List of predictions
        """
        # Extract full constraint set
        osd = self.extract_constraints(image, objects, tau)

        # Build lookup from constraint set
        constraint_lookup = {}
        for c in osd.get("qrr", []):
            pair1 = tuple(sorted(c.get("pair1", [])))
            pair2 = tuple(sorted(c.get("pair2", [])))
            key = (pair1, pair2)
            constraint_lookup[key] = c.get("comparator", "~=")
            # Also store reversed
            constraint_lookup[(pair2, pair1)] = self._flip_comparator(c.get("comparator", "~="))

        # Match queries to constraints
        predictions = []
        for query in queries:
            pair1 = tuple(sorted([query.get("A"), query.get("B")]))
            pair2 = tuple(sorted([query.get("C"), query.get("D")]))
            key = (pair1, pair2)

            if key in constraint_lookup:
                comparator = constraint_lookup[key]
            else:
                # Fall back to individual prediction
                pred = self.vlm.predict_qrr(
                    image, objects,
                    (query["A"], query["B"]),
                    (query["C"], query["D"]),
                    tau=tau,
                )
                comparator = pred.get("comparator", "~=")

            predictions.append({
                "query_id": query.get("query_id", ""),
                "comparator": comparator,
                "confidence": osd.get("confidence", 0.5),
                "from_batch": True,
            })

        return predictions

    def _flip_comparator(self, comp: str) -> str:
        """
        翻转比较器以适应反向的对顺序。

        Flip comparator for reversed pair order.
        """
        if comp == "<":
            return ">"
        elif comp == ">":
            return "<"
        return comp


class IncrementalHybridBaseline:
    """
    逐个添加约束的增量混合基线。

    该变体在添加每个约束后检查一致性，
    允许更早地检测冲突。

    Incremental hybrid baseline that adds constraints one at a time.

    This variant checks consistency after each constraint is added,
    allowing earlier detection of conflicts.
    """

    def __init__(self, config: HybridConfig = None):
        """
        初始化增量基线。

        参数:
            config: 混合配置

        Initialize incremental baseline.

        Args:
            config: Hybrid configuration
        """
        self.config = config or HybridConfig()
        self.vlm = VLMDirectBaseline(self.config.vlm_config)

    def extract_constraints_incremental(
        self,
        image: Union[str, bytes],
        objects: Dict[str, Dict],
        tau: float = 0.10,
    ) -> Dict[str, Any]:
        """
        带逐约束检查的增量约束提取。

        参数:
            image: 图像
            objects: 物体数据
            tau: 容差

        返回:
            一致的约束预测

        Extract constraints incrementally with per-constraint checking.

        Args:
            image: Image
            objects: Object data
            tau: Tolerance

        Returns:
            Consistent constraint prediction
        """
        checker = ConsistencyChecker()
        accepted_qrr = []
        rejected_qrr = []

        # Get initial prediction
        prediction = self.vlm.extract_constraints(image, objects, tau)
        qrr_constraints = prediction.get("qrr", [])

        # Add constraints one by one
        for constraint in qrr_constraints:
            cycle = checker.add_qrr(constraint)

            if cycle is None:
                accepted_qrr.append(constraint)
            else:
                rejected_qrr.append({
                    "constraint": constraint,
                    "conflict_cycle": str(cycle),
                })

        prediction["qrr"] = accepted_qrr
        prediction["rejected_qrr"] = rejected_qrr
        prediction["n_accepted"] = len(accepted_qrr)
        prediction["n_rejected"] = len(rejected_qrr)
        prediction["final_consistent"] = True  # By construction

        return prediction
