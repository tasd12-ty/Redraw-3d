"""
ORDINAL-SPATIAL 基准测试的 Oracle 基线。

Oracle 基线直接从真值 3D 位置计算预测结果，
提供 100% 准确率的评估上界。

特点：
- 从已知的 3D 几何直接计算约束
- 理论上达到 100% 准确率
- 用作其他基线的性能上界
- 不依赖图像或视觉处理
- 支持所有任务类型（T1/T2/T3）

用途：
1. 验证评估指标的正确性
2. 提供性能上界参考
3. 生成数据集真值标签
4. 调试和测试框架

Oracle baseline for ORDINAL-SPATIAL benchmark.

The oracle baseline computes predictions directly from ground-truth
3D positions, providing a 100% accuracy upper bound for evaluation.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ordinal_spatial.dsl.predicates import (
    QRRConstraint,
    TRRConstraint,
    MetricType,
    compute_qrr,
    compute_trr,
    extract_all_qrr,
    extract_all_trr,
)
from ordinal_spatial.dsl.schema import (
    OrdinalSceneDescription,
    QRRQuery,
    TRRQuery,
    QRRPrediction,
    TRRPrediction,
    OSDPrediction,
)
from ordinal_spatial.generation.constraint_extractor import (
    ConstraintExtractor,
    ExtractionConfig,
)


@dataclass
class OraclePrediction:
    """
    Oracle 基线的预测结果。

    Prediction from oracle baseline.
    """
    query_id: str
    prediction: Any
    confidence: float = 1.0
    reasoning: str = "Computed from ground-truth 3D positions"


class OracleBaseline:
    """
    从真值计算答案的 Oracle 基线。

    该基线理论上达到 100% 准确率，因为它直接从已知的
    3D 几何信息计算序关系。用作评估的上界参考。

    Oracle baseline that computes answers from ground truth.

    This baseline achieves 100% accuracy by definition, as it
    computes ordinal relations directly from known 3D geometry.
    Used as an upper bound for evaluation.
    """

    def __init__(self, tau: float = 0.10):
        """
        初始化 Oracle 基线。

        参数:
            tau: 比较的容差参数

        Initialize oracle baseline.

        Args:
            tau: Tolerance parameter for comparisons
        """
        self.tau = tau
        self.extractor = ConstraintExtractor(
            ExtractionConfig(tau=tau)
        )

    def predict_qrr(
        self,
        objects: Dict[str, Dict],
        query: QRRQuery,
    ) -> QRRPrediction:
        """
        从真值预测 QRR 比较。

        参数:
            objects: 物体数据字典
            query: QRR 查询规范

        返回:
            包含正确答案的 QRRPrediction

        Predict QRR comparison from ground truth.

        Args:
            objects: Dictionary of object data
            query: QRR query specification

        Returns:
            QRRPrediction with correct answer
        """
        # Extract object IDs from query
        obj_mapping = query.objects
        pair1 = (obj_mapping["A"], obj_mapping["B"])
        pair2 = (obj_mapping["C"], obj_mapping["D"])

        # Compute constraint
        metric = MetricType.from_string(query.metric)
        constraint = compute_qrr(objects, pair1, pair2, metric, self.tau)

        return QRRPrediction(
            query_id=query.query_id,
            comparator=str(constraint.comparator),
            confidence=1.0,
            reasoning="Oracle: computed from ground-truth 3D positions",
        )

    def predict_trr(
        self,
        objects: Dict[str, Dict],
        query: TRRQuery,
    ) -> TRRPrediction:
        """
        从真值预测 TRR 时钟位置。

        参数:
            objects: 物体数据字典
            query: TRR 查询规范

        返回:
            包含正确答案的 TRRPrediction

        Predict TRR clock position from ground truth.

        Args:
            objects: Dictionary of object data
            query: TRR query specification

        Returns:
            TRRPrediction with correct answer
        """
        constraint = compute_trr(
            objects,
            query.target,
            query.ref1,
            query.ref2,
            use_3d=False,  # Use 2D for image-based evaluation
        )

        return TRRPrediction(
            query_id=query.query_id,
            hour=constraint.hour,
            quadrant=constraint.quadrant,
            confidence=1.0,
            reasoning="Oracle: computed from ground-truth positions",
        )

    def extract_osd(
        self,
        scene: Dict,
    ) -> OSDPrediction:
        """
        从场景提取完整的 OSD。

        参数:
            scene: 包含物体的场景字典

        返回:
            包含所有约束的 OSDPrediction

        Extract complete OSD from scene.

        Args:
            scene: Scene dictionary with objects

        Returns:
            OSDPrediction with all constraints
        """
        osd = self.extractor.extract(scene)

        # Convert to prediction format
        qrr_list = [
            {
                "pair1": list(c.pair1),
                "pair2": list(c.pair2),
                "metric": c.metric,
                "comparator": c.comparator,
            }
            for c in osd.world.qrr
        ]

        trr_list = []
        for view in osd.views:
            for c in view.trr:
                trr_list.append({
                    "target": c.target,
                    "ref1": c.ref1,
                    "ref2": c.ref2,
                    "hour": c.hour,
                })

        return OSDPrediction(
            scene_id=osd.scene_id,
            objects=[obj.id for obj in osd.objects],
            qrr=qrr_list,
            trr=trr_list,
            confidence=1.0,
        )

    def run_t1_qrr(
        self,
        dataset: List[Dict],
    ) -> List[QRRPrediction]:
        """
        在数据集上运行 T1-Q 任务。

        参数:
            dataset: 包含查询的场景数据列表

        返回:
            QRR 预测列表

        Run T1-Q task on a dataset.

        Args:
            dataset: List of scene data with queries

        Returns:
            List of QRR predictions
        """
        predictions = []

        for item in dataset:
            scene = item.get("scene", item)
            queries = item.get("qrr_queries", [])

            # Build objects dict
            objects = {
                obj.get("id", f"obj_{i}"): obj
                for i, obj in enumerate(scene.get("objects", []))
            }

            for query_data in queries:
                query = QRRQuery(**query_data)
                pred = self.predict_qrr(objects, query)
                predictions.append(pred)

        return predictions

    def run_t1_trr(
        self,
        dataset: List[Dict],
    ) -> List[TRRPrediction]:
        """
        在数据集上运行 T1-C 任务。

        参数:
            dataset: 包含查询的场景数据列表

        返回:
            TRR 预测列表

        Run T1-C task on a dataset.

        Args:
            dataset: List of scene data with queries

        Returns:
            List of TRR predictions
        """
        predictions = []

        for item in dataset:
            scene = item.get("scene", item)
            queries = item.get("trr_queries", [])

            objects = {
                obj.get("id", f"obj_{i}"): obj
                for i, obj in enumerate(scene.get("objects", []))
            }

            for query_data in queries:
                query = TRRQuery(**query_data)
                pred = self.predict_trr(objects, query)
                predictions.append(pred)

        return predictions


# =============================================================================
# Standalone Functions
# =============================================================================

def oracle_predict_qrr(
    objects: Dict[str, Dict],
    pair1: tuple,
    pair2: tuple,
    metric: str = "dist3D",
    tau: float = 0.10,
) -> Dict[str, Any]:
    """
    快速 Oracle QRR 查询预测。

    参数:
        objects: 物体数据字典
        pair1: 第一对物体 ID
        pair2: 第二对物体 ID
        metric: 度量类型
        tau: 容差

    返回:
        预测字典

    Quick oracle prediction for a QRR query.

    Args:
        objects: Object data dictionary
        pair1: First pair of object IDs
        pair2: Second pair of object IDs
        metric: Metric type
        tau: Tolerance

    Returns:
        Prediction dictionary
    """
    metric_type = MetricType.from_string(metric)
    constraint = compute_qrr(objects, pair1, pair2, metric_type, tau)

    return {
        "comparator": str(constraint.comparator),
        "ratio": constraint.ratio,
        "difficulty": constraint.difficulty,
        "confidence": 1.0,
    }


def oracle_predict_trr(
    objects: Dict[str, Dict],
    target: str,
    ref1: str,
    ref2: str,
) -> Dict[str, Any]:
    """
    快速 Oracle TRR 查询预测。

    参数:
        objects: 物体数据字典
        target: 目标物体 ID
        ref1: 原点物体 ID
        ref2: 方向参考物体 ID

    返回:
        预测字典

    Quick oracle prediction for a TRR query.

    Args:
        objects: Object data dictionary
        target: Target object ID
        ref1: Origin object ID
        ref2: Direction reference object ID

    Returns:
        Prediction dictionary
    """
    constraint = compute_trr(objects, target, ref1, ref2)

    return {
        "hour": constraint.hour,
        "quadrant": constraint.quadrant,
        "angle_deg": constraint.angle_deg,
        "confidence": 1.0,
    }


def oracle_extract_osd(
    scene: Dict,
    tau: float = 0.10,
) -> OrdinalSceneDescription:
    """
    使用 Oracle 提取完整的 OSD。

    参数:
        scene: 场景字典
        tau: 容差参数

    返回:
        完整的 OrdinalSceneDescription

    Extract complete OSD using oracle.

    Args:
        scene: Scene dictionary
        tau: Tolerance parameter

    Returns:
        Complete OrdinalSceneDescription
    """
    baseline = OracleBaseline(tau=tau)
    return baseline.extractor.extract(scene)
