"""
通用约束比较引擎。

支持 VLM 输出与任意比较目标 (GT / 感知上界) 的比较。
错误分类: correct, wrong_value, missing, hallucinated。

Universal constraint comparison engine.

Compares VLM output against any comparison target (GT, Perception UB).
Error categories: correct, wrong_value, missing, hallucinated.
"""

from dataclasses import dataclass, field
from typing import (
    Dict, List, Any, Optional, Tuple, Set, FrozenSet,
)
from collections import defaultdict

from ordinal_spatial.dsl.comparators import (
    Comparator, is_flip,
)

# -----------------------------------------------------------
# 比较器值归一化
# -----------------------------------------------------------

_COMP_MAP = {
  "<": "<", "lt": "<", "less": "<",
  ">": ">", "gt": ">", "greater": ">",
  "~=": "~=", "eq": "~=", "equal": "~=",
  "approximately_equal": "~=",
}

# axial 反义词对
_AXIAL_INVERSE = {
  "left_of": "right_of", "right_of": "left_of",
  "above": "below", "below": "above",
  "in_front_of": "behind", "behind": "in_front_of",
}

# size 反义词对
_SIZE_INVERSE = {
  "bigger": "smaller", "smaller": "bigger",
  "larger": "smaller",
}


def _norm_comp(comp: str) -> str:
  """归一化比较器字符串。"""
  return _COMP_MAP.get(
    str(comp).strip().lower(), str(comp).strip().lower()
  )


# -----------------------------------------------------------
# 数据结构
# -----------------------------------------------------------

@dataclass
class ConstraintMatch:
  """单个约束的匹配结果。"""
  constraint_type: str
  key: Tuple
  category: str  # correct | wrong_value | missing | hallucinated
  predicted: Optional[Dict] = None
  target: Optional[Dict] = None
  details: Dict = field(default_factory=dict)
  # details 可含: is_flip, ordinal_distance, sub_type,
  #               gt_exists (hallucinated 时)


@dataclass
class ComparisonResult:
  """单场景完整比较结果。"""
  scene_id: str = ""
  matches: List[ConstraintMatch] = field(
    default_factory=list
  )

  # 计数
  n_target: int = 0
  n_predicted: int = 0
  n_correct: int = 0
  n_wrong: int = 0
  n_missing: int = 0
  n_hallucinated: int = 0

  # 按类型分组
  by_type: Dict[str, "ComparisonResult"] = field(
    default_factory=dict
  )

  # 标准指标
  precision: float = 0.0
  recall: float = 0.0
  f1: float = 0.0

  def _compute_rates(self):
    """从计数计算 precision/recall/f1。"""
    if self.n_predicted > 0:
      self.precision = self.n_correct / self.n_predicted
    if self.n_target > 0:
      self.recall = self.n_correct / self.n_target
    if self.precision + self.recall > 0:
      self.f1 = (
        2 * self.precision * self.recall
        / (self.precision + self.recall)
      )

  def to_dict(self) -> Dict[str, Any]:
    """序列化为 dict。"""
    d = {
      "scene_id": self.scene_id,
      "n_target": self.n_target,
      "n_predicted": self.n_predicted,
      "n_correct": self.n_correct,
      "n_wrong": self.n_wrong,
      "n_missing": self.n_missing,
      "n_hallucinated": self.n_hallucinated,
      "precision": round(self.precision, 4),
      "recall": round(self.recall, 4),
      "f1": round(self.f1, 4),
    }
    if self.by_type:
      d["by_type"] = {
        k: v.to_dict() for k, v in self.by_type.items()
      }
    return d

  def summary(self) -> str:
    """单行摘要。"""
    return (
      f"Target:{self.n_target} Pred:{self.n_predicted} "
      f"C:{self.n_correct} W:{self.n_wrong} "
      f"M:{self.n_missing} H:{self.n_hallucinated} "
      f"F1:{self.f1:.3f}"
    )


# -----------------------------------------------------------
# Canonical Key 构建
# -----------------------------------------------------------

def _qrr_canonical(c: Dict) -> Tuple[Tuple, Tuple, str, str]:
  """
  QRR canonical: (sorted_pair1, sorted_pair2, metric, comp).

  如果 pair 顺序翻转, comparator 同时翻转。
  """
  p1 = tuple(sorted(c.get("pair1", [])))
  p2 = tuple(sorted(c.get("pair2", [])))
  metric = c.get("metric", "dist3D")
  comp = _norm_comp(c.get("comparator", "~="))

  if p1 > p2:
    p1, p2 = p2, p1
    # 翻转 comparator
    if comp == "<":
      comp = ">"
    elif comp == ">":
      comp = "<"

  key = (p1, p2, metric)
  return key, comp


def _axial_canonical(c: Dict) -> Tuple[Tuple, str]:
  """
  Axial canonical: ((min_obj, max_obj), normalized_relation).

  无序 obj pair + 归一化方向。
  """
  obj1 = c.get("object1", c.get("obj1", ""))
  obj2 = c.get("object2", c.get("obj2", ""))
  rel = c.get("relation", "")

  # 归一化: 保证 obj1 < obj2, 必要时翻转 relation
  if obj1 > obj2:
    obj1, obj2 = obj2, obj1
    rel = _AXIAL_INVERSE.get(rel, rel)

  key = (obj1, obj2)
  return key, rel


def _size_canonical(c: Dict) -> Tuple[Tuple, str]:
  """
  Size canonical: ((min_obj, max_obj), normalized_relation).
  """
  bigger = c.get("bigger", c.get("obj1", ""))
  smaller = c.get("smaller", c.get("obj2", ""))
  rel = "bigger"

  # 归一化
  if bigger > smaller:
    bigger, smaller = smaller, bigger
    rel = "smaller"

  key = (bigger, smaller)
  return key, rel


def _topology_canonical(c: Dict) -> Tuple[Tuple, str]:
  """
  Topology canonical: ((min_obj, max_obj), relation).
  """
  obj1 = c.get("object1", c.get("obj1", ""))
  obj2 = c.get("object2", c.get("obj2", ""))
  rel = c.get("relation", c.get("rel", ""))

  if obj1 > obj2:
    obj1, obj2 = obj2, obj1

  key = (obj1, obj2)
  return key, rel


def _occlusion_canonical(c: Dict) -> Tuple[Tuple, str]:
  """
  Occlusion canonical: 有向对 (occluder, occluded).
  """
  occluder = c.get("occluder", "")
  occluded = c.get("occluded", "")
  key = (occluder, occluded)
  return key, "occludes"


def _closer_canonical(c: Dict) -> Tuple[Tuple, str]:
  """
  Closer canonical: (anchor, frozenset({obj1, obj2})), closer_id.
  """
  anchor = c.get("anchor", "")
  closer = c.get("closer", "")
  farther = c.get("farther", "")
  pair = tuple(sorted([closer, farther]))
  key = (anchor, pair)
  return key, closer


def _trr_canonical(c: Dict) -> Tuple[Tuple, int]:
  """
  TRR canonical: (target, ref1, ref2), hour.
  """
  target = c.get("target", "")
  ref1 = c.get("ref1", "")
  ref2 = c.get("ref2", "")
  hour = c.get("hour", 0)
  key = (target, ref1, ref2)
  return key, hour


# -----------------------------------------------------------
# 值比较逻辑
# -----------------------------------------------------------

def _classify_qrr(pred_comp: str, tgt_comp: str) -> Dict:
  """比较 QRR 值, 返回分类 details。"""
  if pred_comp == tgt_comp:
    return {"match": True}

  flip = (
    (pred_comp == "<" and tgt_comp == ">")
    or (pred_comp == ">" and tgt_comp == "<")
  )
  shift = (
    (pred_comp == "~=" and tgt_comp in ("<", ">"))
    or (tgt_comp == "~=" and pred_comp in ("<", ">"))
  )

  return {
    "match": False,
    "is_flip": flip,
    "sub_type": "flip" if flip else (
      "shift" if shift else "other"
    ),
    "pred_value": pred_comp,
    "target_value": tgt_comp,
  }


def _classify_directional(
  pred_val: str, tgt_val: str, inverse_map: Dict
) -> Dict:
  """比较有方向的值 (axial, size)。"""
  if pred_val == tgt_val:
    return {"match": True}

  flip = (
    inverse_map.get(pred_val) == tgt_val
  )
  return {
    "match": False,
    "is_flip": flip,
    "sub_type": "flip" if flip else "other",
    "pred_value": pred_val,
    "target_value": tgt_val,
  }


def _classify_trr(pred_hour: int, tgt_hour: int) -> Dict:
  """比较 TRR 时钟方向。"""
  if pred_hour == tgt_hour:
    return {"match": True}

  # 计算角度差 (环形)
  diff = abs(pred_hour - tgt_hour)
  diff = min(diff, 12 - diff)

  return {
    "match": False,
    "is_flip": diff >= 5,  # >=5 小时差视为翻转
    "sub_type": "flip" if diff >= 5 else "shift",
    "angular_diff_hours": diff,
    "pred_value": pred_hour,
    "target_value": tgt_hour,
  }


# -----------------------------------------------------------
# 核心比较器
# -----------------------------------------------------------

class ConstraintComparator:
  """
  通用约束比较器。

  支持 VLM 输出与任意比较目标的比较。

  Args:
      gt_for_annotation: 可选, 完整 GT dict。
          仅用于对 hallucinated 约束附注 gt_exists 标记。
          不影响分类 (hallucinated 仍是 hallucinated)。
  """

  # 支持的约束类型 → (canonical_fn, value_classify_fn)
  SUPPORTED_TYPES = [
    "qrr", "axial", "size", "topology",
    "occlusion", "closer", "trr",
  ]

  def __init__(
    self,
    gt_for_annotation: Optional[Dict] = None,
  ):
    self._gt_ann = gt_for_annotation

  def compare(
    self,
    predicted: Dict[str, Any],
    target: Dict[str, Any],
    constraint_types: Optional[List[str]] = None,
  ) -> ComparisonResult:
    """
    比较预测约束集与目标约束集。

    Args:
        predicted: VLM 输出的约束集
        target: 比较目标 (GT 或 Perception UB)
        constraint_types: 要比较的类型; None=全部支持类型

    Returns:
        ComparisonResult
    """
    if constraint_types is None:
      constraint_types = self.SUPPORTED_TYPES

    all_matches = []
    by_type = {}

    for ctype in constraint_types:
      pred_list = predicted.get(ctype, [])
      tgt_list = target.get(ctype, [])
      gt_list = (
        self._gt_ann.get(ctype, [])
        if self._gt_ann else None
      )

      matches = self._compare_type(
        ctype, pred_list, tgt_list, gt_list
      )
      all_matches.extend(matches)

      # 按类型统计
      type_result = self._aggregate(matches)
      type_result.scene_id = ""
      by_type[ctype] = type_result

    result = self._aggregate(all_matches)
    result.matches = all_matches
    result.by_type = by_type
    return result

  def compare_batch(
    self,
    predictions: List[Dict],
    targets: List[Dict],
    scene_ids: Optional[List[str]] = None,
    constraint_types: Optional[List[str]] = None,
  ) -> ComparisonResult:
    """
    批量比较并聚合指标。
    """
    all_matches = []
    for i, (pred, tgt) in enumerate(
      zip(predictions, targets)
    ):
      r = self.compare(pred, tgt, constraint_types)
      sid = (
        scene_ids[i] if scene_ids
        else f"scene_{i}"
      )
      for m in r.matches:
        m.details["scene_id"] = sid
      all_matches.extend(r.matches)

    return self._aggregate(all_matches)

  # -------------------------------------------------------
  # 内部: 分类型比较
  # -------------------------------------------------------

  def _compare_type(
    self,
    ctype: str,
    pred_list: List[Dict],
    tgt_list: List[Dict],
    gt_list: Optional[List[Dict]],
  ) -> List[ConstraintMatch]:
    """分发到具体类型的比较函数。"""
    dispatch = {
      "qrr": self._compare_qrr,
      "axial": self._compare_axial,
      "size": self._compare_size,
      "topology": self._compare_topology,
      "occlusion": self._compare_occlusion,
      "closer": self._compare_closer,
      "trr": self._compare_trr,
    }
    fn = dispatch.get(ctype)
    if fn is None:
      return []
    return fn(pred_list, tgt_list, gt_list)

  def _compare_qrr(
    self, pred_list, tgt_list, gt_list
  ) -> List[ConstraintMatch]:
    return self._generic_compare(
      "qrr", pred_list, tgt_list, gt_list,
      _qrr_canonical, _classify_qrr,
    )

  def _compare_axial(
    self, pred_list, tgt_list, gt_list
  ) -> List[ConstraintMatch]:
    return self._generic_compare(
      "axial", pred_list, tgt_list, gt_list,
      _axial_canonical,
      lambda pv, tv: _classify_directional(
        pv, tv, _AXIAL_INVERSE
      ),
    )

  def _compare_size(
    self, pred_list, tgt_list, gt_list
  ) -> List[ConstraintMatch]:
    return self._generic_compare(
      "size", pred_list, tgt_list, gt_list,
      _size_canonical,
      lambda pv, tv: _classify_directional(
        pv, tv, _SIZE_INVERSE
      ),
    )

  def _compare_topology(
    self, pred_list, tgt_list, gt_list
  ) -> List[ConstraintMatch]:
    return self._generic_compare(
      "topology", pred_list, tgt_list, gt_list,
      _topology_canonical,
      lambda pv, tv: (
        {"match": True} if pv == tv
        else {"match": False, "is_flip": False,
              "sub_type": "other",
              "pred_value": pv, "target_value": tv}
      ),
    )

  def _compare_occlusion(
    self, pred_list, tgt_list, gt_list
  ) -> List[ConstraintMatch]:
    return self._generic_compare(
      "occlusion", pred_list, tgt_list, gt_list,
      _occlusion_canonical,
      lambda pv, tv: {"match": True},
    )

  def _compare_closer(
    self, pred_list, tgt_list, gt_list
  ) -> List[ConstraintMatch]:
    return self._generic_compare(
      "closer", pred_list, tgt_list, gt_list,
      _closer_canonical,
      lambda pv, tv: (
        {"match": True} if pv == tv
        else {"match": False, "is_flip": True,
              "sub_type": "flip",
              "pred_value": pv, "target_value": tv}
      ),
    )

  def _compare_trr(
    self, pred_list, tgt_list, gt_list
  ) -> List[ConstraintMatch]:
    return self._generic_compare(
      "trr", pred_list, tgt_list, gt_list,
      _trr_canonical, _classify_trr,
    )

  # -------------------------------------------------------
  # 通用比较框架
  # -------------------------------------------------------

  def _generic_compare(
    self,
    ctype: str,
    pred_list: List[Dict],
    tgt_list: List[Dict],
    gt_list: Optional[List[Dict]],
    canonical_fn,
    classify_fn,
  ) -> List[ConstraintMatch]:
    """
    通用比较模式:
    1. 构建 lookup
    2. matched → correct 或 wrong_value
    3. target-only → missing
    4. pred-only → hallucinated
    """
    # 构建 lookup: key → (value, original_dict)
    pred_lookup = {}
    for c in pred_list:
      key, val = canonical_fn(c)
      pred_lookup[key] = (val, c)

    tgt_lookup = {}
    for c in tgt_list:
      key, val = canonical_fn(c)
      tgt_lookup[key] = (val, c)

    # GT lookup (用于 hallucinated 附注)
    gt_lookup = {}
    if gt_list:
      for c in gt_list:
        key, val = canonical_fn(c)
        gt_lookup[key] = (val, c)

    matches = []

    # 1. 匹配的 keys
    matched_keys = (
      set(pred_lookup.keys()) & set(tgt_lookup.keys())
    )
    for key in matched_keys:
      pred_val, pred_c = pred_lookup[key]
      tgt_val, tgt_c = tgt_lookup[key]
      result = classify_fn(pred_val, tgt_val)

      if result.get("match"):
        category = "correct"
      else:
        category = "wrong_value"

      matches.append(ConstraintMatch(
        constraint_type=ctype,
        key=key,
        category=category,
        predicted=pred_c,
        target=tgt_c,
        details=result,
      ))

    # 2. Missing: target 有, pred 无
    missing_keys = (
      set(tgt_lookup.keys()) - set(pred_lookup.keys())
    )
    for key in missing_keys:
      _, tgt_c = tgt_lookup[key]
      matches.append(ConstraintMatch(
        constraint_type=ctype,
        key=key,
        category="missing",
        target=tgt_c,
      ))

    # 3. Hallucinated: pred 有, target 无
    extra_keys = (
      set(pred_lookup.keys()) - set(tgt_lookup.keys())
    )
    for key in extra_keys:
      pred_val, pred_c = pred_lookup[key]
      details = {}

      # 检查 occlusion 方向反转: pred 有 (A,B),
      # target 有 (B,A)? 这是 wrong_value 而非 hallucinated
      if ctype == "occlusion":
        reversed_key = (key[1], key[0])
        if reversed_key in tgt_lookup:
          tgt_val, tgt_c = tgt_lookup[reversed_key]
          matches.append(ConstraintMatch(
            constraint_type=ctype,
            key=key,
            category="wrong_value",
            predicted=pred_c,
            target=tgt_c,
            details={
              "match": False, "is_flip": True,
              "sub_type": "flip",
            },
          ))
          continue

      # 可选: GT 附注
      if gt_lookup and key in gt_lookup:
        gt_val, _ = gt_lookup[key]
        details["gt_exists"] = True
        details["gt_value"] = gt_val
        # 检查 VLM 是否猜对了 GT 的值
        gt_check = classify_fn(pred_val, gt_val)
        details["gt_match"] = gt_check.get(
          "match", False
        )
      elif gt_lookup:
        details["gt_exists"] = False

      matches.append(ConstraintMatch(
        constraint_type=ctype,
        key=key,
        category="hallucinated",
        predicted=pred_c,
        details=details,
      ))

    return matches

  # -------------------------------------------------------
  # 聚合
  # -------------------------------------------------------

  @staticmethod
  def _aggregate(
    matches: List[ConstraintMatch],
  ) -> ComparisonResult:
    """从 matches 列表计算聚合指标。"""
    result = ComparisonResult()

    for m in matches:
      if m.category == "correct":
        result.n_correct += 1
        result.n_target += 1
        result.n_predicted += 1
      elif m.category == "wrong_value":
        result.n_wrong += 1
        result.n_target += 1
        result.n_predicted += 1
      elif m.category == "missing":
        result.n_missing += 1
        result.n_target += 1
      elif m.category == "hallucinated":
        result.n_hallucinated += 1
        result.n_predicted += 1

    result._compute_rates()
    return result
