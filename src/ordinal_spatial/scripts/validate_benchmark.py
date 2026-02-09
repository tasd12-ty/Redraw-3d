#!/usr/bin/env python3
"""
Validate ORDINAL-SPATIAL benchmark dataset (flat format).

验证扁平格式数据集: dataset.json + images/ + metadata/。

Usage:
    uv run os-validate -d ./data/benchmark
    uv run os-validate -d ./data/benchmark -o report.json
"""

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
  from PIL import Image
  PIL_AVAILABLE = True
except ImportError:
  PIL_AVAILABLE = False

logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
  """验证错误。"""
  severity: str   # CRITICAL, WARNING, INFO
  scene_id: str
  category: str
  message: str

  def to_dict(self) -> Dict:
    return asdict(self)


@dataclass
class DatasetStatistics:
  """数据集统计信息。"""
  n_scenes: int = 0
  n_single_view_images: int = 0
  n_multi_view_images: int = 0
  avg_objects: float = 0.0
  min_objects: int = 0
  max_objects: int = 0
  object_distribution: Dict[int, int] = field(
    default_factory=dict
  )
  avg_qrr_constraints: float = 0.0
  avg_trr_constraints: float = 0.0
  avg_axial_constraints: float = 0.0
  tau_values: List[float] = field(
    default_factory=list
  )

  def to_dict(self) -> Dict:
    return asdict(self)


@dataclass
class ValidationReport:
  """完整验证报告。"""
  dataset_dir: str
  timestamp: str
  valid: bool = True
  n_critical: int = 0
  n_warnings: int = 0
  n_info: int = 0
  errors: List[ValidationError] = field(
    default_factory=list
  )
  stats: Optional[DatasetStatistics] = None
  total_scenes: int = 0
  total_images: int = 0

  def add_error(self, error: ValidationError):
    """添加错误。"""
    self.errors.append(error)
    if error.severity == "CRITICAL":
      self.n_critical += 1
      self.valid = False
    elif error.severity == "WARNING":
      self.n_warnings += 1
    else:
      self.n_info += 1

  def to_dict(self) -> Dict:
    return {
      "dataset_dir": self.dataset_dir,
      "timestamp": self.timestamp,
      "valid": self.valid,
      "summary": {
        "critical_errors": self.n_critical,
        "warnings": self.n_warnings,
        "info": self.n_info,
        "total_scenes": self.total_scenes,
        "total_images": self.total_images,
      },
      "statistics": (
        self.stats.to_dict() if self.stats else {}
      ),
      "errors": [
        e.to_dict() for e in self.errors
      ],
    }


class BenchmarkValidator:
  """验证 ORDINAL-SPATIAL 扁平数据集。"""

  def __init__(self, dataset_dir: str):
    self.dataset_dir = Path(dataset_dir)
    self.report = ValidationReport(
      dataset_dir=str(dataset_dir),
      timestamp=datetime.now().isoformat(),
    )

  def validate(self) -> ValidationReport:
    """运行所有验证检查。"""
    logger.info("=" * 60)
    logger.info("ORDINAL-SPATIAL Benchmark Validator")
    logger.info(f"Dataset: {self.dataset_dir}")
    logger.info("=" * 60)

    # 检查目录结构
    self._validate_structure()

    # 检查 dataset_info.json
    self._validate_dataset_info()

    # 验证 dataset.json
    dataset_file = self.dataset_dir / "dataset.json"
    if dataset_file.exists():
      self._validate_dataset(dataset_file)
    else:
      self.report.add_error(ValidationError(
        severity="CRITICAL",
        scene_id="",
        category="structure",
        message="Missing dataset.json",
      ))

    return self.report

  def _validate_structure(self):
    """验证目录结构。"""
    logger.info("Checking directory structure...")

    required_dirs = [
      "images/single_view",
      "images/multi_view",
      "metadata",
    ]

    for dir_name in required_dirs:
      dir_path = self.dataset_dir / dir_name
      if not dir_path.exists():
        self.report.add_error(ValidationError(
          severity="CRITICAL",
          scene_id="",
          category="structure",
          message=(
            f"Missing required directory: {dir_name}"
          ),
        ))
      else:
        logger.info(f"  [OK] {dir_name}")

  def _validate_dataset_info(self):
    """验证 dataset_info.json。"""
    logger.info("Checking dataset_info.json...")

    info_file = self.dataset_dir / "dataset_info.json"
    if not info_file.exists():
      self.report.add_error(ValidationError(
        severity="WARNING",
        scene_id="",
        category="info",
        message="Missing dataset_info.json",
      ))
      return

    try:
      with open(info_file) as f:
        info = json.load(f)
      logger.info(
        f"  [OK] {info.get('name', 'Unknown')}"
      )
    except json.JSONDecodeError as e:
      self.report.add_error(ValidationError(
        severity="CRITICAL",
        scene_id="",
        category="info",
        message=f"Invalid JSON: {e}",
      ))

  def _validate_dataset(self, dataset_file: Path):
    """验证 dataset.json 及其引用的所有文件。"""
    logger.info("Validating dataset.json...")

    try:
      with open(dataset_file) as f:
        dataset = json.load(f)
    except json.JSONDecodeError as e:
      self.report.add_error(ValidationError(
        severity="CRITICAL",
        scene_id="",
        category="json",
        message=f"Invalid JSON: {e}",
      ))
      return

    if not isinstance(dataset, list):
      self.report.add_error(ValidationError(
        severity="CRITICAL",
        scene_id="",
        category="schema",
        message=(
          f"dataset.json should be a list, "
          f"got {type(dataset).__name__}"
        ),
      ))
      return

    stats = DatasetStatistics()
    stats.n_scenes = len(dataset)
    logger.info(f"  Scenes: {stats.n_scenes}")

    object_counts = []
    qrr_counts = []
    trr_counts = []
    axial_counts = []
    taus = []

    for entry in dataset:
      scene_id = entry.get("scene_id", "unknown")

      # 检查必需字段
      for fld in [
        "scene_id", "single_view_image",
        "multi_view_images",
      ]:
        if fld not in entry:
          self.report.add_error(ValidationError(
            severity="CRITICAL",
            scene_id=scene_id,
            category="schema",
            message=f"Missing field: {fld}",
          ))

      # 验证单视角图片
      sv = entry.get("single_view_image", "")
      sv_path = self.dataset_dir / sv
      if sv_path.exists():
        stats.n_single_view_images += 1
        self._validate_image(sv_path, scene_id)
      else:
        self.report.add_error(ValidationError(
          severity="CRITICAL",
          scene_id=scene_id,
          category="image",
          message=f"Missing: {sv}",
        ))

      # 验证多视角图片
      for mv in entry.get("multi_view_images", []):
        mv_path = self.dataset_dir / mv
        if mv_path.exists():
          stats.n_multi_view_images += 1
          self._validate_image(mv_path, scene_id)
        else:
          self.report.add_error(ValidationError(
            severity="CRITICAL",
            scene_id=scene_id,
            category="image",
            message=f"Missing: {mv}",
          ))

      # 验证元数据
      mp_ = entry.get("metadata_path", "")
      if mp_:
        full_mp = self.dataset_dir / mp_
        if full_mp.exists():
          metadata = self._validate_metadata(
            full_mp, scene_id
          )
          if metadata:
            objects = metadata.get("objects", [])
            object_counts.append(len(objects))

            cs = metadata.get("constraints", {})
            qrr_counts.append(
              len(cs.get("qrr", []))
            )
            trr_counts.append(
              len(cs.get("trr", []))
            )
            axial_counts.append(
              len(cs.get("axial", []))
            )
        else:
          self.report.add_error(ValidationError(
            severity="WARNING",
            scene_id=scene_id,
            category="metadata",
            message=f"Missing: {mp_}",
          ))

      taus.append(entry.get("tau", 0.1))

    # 计算统计信息
    if object_counts:
      stats.avg_objects = (
        sum(object_counts) / len(object_counts)
      )
      stats.min_objects = min(object_counts)
      stats.max_objects = max(object_counts)
      stats.object_distribution = dict(
        sorted(Counter(object_counts).items())
      )

    if qrr_counts:
      stats.avg_qrr_constraints = (
        sum(qrr_counts) / len(qrr_counts)
      )
    if trr_counts:
      stats.avg_trr_constraints = (
        sum(trr_counts) / len(trr_counts)
      )
    if axial_counts:
      stats.avg_axial_constraints = (
        sum(axial_counts) / len(axial_counts)
      )

    stats.tau_values = sorted(set(taus))

    self.report.stats = stats
    self.report.total_scenes = stats.n_scenes
    self.report.total_images = (
      stats.n_single_view_images
      + stats.n_multi_view_images
    )

  def _validate_image(
    self, image_path: Path, scene_id: str
  ) -> bool:
    """验证图片文件。"""
    if not PIL_AVAILABLE:
      return True

    try:
      with Image.open(image_path) as img:
        if img.format not in ["PNG", "JPEG"]:
          self.report.add_error(ValidationError(
            severity="WARNING",
            scene_id=scene_id,
            category="image",
            message=(
              f"Unexpected format {img.format}: "
              f"{image_path.name}"
            ),
          ))

        if img.width < 100 or img.height < 100:
          self.report.add_error(ValidationError(
            severity="WARNING",
            scene_id=scene_id,
            category="image",
            message=(
              f"Too small "
              f"({img.width}x{img.height}): "
              f"{image_path.name}"
            ),
          ))
      return True

    except Exception as e:
      self.report.add_error(ValidationError(
        severity="CRITICAL",
        scene_id=scene_id,
        category="image",
        message=(
          f"Cannot read {image_path.name}: {e}"
        ),
      ))
      return False

  def _validate_metadata(
    self, metadata_path: Path, scene_id: str
  ) -> Optional[Dict]:
    """验证元数据文件。"""
    try:
      with open(metadata_path) as f:
        metadata = json.load(f)

      if "objects" not in metadata:
        self.report.add_error(ValidationError(
          severity="WARNING",
          scene_id=scene_id,
          category="metadata",
          message="Missing 'objects' field",
        ))

      for i, obj in enumerate(
        metadata.get("objects", [])
      ):
        for fld in ["3d_coords", "shape", "color"]:
          if fld not in obj:
            self.report.add_error(ValidationError(
              severity="WARNING",
              scene_id=scene_id,
              category="metadata",
              message=(
                f"Object {i} missing: {fld}"
              ),
            ))

      return metadata

    except json.JSONDecodeError as e:
      self.report.add_error(ValidationError(
        severity="CRITICAL",
        scene_id=scene_id,
        category="metadata",
        message=f"Invalid JSON: {e}",
      ))
      return None

  def print_report(self):
    """打印验证报告。"""
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    print(f"\nDataset: {self.report.dataset_dir}")

    # 结论
    print("\n--- Summary ---")
    if self.report.valid:
      print("[PASS] Dataset is valid")
    else:
      print("[FAIL] Dataset has critical errors")

    print(
      f"  Critical errors: {self.report.n_critical}"
    )
    print(f"  Warnings: {self.report.n_warnings}")
    print(
      f"  Total scenes: {self.report.total_scenes}"
    )
    print(
      f"  Total images: {self.report.total_images}"
    )

    # 统计
    if self.report.stats:
      s = self.report.stats
      print("\n--- Statistics ---")
      print(
        f"  Single-view images: "
        f"{s.n_single_view_images}"
      )
      print(
        f"  Multi-view images: "
        f"{s.n_multi_view_images}"
      )
      print(
        f"  Objects range: "
        f"{s.min_objects}-{s.max_objects}"
      )
      print(f"  Avg objects: {s.avg_objects:.1f}")
      print(
        f"  Avg QRR constraints: "
        f"{s.avg_qrr_constraints:.1f}"
      )

      # 物体数量分布
      if s.object_distribution:
        print("\n--- Object Count Distribution ---")
        print(
          f"{'Objects':>8} {'Count':>8} "
          f"{'Percent':>8}"
        )
        print("-" * 28)
        total = sum(s.object_distribution.values())
        for k, v in s.object_distribution.items():
          pct = v / total * 100 if total else 0
          print(f"{k:>8} {v:>8} {pct:>7.1f}%")

    # 错误
    if self.report.errors:
      print("\n--- Errors ---")
      for err in self.report.errors[:20]:
        print(
          f"[{err.severity}] "
          f"{err.scene_id}: {err.message}"
        )

      if len(self.report.errors) > 20:
        print(
          f"... and "
          f"{len(self.report.errors) - 20} more"
        )

    print("\n" + "=" * 60)


def main():
  """Main entry point."""
  parser = argparse.ArgumentParser(
    description=(
      "Validate ORDINAL-SPATIAL benchmark dataset"
    )
  )

  parser.add_argument(
    "--dataset-dir", "-d", required=True,
    help="Path to benchmark dataset"
  )
  parser.add_argument(
    "--output", "-o",
    help="Output path for JSON report"
  )
  parser.add_argument(
    "--verbose", "-v", action="store_true",
    help="Verbose output"
  )

  args = parser.parse_args()

  if args.verbose:
    logging.getLogger().setLevel(logging.DEBUG)

  validator = BenchmarkValidator(args.dataset_dir)
  report = validator.validate()

  validator.print_report()

  if args.output:
    with open(args.output, 'w') as f:
      json.dump(report.to_dict(), f, indent=2)
    print(f"\nSaved report to: {args.output}")

  if not report.valid:
    sys.exit(1)


if __name__ == "__main__":
  main()
