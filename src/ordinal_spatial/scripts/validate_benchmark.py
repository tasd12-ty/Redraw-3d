#!/usr/bin/env python3
"""
Validate ORDINAL-SPATIAL benchmark dataset.

Performs comprehensive validation checks:
1. Image integrity (files exist, readable, correct size)
2. JSON validity (parseable, correct schema)
3. Data consistency (objects match, constraints valid)
4. Statistics reporting

Usage:
    python -m ordinal_spatial.scripts.validate_benchmark \
        --dataset-dir ./data/benchmark \
        --output report.json
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

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
    """A validation error."""
    severity: str  # CRITICAL, WARNING, INFO
    scene_id: str
    category: str
    message: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SplitStatistics:
    """Statistics for a dataset split."""
    name: str
    n_scenes: int = 0
    n_single_view_images: int = 0
    n_multi_view_images: int = 0
    avg_objects: float = 0.0
    min_objects: int = 0
    max_objects: int = 0
    avg_qrr_constraints: float = 0.0
    avg_trr_constraints: float = 0.0
    avg_axial_constraints: float = 0.0
    tau_values: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ValidationReport:
    """Complete validation report."""
    dataset_dir: str
    timestamp: str
    valid: bool = True
    n_critical: int = 0
    n_warnings: int = 0
    n_info: int = 0
    errors: List[ValidationError] = field(default_factory=list)
    split_stats: Dict[str, SplitStatistics] = field(default_factory=dict)
    total_scenes: int = 0
    total_images: int = 0

    def add_error(self, error: ValidationError):
        """Add an error to the report."""
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
            "split_statistics": {
                name: stats.to_dict()
                for name, stats in self.split_stats.items()
            },
            "errors": [e.to_dict() for e in self.errors]
        }


class BenchmarkValidator:
    """
    Validates ORDINAL-SPATIAL benchmark dataset.
    """

    def __init__(self, dataset_dir: str):
        """
        Initialize validator.

        Args:
            dataset_dir: Path to benchmark dataset
        """
        self.dataset_dir = Path(dataset_dir)
        self.report = ValidationReport(
            dataset_dir=str(dataset_dir),
            timestamp=datetime.now().isoformat()
        )

    def validate(self) -> ValidationReport:
        """
        Run all validation checks.

        Returns:
            Validation report
        """
        logger.info("=" * 60)
        logger.info("ORDINAL-SPATIAL Benchmark Validator")
        logger.info(f"Dataset: {self.dataset_dir}")
        logger.info("=" * 60)

        # Check directory structure
        self._validate_structure()

        # Load and validate dataset info
        self._validate_dataset_info()

        # Validate each split
        splits_dir = self.dataset_dir / "splits"
        if splits_dir.exists():
            for split_file in sorted(splits_dir.glob("*.json")):
                split_name = split_file.stem
                logger.info(f"\nValidating split: {split_name}")
                self._validate_split(split_file)

        # Compute totals
        self.report.total_scenes = sum(
            s.n_scenes for s in self.report.split_stats.values()
        )
        self.report.total_images = sum(
            s.n_single_view_images + s.n_multi_view_images
            for s in self.report.split_stats.values()
        )

        return self.report

    def _validate_structure(self):
        """Validate directory structure."""
        logger.info("Checking directory structure...")

        required_dirs = [
            "splits",
            "images/single_view",
            "images/multi_view",
            "metadata"
        ]

        for dir_name in required_dirs:
            dir_path = self.dataset_dir / dir_name
            if not dir_path.exists():
                self.report.add_error(ValidationError(
                    severity="CRITICAL",
                    scene_id="",
                    category="structure",
                    message=f"Missing required directory: {dir_name}"
                ))
            else:
                logger.info(f"  [OK] {dir_name}")

    def _validate_dataset_info(self):
        """Validate dataset_info.json."""
        logger.info("Checking dataset_info.json...")

        info_file = self.dataset_dir / "dataset_info.json"
        if not info_file.exists():
            self.report.add_error(ValidationError(
                severity="WARNING",
                scene_id="",
                category="info",
                message="Missing dataset_info.json"
            ))
            return

        try:
            with open(info_file) as f:
                info = json.load(f)
            logger.info(f"  [OK] Dataset: {info.get('name', 'Unknown')}")
            logger.info(f"  [OK] Version: {info.get('version', 'Unknown')}")
        except json.JSONDecodeError as e:
            self.report.add_error(ValidationError(
                severity="CRITICAL",
                scene_id="",
                category="info",
                message=f"Invalid JSON in dataset_info.json: {e}"
            ))

    def _validate_split(self, split_file: Path):
        """
        Validate a single split.

        Args:
            split_file: Path to split JSON file
        """
        split_name = split_file.stem
        stats = SplitStatistics(name=split_name)

        # Load split file
        try:
            with open(split_file) as f:
                split_data = json.load(f)
        except json.JSONDecodeError as e:
            self.report.add_error(ValidationError(
                severity="CRITICAL",
                scene_id="",
                category="json",
                message=f"Invalid JSON in {split_file.name}: {e}"
            ))
            return

        if not isinstance(split_data, list):
            self.report.add_error(ValidationError(
                severity="CRITICAL",
                scene_id="",
                category="schema",
                message=f"Split file should contain a list, got {type(split_data)}"
            ))
            return

        stats.n_scenes = len(split_data)
        logger.info(f"  Scenes: {stats.n_scenes}")

        # Validate each scene
        object_counts = []
        qrr_counts = []
        trr_counts = []
        axial_counts = []
        taus = []

        for entry in split_data:
            scene_id = entry.get("scene_id", "unknown")

            # Check required fields
            required_fields = ["scene_id", "single_view_image", "multi_view_images"]
            for field in required_fields:
                if field not in entry:
                    self.report.add_error(ValidationError(
                        severity="CRITICAL",
                        scene_id=scene_id,
                        category="schema",
                        message=f"Missing required field: {field}"
                    ))

            # Validate single-view image
            single_view = entry.get("single_view_image", "")
            single_view_path = self.dataset_dir / single_view
            if single_view_path.exists():
                stats.n_single_view_images += 1
                self._validate_image(single_view_path, scene_id)
            else:
                self.report.add_error(ValidationError(
                    severity="CRITICAL",
                    scene_id=scene_id,
                    category="image",
                    message=f"Missing single-view image: {single_view}"
                ))

            # Validate multi-view images
            multi_views = entry.get("multi_view_images", [])
            for mv_path in multi_views:
                full_path = self.dataset_dir / mv_path
                if full_path.exists():
                    stats.n_multi_view_images += 1
                    self._validate_image(full_path, scene_id)
                else:
                    self.report.add_error(ValidationError(
                        severity="CRITICAL",
                        scene_id=scene_id,
                        category="image",
                        message=f"Missing multi-view image: {mv_path}"
                    ))

            # Validate metadata
            metadata_path = entry.get("metadata_path", "")
            if metadata_path:
                full_metadata = self.dataset_dir / metadata_path
                if full_metadata.exists():
                    metadata = self._validate_metadata(full_metadata, scene_id)
                    if metadata:
                        # Collect statistics
                        objects = metadata.get("objects", [])
                        object_counts.append(len(objects))

                        constraints = metadata.get("constraints", {})
                        qrr_counts.append(len(constraints.get("qrr", [])))
                        trr_counts.append(len(constraints.get("trr", [])))
                        axial_counts.append(len(constraints.get("axial", [])))
                else:
                    self.report.add_error(ValidationError(
                        severity="WARNING",
                        scene_id=scene_id,
                        category="metadata",
                        message=f"Missing metadata file: {metadata_path}"
                    ))

            # Collect tau
            tau = entry.get("tau", 0.1)
            taus.append(tau)

        # Compute statistics
        if object_counts:
            stats.avg_objects = sum(object_counts) / len(object_counts)
            stats.min_objects = min(object_counts)
            stats.max_objects = max(object_counts)

        if qrr_counts:
            stats.avg_qrr_constraints = sum(qrr_counts) / len(qrr_counts)
        if trr_counts:
            stats.avg_trr_constraints = sum(trr_counts) / len(trr_counts)
        if axial_counts:
            stats.avg_axial_constraints = sum(axial_counts) / len(axial_counts)

        stats.tau_values = list(set(taus))

        # Log statistics
        logger.info(f"  Single-view images: {stats.n_single_view_images}")
        logger.info(f"  Multi-view images: {stats.n_multi_view_images}")
        logger.info(f"  Avg objects: {stats.avg_objects:.1f}")
        logger.info(f"  Avg QRR constraints: {stats.avg_qrr_constraints:.1f}")

        self.report.split_stats[split_name] = stats

    def _validate_image(self, image_path: Path, scene_id: str) -> bool:
        """
        Validate a single image.

        Args:
            image_path: Path to image file
            scene_id: Scene identifier for error reporting

        Returns:
            True if valid
        """
        if not PIL_AVAILABLE:
            return True

        try:
            with Image.open(image_path) as img:
                # Check format
                if img.format not in ["PNG", "JPEG"]:
                    self.report.add_error(ValidationError(
                        severity="WARNING",
                        scene_id=scene_id,
                        category="image",
                        message=f"Unexpected format {img.format}: {image_path.name}"
                    ))

                # Check size is reasonable
                if img.width < 100 or img.height < 100:
                    self.report.add_error(ValidationError(
                        severity="WARNING",
                        scene_id=scene_id,
                        category="image",
                        message=f"Image too small ({img.width}x{img.height}): {image_path.name}"
                    ))

            return True

        except Exception as e:
            self.report.add_error(ValidationError(
                severity="CRITICAL",
                scene_id=scene_id,
                category="image",
                message=f"Cannot read image {image_path.name}: {e}"
            ))
            return False

    def _validate_metadata(
        self,
        metadata_path: Path,
        scene_id: str
    ) -> Optional[Dict]:
        """
        Validate metadata file.

        Args:
            metadata_path: Path to metadata JSON
            scene_id: Scene identifier

        Returns:
            Parsed metadata or None if invalid
        """
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)

            # Check required fields
            if "objects" not in metadata:
                self.report.add_error(ValidationError(
                    severity="WARNING",
                    scene_id=scene_id,
                    category="metadata",
                    message="Missing 'objects' field in metadata"
                ))

            # Validate objects have required fields
            for i, obj in enumerate(metadata.get("objects", [])):
                for field in ["3d_coords", "shape", "color"]:
                    if field not in obj:
                        self.report.add_error(ValidationError(
                            severity="WARNING",
                            scene_id=scene_id,
                            category="metadata",
                            message=f"Object {i} missing field: {field}"
                        ))

            return metadata

        except json.JSONDecodeError as e:
            self.report.add_error(ValidationError(
                severity="CRITICAL",
                scene_id=scene_id,
                category="metadata",
                message=f"Invalid JSON: {e}"
            ))
            return None

    def print_report(self):
        """Print validation report to console."""
        print("\n" + "=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)

        print(f"\nDataset: {self.report.dataset_dir}")
        print(f"Timestamp: {self.report.timestamp}")

        # Summary
        print("\n--- Summary ---")
        if self.report.valid:
            print("[PASS] Dataset is valid")
        else:
            print("[FAIL] Dataset has critical errors")

        print(f"  Critical errors: {self.report.n_critical}")
        print(f"  Warnings: {self.report.n_warnings}")
        print(f"  Info: {self.report.n_info}")
        print(f"  Total scenes: {self.report.total_scenes}")
        print(f"  Total images: {self.report.total_images}")

        # Split statistics
        print("\n--- Split Statistics ---")
        print(f"{'Split':<12} {'Scenes':>8} {'Objects':>10} {'QRR':>8} {'Images':>10}")
        print("-" * 50)

        for name, stats in sorted(self.report.split_stats.items()):
            total_images = stats.n_single_view_images + stats.n_multi_view_images
            print(
                f"{name:<12} {stats.n_scenes:>8} "
                f"{stats.avg_objects:>10.1f} {stats.avg_qrr_constraints:>8.1f} "
                f"{total_images:>10}"
            )

        # Errors
        if self.report.errors:
            print("\n--- Errors ---")
            for error in self.report.errors[:20]:  # Show first 20
                print(f"[{error.severity}] {error.scene_id}: {error.message}")

            if len(self.report.errors) > 20:
                print(f"... and {len(self.report.errors) - 20} more errors")

        print("\n" + "=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate ORDINAL-SPATIAL benchmark dataset"
    )

    parser.add_argument(
        "--dataset-dir", "-d",
        required=True,
        help="Path to benchmark dataset"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output path for JSON report"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run validation
    validator = BenchmarkValidator(args.dataset_dir)
    report = validator.validate()

    # Print report
    validator.print_report()

    # Save JSON report
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nSaved report to: {args.output}")

    # Exit with error code if validation failed
    if not report.valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
