#!/usr/bin/env python3
"""
Build ORDINAL-SPATIAL benchmark dataset.

This script orchestrates the complete benchmark generation pipeline:
1. Generate scene configurations
2. Render images (single-view and multi-view)
3. Extract ground truth constraints
4. Create dataset splits

Usage:
    python -m ordinal_spatial.scripts.build_benchmark \
        --output-dir ./data/benchmark \
        --blender-path /path/to/blender \
        --n-train 1000 --n-val 200 --n-test 200

For small test run:
    python -m ordinal_spatial.scripts.build_benchmark \
        --output-dir ./data/benchmark_test \
        --blender-path /path/to/blender \
        --small
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """Configuration for a dataset split."""
    name: str
    n_scenes: int
    min_objects: int
    max_objects: int
    tau: float
    description: str


@dataclass
class BenchmarkConfig:
    """Complete benchmark configuration."""
    output_dir: str
    blender_path: str
    n_views: int = 4
    image_width: int = 480
    image_height: int = 320
    camera_distance: float = 12.0
    elevation: float = 30.0
    use_gpu: bool = False
    render_samples: int = 256
    random_seed: int = 42

    # Split configurations
    splits: List[SplitConfig] = None

    def __post_init__(self):
        if self.splits is None:
            self.splits = [
                SplitConfig("train", 1000, 4, 10, 0.10,
                           "Training set with standard configuration"),
                SplitConfig("val", 200, 4, 10, 0.10,
                           "Validation set, same distribution as train"),
                SplitConfig("test_iid", 200, 4, 10, 0.10,
                           "IID test set, same distribution as train"),
                SplitConfig("test_comp", 100, 10, 15, 0.10,
                           "Compositional test: more objects"),
                SplitConfig("test_hard", 100, 4, 10, 0.05,
                           "Hard test: stricter tolerance (tau=0.05)"),
            ]


class BenchmarkBuilder:
    """
    Orchestrates benchmark dataset generation.
    """

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize benchmark builder.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        # 渲染模块路径 (相对于包根目录)
        self.rendering_dir = Path(__file__).parent.parent / "rendering"

    def build(self) -> Dict[str, Any]:
        """
        Build the complete benchmark dataset.

        Returns:
            Dataset statistics
        """
        logger.info("=" * 60)
        logger.info("ORDINAL-SPATIAL Benchmark Builder")
        logger.info("=" * 60)

        # Create directory structure
        self._create_directories()

        # Build each split
        all_stats = {}
        for split_config in self.config.splits:
            logger.info(f"\n{'='*40}")
            logger.info(f"Building split: {split_config.name}")
            logger.info(f"  Scenes: {split_config.n_scenes}")
            logger.info(f"  Objects: {split_config.min_objects}-{split_config.max_objects}")
            logger.info(f"  Tau: {split_config.tau}")
            logger.info(f"{'='*40}")

            stats = self._build_split(split_config)
            all_stats[split_config.name] = stats

        # Generate dataset info
        self._save_dataset_info(all_stats)

        logger.info("\n" + "=" * 60)
        logger.info("Benchmark generation complete!")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 60)

        return all_stats

    def _create_directories(self):
        """Create output directory structure."""
        dirs = [
            self.output_dir,
            self.output_dir / "images" / "single_view",
            self.output_dir / "images" / "multi_view",
            self.output_dir / "splits",
            self.output_dir / "metadata",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {d}")

    def _build_split(self, split_config: SplitConfig) -> Dict[str, Any]:
        """
        Build a single dataset split.

        Args:
            split_config: Split configuration

        Returns:
            Split statistics
        """
        # Step 1: Render images
        logger.info(f"Step 1: Rendering {split_config.n_scenes} scenes...")
        render_output = self._render_split(split_config)

        # Step 2: Extract constraints
        logger.info("Step 2: Extracting ground truth constraints...")
        constraints_data = self._extract_constraints(render_output, split_config)

        # Step 3: Build split file
        logger.info("Step 3: Building split file...")
        split_data = self._build_split_file(
            render_output, constraints_data, split_config
        )

        # Save split file
        split_file = self.output_dir / "splits" / f"{split_config.name}.json"
        with open(split_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        logger.info(f"Saved split file: {split_file}")

        return {
            "n_scenes": len(split_data),
            "n_single_view_images": len(split_data),
            "n_multi_view_images": len(split_data) * self.config.n_views,
        }

    def _render_split(self, split_config: SplitConfig) -> Path:
        """
        Render images for a split using Blender.

        Args:
            split_config: Split configuration

        Returns:
            Path to render output directory
        """
        # 使用本地路径
        render_output = self.output_dir / "render_temp" / split_config.name
        blender_output = str(render_output)

        render_output.mkdir(parents=True, exist_ok=True)

        # 构建 Blender 命令
        render_script = str(self.rendering_dir / "render_multiview.py")
        data_dir = self.rendering_dir / "data"
        cmd = [
            self.config.blender_path,
            "--background",
            "--python", render_script,
            "--",
            "--base_scene_blendfile", str(data_dir / "base_scene_v5.blend"),
            "--properties_json", str(data_dir / "properties.json"),
            "--shape_dir", str(data_dir / "shapes_v5"),
            "--material_dir", str(data_dir / "materials_v5"),
            "--output_dir", blender_output,
            "--split", split_config.name,
            "--num_images", str(split_config.n_scenes),
            "--min_objects", str(split_config.min_objects),
            "--max_objects", str(split_config.max_objects),
            "--n_views", str(self.config.n_views),
            "--camera_distance", str(self.config.camera_distance),
            "--elevation", str(self.config.elevation),
            "--width", str(self.config.image_width),
            "--height", str(self.config.image_height),
            "--render_num_samples", str(self.config.render_samples),
        ]

        if self.config.use_gpu:
            cmd.extend(["--use_gpu", "1"])

        logger.info(f"Running Blender render to {blender_output}...")

        try:
            result = subprocess.run(
                cmd,
                timeout=3600 * 4
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"Render failed for split {split_config.name}"
                )

            logger.info("Render completed successfully")

        except subprocess.TimeoutExpired:
            logger.error("Render timed out")
            raise

        return render_output

    def _extract_constraints(
        self,
        render_output: Path,
        split_config: SplitConfig
    ) -> Dict[str, Dict]:
        """
        Extract ground truth constraints from rendered scenes.

        Args:
            render_output: Path to render output
            split_config: Split configuration

        Returns:
            Dictionary mapping scene_id to constraints
        """
        # Import constraint extraction
        try:
            from ordinal_spatial.agents import BlenderConstraintAgent
        except ImportError:
            logger.warning("BlenderConstraintAgent not available, using scene data directly")
            return {}

        # Load scene data
        scenes_file = render_output / f"{split_config.name}_scenes.json"
        if not scenes_file.exists():
            logger.warning(f"Scenes file not found: {scenes_file}")
            return {}

        with open(scenes_file) as f:
            scenes_data = json.load(f)

        constraints_data = {}
        agent = BlenderConstraintAgent()

        for scene in scenes_data.get("scenes", []):
            scene_id = scene.get("scene_id", "")
            try:
                # Extract constraints using agent
                # extract_from_single_view accepts dict input
                constraint_set = agent.extract_from_single_view(
                    image=scene,  # Pass scene dict directly
                    tau=split_config.tau
                )
                constraints_data[scene_id] = constraint_set.to_dict()
            except Exception as e:
                logger.warning(f"Failed to extract constraints for {scene_id}: {e}")
                # Use basic constraints from scene
                constraints_data[scene_id] = scene.get("world_constraints", {})

        return constraints_data

    def _build_split_file(
        self,
        render_output: Path,
        constraints_data: Dict[str, Dict],
        split_config: SplitConfig
    ) -> List[Dict]:
        """
        Build the split JSON file.

        Args:
            render_output: Path to render output
            constraints_data: Extracted constraints
            split_config: Split configuration

        Returns:
            List of scene entries for split file
        """
        split_data = []

        # Load scene data
        scenes_file = render_output / f"{split_config.name}_scenes.json"
        if not scenes_file.exists():
            logger.error(f"Scenes file not found: {scenes_file}")
            return split_data

        with open(scenes_file) as f:
            scenes_data = json.load(f)

        # Process each scene
        for scene in scenes_data.get("scenes", []):
            scene_id = scene.get("scene_id", "")

            # Copy images to final location
            src_multiview = render_output / "multi_view" / scene_id
            dst_multiview = self.output_dir / "images" / "multi_view" / scene_id

            if src_multiview.exists():
                if dst_multiview.exists():
                    shutil.rmtree(dst_multiview)
                shutil.copytree(src_multiview, dst_multiview)

            # Copy single view
            src_single = render_output / "single_view" / f"{scene_id}.png"
            dst_single = self.output_dir / "images" / "single_view" / f"{scene_id}.png"

            if src_single.exists():
                shutil.copy(src_single, dst_single)

            # Save metadata
            metadata_file = self.output_dir / "metadata" / f"{scene_id}.json"
            scene_metadata = {
                **scene,
                "constraints": constraints_data.get(scene_id, {})
            }
            with open(metadata_file, 'w') as f:
                json.dump(scene_metadata, f, indent=2)

            # Build split entry
            multi_view_images = [
                f"images/multi_view/{scene_id}/view_{i}.png"
                for i in range(self.config.n_views)
            ]

            entry = {
                "scene_id": scene_id,
                "single_view_image": f"images/single_view/{scene_id}.png",
                "multi_view_images": multi_view_images,
                "metadata_path": f"metadata/{scene_id}.json",
                "n_objects": scene.get("n_objects", len(scene.get("objects", []))),
                "tau": split_config.tau,
                "split": split_config.name
            }

            split_data.append(entry)

        # Cleanup temp render output
        try:
            shutil.rmtree(render_output)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")

        return split_data

    def _save_dataset_info(self, stats: Dict[str, Any]):
        """Save dataset information file."""
        info = {
            "name": "ORDINAL-SPATIAL Benchmark",
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "config": {
                "n_views": self.config.n_views,
                "image_size": [self.config.image_width, self.config.image_height],
                "camera_distance": self.config.camera_distance,
                "elevation": self.config.elevation,
            },
            "splits": {
                split.name: {
                    "n_scenes": split.n_scenes,
                    "min_objects": split.min_objects,
                    "max_objects": split.max_objects,
                    "tau": split.tau,
                    "description": split.description,
                }
                for split in self.config.splits
            },
            "statistics": stats,
            "total_scenes": sum(s.get("n_scenes", 0) for s in stats.values()),
            "total_images": sum(
                s.get("n_single_view_images", 0) + s.get("n_multi_view_images", 0)
                for s in stats.values()
            ),
        }

        info_file = self.output_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)

        logger.info(f"Saved dataset info: {info_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build ORDINAL-SPATIAL benchmark dataset"
    )

    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Output directory for benchmark"
    )
    parser.add_argument(
        "--blender-path", "-b",
        default="blender",
        help="Path to Blender executable"
    )

    # Dataset size
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--n-test-comp", type=int, default=100)
    parser.add_argument("--n-test-hard", type=int, default=100)

    # Rendering settings
    parser.add_argument("--n-views", type=int, default=4)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--render-samples", type=int, default=256)

    # Camera settings
    parser.add_argument("--camera-distance", type=float, default=12.0)
    parser.add_argument("--elevation", type=float, default=30.0)

    # Presets
    parser.add_argument(
        "--small", action="store_true",
        help="Generate small test dataset"
    )
    parser.add_argument(
        "--tiny", action="store_true",
        help="Generate tiny dataset for debugging"
    )

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Apply presets
    if args.tiny:
        args.n_train = 5
        args.n_val = 2
        args.n_test = 2
        args.n_test_comp = 2
        args.n_test_hard = 2
        args.render_samples = 64
    elif args.small:
        args.n_train = 100
        args.n_val = 20
        args.n_test = 20
        args.n_test_comp = 10
        args.n_test_hard = 10
        args.render_samples = 128

    # Build config
    splits = [
        SplitConfig("train", args.n_train, 4, 10, 0.10,
                   "Training set"),
        SplitConfig("val", args.n_val, 4, 10, 0.10,
                   "Validation set"),
        SplitConfig("test_iid", args.n_test, 4, 10, 0.10,
                   "IID test set"),
        SplitConfig("test_comp", args.n_test_comp, 10, 15, 0.10,
                   "Compositional test (more objects)"),
        SplitConfig("test_hard", args.n_test_hard, 4, 10, 0.05,
                   "Hard test (stricter tau)"),
    ]

    config = BenchmarkConfig(
        output_dir=args.output_dir,
        blender_path=args.blender_path,
        n_views=args.n_views,
        image_width=args.width,
        image_height=args.height,
        camera_distance=args.camera_distance,
        elevation=args.elevation,
        use_gpu=args.use_gpu,
        render_samples=args.render_samples,
        random_seed=args.seed,
        splits=splits
    )

    # Build benchmark
    builder = BenchmarkBuilder(config)
    stats = builder.build()

    # Print summary
    print("\n" + "=" * 50)
    print("BENCHMARK GENERATION SUMMARY")
    print("=" * 50)
    for split_name, split_stats in stats.items():
        print(f"\n{split_name}:")
        for key, value in split_stats.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
