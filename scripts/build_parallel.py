#!/usr/bin/env python3
"""
多 GPU 并行生成 ORDINAL-SPATIAL 数据集（扁平模式）。
Multi-GPU parallel dataset generation for ORDINAL-SPATIAL.

Usage:
  python scripts/build_parallel.py -o ./data/bench -n 1000 --n-gpus 4
  python scripts/build_parallel.py -o ./data/bench -n 8000 --n-gpus 8 \
      --quality high --min-objects 3 --max-objects 10
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.lib.multi_gpu_builder import MultiGPUBuilder


def main():
  parser = argparse.ArgumentParser(
    description="Multi-GPU parallel dataset generation",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python scripts/build_parallel.py -o ./data/test -n 80 --n-gpus 2
  python scripts/build_parallel.py -o ./data/full -n 10000 --n-gpus 8
  python scripts/build_parallel.py -o ./data/full -n 1000 --n-gpus 0
    """
  )

  parser.add_argument(
    "--output", "-o", required=True,
    help="Output directory"
  )
  parser.add_argument(
    "--n-scenes", "-n", type=int, default=1000,
    help="Total number of scenes (default: 1000)"
  )

  # 物体数量范围
  parser.add_argument(
    "--min-objects", type=int, default=3,
    help="Min objects per scene (default: 3)"
  )
  parser.add_argument(
    "--max-objects", type=int, default=10,
    help="Max objects per scene (default: 10)"
  )
  parser.add_argument(
    "--tau", type=float, default=0.10,
    help="Tolerance threshold (default: 0.10)"
  )

  # 渲染质量
  parser.add_argument(
    "--quality",
    choices=["draft", "normal", "high"],
    default="normal",
    help="Render quality (default: normal)"
  )

  # GPU / Blender
  default_blender = (
    os.environ.get("BLENDER_PATH")
    or os.environ.get("BLENDER_BIN")
    or "blender"
  )
  parser.add_argument(
    "--n-gpus", type=int, default=1,
    help="Number of GPUs; 0 = CPU only (default: 1)"
  )
  parser.add_argument(
    "--blender", default=default_blender,
    help="Blender path (env: BLENDER_PATH)"
  )
  parser.add_argument(
    "--seed", type=int, default=42,
    help="Random seed (default: 42)"
  )
  parser.add_argument(
    "--yes", "-y", action="store_true",
    help="Skip confirmation prompt"
  )

  args = parser.parse_args()

  n_levels = args.max_objects - args.min_objects + 1
  per_level = args.n_scenes // n_levels
  device = (
    f"{args.n_gpus} GPU(s)" if args.n_gpus > 0
    else "CPU"
  )

  print("\n" + "=" * 60)
  print("ORDINAL-SPATIAL Parallel Dataset Builder")
  print("=" * 60)
  print(f"Output  : {args.output}")
  print(f"Scenes  : {args.n_scenes}")
  print(
    f"Objects : {args.min_objects}-{args.max_objects} "
    f"({n_levels} levels, ~{per_level}/level)"
  )
  print(f"Tau     : {args.tau}")
  print(f"Quality : {args.quality}")
  print(f"Device  : {device}")
  print(f"Seed    : {args.seed}")
  print("=" * 60 + "\n")

  if not args.yes:
    response = input("Start generation? (y/N): ")
    if response.lower() != 'y':
      print("Cancelled")
      return

  builder = MultiGPUBuilder(
    output_dir=args.output,
    blender_path=args.blender,
    n_gpus=args.n_gpus,
    n_scenes=args.n_scenes,
    min_objects=args.min_objects,
    max_objects=args.max_objects,
    tau=args.tau,
    quality=args.quality,
    seed=args.seed,
  )

  builder.build()


if __name__ == "__main__":
  main()
