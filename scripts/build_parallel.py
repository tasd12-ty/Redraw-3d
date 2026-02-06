#!/usr/bin/env python3
"""
多 GPU 并行生成 ORDINAL-SPATIAL 数据集。
Multi-GPU parallel dataset generation for ORDINAL-SPATIAL.

Usage:
  python scripts/build_parallel.py --output ./data/benchmark_full --size large
  python scripts/build_parallel.py --output ./data/test --size tiny --n-gpus 2
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.lib.multi_gpu_builder import MultiGPUBuilder, DatasetSize


def main():
  parser = argparse.ArgumentParser(
    description="Multi-GPU parallel dataset generation",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Dataset size presets:
  tiny   : 40 scenes
  small  : 1,520 scenes
  medium : 15,200 scenes
  large  : 152,000 scenes

Examples:
  python scripts/build_parallel.py --output ./data/test --size tiny --n-gpus 2
  python scripts/build_parallel.py --output ./data/full --size large --n-gpus 8
  python scripts/build_parallel.py --output ./data/full --size large --seed 42
    """
  )

  parser.add_argument(
    "--output", "-o", required=True,
    help="Output directory"
  )
  parser.add_argument(
    "--size",
    choices=["tiny", "small", "medium", "large"],
    default="small",
    help="Dataset size (default: small)"
  )
  parser.add_argument(
    "--quality",
    choices=["draft", "normal", "high"],
    default="normal",
    help="Render quality (default: normal)"
  )

  # 可选高级参数
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
    help="Blender path (env: BLENDER_PATH or BLENDER_BIN)"
  )
  parser.add_argument(
    "--seed", type=int, default=42,
    help="Random seed for reproducibility (default: 42)"
  )
  parser.add_argument(
    "--yes", "-y", action="store_true",
    help="Skip confirmation prompt (for non-interactive / server use)"
  )

  args = parser.parse_args()

  device = f"{args.n_gpus} GPU(s)" if args.n_gpus > 0 else "CPU"
  print("\n" + "=" * 70)
  print("ORDINAL-SPATIAL Parallel Dataset Generation")
  print("=" * 70)
  print(f"Output : {args.output}")
  print(f"Size   : {args.size}")
  print(f"Quality: {args.quality}")
  print(f"Device : {device}")
  print(f"Seed   : {args.seed}")
  print("=" * 70 + "\n")

  if not args.yes:
    response = input("Start generation? (y/N): ")
    if response.lower() != 'y':
      print("Cancelled")
      return

  builder = MultiGPUBuilder(
    output_dir=args.output,
    blender_path=args.blender,
    n_gpus=args.n_gpus,
    dataset_size=DatasetSize[args.size.upper()],
    quality=args.quality,
    seed=args.seed,
  )

  builder.build()


if __name__ == "__main__":
  main()
