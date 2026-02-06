#!/usr/bin/env python3
"""
命令行工具 - 从约束 DSL 重建 3D 场景。

CLI Tool - Reconstruct 3D scene from constraint DSL.

Usage:
    python -m ordinal_spatial.reconstruction.cli -i constraints.json -o ./output
    python -m ordinal_spatial.reconstruction.cli --input constraints.json --visualizer plotly --show
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from .pipeline import reconstruct_and_visualize, PipelineConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(
        description="Reconstruct 3D scene from constraint DSL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic reconstruction
  python -m ordinal_spatial.reconstruction.cli -i constraints.json -o ./output

  # With Plotly visualization
  python -m ordinal_spatial.reconstruction.cli -i constraints.json -o ./output --visualizer plotly

  # Show interactive visualization
  python -m ordinal_spatial.reconstruction.cli -i constraints.json --visualizer plotly --show

  # Custom solver settings
  python -m ordinal_spatial.reconstruction.cli -i constraints.json -o ./output --max-iterations 5000 --tau 0.05
        """,
    )

    # 输入输出
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input constraints JSON file",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory (optional)",
    )

    # 求解器选项
    parser.add_argument(
        "--solver",
        choices=["pytorch", "numpy"],
        default="pytorch",
        help="Solver type (default: pytorch)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2000,
        help="Maximum optimization iterations (default: 2000)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate (default: 0.05)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.10,
        help="Tolerance parameter (default: 0.10)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU if available",
    )

    # 可视化选项
    parser.add_argument(
        "--visualizer",
        choices=["matplotlib", "plotly", "simple"],
        default="matplotlib",
        help="Visualization backend (default: matplotlib)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show visualization (opens window or browser)",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Hide object labels in visualization",
    )
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        default=[10, 10],
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches (default: 10 10)",
    )

    # 其他选项
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output",
    )

    args = parser.parse_args()

    # 设置日志级别
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 检查输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # 创建配置
    config = PipelineConfig(
        solver_type=args.solver,
        max_iterations=args.max_iterations,
        learning_rate=args.learning_rate,
        tau=args.tau,
        use_gpu=args.gpu,
        visualizer=args.visualizer,
        figsize=tuple(args.figsize),
        show_labels=not args.no_labels,
    )

    try:
        # 运行重建
        result = reconstruct_and_visualize(
            constraints_input=str(input_path),
            output_dir=args.output,
            config=config,
            show=args.show,
        )

        # 打印摘要
        if not args.quiet:
            print("\n" + result.summary())

        # 成功退出
        sys.exit(0)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        sys.exit(1)
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
