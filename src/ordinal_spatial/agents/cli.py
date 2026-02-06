#!/usr/bin/env python3
"""
VLM 约束提取智能体的命令行接口。

CLI for VLM Constraint Extraction Agent.

Usage:
    # Single-view extraction (Task-3)
    python -m ordinal_spatial.agents.cli extract --image scene.png --output constraints.json

    # Multi-view extraction (Task-2)
    python -m ordinal_spatial.agents.cli extract --images view1.png view2.png --output constraints.json

    # With custom model
    python -m ordinal_spatial.agents.cli extract --image scene.png --model openai/gpt-4o --output constraints.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from ordinal_spatial.agents.vlm_constraint_agent import VLMConstraintAgent, VLMAgentConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def extract_command(args: argparse.Namespace) -> int:
    """
    执行约束提取命令。

    Execute constraint extraction command.
    """
    # Collect images
    images: List[str] = []

    if args.image:
        images.append(args.image)

    if args.images:
        images.extend(args.images)

    if not images:
        logger.error("No images provided. Use --image or --images")
        return 1

    # Validate images exist
    for img_path in images:
        if not Path(img_path).exists():
            logger.error(f"Image not found: {img_path}")
            return 1

    # Create config
    config = VLMAgentConfig(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    if args.api_key:
        config.api_key = args.api_key

    # Create agent
    agent = VLMConstraintAgent(config)

    # Load objects if provided
    objects = None
    if args.objects:
        try:
            with open(args.objects) as f:
                objects = json.load(f)
                if isinstance(objects, dict) and "objects" in objects:
                    objects = objects["objects"]
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load objects file: {e}")
            return 1

    # Extract constraints
    try:
        logger.info(f"Extracting constraints from {len(images)} image(s)...")

        if len(images) == 1:
            result = agent.extract_from_single_view(
                images[0],
                objects=objects,
                tau=args.tau,
            )
        else:
            result = agent.extract_from_multi_view(
                images,
                objects=objects,
                tau=args.tau,
            )

        logger.info(f"Extraction complete. Confidence: {result.confidence:.2f}")
        logger.info(f"Total constraints: {result.total_constraints()}")

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    # Output result
    output_data = result.to_dict()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {output_path}")
    else:
        # Print to stdout
        print(json.dumps(output_data, indent=2, ensure_ascii=False))

    # Print summary if not quiet
    if not args.quiet:
        print("\n" + "=" * 50)
        print("Extraction Summary")
        print("=" * 50)
        print(result.summary())

    return 0


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(
        description="VLM Constraint Extraction Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-view extraction (Task-3)
  python -m ordinal_spatial.agents.cli extract --image scene.png -o constraints.json

  # Multi-view extraction (Task-2)
  python -m ordinal_spatial.agents.cli extract --images v1.png v2.png v3.png -o constraints.json

  # With known objects
  python -m ordinal_spatial.agents.cli extract --image scene.png --objects scene_objects.json -o constraints.json

  # Custom model
  python -m ordinal_spatial.agents.cli extract --image scene.png --model openai/gpt-4o -o constraints.json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Extract command
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract constraints from image(s)",
    )

    # Input options
    input_group = extract_parser.add_argument_group("Input")
    input_group.add_argument(
        "--image", "-i",
        help="Single image file path",
    )
    input_group.add_argument(
        "--images",
        nargs="+",
        help="Multiple image file paths (for multi-view)",
    )
    input_group.add_argument(
        "--objects",
        help="JSON file with known objects in the scene",
    )

    # Output options
    output_group = extract_parser.add_argument_group("Output")
    output_group.add_argument(
        "--output", "-o",
        help="Output JSON file path (default: stdout)",
    )

    # Model options
    model_group = extract_parser.add_argument_group("Model")
    model_group.add_argument(
        "--model", "-m",
        default="google/gemma-3-27b-it",
        help="VLM model name (default: google/gemma-3-27b-it)",
    )
    model_group.add_argument(
        "--api-key",
        help="API key (default: OPENROUTER_API_KEY env var)",
    )
    model_group.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    model_group.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens (default: 4096)",
    )

    # Extraction options
    extract_group = extract_parser.add_argument_group("Extraction")
    extract_group.add_argument(
        "--tau",
        type=float,
        default=0.10,
        help="Tolerance parameter (default: 0.10)",
    )

    # General options
    extract_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    extract_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress summary output",
    )

    extract_parser.set_defaults(func=extract_command)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Set logging level
    if hasattr(args, 'verbose') and args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    if hasattr(args, 'quiet') and args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Execute command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
