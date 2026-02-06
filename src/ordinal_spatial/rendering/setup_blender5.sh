#!/bin/bash
# Setup script for CLEVR dataset generation with Blender 5.0
#
# Usage:
#   ./setup_blender5.sh [BLENDER_PATH]
#
# Example:
#   ./setup_blender5.sh /usr/bin/blender
#   ./setup_blender5.sh /Applications/Blender.app/Contents/MacOS/Blender

set -e

# Default Blender path
BLENDER="${1:-blender}"

echo "=============================================="
echo "CLEVR Dataset Generator - Blender 5.0 Setup"
echo "=============================================="
echo ""

# Check Blender version
echo "Checking Blender installation..."
BLENDER_VERSION=$($BLENDER --version 2>/dev/null | head -n1 || echo "Not found")
echo "Found: $BLENDER_VERSION"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Working directory: $SCRIPT_DIR"
echo ""

# Step 1: Create base scene
echo "[1/3] Creating base scene..."
$BLENDER --background --python create_base_scene.py
echo ""

# Step 2: Create materials
echo "[2/3] Creating materials..."
$BLENDER --background --python create_materials.py
echo ""

# Step 3: Create shapes
echo "[3/3] Creating shapes..."
$BLENDER --background --python create_shapes.py
echo ""

# Step 4: Add clevr.pth to Blender's Python
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "New files created:"
echo "  - data/base_scene_v5.blend"
echo "  - data/materials_v5/Rubber.blend"
echo "  - data/materials_v5/MyMetal.blend"
echo "  - data/shapes_v5/SmoothCube_v2.blend"
echo "  - data/shapes_v5/Sphere.blend"
echo "  - data/shapes_v5/SmoothCylinder.blend"
echo ""
echo "To render images, run:"
echo ""
echo "  $BLENDER --background --python render_images.py -- \\"
echo "    --base_scene_blendfile data/base_scene_v5.blend \\"
echo "    --material_dir data/materials_v5 \\"
echo "    --shape_dir data/shapes_v5 \\"
echo "    --num_images 10"
echo ""
echo "For GPU acceleration (NVIDIA CUDA):"
echo ""
echo "  $BLENDER --background --python render_images.py -- \\"
echo "    --base_scene_blendfile data/base_scene_v5.blend \\"
echo "    --material_dir data/materials_v5 \\"
echo "    --shape_dir data/shapes_v5 \\"
echo "    --num_images 10 \\"
echo "    --use_gpu 1"
echo ""

# Note about clevr.pth
echo "=============================================="
echo "IMPORTANT: Python Path Setup"
echo "=============================================="
echo ""
echo "You may need to add this directory to Blender's Python path."
echo "Find your Blender Python site-packages directory and run:"
echo ""
echo "  echo '$SCRIPT_DIR' >> \$BLENDER_PATH/\$VERSION/python/lib/python*/site-packages/clevr.pth"
echo ""
echo "Example for Blender 5.0:"
echo "  echo '$SCRIPT_DIR' >> ~/.config/blender/5.0/python/lib/python3.11/site-packages/clevr.pth"
echo ""
