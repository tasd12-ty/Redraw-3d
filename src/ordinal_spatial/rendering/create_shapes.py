# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

"""
Create shape .blend files compatible with Blender 5.0 / 4.x / 2.80+

Run this script from Blender to generate new shape files:
    blender --background --python create_shapes_blender5.py

This will create shape files in 'data/shapes_v5/' directory:
- SmoothCube_v2.blend - Beveled cube
- Sphere.blend - UV sphere
- SmoothCylinder.blend - Smooth cylinder
"""

import bpy
import os
import math

def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Clear orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

def create_smooth_cube():
    """
    Create a smooth/beveled cube centered at origin with unit size.
    The cube is slightly beveled for a smoother appearance.
    """
    # Create cube
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.5))
    cube = bpy.context.object
    cube.name = 'SmoothCube_v2'

    # Add bevel modifier for smooth edges
    bevel = cube.modifiers.new(name='Bevel', type='BEVEL')
    bevel.width = 0.05
    bevel.segments = 3

    # Apply the modifier
    bpy.context.view_layer.objects.active = cube
    bpy.ops.object.modifier_apply(modifier='Bevel')

    # Smooth shading
    bpy.ops.object.shade_smooth()

    # Move origin to bottom center
    bpy.context.scene.cursor.location = (0, 0, 0)
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')

    return cube

def create_sphere():
    """
    Create a UV sphere centered at origin with unit diameter.
    """
    # Create sphere with radius 0.5 (diameter 1.0)
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=0.5,
        segments=32,
        ring_count=16,
        location=(0, 0, 0.5)
    )
    sphere = bpy.context.object
    sphere.name = 'Sphere'

    # Smooth shading
    bpy.ops.object.shade_smooth()

    # Move origin to bottom center
    bpy.context.scene.cursor.location = (0, 0, 0)
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')

    return sphere

def create_smooth_cylinder():
    """
    Create a smooth cylinder centered at origin with unit size.
    """
    # Create cylinder with radius 0.5, height 1.0
    bpy.ops.mesh.primitive_cylinder_add(
        radius=0.5,
        depth=1.0,
        vertices=32,
        location=(0, 0, 0.5)
    )
    cylinder = bpy.context.object
    cylinder.name = 'SmoothCylinder'

    # Add bevel modifier for smooth top/bottom edges
    bevel = cylinder.modifiers.new(name='Bevel', type='BEVEL')
    bevel.width = 0.03
    bevel.segments = 2

    # Apply the modifier
    bpy.context.view_layer.objects.active = cylinder
    bpy.ops.object.modifier_apply(modifier='Bevel')

    # Smooth shading
    bpy.ops.object.shade_smooth()

    # Move origin to bottom center
    bpy.context.scene.cursor.location = (0, 0, 0)
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')

    return cylinder

def save_object(obj_name, create_func, output_dir):
    """Save an object to its own .blend file."""
    # Clear scene
    clear_scene()

    # Create the object
    obj = create_func()

    # Ensure object is selected and active
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Save the file
    filepath = os.path.join(output_dir, f'{obj_name}.blend')
    bpy.ops.wm.save_as_mainfile(filepath=filepath)
    print(f"  Saved: {filepath}")

def main():
    print("Creating CLEVR shapes for Blender 5.0...")
    print(f"Blender version: {bpy.app.version_string}")

    # Get output directory
    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_dir = os.path.join(script_dir, 'data', 'shapes_v5')

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")
    print("\nCreating shapes...")

    # Create and save each shape
    shapes = [
        ('SmoothCube_v2', create_smooth_cube),
        ('Sphere', create_sphere),
        ('SmoothCylinder', create_smooth_cylinder),
    ]

    for name, create_func in shapes:
        save_object(name, create_func, output_dir)

    print("\nShape files created successfully!")
    print("\nTo use these shapes, update your render command:")
    print("  blender --background --python render_images.py -- --shape_dir data/shapes_v5")

if __name__ == '__main__':
    main()
