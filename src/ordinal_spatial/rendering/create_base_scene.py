# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

"""
Create a base_scene.blend file compatible with Blender 5.0 / 4.x / 2.80+

Run this script from Blender to generate a new base_scene.blend file:
    blender --background --python create_base_scene_blender5.py

This will create 'data/base_scene_v5.blend' with:
- Ground plane
- Camera at the correct position
- Three-point lighting setup (Key, Fill, Back)
"""

import bpy
import math
import os

def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Clear orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.lights:
        if block.users == 0:
            bpy.data.lights.remove(block)

def create_ground():
    """Create the ground plane."""
    # Create a large plane for the ground
    bpy.ops.mesh.primitive_plane_add(size=100, location=(0, 0, 0))
    ground = bpy.context.object
    ground.name = 'Ground'

    # Create a gray material for the ground
    mat = bpy.data.materials.new(name='GroundMaterial')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled = nodes.get('Principled BSDF')
    if principled:
        principled.inputs['Base Color'].default_value = (0.5, 0.5, 0.5, 1.0)
        principled.inputs['Roughness'].default_value = 0.8

    ground.data.materials.append(mat)
    return ground

def create_camera():
    """Create and position the camera."""
    # Create camera
    bpy.ops.object.camera_add(location=(7.48, -6.51, 5.34))
    camera = bpy.context.object
    camera.name = 'Camera'

    # Set rotation to look at origin
    camera.rotation_euler = (math.radians(63.6), 0, math.radians(46.7))

    # Set camera properties
    camera.data.lens = 35
    camera.data.sensor_width = 32

    # Set as active camera
    bpy.context.scene.camera = camera

    return camera

def create_lights():
    """Create three-point lighting setup."""
    lights = []

    # Key Light - Main light source
    bpy.ops.object.light_add(type='AREA', location=(-5, -5, 8))
    key_light = bpy.context.object
    key_light.name = 'Lamp_Key'
    key_light.data.energy = 500
    key_light.data.size = 3
    key_light.rotation_euler = (math.radians(37), 0, math.radians(-45))
    lights.append(key_light)

    # Fill Light - Softer fill
    bpy.ops.object.light_add(type='AREA', location=(5, -5, 5))
    fill_light = bpy.context.object
    fill_light.name = 'Lamp_Fill'
    fill_light.data.energy = 200
    fill_light.data.size = 5
    fill_light.rotation_euler = (math.radians(50), 0, math.radians(45))
    lights.append(fill_light)

    # Back Light - Rim lighting
    bpy.ops.object.light_add(type='AREA', location=(0, 8, 6))
    back_light = bpy.context.object
    back_light.name = 'Lamp_Back'
    back_light.data.energy = 300
    back_light.data.size = 3
    back_light.rotation_euler = (math.radians(-30), 0, math.radians(180))
    lights.append(back_light)

    return lights

def setup_world():
    """Setup world/environment settings."""
    world = bpy.data.worlds['World']
    world.use_nodes = True
    nodes = world.node_tree.nodes

    # Set a simple gray background
    bg_node = nodes.get('Background')
    if bg_node:
        bg_node.inputs['Color'].default_value = (0.8, 0.8, 0.8, 1.0)
        bg_node.inputs['Strength'].default_value = 0.5

def setup_render_settings():
    """Configure render settings for Cycles."""
    scene = bpy.context.scene

    # Use Cycles renderer
    scene.render.engine = 'CYCLES'

    # Set resolution
    scene.render.resolution_x = 320
    scene.render.resolution_y = 240
    scene.render.resolution_percentage = 100

    # Cycles settings
    scene.cycles.samples = 512
    scene.cycles.use_denoising = True

    # Film settings
    scene.render.film_transparent = False

def main():
    print("Creating CLEVR base scene for Blender 5.0...")
    print(f"Blender version: {bpy.app.version_string}")

    # Clear existing scene
    clear_scene()

    # Create scene elements
    ground = create_ground()
    camera = create_camera()
    lights = create_lights()

    # Setup environment
    setup_world()
    setup_render_settings()

    # Get output path
    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_path = os.path.join(script_dir, 'data', 'base_scene_v5.blend')

    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the file
    bpy.ops.wm.save_as_mainfile(filepath=output_path)

    print(f"\nBase scene created successfully!")
    print(f"Saved to: {output_path}")
    print("\nScene contents:")
    print(f"  - Ground plane: {ground.name}")
    print(f"  - Camera: {camera.name}")
    print(f"  - Lights: {[l.name for l in lights]}")
    print("\nTo use this scene, update your render command:")
    print("  blender --background --python render_images.py -- --base_scene_blendfile data/base_scene_v5.blend")

if __name__ == '__main__':
    main()
