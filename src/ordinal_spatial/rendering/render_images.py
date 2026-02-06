# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# 简化版: 仅支持 Blender 5.0+

"""
使用 Blender 渲染随机 CLEVR 场景 (仅 Blender 5.0+)。
Render random CLEVR scenes using Blender (Blender 5.0+ only).

每个场景包含随机数量的对象，每个对象具有随机的大小、位置、
颜色和形状。对象之间不会相交，但可能部分遮挡。

用法 / Usage:
  blender --background --python render_images.py -- [arguments]
"""

import math
import sys
import random
import argparse
import json
import os
import tempfile
from datetime import datetime as dt
from collections import Counter

# 检测 Blender 环境
INSIDE_BLENDER = True
try:
  import bpy
  import bpy_extras
  from mathutils import Vector
except ImportError:
  INSIDE_BLENDER = False

if INSIDE_BLENDER:
  try:
    from ordinal_spatial.rendering import blender_utils as utils
  except ImportError:
    # 如果包导入失败，尝试从脚本目录导入
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
      sys.path.insert(0, script_dir)
    try:
      import blender_utils as utils
    except ImportError:
      print("\nERROR: Cannot import blender_utils.")
      print("Add the rendering directory to Blender's Python path.")
      sys.exit(1)


def get_object_by_name(name, alternative_names=None):
  """
  按名称获取 Blender 对象，支持备选名称回退。
  用于处理不同 Blender 版本之间的命名差异。
  """
  if name in bpy.data.objects:
    return bpy.data.objects[name]
  if alternative_names:
    for alt_name in alternative_names:
      if alt_name in bpy.data.objects:
        return bpy.data.objects[alt_name]
  raise KeyError(
    f"Object not found: {name} (also tried: {alternative_names})"
  )


# === 参数定义 ===
parser = argparse.ArgumentParser()

# 输入选项
parser.add_argument(
  '--base_scene_blendfile', default='data/base_scene_v5.blend',
  help="Base blender scene file (Blender 5.0+)")
parser.add_argument(
  '--properties_json', default='data/properties.json',
  help="JSON file defining objects, materials, sizes, and colors")
parser.add_argument(
  '--shape_dir', default='data/shapes_v5',
  help="Directory with .blend shape files (v5 format)")
parser.add_argument(
  '--material_dir', default='data/materials_v5',
  help="Directory with .blend material files (v5 format)")
parser.add_argument(
  '--shape_color_combos_json', default=None,
  help="Optional JSON for CLEVR-CoGenT color restrictions")

# 对象设置
parser.add_argument('--min_objects', default=3, type=int)
parser.add_argument('--max_objects', default=10, type=int)
parser.add_argument('--min_dist', default=0.25, type=float,
  help="Minimum distance between object centers")
parser.add_argument('--margin', default=0.4, type=float,
  help="Minimum margin along cardinal directions")
parser.add_argument('--min_pixels_per_object', default=200, type=int,
  help="Minimum visible pixels per object (0 to skip check)")
parser.add_argument('--max_retries', default=50, type=int)

# 输出设置
parser.add_argument('--start_idx', default=0, type=int)
parser.add_argument('--num_images', default=5, type=int)
parser.add_argument('--filename_prefix', default='CLEVR')
parser.add_argument('--split', default='new')
parser.add_argument('--output_image_dir', default='../output/images/')
parser.add_argument('--output_scene_dir', default='../output/scenes/')
parser.add_argument(
  '--output_scene_file', default='../output/CLEVR_scenes.json')
parser.add_argument('--output_blend_dir', default='output/blendfiles')
parser.add_argument('--save_blendfiles', type=int, default=0)
parser.add_argument('--version', default='1.0')
parser.add_argument('--license',
  default="Creative Commons Attribution (CC-BY 4.0)")
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"))

# 渲染选项
parser.add_argument('--use_gpu', default=0, type=int,
  help="Enable GPU rendering (1=on, 0=off)")
parser.add_argument('--width', default=320, type=int)
parser.add_argument('--height', default=240, type=int)
parser.add_argument('--key_light_jitter', default=1.0, type=float)
parser.add_argument('--fill_light_jitter', default=1.0, type=float)
parser.add_argument('--back_light_jitter', default=1.0, type=float)
parser.add_argument('--camera_jitter', default=0.5, type=float)
parser.add_argument('--render_num_samples', default=512, type=int)
parser.add_argument('--render_max_bounces', default=8, type=int)
parser.add_argument('--render_tile_size', default=256, type=int)


def main(args):
  """渲染主入口: 批量生成随机场景图像。"""
  num_digits = 6
  prefix = '%s_%s_' % (args.filename_prefix, args.split)
  img_template = '%s%%0%dd.png' % (prefix, num_digits)
  scene_template = '%s%%0%dd.json' % (prefix, num_digits)
  blend_template = '%s%%0%dd.blend' % (prefix, num_digits)
  img_template = os.path.join(args.output_image_dir, img_template)
  scene_template = os.path.join(args.output_scene_dir, scene_template)
  blend_template = os.path.join(args.output_blend_dir, blend_template)

  if not os.path.isdir(args.output_image_dir):
    os.makedirs(args.output_image_dir)
  if not os.path.isdir(args.output_scene_dir):
    os.makedirs(args.output_scene_dir)
  if args.save_blendfiles == 1 and not os.path.isdir(args.output_blend_dir):
    os.makedirs(args.output_blend_dir)

  all_scene_paths = []
  successful_scenes = 0
  failed_scenes = 0
  for i in range(args.num_images):
    img_path = img_template % (i + args.start_idx)
    scene_path = scene_template % (i + args.start_idx)
    all_scene_paths.append(scene_path)
    blend_path = None
    if args.save_blendfiles == 1:
      blend_path = blend_template % (i + args.start_idx)
    num_objects = random.randint(args.min_objects, args.max_objects)
    try:
      render_scene(
        args,
        num_objects=num_objects,
        output_index=(i + args.start_idx),
        output_split=args.split,
        output_image=img_path,
        output_scene=scene_path,
        output_blendfile=blend_path,
      )
      successful_scenes += 1
    except RuntimeError as e:
      print(f"Warning: Failed to render scene {i}: {e}")
      print("Skipping this scene and continuing...")
      failed_scenes += 1
      continue

  print(
    f"\nRendering complete: "
    f"{successful_scenes} successful, {failed_scenes} failed"
  )

  # 合并所有场景 JSON 到一个文件
  all_scenes = []
  for scene_path in all_scene_paths:
    if os.path.exists(scene_path):
      with open(scene_path, 'r') as f:
        all_scenes.append(json.load(f))
  output = {
    'info': {
      'date': args.date,
      'version': args.version,
      'split': args.split,
      'license': args.license,
    },
    'scenes': all_scenes
  }
  with open(args.output_scene_file, 'w') as f:
    json.dump(output, f)


def render_scene(
    args,
    num_objects=5,
    output_index=0,
    output_split='none',
    output_image='render.png',
    output_scene='render_json',
    output_blendfile=None,
):
  """渲染单个随机场景。"""
  # 加载基础场景
  bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

  # 加载材质
  utils.load_materials(args.material_dir)

  # 配置渲染参数 (Cycles 引擎)
  render_args = bpy.context.scene.render
  render_args.engine = "CYCLES"
  render_args.filepath = output_image
  render_args.resolution_x = args.width
  render_args.resolution_y = args.height
  render_args.resolution_percentage = 100

  # GPU 加速配置
  if args.use_gpu == 1:
    cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
    # 依次尝试 CUDA → OptiX → HIP → OneAPI
    for compute_type in ['CUDA', 'OPTIX', 'HIP', 'ONEAPI']:
      try:
        cycles_prefs.compute_device_type = compute_type
        for device in cycles_prefs.devices:
          device.use = True
        break
      except Exception:
        continue
    bpy.context.scene.cycles.device = 'GPU'

  # Cycles 渲染质量设置
  bpy.context.scene.cycles.blur_glossy = 2.0
  bpy.context.scene.cycles.samples = args.render_num_samples
  bpy.context.scene.cycles.transparent_max_bounces = (
    args.render_max_bounces
  )

  # 构建场景结构数据
  scene_struct = {
    'split': output_split,
    'image_index': output_index,
    'image_filename': os.path.basename(output_image),
    'objects': [],
    'directions': {},
  }

  # 用临时平面计算主方向
  bpy.ops.mesh.primitive_plane_add(size=10)
  plane = bpy.context.object

  def rand(L):
    return 2.0 * L * (random.random() - 0.5)

  # 相机随机抖动
  if args.camera_jitter > 0:
    for i in range(3):
      bpy.data.objects['Camera'].location[i] += rand(
        args.camera_jitter
      )

  # 计算相机坐标系下的主方向
  camera = bpy.data.objects['Camera']
  plane_normal = plane.data.vertices[0].normal

  # Blender 5.0 使用 @ 运算符进行矩阵乘法
  cam_behind = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
  cam_left = camera.matrix_world.to_quaternion() @ Vector((-1, 0, 0))
  cam_up = camera.matrix_world.to_quaternion() @ Vector((0, 1, 0))

  plane_behind = (
    cam_behind - cam_behind.project(plane_normal)
  ).normalized()
  plane_left = (
    cam_left - cam_left.project(plane_normal)
  ).normalized()
  plane_up = cam_up.project(plane_normal).normalized()

  # 删除临时平面
  utils.delete_object(plane)

  # 保存六个方向向量
  scene_struct['directions']['behind'] = tuple(plane_behind)
  scene_struct['directions']['front'] = tuple(-plane_behind)
  scene_struct['directions']['left'] = tuple(plane_left)
  scene_struct['directions']['right'] = tuple(-plane_left)
  scene_struct['directions']['above'] = tuple(plane_up)
  scene_struct['directions']['below'] = tuple(-plane_up)

  # 灯光随机抖动
  lamp_key = get_object_by_name(
    'Lamp_Key', ['Light_Key', 'Key', 'KeyLight']
  )
  lamp_back = get_object_by_name(
    'Lamp_Back', ['Light_Back', 'Back', 'BackLight']
  )
  lamp_fill = get_object_by_name(
    'Lamp_Fill', ['Light_Fill', 'Fill', 'FillLight']
  )

  if args.key_light_jitter > 0:
    for i in range(3):
      lamp_key.location[i] += rand(args.key_light_jitter)
  if args.back_light_jitter > 0:
    for i in range(3):
      lamp_back.location[i] += rand(args.back_light_jitter)
  if args.fill_light_jitter > 0:
    for i in range(3):
      lamp_fill.location[i] += rand(args.fill_light_jitter)

  # 放置随机对象
  objects, blender_objects = add_random_objects(
    scene_struct, num_objects, args, camera
  )

  # 渲染并保存
  scene_struct['objects'] = objects
  scene_struct['relationships'] = compute_all_relationships(
    scene_struct
  )
  while True:
    try:
      bpy.ops.render.render(write_still=True)
      break
    except Exception as e:
      print(e)

  with open(output_scene, 'w') as f:
    json.dump(scene_struct, f, indent=2)

  if output_blendfile is not None:
    bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)


def add_random_objects(
    scene_struct, num_objects, args, camera, _retry_count=0
):
  """
  向场景中添加随机对象。
  包含防止无限递归的重试限制。
  """
  MAX_OCCLUSION_RETRIES = 50
  if _retry_count >= MAX_OCCLUSION_RETRIES:
    raise RuntimeError(
      f"Failed to place objects after {MAX_OCCLUSION_RETRIES} "
      "attempts due to occlusion. Try fewer objects or smaller sizes."
    )

  # 加载对象属性
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties['colors'].items():
      rgba = [float(c) / 255.0 for c in rgb] + [1.0]
      color_name_to_rgba[name] = rgba
    material_mapping = [
      (v, k) for k, v in properties['materials'].items()
    ]
    object_mapping = [
      (v, k) for k, v in properties['shapes'].items()
    ]
    size_mapping = list(properties['sizes'].items())

  shape_color_combos = None
  if args.shape_color_combos_json is not None:
    with open(args.shape_color_combos_json, 'r') as f:
      shape_color_combos = list(json.load(f).items())

  positions = []
  objects = []
  blender_objects = []

  for i in range(num_objects):
    size_name, r = random.choice(size_mapping)

    # 尝试放置对象，确保不与已有对象重叠
    num_tries = 0
    while True:
      num_tries += 1
      if num_tries > args.max_retries:
        # 放置失败，清理所有对象并重试
        for obj in blender_objects:
          utils.delete_object(obj)
        return add_random_objects(
          scene_struct, num_objects, args, camera
        )
      x = random.uniform(-3, 3)
      y = random.uniform(-3, 3)

      # 检查距离和边距约束
      dists_good = True
      margins_good = True
      for (xx, yy, rr) in positions:
        dx, dy = x - xx, y - yy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist - r - rr < args.min_dist:
          dists_good = False
          break
        for direction_name in ['left', 'right', 'front', 'behind']:
          direction_vec = scene_struct['directions'][direction_name]
          assert direction_vec[2] == 0
          margin = dx * direction_vec[0] + dy * direction_vec[1]
          if 0 < margin < args.margin:
            print(margin, args.margin, direction_name)
            print('BROKEN MARGIN!')
            margins_good = False
            break
        if not margins_good:
          break
      if dists_good and margins_good:
        break

    # 随机选择颜色和形状
    if shape_color_combos is None:
      obj_name, obj_name_out = random.choice(object_mapping)
      color_name, rgba = random.choice(
        list(color_name_to_rgba.items())
      )
    else:
      obj_name_out, color_choices = random.choice(shape_color_combos)
      color_name = random.choice(color_choices)
      obj_name = [
        k for k, v in object_mapping if v == obj_name_out
      ][0]
      rgba = color_name_to_rgba[color_name]

    # 立方体大小调整
    if obj_name == 'Cube':
      r /= math.sqrt(2)

    theta = 360.0 * random.random()

    # 实际添加对象到场景
    utils.add_object(
      args.shape_dir, obj_name, r, (x, y), theta=theta
    )
    obj = bpy.context.object
    blender_objects.append(obj)
    positions.append((x, y, r))

    # 随机分配材质
    mat_name, mat_name_out = random.choice(material_mapping)
    utils.add_material(mat_name, Color=rgba)

    # 记录对象信息
    pixel_coords = utils.get_camera_coords(camera, obj.location)
    objects.append({
      'shape': obj_name_out,
      'size': size_name,
      'material': mat_name_out,
      '3d_coords': tuple(obj.location),
      'rotation': theta,
      'pixel_coords': pixel_coords,
      'color': color_name,
    })

  # 检查所有对象是否至少部分可见
  all_visible = check_visibility(
    blender_objects, args.min_pixels_per_object
  )
  if not all_visible:
    print('Some objects are occluded; replacing objects')
    for obj in blender_objects:
      utils.delete_object(obj)
    return add_random_objects(
      scene_struct, num_objects, args, camera,
      _retry_count=_retry_count + 1
    )

  return objects, blender_objects


def compute_all_relationships(scene_struct, eps=0.2):
  """
  计算场景中所有对象对之间的空间关系。
  返回字典: {关系名: [[与对象 i 具有该关系的对象索引列表]]}
  """
  all_relationships = {}
  for name, direction_vec in scene_struct['directions'].items():
    if name == 'above' or name == 'below':
      continue
    all_relationships[name] = []
    for i, obj1 in enumerate(scene_struct['objects']):
      coords1 = obj1['3d_coords']
      related = set()
      for j, obj2 in enumerate(scene_struct['objects']):
        if obj1 == obj2:
          continue
        coords2 = obj2['3d_coords']
        diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
        dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
        if dot > eps:
          related.add(j)
      all_relationships[name].append(sorted(list(related)))
  return all_relationships


def check_visibility(blender_objects, min_pixels_per_object):
  """
  检查所有对象是否至少有 min_pixels_per_object 个可见像素。
  通过为对象分配纯色发射材质并渲染来实现。
  """
  if min_pixels_per_object <= 0:
    return True

  f, path = tempfile.mkstemp(suffix='.png')
  os.close(f)
  object_colors = render_shadeless(blender_objects, path=path)
  img = bpy.data.images.load(path)
  p = list(img.pixels)
  color_count = Counter(
    (p[i], p[i + 1], p[i + 2], p[i + 3])
    for i in range(0, len(p), 4)
  )
  bpy.data.images.remove(img)
  try:
    os.remove(path)
  except PermissionError:
    pass
  if len(color_count) != len(blender_objects) + 1:
    return False
  for _, count in color_count.most_common():
    if count < min_pixels_per_object:
      return False
  return True


def render_shadeless(blender_objects, path='flat.png'):
  """
  使用发射着色器渲染纯色版本，用于可见性检测。
  返回所有使用的颜色集合。
  """
  render_args = bpy.context.scene.render
  old_filepath = render_args.filepath

  render_args.filepath = path
  old_samples = bpy.context.scene.cycles.samples
  bpy.context.scene.cycles.samples = 1

  # 隐藏灯光和地面
  lamp_key = get_object_by_name(
    'Lamp_Key', ['Light_Key', 'Key', 'KeyLight']
  )
  lamp_fill = get_object_by_name(
    'Lamp_Fill', ['Light_Fill', 'Fill', 'FillLight']
  )
  lamp_back = get_object_by_name(
    'Lamp_Back', ['Light_Back', 'Back', 'BackLight']
  )
  ground = get_object_by_name('Ground', ['ground', 'Floor', 'Plane'])

  utils.set_layer(lamp_key, 2)
  utils.set_layer(lamp_fill, 2)
  utils.set_layer(lamp_back, 2)
  utils.set_layer(ground, 2)

  # 为每个对象创建唯一的发射材质
  object_colors = set()
  old_materials = []
  for i, obj in enumerate(blender_objects):
    old_materials.append(obj.data.materials[0])

    mat = bpy.data.materials.new(name='FlatMaterial_%d' % i)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    # 生成唯一颜色
    while True:
      r, g, b = [random.random() for _ in range(3)]
      if (r, g, b) not in object_colors:
        break
    object_colors.add((r, g, b))

    # 创建发射着色器节点
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = [r, g, b, 1.0]
    emission.inputs['Strength'].default_value = 1.0

    output = nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(
      emission.outputs['Emission'], output.inputs['Surface']
    )

    obj.data.materials[0] = mat

  bpy.ops.render.render(write_still=True)

  # 恢复原始材质
  for mat, obj in zip(old_materials, blender_objects):
    obj.data.materials[0] = mat

  # 恢复灯光和地面
  utils.set_layer(lamp_key, 0)
  utils.set_layer(lamp_fill, 0)
  utils.set_layer(lamp_back, 0)
  utils.set_layer(ground, 0)

  # 恢复渲染设置
  render_args.filepath = old_filepath
  bpy.context.scene.cycles.samples = old_samples

  return object_colors


if __name__ == '__main__':
  if INSIDE_BLENDER:
    argv = utils.extract_args()
    args = parser.parse_args(argv)
    main(args)
  elif '--help' in sys.argv or '-h' in sys.argv:
    parser.print_help()
  else:
    print('This script is intended to be called from Blender:')
    print()
    print('  blender --background --python render_images.py -- [args]')
    print()
    print('For help:')
    print('  python render_images.py --help')
