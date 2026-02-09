# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# 简化版: 仅支持 Blender 5.0+

"""
多视角渲染脚本 (仅 Blender 5.0+)。
Multi-view rendering for ORDINAL-SPATIAL benchmark (Blender 5.0+ only).

从多个相机视角渲染同一场景，用于多视角空间推理评估。

用法 / Usage:
  blender --background --python render_multiview.py -- [arguments]
"""

import math
import sys
import random
import argparse
import json
import os
from datetime import datetime as dt
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

# 检测 Blender 环境
INSIDE_BLENDER = True
try:
  import bpy
  import bpy_extras
  from mathutils import Vector, Euler
except ImportError:
  INSIDE_BLENDER = False

if INSIDE_BLENDER:
  try:
    from ordinal_spatial.rendering import blender_utils as utils
  except ImportError:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
      sys.path.insert(0, script_dir)
    try:
      import blender_utils as utils
    except ImportError:
      print("\nERROR: Cannot import blender_utils.")
      print("Add the rendering directory to Blender's Python path.")
      sys.exit(1)


@dataclass
class CameraConfig:
  """单个相机视角的配置。"""
  camera_id: str
  azimuth: float       # 方位角 (度)，0 = +X 方向
  elevation: float     # 仰角 (度)，0 = 水平
  distance: float      # 到场景中心的距离
  look_at: Tuple[float, float, float] = (0.0, 0.0, 0.0)

  def to_cartesian(self) -> Tuple[float, float, float]:
    """将球坐标转换为笛卡尔坐标。"""
    az_rad = math.radians(self.azimuth)
    el_rad = math.radians(self.elevation)
    x = self.distance * math.cos(el_rad) * math.cos(az_rad)
    y = self.distance * math.cos(el_rad) * math.sin(az_rad)
    z = self.distance * math.sin(el_rad)
    return (
      x + self.look_at[0],
      y + self.look_at[1],
      z + self.look_at[2]
    )

  def to_dict(self) -> Dict[str, Any]:
    """转换为可 JSON 序列化的字典。"""
    pos = self.to_cartesian()
    return {
      "camera_id": self.camera_id,
      "azimuth": self.azimuth,
      "elevation": self.elevation,
      "distance": self.distance,
      "position": list(pos),
      "look_at": list(self.look_at)
    }


@dataclass
class MultiViewConfig:
  """多视角渲染配置。"""
  n_views: int = 4
  camera_distance: float = 12.0
  elevation: float = 30.0
  azimuth_start: float = 45.0  # 起始角 45° 以获得更好的覆盖
  look_at: Tuple[float, float, float] = (0.0, 0.0, 0.0)

  def generate_cameras(self) -> List[CameraConfig]:
    """生成所有视角的相机配置，等间隔分布。"""
    cameras = []
    azimuth_step = 360.0 / self.n_views
    for i in range(self.n_views):
      azimuth = (self.azimuth_start + i * azimuth_step) % 360.0
      cameras.append(CameraConfig(
        camera_id=f"view_{i}",
        azimuth=azimuth,
        elevation=self.elevation,
        distance=self.camera_distance,
        look_at=self.look_at
      ))
    return cameras


def get_object_by_name(
    name: str, alternative_names: Optional[List[str]] = None
):
  """按名称获取 Blender 对象，支持备选名称回退。"""
  if name in bpy.data.objects:
    return bpy.data.objects[name]
  if alternative_names:
    for alt_name in alternative_names:
      if alt_name in bpy.data.objects:
        return bpy.data.objects[alt_name]
  raise KeyError(f"Object not found: {name}")


def set_camera_position(camera_config: CameraConfig) -> None:
  """
  设置 Blender 相机的位置和朝向。
  使用 track-to 约束自动朝向目标点。
  """
  camera = bpy.data.objects['Camera']
  position = camera_config.to_cartesian()
  look_at = camera_config.look_at

  camera.location = position

  # 计算朝向并设置旋转
  direction = Vector(look_at) - Vector(position)
  rot_quat = direction.to_track_quat('-Z', 'Y')
  camera.rotation_euler = rot_quat.to_euler()


def compute_pixel_coords_for_view(
    camera, objects_3d: List[Dict]
) -> List[Dict]:
  """为当前相机视角计算所有对象的像素坐标。"""
  updated_objects = []
  for obj in objects_3d:
    obj_copy = obj.copy()
    coords_3d = obj['3d_coords']
    pixel_coords = utils.get_camera_coords(
      camera, Vector(coords_3d)
    )
    obj_copy['pixel_coords'] = pixel_coords
    updated_objects.append(obj_copy)
  return updated_objects


def compute_directions_for_view(
    camera,
) -> Dict[str, Tuple[float, float, float]]:
  """计算当前相机视角下的主方向向量。"""
  bpy.ops.mesh.primitive_plane_add(size=10)
  plane = bpy.context.object
  plane_normal = plane.data.vertices[0].normal

  # Blender 5.0 使用 @ 运算符
  cam_behind = camera.matrix_world.to_quaternion() @ Vector(
    (0, 0, -1)
  )
  cam_left = camera.matrix_world.to_quaternion() @ Vector((-1, 0, 0))
  cam_up = camera.matrix_world.to_quaternion() @ Vector((0, 1, 0))

  plane_behind = (
    cam_behind - cam_behind.project(plane_normal)
  ).normalized()
  plane_left = (
    cam_left - cam_left.project(plane_normal)
  ).normalized()
  plane_up = cam_up.project(plane_normal).normalized()

  utils.delete_object(plane)

  return {
    'behind': tuple(plane_behind),
    'front': tuple(-plane_behind),
    'left': tuple(plane_left),
    'right': tuple(-plane_left),
    'above': tuple(plane_up),
    'below': tuple(-plane_up)
  }


def render_single_view(
    camera_config: CameraConfig,
    output_image: str,
    objects_3d: List[Dict],
    args
) -> Dict[str, Any]:
  """从单个相机视角渲染场景，返回视图元数据。"""
  set_camera_position(camera_config)
  camera = bpy.data.objects['Camera']

  objects_with_pixels = compute_pixel_coords_for_view(
    camera, objects_3d
  )
  directions = compute_directions_for_view(camera)

  bpy.context.scene.render.filepath = output_image
  bpy.ops.render.render(write_still=True)

  return {
    "view_id": camera_config.camera_id,
    "image_path": os.path.basename(output_image),
    "camera": camera_config.to_dict(),
    "directions": directions,
    "objects": objects_with_pixels
  }


def render_multiview_scene(
    args,
    num_objects: int,
    output_index: int,
    output_split: str,
    output_dir: str,
    mv_config: MultiViewConfig
) -> Dict[str, Any]:
  """渲染完整的多视角场景。"""
  os.makedirs(output_dir, exist_ok=True)

  # 加载基础场景和材质
  bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)
  utils.load_materials(args.material_dir)

  # 渲染设置
  render_args = bpy.context.scene.render
  render_args.engine = "CYCLES"
  render_args.resolution_x = args.width
  render_args.resolution_y = args.height
  render_args.resolution_percentage = 100

  # GPU 加速
  if args.use_gpu == 1:
    cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
    for compute_type in ['CUDA', 'OPTIX', 'HIP', 'ONEAPI']:
      try:
        cycles_prefs.compute_device_type = compute_type
        for device in cycles_prefs.devices:
          device.use = True
        break
      except Exception:
        continue
    bpy.context.scene.cycles.device = 'GPU'

  bpy.context.scene.cycles.samples = args.render_num_samples
  bpy.context.scene.cycles.blur_glossy = 2.0
  bpy.context.scene.cycles.transparent_max_bounces = (
    args.render_max_bounces
  )

  # 初始化场景结构
  scene_id = f"{output_split}_{output_index:06d}"
  scene_struct = {
    "scene_id": scene_id,
    "split": output_split,
    "image_index": output_index,
    "n_objects": num_objects,
    "objects": [],
    "world_constraints": {},
    "views": []
  }

  # 灯光随机抖动
  def rand(L):
    return 2.0 * L * (random.random() - 0.5)

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

  # 用第一个相机视角放置对象
  cameras = mv_config.generate_cameras()
  set_camera_position(cameras[0])

  # 计算对象放置方向
  bpy.ops.mesh.primitive_plane_add(size=10)
  plane = bpy.context.object
  camera = bpy.data.objects['Camera']
  plane_normal = plane.data.vertices[0].normal

  cam_behind = camera.matrix_world.to_quaternion() @ Vector(
    (0, 0, -1)
  )
  cam_left = camera.matrix_world.to_quaternion() @ Vector((-1, 0, 0))

  plane_behind = (
    cam_behind - cam_behind.project(plane_normal)
  ).normalized()
  plane_left = (
    cam_left - cam_left.project(plane_normal)
  ).normalized()

  temp_directions = {
    'behind': tuple(plane_behind),
    'front': tuple(-plane_behind),
    'left': tuple(plane_left),
    'right': tuple(-plane_left),
  }

  utils.delete_object(plane)

  # 添加随机对象
  objects_3d, blender_objects = add_random_objects(
    temp_directions, num_objects, args, camera
  )
  scene_struct["objects"] = objects_3d

  # 从每个视角渲染
  for cam_config in cameras:
    img_path = os.path.join(
      output_dir, f"{cam_config.camera_id}.png"
    )
    view_data = render_single_view(
      cam_config, img_path, objects_3d, args
    )
    scene_struct["views"].append(view_data)

  # 保存场景元数据
  metadata_path = os.path.join(output_dir, "metadata.json")
  with open(metadata_path, 'w') as f:
    json.dump(scene_struct, f, indent=2)

  # 复制 view_0 到单视角目录
  if args.output_single_view_dir:
    single_view_dir = args.output_single_view_dir
    os.makedirs(single_view_dir, exist_ok=True)
    single_img_path = os.path.join(
      single_view_dir, f"{scene_id}.png"
    )
    import shutil
    view0_path = os.path.join(output_dir, "view_0.png")
    if os.path.exists(view0_path):
      shutil.copy(view0_path, single_img_path)

  return scene_struct


def add_random_objects(
    directions, num_objects, args, camera, _retry_count=0
):
  """
  向场景中添加随机对象。
  放置区域根据对象数量自动缩放以适应密集场景。
  """
  MAX_RETRIES = 100
  if _retry_count >= MAX_RETRIES:
    raise RuntimeError(
      f"Failed to place objects after {MAX_RETRIES} attempts"
    )

  # 根据对象数量缩放放置区域
  if num_objects <= 6:
    placement_range = 3.0
  elif num_objects <= 10:
    placement_range = 3.5
  else:
    placement_range = 4.0  # 11-15 个对象使用更大区域

  # 密集场景减小间距要求
  effective_min_dist = args.min_dist
  effective_margin = args.margin
  if num_objects > 10:
    effective_min_dist = max(0.15, args.min_dist * 0.7)
    effective_margin = max(0.25, args.margin * 0.7)

  # 加载属性
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

  positions = []
  objects = []
  blender_objects = []

  for i in range(num_objects):
    size_name, r = random.choice(size_mapping)

    num_tries = 0
    while True:
      num_tries += 1
      if num_tries > args.max_retries:
        for obj in blender_objects:
          utils.delete_object(obj)
        return add_random_objects(
          directions, num_objects, args, camera,
          _retry_count=_retry_count + 1
        )
      x = random.uniform(-placement_range, placement_range)
      y = random.uniform(-placement_range, placement_range)

      dists_good = True
      margins_good = True
      for (xx, yy, rr) in positions:
        dx, dy = x - xx, y - yy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist - r - rr < effective_min_dist:
          dists_good = False
          break
        for direction_name in ['left', 'right', 'front', 'behind']:
          direction_vec = directions[direction_name]
          margin_val = (
            dx * direction_vec[0] + dy * direction_vec[1]
          )
          if 0 < margin_val < effective_margin:
            margins_good = False
            break
        if not margins_good:
          break
      if dists_good and margins_good:
        break

    obj_name, obj_name_out = random.choice(object_mapping)
    color_name, rgba = random.choice(
      list(color_name_to_rgba.items())
    )

    if obj_name == 'Cube':
      r /= math.sqrt(2)

    theta = 360.0 * random.random()

    utils.add_object(
      args.shape_dir, obj_name, r, (x, y), theta=theta
    )
    obj = bpy.context.object
    blender_objects.append(obj)
    positions.append((x, y, r))

    mat_name, mat_name_out = random.choice(material_mapping)
    utils.add_material(mat_name, Color=rgba)

    pixel_coords = utils.get_camera_coords(camera, obj.location)
    objects.append({
      "id": f"obj_{i}",
      "shape": obj_name_out,
      "size": size_name,
      "material": mat_name_out,
      "3d_coords": tuple(obj.location),
      "rotation": theta,
      "pixel_coords": pixel_coords,
      "color": color_name,
    })

  return objects, blender_objects


def _build_object_count_schedule(
    num_images, min_objects, max_objects, balanced=True
):
  """
  预生成每个场景的物体数量列表。
  balanced=True 时严格均分，每个数量级别的场景数最多相差 1。
  """
  n_levels = max_objects - min_objects + 1
  if not balanced or n_levels <= 0:
    return [
      random.randint(min_objects, max_objects)
      for _ in range(num_images)
    ]

  per_level = num_images // n_levels
  remainder = num_images % n_levels
  schedule = []
  for k in range(n_levels):
    obj_count = min_objects + k
    # 余数优先分配给较小的物体数量
    n = per_level + (1 if k < remainder else 0)
    schedule.extend([obj_count] * n)
  random.shuffle(schedule)
  return schedule


def main(args):
  """多视角渲染主入口。"""
  # 设置随机种子，确保可复现
  if args.seed is not None:
    random.seed(args.seed)
    print(f"Random seed: {args.seed}")

  print(
    f"Starting multi-view rendering: "
    f"{args.num_images} scenes, {args.n_views} views each"
  )

  mv_config = MultiViewConfig(
    n_views=args.n_views,
    camera_distance=args.camera_distance,
    elevation=args.elevation,
    azimuth_start=args.azimuth_start
  )

  multiview_dir = os.path.join(args.output_dir, "multi_view")
  single_view_dir = os.path.join(args.output_dir, "single_view")
  os.makedirs(multiview_dir, exist_ok=True)
  os.makedirs(single_view_dir, exist_ok=True)
  args.output_single_view_dir = single_view_dir

  # 预生成物体数量分配表
  balanced = getattr(args, 'balanced_objects', True)
  object_schedule = _build_object_count_schedule(
    args.num_images, args.min_objects, args.max_objects,
    balanced=balanced
  )
  if balanced:
    from collections import Counter
    dist = Counter(object_schedule)
    print(f"Object count distribution (balanced={balanced}):")
    for k in sorted(dist):
      print(f"  {k} objects: {dist[k]} scenes")

  all_scenes = []
  successful = 0
  failed = 0

  for i in range(args.num_images):
    scene_id = f"{args.split}_{(i + args.start_idx):06d}"
    scene_output_dir = os.path.join(multiview_dir, scene_id)
    num_objects = object_schedule[i]

    try:
      scene_struct = render_multiview_scene(
        args,
        num_objects=num_objects,
        output_index=(i + args.start_idx),
        output_split=args.split,
        output_dir=scene_output_dir,
        mv_config=mv_config
      )
      all_scenes.append(scene_struct)
      successful += 1
      print(f"  [{successful}/{args.num_images}] Rendered {scene_id}")
    except Exception as e:
      print(f"  [ERROR] Failed to render {scene_id}: {e}")
      failed += 1
      continue

  print(
    f"\nRendering complete: "
    f"{successful} successful, {failed} failed"
  )

  # 保存合并的场景文件
  output_file = os.path.join(
    args.output_dir, f"{args.split}_scenes.json"
  )
  output_data = {
    "info": {
      "date": dt.today().strftime("%Y-%m-%d"),
      "split": args.split,
      "n_views": args.n_views,
      "camera_config": asdict(mv_config)
    },
    "scenes": all_scenes
  }
  with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

  print(f"Saved scenes to {output_file}")


# === 参数定义 ===
parser = argparse.ArgumentParser(
  description="Multi-view scene rendering (Blender 5.0+)"
)

# 输入选项
parser.add_argument(
  '--base_scene_blendfile', default='data/base_scene_v5.blend')
parser.add_argument(
  '--properties_json', default='data/properties.json')
parser.add_argument('--shape_dir', default='data/shapes_v5')
parser.add_argument('--material_dir', default='data/materials_v5')

# 对象设置
parser.add_argument('--min_objects', default=3, type=int)
parser.add_argument('--max_objects', default=10, type=int)
parser.add_argument('--min_dist', default=0.25, type=float)
parser.add_argument('--margin', default=0.4, type=float)
parser.add_argument('--max_retries', default=50, type=int)
parser.add_argument('--balanced_objects', default=1, type=int,
  help="1=balanced object counts across scenes, 0=random")

# 多视角设置
parser.add_argument('--n_views', default=4, type=int,
  help="Number of camera viewpoints")
parser.add_argument('--camera_distance', default=12.0, type=float,
  help="Camera distance from scene center")
parser.add_argument('--elevation', default=30.0, type=float,
  help="Camera elevation angle in degrees")
parser.add_argument('--azimuth_start', default=45.0, type=float,
  help="Starting azimuth angle in degrees")

# 输出设置
parser.add_argument('--output_dir', default='../output/multiview/')
parser.add_argument('--start_idx', default=0, type=int)
parser.add_argument('--num_images', default=5, type=int)
parser.add_argument('--split', default='train')

# 渲染设置
parser.add_argument('--use_gpu', default=0, type=int)
parser.add_argument('--width', default=480, type=int)
parser.add_argument('--height', default=320, type=int)
parser.add_argument('--render_num_samples', default=256, type=int)
parser.add_argument('--render_max_bounces', default=8, type=int)
parser.add_argument('--key_light_jitter', default=1.0, type=float)
parser.add_argument('--fill_light_jitter', default=1.0, type=float)
parser.add_argument('--back_light_jitter', default=1.0, type=float)

# 随机种子
parser.add_argument('--seed', default=None, type=int,
  help="Random seed for reproducible scene generation")


if __name__ == '__main__':
  if INSIDE_BLENDER:
    argv = utils.extract_args()
    args = parser.parse_args(argv)
    main(args)
  elif '--help' in sys.argv or '-h' in sys.argv:
    parser.print_help()
  else:
    print('Run from Blender:')
    print(
      '  blender --background --python render_multiview.py -- [args]'
    )
    print()
    print('For help:')
    print('  python render_multiview.py --help')
