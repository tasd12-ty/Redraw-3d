# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# 简化版: 仅支持 Blender 5.0+

"""
Blender 工具函数 (仅 Blender 5.0+)。
Blender utility functions (Blender 5.0+ only).

提供与 Blender 交互的基础操作:
- 参数解析
- 对象管理 (添加/删除/图层控制)
- 材质加载与分配
- 相机坐标转换
"""

import sys
import os
import random

import bpy
import bpy_extras


def extract_args(input_argv=None):
  """
  提取 '--' 之后的命令行参数。
  Blender 会忽略 '--' 之后的参数，因此我们可以通过它
  将自定义参数传递给脚本。
  """
  if input_argv is None:
    input_argv = sys.argv
  output_argv = []
  if '--' in input_argv:
    idx = input_argv.index('--')
    output_argv = input_argv[(idx + 1):]
  return output_argv


def parse_args(parser, argv=None):
  """使用 extract_args 解析命令行参数。"""
  return parser.parse_args(extract_args(argv))


def delete_object(obj):
  """
  删除指定的 Blender 对象。
  先取消所有对象的选择，再选中目标对象后删除。
  """
  for o in bpy.data.objects:
    o.select_set(False)
  obj.select_set(True)
  bpy.ops.object.delete()


def get_camera_coords(cam, pos):
  """
  获取指定 3D 点在相机视角下的像素坐标。

  参数:
    cam: 相机对象
    pos: 3D 世界坐标 (Vector)

  返回:
    (px, py, pz): px/py 为像素坐标，pz 为深度 [-1, 1]
  """
  scene = bpy.context.scene
  x, y, z = bpy_extras.object_utils.world_to_camera_view(
    scene, cam, pos
  )
  scale = scene.render.resolution_percentage / 100.0
  w = int(scale * scene.render.resolution_x)
  h = int(scale * scene.render.resolution_y)
  px = int(round(x * w))
  py = int(round(h - y * h))
  return (px, py, z)


def set_layer(obj, layer_idx):
  """
  将对象移动到指定图层。
  layer_idx 0 = 可见 (Scene Collection)
  layer_idx > 0 = 隐藏集合

  使用 Blender 5.0+ 的 Collections 系统实现。
  """
  # 从当前所有集合中移除
  for col in obj.users_collection:
    col.objects.unlink(obj)

  if layer_idx == 0:
    # 链接到场景集合 (可见)
    bpy.context.scene.collection.objects.link(obj)
  else:
    # 创建或获取隐藏集合
    hidden_col_name = f"HiddenLayer_{layer_idx}"
    if hidden_col_name not in bpy.data.collections:
      hidden_col = bpy.data.collections.new(hidden_col_name)
      bpy.context.scene.collection.children.link(hidden_col)
      # 从视图层中排除以隐藏
      layer_col = bpy.context.view_layer.layer_collection
      layer_col.children[hidden_col_name].exclude = True
    else:
      hidden_col = bpy.data.collections[hidden_col_name]
    hidden_col.objects.link(obj)


def add_object(object_dir, name, scale, loc, theta=0):
  """
  从 .blend 文件中加载对象并放置到场景中。

  假设 object_dir 中有一个名为 "$name.blend" 的文件，
  其中包含一个名为 "$name" 的单位大小对象。

  参数:
    object_dir: .blend 文件所在目录
    name: 对象名称
    scale: 缩放因子
    loc: (x, y) 地面平面上的坐标
    theta: 旋转角度 (度)
  """
  # 计算场景中已有的同名对象数量，避免命名冲突
  count = 0
  for obj in bpy.data.objects:
    if obj.name.startswith(name):
      count += 1

  blend_path = os.path.join(object_dir, '%s.blend' % name)
  directory = os.path.join(blend_path, 'Object') + os.sep
  bpy.ops.wm.append(
    filepath=os.path.join(directory, name),
    directory=directory,
    filename=name
  )

  # 重命名以避免冲突
  new_name = '%s_%d' % (name, count)
  bpy.data.objects[name].name = new_name

  # 设置为活动对象，然后旋转、缩放、平移
  x, y = loc
  obj = bpy.data.objects[new_name]
  bpy.context.view_layer.objects.active = obj
  obj.select_set(True)

  bpy.context.object.rotation_euler[2] = theta
  bpy.ops.transform.resize(value=(scale, scale, scale))
  # v5 形状原点在底部，z=0 即放置在地面上
  bpy.ops.transform.translate(value=(x, y, 0))


def load_materials(material_dir):
  """
  从目录中加载材质。
  假设目录中的每个 .blend 文件包含一个 NodeTree。
  文件 X.blend 包含一个名为 X 的 NodeTree，
  它必须有一个接受 RGBA 值的 "Color" 输入。
  """
  for fn in os.listdir(material_dir):
    if not fn.endswith('.blend'):
      continue
    name = os.path.splitext(fn)[0]
    directory = os.path.join(material_dir, fn, 'NodeTree') + os.sep
    bpy.ops.wm.append(
      filepath=os.path.join(directory, name),
      directory=directory,
      filename=name
    )


def add_material(name, **properties):
  """
  创建新材质并分配给活动对象。
  name 应为之前通过 load_materials 加载的材质名称。
  """
  mat_count = len(bpy.data.materials)
  mat = bpy.data.materials.new(name='Material_%d' % mat_count)
  mat.use_nodes = True

  # 将新材质附加到活动对象
  obj = bpy.context.active_object
  assert len(obj.data.materials) == 0
  obj.data.materials.append(mat)

  # 通过节点类型查找输出节点（避免本地化名称问题）
  output_node = None
  for n in mat.node_tree.nodes:
    if n.type == 'OUTPUT_MATERIAL':
      output_node = n
      break

  if output_node is None:
    raise RuntimeError(
      "Could not find Material Output node in material"
    )

  # 创建 GroupNode 并复制预加载的节点组
  group_node = mat.node_tree.nodes.new('ShaderNodeGroup')
  group_node.node_tree = bpy.data.node_groups[name]

  # 设置 "Color" 等输入属性
  for inp in group_node.inputs:
    if inp.name in properties:
      inp.default_value = properties[inp.name]

  # 将节点组输出连接到材质输出
  mat.node_tree.links.new(
    group_node.outputs['Shader'],
    output_node.inputs['Surface'],
  )
