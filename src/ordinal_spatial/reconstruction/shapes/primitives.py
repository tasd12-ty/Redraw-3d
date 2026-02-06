"""
3D 形状基元定义。

为各种 3D 基本形状生成网格数据。

3D Shape Primitives.

Generates mesh data for various 3D primitive shapes.
"""

from typing import Tuple, Optional
import numpy as np


# 尺寸比例映射
SIZE_SCALE_MAP = {
    "tiny": 0.25,
    "small": 0.35,
    "medium": 0.5,
    "large": 0.7,
    "custom": 0.5,
}


def sphere_mesh(
    center: np.ndarray,
    radius: float = 0.5,
    resolution: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成球体网格。

    Generate sphere mesh.

    Args:
        center: 球心坐标 [x, y, z]
        radius: 半径
        resolution: 分辨率（经纬线数量）

    Returns:
        (vertices, faces) 元组
        - vertices: (N, 3) 顶点坐标
        - faces: (M, 3) 三角面索引
    """
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    # 生成顶点
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))

    vertices = np.stack([
        x.flatten() + center[0],
        y.flatten() + center[1],
        z.flatten() + center[2],
    ], axis=1)

    # 生成面
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            p1 = i * resolution + j
            p2 = i * resolution + (j + 1)
            p3 = (i + 1) * resolution + j
            p4 = (i + 1) * resolution + (j + 1)
            faces.append([p1, p2, p3])
            faces.append([p2, p4, p3])

    return vertices, np.array(faces)


def cube_mesh(
    center: np.ndarray,
    size: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成立方体网格。

    Generate cube mesh.

    Args:
        center: 中心坐标 [x, y, z]
        size: 边长

    Returns:
        (vertices, faces) 元组
    """
    half = size / 2
    cx, cy, cz = center

    vertices = np.array([
        [cx - half, cy - half, cz - half],  # 0
        [cx + half, cy - half, cz - half],  # 1
        [cx + half, cy + half, cz - half],  # 2
        [cx - half, cy + half, cz - half],  # 3
        [cx - half, cy - half, cz + half],  # 4
        [cx + half, cy - half, cz + half],  # 5
        [cx + half, cy + half, cz + half],  # 6
        [cx - half, cy + half, cz + half],  # 7
    ])

    # 12 个三角形面（6 个面，每面 2 个三角形）
    faces = np.array([
        # 底面 (z-)
        [0, 1, 2], [0, 2, 3],
        # 顶面 (z+)
        [4, 6, 5], [4, 7, 6],
        # 前面 (y-)
        [0, 5, 1], [0, 4, 5],
        # 后面 (y+)
        [2, 7, 3], [2, 6, 7],
        # 左面 (x-)
        [0, 7, 4], [0, 3, 7],
        # 右面 (x+)
        [1, 5, 6], [1, 6, 2],
    ])

    return vertices, faces


def cylinder_mesh(
    center: np.ndarray,
    radius: float = 0.5,
    height: float = 1.0,
    resolution: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成圆柱体网格。

    Generate cylinder mesh.

    Args:
        center: 中心坐标 [x, y, z]
        radius: 底面半径
        height: 高度
        resolution: 圆周分辨率

    Returns:
        (vertices, faces) 元组
    """
    theta = np.linspace(0, 2 * np.pi, resolution, endpoint=False)

    # 底面和顶面圆周顶点
    bottom_x = radius * np.cos(theta) + center[0]
    bottom_y = radius * np.sin(theta) + center[1]
    bottom_z = np.full(resolution, center[2] - height / 2)

    top_x = radius * np.cos(theta) + center[0]
    top_y = radius * np.sin(theta) + center[1]
    top_z = np.full(resolution, center[2] + height / 2)

    # 中心点
    bottom_center = center.copy()
    bottom_center[2] -= height / 2
    top_center = center.copy()
    top_center[2] += height / 2

    # 组合顶点
    vertices = []
    # 底面圆周 (0 to resolution-1)
    for i in range(resolution):
        vertices.append([bottom_x[i], bottom_y[i], bottom_z[i]])
    # 顶面圆周 (resolution to 2*resolution-1)
    for i in range(resolution):
        vertices.append([top_x[i], top_y[i], top_z[i]])
    # 底面中心 (2*resolution)
    vertices.append(bottom_center.tolist())
    # 顶面中心 (2*resolution + 1)
    vertices.append(top_center.tolist())

    vertices = np.array(vertices)

    # 生成面
    faces = []
    bottom_center_idx = 2 * resolution
    top_center_idx = 2 * resolution + 1

    for i in range(resolution):
        next_i = (i + 1) % resolution

        # 侧面
        faces.append([i, next_i, resolution + i])
        faces.append([next_i, resolution + next_i, resolution + i])

        # 底面
        faces.append([bottom_center_idx, next_i, i])

        # 顶面
        faces.append([top_center_idx, resolution + i, resolution + next_i])

    return vertices, np.array(faces)


def cone_mesh(
    center: np.ndarray,
    radius: float = 0.5,
    height: float = 1.0,
    resolution: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成圆锥网格。

    Generate cone mesh.

    Args:
        center: 底面中心坐标 [x, y, z]
        radius: 底面半径
        height: 高度
        resolution: 圆周分辨率

    Returns:
        (vertices, faces) 元组
    """
    theta = np.linspace(0, 2 * np.pi, resolution, endpoint=False)

    # 底面圆周顶点
    base_x = radius * np.cos(theta) + center[0]
    base_y = radius * np.sin(theta) + center[1]
    base_z = np.full(resolution, center[2] - height / 2)

    # 顶点和底面中心
    apex = center.copy()
    apex[2] += height / 2
    base_center = center.copy()
    base_center[2] -= height / 2

    # 组合顶点
    vertices = []
    # 底面圆周 (0 to resolution-1)
    for i in range(resolution):
        vertices.append([base_x[i], base_y[i], base_z[i]])
    # 顶点 (resolution)
    vertices.append(apex.tolist())
    # 底面中心 (resolution + 1)
    vertices.append(base_center.tolist())

    vertices = np.array(vertices)

    # 生成面
    faces = []
    apex_idx = resolution
    base_center_idx = resolution + 1

    for i in range(resolution):
        next_i = (i + 1) % resolution

        # 侧面
        faces.append([i, next_i, apex_idx])

        # 底面
        faces.append([base_center_idx, next_i, i])

    return vertices, np.array(faces)


def pyramid_mesh(
    center: np.ndarray,
    base_size: float = 1.0,
    height: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成四棱锥网格。

    Generate pyramid mesh.

    Args:
        center: 底面中心坐标 [x, y, z]
        base_size: 底面边长
        height: 高度

    Returns:
        (vertices, faces) 元组
    """
    half = base_size / 2
    cx, cy, cz = center

    vertices = np.array([
        # 底面四角
        [cx - half, cy - half, cz - height / 2],  # 0
        [cx + half, cy - half, cz - height / 2],  # 1
        [cx + half, cy + half, cz - height / 2],  # 2
        [cx - half, cy + half, cz - height / 2],  # 3
        # 顶点
        [cx, cy, cz + height / 2],  # 4
    ])

    faces = np.array([
        # 底面
        [0, 2, 1], [0, 3, 2],
        # 侧面
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4],
    ])

    return vertices, faces


def tetrahedron_mesh(
    center: np.ndarray,
    size: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成正四面体网格。

    Generate tetrahedron mesh.

    Args:
        center: 中心坐标 [x, y, z]
        size: 边长

    Returns:
        (vertices, faces) 元组
    """
    # 正四面体顶点（单位边长）
    a = size / np.sqrt(2)
    vertices = np.array([
        [1, 0, -1 / np.sqrt(2)],
        [-1, 0, -1 / np.sqrt(2)],
        [0, 1, 1 / np.sqrt(2)],
        [0, -1, 1 / np.sqrt(2)],
    ]) * a / 2

    # 平移到中心
    vertices = vertices + center

    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 1],
        [1, 3, 2],
    ])

    return vertices, faces


def octahedron_mesh(
    center: np.ndarray,
    size: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成正八面体网格。

    Generate octahedron mesh.

    Args:
        center: 中心坐标 [x, y, z]
        size: 顶点到中心的距离

    Returns:
        (vertices, faces) 元组
    """
    r = size / 2
    cx, cy, cz = center

    vertices = np.array([
        [cx + r, cy, cz],      # 0: +x
        [cx - r, cy, cz],      # 1: -x
        [cx, cy + r, cz],      # 2: +y
        [cx, cy - r, cz],      # 3: -y
        [cx, cy, cz + r],      # 4: +z
        [cx, cy, cz - r],      # 5: -z
    ])

    faces = np.array([
        # 上半部分
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        # 下半部分
        [0, 5, 2],
        [2, 5, 1],
        [1, 5, 3],
        [3, 5, 0],
    ])

    return vertices, faces


def torus_mesh(
    center: np.ndarray,
    major_radius: float = 0.5,
    minor_radius: float = 0.2,
    major_resolution: int = 20,
    minor_resolution: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成圆环网格。

    Generate torus mesh.

    Args:
        center: 中心坐标 [x, y, z]
        major_radius: 主半径（圆环中心到管道中心）
        minor_radius: 次半径（管道半径）
        major_resolution: 主圆分辨率
        minor_resolution: 管道圆分辨率

    Returns:
        (vertices, faces) 元组
    """
    u = np.linspace(0, 2 * np.pi, major_resolution, endpoint=False)
    v = np.linspace(0, 2 * np.pi, minor_resolution, endpoint=False)

    vertices = []
    for i, ui in enumerate(u):
        for j, vi in enumerate(v):
            x = (major_radius + minor_radius * np.cos(vi)) * np.cos(ui) + center[0]
            y = (major_radius + minor_radius * np.cos(vi)) * np.sin(ui) + center[1]
            z = minor_radius * np.sin(vi) + center[2]
            vertices.append([x, y, z])

    vertices = np.array(vertices)

    faces = []
    for i in range(major_resolution):
        next_i = (i + 1) % major_resolution
        for j in range(minor_resolution):
            next_j = (j + 1) % minor_resolution

            p1 = i * minor_resolution + j
            p2 = i * minor_resolution + next_j
            p3 = next_i * minor_resolution + j
            p4 = next_i * minor_resolution + next_j

            faces.append([p1, p2, p3])
            faces.append([p2, p4, p3])

    return vertices, np.array(faces)


def get_mesh_for_shape(
    shape: str,
    center: np.ndarray,
    size: float = 0.5,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据形状名称获取网格数据。

    Get mesh data by shape name.

    Args:
        shape: 形状名称
        center: 中心坐标
        size: 尺寸
        **kwargs: 其他参数

    Returns:
        (vertices, faces) 元组
    """
    shape = shape.lower()

    if shape == "sphere":
        return sphere_mesh(center, radius=size, **kwargs)
    elif shape in ["cube", "box"]:
        return cube_mesh(center, size=size * 2)
    elif shape == "cylinder":
        return cylinder_mesh(center, radius=size, height=size * 2, **kwargs)
    elif shape == "cone":
        return cone_mesh(center, radius=size, height=size * 2, **kwargs)
    elif shape in ["pyramid", "square_pyramid"]:
        return pyramid_mesh(center, base_size=size * 2, height=size * 2)
    elif shape == "tetrahedron":
        return tetrahedron_mesh(center, size=size * 2)
    elif shape == "octahedron":
        return octahedron_mesh(center, size=size * 2)
    elif shape == "torus":
        return torus_mesh(center, major_radius=size, minor_radius=size * 0.3, **kwargs)
    elif shape in ["cuboid", "rectangular_prism"]:
        # 长方体：默认为 2:1:1 比例
        return cuboid_mesh(center, size * 2, size, size)
    elif shape in ["triangular_prism", "prism"]:
        return triangular_prism_mesh(center, size * 2, size * 2)
    elif shape in ["hexagonal_prism", "hex_prism"]:
        return hexagonal_prism_mesh(center, size, size * 2)
    elif shape == "ellipsoid":
        return ellipsoid_mesh(center, size, size * 0.7, size * 0.5)
    else:
        # 默认使用立方体
        return cube_mesh(center, size=size * 2)


def cuboid_mesh(
    center: np.ndarray,
    length: float,
    width: float,
    height: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成长方体网格。

    Generate cuboid mesh.
    """
    hl, hw, hh = length / 2, width / 2, height / 2
    cx, cy, cz = center

    vertices = np.array([
        [cx - hl, cy - hw, cz - hh],
        [cx + hl, cy - hw, cz - hh],
        [cx + hl, cy + hw, cz - hh],
        [cx - hl, cy + hw, cz - hh],
        [cx - hl, cy - hw, cz + hh],
        [cx + hl, cy - hw, cz + hh],
        [cx + hl, cy + hw, cz + hh],
        [cx - hl, cy + hw, cz + hh],
    ])

    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 5, 1], [0, 4, 5],
        [2, 7, 3], [2, 6, 7],
        [0, 7, 4], [0, 3, 7],
        [1, 5, 6], [1, 6, 2],
    ])

    return vertices, faces


def triangular_prism_mesh(
    center: np.ndarray,
    base_size: float,
    height: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成三角棱柱网格。

    Generate triangular prism mesh.
    """
    h = height / 2
    s = base_size / 2

    # 等边三角形顶点
    triangle_h = s * np.sqrt(3) / 2
    cx, cy, cz = center

    vertices = np.array([
        # 底面三角形
        [cx - s, cy - triangle_h / 2, cz - h],
        [cx + s, cy - triangle_h / 2, cz - h],
        [cx, cy + triangle_h / 2, cz - h],
        # 顶面三角形
        [cx - s, cy - triangle_h / 2, cz + h],
        [cx + s, cy - triangle_h / 2, cz + h],
        [cx, cy + triangle_h / 2, cz + h],
    ])

    faces = np.array([
        # 底面
        [0, 2, 1],
        # 顶面
        [3, 4, 5],
        # 侧面
        [0, 1, 4], [0, 4, 3],
        [1, 2, 5], [1, 5, 4],
        [2, 0, 3], [2, 3, 5],
    ])

    return vertices, faces


def hexagonal_prism_mesh(
    center: np.ndarray,
    radius: float,
    height: float,
    resolution: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成六角棱柱网格。

    Generate hexagonal prism mesh.
    """
    h = height / 2
    theta = np.linspace(0, 2 * np.pi, resolution, endpoint=False)

    vertices = []
    cx, cy, cz = center

    # 底面顶点
    for t in theta:
        vertices.append([cx + radius * np.cos(t), cy + radius * np.sin(t), cz - h])
    # 顶面顶点
    for t in theta:
        vertices.append([cx + radius * np.cos(t), cy + radius * np.sin(t), cz + h])
    # 底面中心
    vertices.append([cx, cy, cz - h])
    # 顶面中心
    vertices.append([cx, cy, cz + h])

    vertices = np.array(vertices)

    faces = []
    n = resolution
    bottom_center = 2 * n
    top_center = 2 * n + 1

    for i in range(n):
        next_i = (i + 1) % n

        # 底面
        faces.append([bottom_center, next_i, i])
        # 顶面
        faces.append([top_center, n + i, n + next_i])
        # 侧面
        faces.append([i, next_i, n + i])
        faces.append([next_i, n + next_i, n + i])

    return vertices, np.array(faces)


def ellipsoid_mesh(
    center: np.ndarray,
    a: float,
    b: float,
    c: float,
    resolution: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成椭球网格。

    Generate ellipsoid mesh.

    Args:
        center: 中心坐标
        a, b, c: 三个半轴长度
        resolution: 分辨率

    Returns:
        (vertices, faces) 元组
    """
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    x = a * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = b * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = c * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

    vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            p1 = i * resolution + j
            p2 = i * resolution + (j + 1)
            p3 = (i + 1) * resolution + j
            p4 = (i + 1) * resolution + (j + 1)
            faces.append([p1, p2, p3])
            faces.append([p2, p4, p3])

    return vertices, np.array(faces)
