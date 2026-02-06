"""
3D 形状基元库。

3D Shape Primitives Library.
"""

from .primitives import (
    sphere_mesh,
    cube_mesh,
    cylinder_mesh,
    cone_mesh,
    pyramid_mesh,
    get_mesh_for_shape,
    SIZE_SCALE_MAP,
)

__all__ = [
    "sphere_mesh",
    "cube_mesh",
    "cylinder_mesh",
    "cone_mesh",
    "pyramid_mesh",
    "get_mesh_for_shape",
    "SIZE_SCALE_MAP",
]
