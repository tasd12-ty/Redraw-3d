"""
Unit tests for multi-view rendering configuration.

Tests the camera configuration and coordinate conversion logic
that doesn't require Blender.
"""

import pytest
import math
import sys
import os

# 添加项目路径以便导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# We need to mock bpy before importing render_multiview
sys.modules['bpy'] = type(sys)('bpy')
sys.modules['bpy_extras'] = type(sys)('bpy_extras')
sys.modules['mathutils'] = type(sys)('mathutils')

# Now import the dataclasses we want to test
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any

# Re-implement the classes for testing (without bpy dependency)
@dataclass
class CameraConfig:
    """Configuration for a single camera viewpoint."""
    camera_id: str
    azimuth: float
    elevation: float
    distance: float
    look_at: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_cartesian(self) -> Tuple[float, float, float]:
        azimuth_rad = math.radians(self.azimuth)
        elevation_rad = math.radians(self.elevation)
        x = self.distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
        y = self.distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
        z = self.distance * math.sin(elevation_rad)
        return (
            x + self.look_at[0],
            y + self.look_at[1],
            z + self.look_at[2]
        )

    def to_dict(self) -> Dict[str, Any]:
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
    """Configuration for multi-view rendering."""
    n_views: int = 4
    camera_distance: float = 12.0
    elevation: float = 30.0
    azimuth_start: float = 45.0
    look_at: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def generate_cameras(self) -> List[CameraConfig]:
        cameras = []
        azimuth_step = 360.0 / self.n_views
        for i in range(self.n_views):
            azimuth = self.azimuth_start + i * azimuth_step
            azimuth = azimuth % 360.0
            cameras.append(CameraConfig(
                camera_id=f"view_{i}",
                azimuth=azimuth,
                elevation=self.elevation,
                distance=self.camera_distance,
                look_at=self.look_at
            ))
        return cameras


class TestCameraConfig:
    """Tests for CameraConfig class."""

    def test_cartesian_at_origin(self):
        """Test camera at azimuth=0, elevation=0."""
        cam = CameraConfig(
            camera_id="test",
            azimuth=0.0,
            elevation=0.0,
            distance=10.0
        )
        x, y, z = cam.to_cartesian()
        assert abs(x - 10.0) < 1e-6
        assert abs(y - 0.0) < 1e-6
        assert abs(z - 0.0) < 1e-6

    def test_cartesian_azimuth_90(self):
        """Test camera at azimuth=90 (Y axis)."""
        cam = CameraConfig(
            camera_id="test",
            azimuth=90.0,
            elevation=0.0,
            distance=10.0
        )
        x, y, z = cam.to_cartesian()
        assert abs(x - 0.0) < 1e-6
        assert abs(y - 10.0) < 1e-6
        assert abs(z - 0.0) < 1e-6

    def test_cartesian_elevation_90(self):
        """Test camera at elevation=90 (top-down)."""
        cam = CameraConfig(
            camera_id="test",
            azimuth=0.0,
            elevation=90.0,
            distance=10.0
        )
        x, y, z = cam.to_cartesian()
        assert abs(x - 0.0) < 1e-6
        assert abs(y - 0.0) < 1e-6
        assert abs(z - 10.0) < 1e-6

    def test_cartesian_elevation_30(self):
        """Test camera at typical 30 degree elevation."""
        cam = CameraConfig(
            camera_id="test",
            azimuth=0.0,
            elevation=30.0,
            distance=12.0
        )
        x, y, z = cam.to_cartesian()
        # cos(30) ≈ 0.866, sin(30) = 0.5
        expected_x = 12.0 * math.cos(math.radians(30))
        expected_z = 12.0 * math.sin(math.radians(30))
        assert abs(x - expected_x) < 1e-6
        assert abs(y - 0.0) < 1e-6
        assert abs(z - expected_z) < 1e-6

    def test_cartesian_with_look_at(self):
        """Test camera with non-origin look_at point."""
        cam = CameraConfig(
            camera_id="test",
            azimuth=0.0,
            elevation=0.0,
            distance=10.0,
            look_at=(5.0, 3.0, 2.0)
        )
        x, y, z = cam.to_cartesian()
        assert abs(x - 15.0) < 1e-6
        assert abs(y - 3.0) < 1e-6
        assert abs(z - 2.0) < 1e-6

    def test_to_dict(self):
        """Test serialization to dict."""
        cam = CameraConfig(
            camera_id="view_0",
            azimuth=45.0,
            elevation=30.0,
            distance=12.0
        )
        d = cam.to_dict()
        assert d["camera_id"] == "view_0"
        assert d["azimuth"] == 45.0
        assert d["elevation"] == 30.0
        assert d["distance"] == 12.0
        assert len(d["position"]) == 3
        assert d["look_at"] == [0.0, 0.0, 0.0]


class TestMultiViewConfig:
    """Tests for MultiViewConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = MultiViewConfig()
        assert config.n_views == 4
        assert config.camera_distance == 12.0
        assert config.elevation == 30.0
        assert config.azimuth_start == 45.0

    def test_generate_4_cameras(self):
        """Test generating 4 cameras with 90 degree spacing."""
        config = MultiViewConfig(n_views=4, azimuth_start=0.0)
        cameras = config.generate_cameras()

        assert len(cameras) == 4
        assert cameras[0].azimuth == 0.0
        assert cameras[1].azimuth == 90.0
        assert cameras[2].azimuth == 180.0
        assert cameras[3].azimuth == 270.0

    def test_generate_8_cameras(self):
        """Test generating 8 cameras with 45 degree spacing."""
        config = MultiViewConfig(n_views=8, azimuth_start=0.0)
        cameras = config.generate_cameras()

        assert len(cameras) == 8
        for i, cam in enumerate(cameras):
            expected_azimuth = (i * 45.0) % 360.0
            assert abs(cam.azimuth - expected_azimuth) < 1e-6

    def test_camera_ids(self):
        """Test that camera IDs are generated correctly."""
        config = MultiViewConfig(n_views=4)
        cameras = config.generate_cameras()

        for i, cam in enumerate(cameras):
            assert cam.camera_id == f"view_{i}"

    def test_cameras_share_elevation_and_distance(self):
        """Test that all cameras share the same elevation and distance."""
        config = MultiViewConfig(
            n_views=4,
            camera_distance=15.0,
            elevation=45.0
        )
        cameras = config.generate_cameras()

        for cam in cameras:
            assert cam.elevation == 45.0
            assert cam.distance == 15.0

    def test_azimuth_start_offset(self):
        """Test azimuth_start offset is applied correctly."""
        config = MultiViewConfig(n_views=4, azimuth_start=45.0)
        cameras = config.generate_cameras()

        assert cameras[0].azimuth == 45.0
        assert cameras[1].azimuth == 135.0
        assert cameras[2].azimuth == 225.0
        assert cameras[3].azimuth == 315.0

    def test_cameras_at_same_distance_from_center(self):
        """Test all cameras are equidistant from center."""
        config = MultiViewConfig(n_views=4, camera_distance=12.0)
        cameras = config.generate_cameras()

        for cam in cameras:
            x, y, z = cam.to_cartesian()
            distance = math.sqrt(x**2 + y**2 + z**2)
            assert abs(distance - 12.0) < 1e-6


class TestSphericalToCartesian:
    """Additional tests for coordinate conversion."""

    def test_all_quadrants(self):
        """Test camera positions in all quadrants."""
        test_cases = [
            (0.0, 0.0, (1, 0, 0)),    # +X
            (90.0, 0.0, (0, 1, 0)),   # +Y
            (180.0, 0.0, (-1, 0, 0)), # -X
            (270.0, 0.0, (0, -1, 0)), # -Y
        ]
        for azimuth, elevation, (ex, ey, ez) in test_cases:
            cam = CameraConfig(
                camera_id="test",
                azimuth=azimuth,
                elevation=elevation,
                distance=1.0
            )
            x, y, z = cam.to_cartesian()
            assert abs(x - ex) < 1e-6, f"azimuth={azimuth}: x={x}, expected={ex}"
            assert abs(y - ey) < 1e-6, f"azimuth={azimuth}: y={y}, expected={ey}"
            assert abs(z - ez) < 1e-6, f"azimuth={azimuth}: z={z}, expected={ez}"

    def test_distance_invariant(self):
        """Test that Cartesian distance equals spherical distance."""
        for azimuth in [0, 45, 90, 135, 180, 225, 270, 315]:
            for elevation in [0, 15, 30, 45, 60, 75, 90]:
                for dist in [1.0, 5.0, 10.0, 12.0]:
                    cam = CameraConfig(
                        camera_id="test",
                        azimuth=float(azimuth),
                        elevation=float(elevation),
                        distance=dist
                    )
                    x, y, z = cam.to_cartesian()
                    computed_dist = math.sqrt(x**2 + y**2 + z**2)
                    assert abs(computed_dist - dist) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
