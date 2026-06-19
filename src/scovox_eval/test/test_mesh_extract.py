"""Unit tests for mesh extraction pipeline."""

import numpy as np
import pytest

from scovox_eval.mesh_extract import (
    extract_mesh,
    occupancy_grid_from_pointcloud,
)


def _try_import_skimage():
    try:
        from skimage.measure import marching_cubes  # noqa: F401
        return True
    except ImportError:
        return False


def _make_sphere_voxels(radius: float = 1.0, resolution: float = 0.05):
    """Generate occupied voxels forming a sphere."""
    coords = np.arange(-radius, radius + resolution, resolution)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    dist = np.linalg.norm(points, axis=1)
    # Occupied on the shell
    mask = np.abs(dist - radius) < resolution * 2
    return points[mask], np.ones(mask.sum(), dtype=np.float32)


class TestOccupancyGrid:
    def test_single_point(self):
        points = np.array([[1.0, 2.0, 3.0]])
        occ = np.array([0.9])
        grid, origin = occupancy_grid_from_pointcloud(points, occ, 0.1)
        assert grid.ndim == 3
        assert grid.max() == pytest.approx(0.9)

    def test_shape_reasonable(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        occ = np.ones(3, dtype=np.float32)
        grid, origin = occupancy_grid_from_pointcloud(points, occ, 0.1)
        # Expect roughly 10+3 cells in x and y, small in z
        assert grid.shape[0] >= 10
        assert grid.shape[1] >= 10


class TestMarchingCubes:
    @pytest.mark.skipif(not _try_import_skimage(), reason="scikit-image not installed")
    def test_sphere_mesh(self):
        points, occ = _make_sphere_voxels(0.5, 0.05)
        grid, origin = occupancy_grid_from_pointcloud(points, occ, 0.05)
        verts, faces = extract_mesh(grid, origin, 0.05, threshold=0.5)
        assert len(verts) > 0
        assert len(faces) > 0
        # Vertices should be near the sphere surface
        dists = np.linalg.norm(verts, axis=1)
        assert np.abs(dists.mean() - 0.5) < 0.15
