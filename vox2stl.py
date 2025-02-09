import numpy as np
import trimesh
from skimage.measure import marching_cubes
import pyvista
def exporter(input_voxel):
    voxel_grid = input_voxel
    # Ensure the voxel grid is within the range [0, 1]
    voxel_grid = np.clip(voxel_grid, 0, 1)
    
    # Determine the min and max values of the voxel grid
    voxel_min = voxel_grid.min()
    voxel_max = voxel_grid.max()

    # Set the level within the voxel range for marching cubes
    level = (voxel_min + voxel_max) / 2  # Middle of the range

    # Perform marching cubes
    vertices, faces, _, _ = marching_cubes(voxel_grid, level=level)
    mesh = trimesh.Trimesh(vertices=vertices,faces=faces)
    pv_mesh = pyvista.wrap(mesh)
    smooth = pv_mesh.smooth(n_iter=100, relaxation_factor=0.1)
    smooth.save("outputs/outputTest1.stl")