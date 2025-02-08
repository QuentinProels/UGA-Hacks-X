import numpy as np
import trimesh
from skimage.measure import marching_cubes
import pyvista
def exporter(input_voxel):
    voxel_grid = input_voxel
    vertices, faces, _, _ = marching_cubes(voxel_grid, level=0.5)
    mesh = trimesh.Trimesh(vertices=vertices,faces=faces)
    pv_mesh = pyvista.wrap(mesh)
    smooth = pv_mesh.smooth(n_iter=100, relaxation_factor=0.1)
    smooth.save("outputs/outputTest1.stl")